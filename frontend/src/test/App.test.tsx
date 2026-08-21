import { act, render, screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { afterEach, describe, expect, it, vi } from 'vitest';
import App from '../App';

const users = [
  { id: 'user-a', tenant_id: 'tenant', name: '张三', created_at: '2026-08-20T10:00:00Z' },
  { id: 'user-b', tenant_id: 'tenant', name: '李四', created_at: '2026-08-20T10:00:00Z' },
];

function json(body: unknown, status = 200): Response {
  return new Response(JSON.stringify(body), { status, headers: { 'Content-Type': 'application/json' } });
}

describe('App user isolation', () => {
  afterEach(() => vi.restoreAllMocks());

  it('clears the previous conversation list when switching users', async () => {
    const fetchMock = vi.spyOn(globalThis, 'fetch').mockImplementation(async (input, init) => {
      const url = String(input);
      if (url.endsWith('/api/users')) return json(users);
      if (url.endsWith('/api/index/status')) return json({ status: 'ready', message: '就绪', dense_enabled: true });
      if (url.endsWith('/api/conversations')) {
        const headers = new Headers(init?.headers);
        if (headers.get('X-User-ID') === 'user-a') {
          return json([{ id: 'conversation-a', tenant_id: 'tenant', user_id: 'user-a', title: '张三的劳动争议', created_at: '2026-08-20T10:00:00Z', updated_at: '2026-08-20T10:00:00Z' }]);
        }
        return json([{ id: 'conversation-b', tenant_id: 'tenant', user_id: 'user-b', title: '李四的租赁问题', created_at: '2026-08-20T11:00:00Z', updated_at: '2026-08-20T11:00:00Z' }]);
      }
      throw new Error(`Unexpected request: ${url}`);
    });
    const user = userEvent.setup();
    render(<App />);

    expect(await screen.findByText('张三的劳动争议')).toBeInTheDocument();
    await user.selectOptions(screen.getByLabelText('当前用户'), 'user-b');
    expect(await screen.findByText('李四的租赁问题')).toBeInTheDocument();
    expect(screen.queryByText('张三的劳动争议')).not.toBeInTheDocument();
    await waitFor(() => expect(fetchMock).toHaveBeenCalled());
  });

  it('creates a conversation and consumes the complete stream lifecycle', async () => {
    const encoder = new TextEncoder();
    const streamBody = [
      'event: message_start\ndata: {"request_id":"r1"}\n\n',
      'event: tool_call_start\ndata: {"name":"search_laws","arguments":{"query":"private"}}\n\n',
      'event: tool_call_result\ndata: {"name":"search_laws","status":"success"}\n\n',
      'event: token\ndata: "根据相关法律，"\n\n',
      'event: token\ndata: "可以依法主张权利。"\n\n',
      'event: citations\ndata: [{"document_id":"law-1","law_name":"劳动合同法","article_number":"第八十二条"}]\n\n',
      'event: memory_status\ndata: {"status":"pending","job_id":"memory-1"}\n\n',
      'event: message_end\ndata: {"message_id":"assistant-1"}\n\n',
    ];
    vi.spyOn(globalThis, 'fetch').mockImplementation(async (input, init) => {
      const url = String(input);
      if (url.endsWith('/api/users')) return json(users.slice(0, 1));
      if (url.endsWith('/api/index/status')) return json({ status: 'ready', message: '就绪', dense_enabled: true });
      if (url.endsWith('/api/conversations') && init?.method === 'POST') {
        return json({ id: 'new-conversation', tenant_id: 'tenant', user_id: 'user-a', title: '法律咨询', created_at: '2026-08-20T12:00:00Z', updated_at: '2026-08-20T12:00:00Z' }, 201);
      }
      if (url.endsWith('/api/conversations')) return json([]);
      if (url.endsWith('/messages/stream')) {
        return new Response(new ReadableStream<Uint8Array>({
          start(controller) {
            streamBody.forEach((block) => controller.enqueue(encoder.encode(block)));
            controller.close();
          },
        }), { status: 200, headers: { 'Content-Type': 'text/event-stream' } });
      }
      if (url.endsWith('/api/memory-jobs/memory-1')) {
        return json({ id: 'memory-1', status: 'completed', candidate_count: 1, summary_updated: false });
      }
      throw new Error(`Unexpected request: ${url}`);
    });
    const user = userEvent.setup();
    render(<App />);
    const textbox = await screen.findByRole('textbox', { name: '法律咨询问题' });

    await user.type(textbox, '公司未签劳动合同怎么办？{enter}');

    expect(await screen.findByText('根据相关法律，可以依法主张权利。')).toBeInTheDocument();
    expect(screen.getByText('劳动合同法 第八十二条')).toBeInTheDocument();
    expect(await screen.findByText('本轮记忆整理已完成，新增 1 条记忆并已生效', {}, { timeout: 2500 })).toBeInTheDocument();
    await waitFor(() => expect(screen.queryByRole('button', { name: '停止生成' })).not.toBeInTheDocument());
  });

  it('keeps user A streaming in the background while user B is visible', async () => {
    const encoder = new TextEncoder();
    let streamController: ReadableStreamDefaultController<Uint8Array> | null = null;
    let streamSignal: AbortSignal | null = null;
    vi.spyOn(globalThis, 'fetch').mockImplementation(async (input, init) => {
      const url = String(input);
      const headers = new Headers(init?.headers);
      const owner = headers.get('X-User-ID');
      if (url.endsWith('/api/users')) return json(users);
      if (url.endsWith('/api/index/status')) return json({ status: 'ready', message: '就绪', dense_enabled: true });
      if (url.endsWith('/api/conversations')) {
        return json(owner === 'user-a'
          ? [{ id: 'conversation-a', tenant_id: 'tenant', user_id: 'user-a', title: '张三会话', created_at: '2026-08-20T10:00:00Z', updated_at: '2026-08-20T10:00:00Z' }]
          : [{ id: 'conversation-b', tenant_id: 'tenant', user_id: 'user-b', title: '李四会话', created_at: '2026-08-20T10:00:00Z', updated_at: '2026-08-20T10:00:00Z' }]);
      }
      if (url.endsWith('/messages/stream')) {
        streamSignal = init?.signal as AbortSignal;
        return new Response(new ReadableStream<Uint8Array>({
          start(controller) {
            streamController = controller;
            controller.enqueue(encoder.encode('event: message_start\ndata: {"request_id":"r-a"}\n\n'));
            controller.enqueue(encoder.encode('event: agent_status\ndata: {"agent":"case_analyst","status":"analyzing","message":"正在分析案情"}\n\n'));
          },
        }), { status: 200, headers: { 'Content-Type': 'text/event-stream' } });
      }
      if (url.endsWith('/messages')) return json([]);
      throw new Error(`Unexpected request: ${url}`);
    });
    const user = userEvent.setup();
    render(<App />);

    await user.click(await screen.findByText('张三会话'));
    await user.type(await screen.findByRole('textbox', { name: '法律咨询问题' }), '问题 A{enter}');
    expect(await screen.findByText('正在分析案情')).toBeInTheDocument();

    await user.selectOptions(screen.getByLabelText('当前用户'), 'user-b');
    expect(await screen.findByText('李四会话')).toBeInTheDocument();
    expect(streamSignal).not.toBeNull();
    expect((streamSignal as unknown as AbortSignal).aborted).toBe(false);

    await act(async () => {
      streamController?.enqueue(encoder.encode('event: token\ndata: "A 的后台回答"\n\n'));
    });
    expect(screen.queryByText('A 的后台回答')).not.toBeInTheDocument();

    await user.selectOptions(screen.getByLabelText('当前用户'), 'user-a');
    await user.click(await screen.findByText('张三会话'));
    expect(await screen.findByText('A 的后台回答')).toBeInTheDocument();
    expect(screen.getByRole('button', { name: '停止生成' })).toBeInTheDocument();
    expect(streamController).not.toBeNull();
    (streamController as unknown as ReadableStreamDefaultController<Uint8Array>).close();
  });

  it('clears the tool spinner when a no-match result moves to general analysis', async () => {
    const encoder = new TextEncoder();
    let streamController: ReadableStreamDefaultController<Uint8Array> | null = null;
    vi.spyOn(globalThis, 'fetch').mockImplementation(async (input) => {
      const url = String(input);
      if (url.endsWith('/api/users')) return json(users.slice(0, 1));
      if (url.endsWith('/api/index/status')) return json({ status: 'ready', message: '就绪', dense_enabled: true });
      if (url.endsWith('/api/conversations')) {
        return json([{ id: 'conversation-a', tenant_id: 'tenant', user_id: 'user-a', title: '无法条测试', created_at: '2026-08-20T10:00:00Z', updated_at: '2026-08-20T10:00:00Z' }]);
      }
      if (url.endsWith('/messages/stream')) {
        return new Response(new ReadableStream<Uint8Array>({
          start(controller) {
            streamController = controller;
            controller.enqueue(encoder.encode('event: message_start\ndata: {"request_id":"r-no-match"}\n\n'));
            controller.enqueue(encoder.encode('event: tool_call_start\ndata: {"name":"search_laws","status":"started"}\n\n'));
          },
        }), { status: 200, headers: { 'Content-Type': 'text/event-stream' } });
      }
      if (url.endsWith('/messages')) return json([]);
      throw new Error(`Unexpected request: ${url}`);
    });
    const user = userEvent.setup();
    render(<App />);

    await user.click(await screen.findByText('无法条测试'));
    await user.type(await screen.findByRole('textbox', { name: '法律咨询问题' }), '样本中没有法条怎么办？{enter}');
    expect(await screen.findByText('正在检索相关法律条款…')).toBeInTheDocument();

    await act(async () => {
      streamController?.enqueue(encoder.encode('event: tool_call_result\ndata: {"name":"search_laws","status":"success"}\n\n'));
      streamController?.enqueue(encoder.encode('event: agent_status\ndata: {"agent":"legal_counsel","status":"drafting","message":"未检索到可引用法条，将基于案情形成一般性分析"}\n\n'));
    });

    expect(await screen.findByText('未检索到可引用法条，将基于案情形成一般性分析')).toBeInTheDocument();
    expect(screen.queryByText('正在检索相关法律条款…')).not.toBeInTheDocument();

    await act(async () => {
      streamController?.enqueue(encoder.encode('event: token\ndata: "一般性分析结果"\n\n'));
      streamController?.enqueue(encoder.encode('event: message_end\ndata: {"message_id":"assistant-no-match"}\n\n'));
      streamController?.close();
    });
    expect(await screen.findByText('一般性分析结果')).toBeInTheDocument();
    expect(screen.queryByText('未检索到可引用法条，将基于案情形成一般性分析')).not.toBeInTheDocument();
    expect(screen.queryByText(/第.+条/)).not.toBeInTheDocument();
  });
});
