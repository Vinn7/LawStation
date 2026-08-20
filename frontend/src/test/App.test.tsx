import { render, screen, waitFor } from '@testing-library/react';
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
      'event: memory_status\ndata: {"compressed":true}\n\n',
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
      throw new Error(`Unexpected request: ${url}`);
    });
    const user = userEvent.setup();
    render(<App />);
    const textbox = await screen.findByRole('textbox', { name: '法律咨询问题' });

    await user.type(textbox, '公司未签劳动合同怎么办？{enter}');

    expect(await screen.findByText('根据相关法律，可以依法主张权利。')).toBeInTheDocument();
    expect(screen.getByText('劳动合同法 第八十二条')).toBeInTheDocument();
    expect(screen.getByText('本轮对话已完成摘要与记忆整理')).toBeInTheDocument();
    await waitFor(() => expect(screen.queryByRole('button', { name: '停止生成' })).not.toBeInTheDocument());
  });
});
