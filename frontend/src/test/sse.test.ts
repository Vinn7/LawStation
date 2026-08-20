import { describe, expect, it, vi } from 'vitest';
import { consumeSse, parseSseBlock } from '../sse';

describe('SSE parser', () => {
  it('parses JSON data and ignores comments', () => {
    expect(parseSseBlock(': heartbeat\nevent: token\ndata: "法"')).toEqual({ event: 'token', data: '法' });
  });

  it('handles events split across stream chunks', async () => {
    const encoder = new TextEncoder();
    const body = new ReadableStream<Uint8Array>({
      start(controller) {
        controller.enqueue(encoder.encode('event: message_start\ndata: {"request_id":"r1"}\n\nevent: tok'));
        controller.enqueue(encoder.encode('en\ndata: "法律"\n\nevent: tool_call_result\ndata: {"name":"search_laws",'));
        controller.enqueue(encoder.encode('"status":"success"}\n\n'));
        controller.close();
      },
    });
    const onEvent = vi.fn();

    await consumeSse(new Response(body, { status: 200 }), onEvent);

    expect(onEvent).toHaveBeenCalledTimes(3);
    expect(onEvent).toHaveBeenNthCalledWith(2, { event: 'token', data: '法律' });
    expect(onEvent).toHaveBeenNthCalledWith(3, {
      event: 'tool_call_result',
      data: { name: 'search_laws', status: 'success' },
    });
  });

  it('rejects a failed HTTP response', async () => {
    await expect(consumeSse(
      new Response(JSON.stringify({ detail: '会话不存在或无权访问' }), {
        status: 404,
        headers: { 'Content-Type': 'application/json' },
      }),
      vi.fn(),
    )).rejects.toThrow('会话不存在或无权访问');
  });
});
