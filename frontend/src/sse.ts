import type { SseEvent, SseEventName } from './types';

export function parseSseBlock(block: string): SseEvent | null {
  let event = 'message' as SseEventName;
  let id: number | undefined;
  const data: string[] = [];

  for (const rawLine of block.split(/\r?\n/)) {
    if (!rawLine || rawLine.startsWith(':')) continue;
    const separator = rawLine.indexOf(':');
    const field = separator === -1 ? rawLine : rawLine.slice(0, separator);
    let value = separator === -1 ? '' : rawLine.slice(separator + 1);
    if (value.startsWith(' ')) value = value.slice(1);
    if (field === 'event') event = value as SseEventName;
    if (field === 'id' && /^\d+$/.test(value)) id = Number(value);
    if (field === 'data') data.push(value);
  }

  if (data.length === 0 || event === ('message' as SseEventName)) return null;
  const rawData = data.join('\n');
  try {
    return { id, event, data: JSON.parse(rawData) };
  } catch {
    return { id, event, data: rawData };
  }
}

export async function consumeSse(
  response: Response,
  onEvent: (event: SseEvent) => void,
): Promise<void> {
  if (!response.ok) {
    let message = `对话请求失败（${response.status}）`;
    try {
      const body = (await response.json()) as { detail?: string };
      if (body.detail) message = body.detail;
    } catch {
      // Keep the status-based message when the response is not JSON.
    }
    throw new Error(message);
  }
  if (!response.body) throw new Error('浏览器未收到流式响应');

  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = '';

  while (true) {
    const { value, done } = await reader.read();
    buffer += decoder.decode(value, { stream: !done });
    const blocks = buffer.split(/\r?\n\r?\n/);
    buffer = blocks.pop() ?? '';
    for (const block of blocks) {
      const parsed = parseSseBlock(block);
      if (parsed) onEvent(parsed);
    }
    if (done) break;
  }

  if (buffer.trim()) {
    const parsed = parseSseBlock(buffer);
    if (parsed) onEvent(parsed);
  }
}
