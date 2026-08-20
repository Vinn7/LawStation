import type { ChatMessage, Conversation, IndexStatus, User } from './types';

const API = '/api';

export class ApiError extends Error {
  constructor(message: string, public readonly status: number) {
    super(message);
    this.name = 'ApiError';
  }
}

async function request<T>(path: string, userId?: string, init: RequestInit = {}): Promise<T> {
  const headers = new Headers(init.headers);
  if (userId) headers.set('X-User-ID', userId);
  const response = await fetch(`${API}${path}`, { ...init, headers });
  if (!response.ok) {
    let message = `请求失败（${response.status}）`;
    try {
      const detail = (await response.json()) as { detail?: string };
      if (detail.detail) message = detail.detail;
    } catch {
      // Keep the status-based message when the response is not JSON.
    }
    throw new ApiError(message, response.status);
  }
  return response.json() as Promise<T>;
}

export const api = {
  users: (signal?: AbortSignal) => request<User[]>('/users', undefined, { signal }),
  indexStatus: (signal?: AbortSignal) => request<IndexStatus>('/index/status', undefined, { signal }),
  conversations: (userId: string, signal?: AbortSignal) =>
    request<Conversation[]>('/conversations', userId, { signal }),
  createConversation: (userId: string, signal?: AbortSignal) =>
    request<Conversation>('/conversations', userId, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ title: '法律咨询' }),
      signal,
    }),
  messages: (userId: string, conversationId: string, signal?: AbortSignal) =>
    request<ChatMessage[]>(`/conversations/${conversationId}/messages`, userId, { signal }),
  streamMessage: (userId: string, conversationId: string, content: string, signal: AbortSignal) =>
    fetch(`${API}/conversations/${conversationId}/messages/stream`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json', 'X-User-ID': userId },
      body: JSON.stringify({ content }),
      signal,
    }),
};
