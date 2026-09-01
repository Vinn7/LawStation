import type {
  AgentRun,
  ChatMessage,
  Conversation,
  DialogueScenario,
  IndexStatus,
  MemoryJob,
  ScenarioDataset,
  ScenarioRunOutcome,
  ScenarioSummary,
  User,
  UserMemory,
} from './types';

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
  if (response.status === 204) return undefined as T;
  return response.json() as Promise<T>;
}

export const api = {
  users: (signal?: AbortSignal) => request<User[]>('/users', undefined, { signal }),
  indexStatus: (signal?: AbortSignal) => request<IndexStatus>('/index/status', undefined, { signal }),
  conversations: (userId: string, signal?: AbortSignal) =>
    request<Conversation[]>('/conversations', userId, { signal }),
  createConversation: (userId: string, title = '法律咨询', signal?: AbortSignal) =>
    request<Conversation>('/conversations', userId, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ title }),
      signal,
    }),
  messages: (userId: string, conversationId: string, signal?: AbortSignal) =>
    request<ChatMessage[]>(`/conversations/${conversationId}/messages`, userId, { signal }),
  memories: (userId: string, query = '', signal?: AbortSignal) =>
    request<UserMemory[]>(`/memories${query ? `?${query}` : ''}`, userId, { signal }),
  confirmMemory: (userId: string, memoryId: string, version: number) =>
    request<{ ok: boolean }>(`/memories/${memoryId}/confirm`, userId, {
      method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ version }),
    }),
  rejectMemory: (userId: string, memoryId: string, version: number) =>
    request<{ ok: boolean }>(`/memories/${memoryId}/reject`, userId, {
      method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ version }),
    }),
  updateMemory: (userId: string, memoryId: string, content: string, version: number) =>
    request<{ ok: boolean }>(`/memories/${memoryId}`, userId, {
      method: 'PATCH', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ content, version }),
    }),
  deleteMemory: (userId: string, memoryId: string) =>
    request<void>(`/memories/${memoryId}`, userId, { method: 'DELETE' }),
  memoryJob: (userId: string, jobId: string, signal?: AbortSignal) =>
    request<MemoryJob>(`/memory-jobs/${jobId}`, userId, { signal }),
  messageFeedback: (userId: string, messageId: string, score: -1 | 1, comment = '') =>
    request<{ ok: boolean; sync_status: string }>(`/messages/${messageId}/feedback`, userId, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ score, comment }),
    }),
  createRun: (userId: string, conversationId: string, content: string) =>
    request<AgentRun>(`/conversations/${conversationId}/runs`, userId, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ content }),
    }),
  agentRun: (userId: string, runId: string, signal?: AbortSignal) =>
    request<AgentRun>(`/agent-runs/${runId}`, userId, { signal }),
  activeRun: (userId: string, conversationId: string, signal?: AbortSignal) =>
    request<AgentRun | null>(`/conversations/${conversationId}/active-run`, userId, { signal }),
  cancelRun: (userId: string, runId: string) =>
    request<AgentRun>(`/agent-runs/${runId}/cancel`, userId, { method: 'POST' }),
  runEvents: (userId: string, runId: string, afterSequence: number, signal: AbortSignal) =>
    fetch(`${API}/agent-runs/${runId}/events?after_sequence=${afterSequence}`, {
      headers: { 'X-User-ID': userId },
      signal,
    }),
  scenarioDatasets: (userId: string, signal?: AbortSignal) =>
    request<ScenarioDataset[]>('/test-scenarios/datasets', userId, { signal }),
  scenarioSummaries: (userId: string, datasetId: string, signal?: AbortSignal) =>
    request<ScenarioSummary[]>(`/test-scenarios/datasets/${encodeURIComponent(datasetId)}/scenarios`, userId, { signal }),
  scenario: (userId: string, datasetId: string, scenarioId: string, signal?: AbortSignal) =>
    request<DialogueScenario>(
      `/test-scenarios/datasets/${encodeURIComponent(datasetId)}/scenarios/${encodeURIComponent(scenarioId)}`,
      userId,
      { signal },
    ),
  scenarioOutcome: (userId: string, runId: string, signal?: AbortSignal) =>
    request<ScenarioRunOutcome>(`/test-scenarios/agent-runs/${runId}/outcome`, userId, { signal }),
  deleteScenarioConversation: (userId: string, conversationId: string) =>
    request<void>(`/test-scenarios/conversations/${conversationId}`, userId, { method: 'DELETE' }),
  streamMessage: (userId: string, conversationId: string, content: string, signal: AbortSignal) =>
    fetch(`${API}/conversations/${conversationId}/messages/stream`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json', 'X-User-ID': userId },
      body: JSON.stringify({ content }),
      signal,
    }),
};
