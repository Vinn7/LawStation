export interface User {
  id: string;
  tenant_id: string;
  name: string;
  created_at: string;
}

export interface Conversation {
  id: string;
  tenant_id: string;
  user_id: string;
  title: string;
  created_at: string;
  updated_at: string;
}

export type MessageRole = 'user' | 'assistant';
export type MessageStatus = 'complete' | 'streaming' | 'interrupted' | 'error';

export interface ChatMessage {
  id: string;
  role: MessageRole;
  content: string;
  status?: MessageStatus;
  created_at?: string;
  citations?: Citation[];
}

export interface Citation {
  document_id?: string;
  law_name?: string;
  article_number?: string;
}

export type IndexState = 'checking' | 'building' | 'ready' | 'degraded' | 'failed';

export interface IndexStatus {
  status: IndexState;
  message: string;
  progress?: number;
  dense_enabled?: boolean;
  source_file?: string;
}

export interface ToolActivity {
  name: string;
  status: 'running' | 'success' | 'failed' | 'timeout';
}

export type SseEventName =
  | 'message_start'
  | 'tool_call_start'
  | 'tool_call_result'
  | 'token'
  | 'citations'
  | 'memory_status'
  | 'message_end'
  | 'error';

export interface SseEvent<T = unknown> {
  event: SseEventName;
  data: T;
}
