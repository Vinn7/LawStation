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
  feedback_score?: -1 | 1 | null;
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
  reranker_enabled?: boolean;
  reranker_status?: 'disabled' | 'checking' | 'ready' | 'degraded' | 'cooldown';
  reranker_provider?: string;
  reranker_model?: string;
  reranker_model_digest?: string;
  reranker_managed?: boolean;
  reranker_candidate_count?: number;
  reranker_message?: string;
}

export interface ToolActivity {
  name: string;
  status: 'running' | 'success' | 'failed' | 'timeout';
}

export type MemoryScope = 'user' | 'conversation';
export type MemoryStatus = 'pending' | 'active' | 'superseded' | 'rejected' | 'expired';

export interface UserMemory {
  id: string;
  tenant_id: string;
  user_id: string;
  conversation_id: string;
  memory_type: string;
  scope: MemoryScope;
  status: MemoryStatus;
  content: string;
  source_excerpt: string;
  confidence: number;
  importance: number;
  version: number;
  created_at: string;
  updated_at: string;
}

export interface MemoryJob {
  id: string;
  status: 'pending' | 'running' | 'completed' | 'failed';
  candidate_count: number;
  summary_updated: boolean;
}

export type AgentName = 'coordinator' | 'case_analyst' | 'legal_researcher' | 'legal_counsel';
export type AgentStage =
  | 'idle'
  | 'queued'
  | 'analyzing'
  | 'researching'
  | 'drafting'
  | 'reviewing'
  | 'completed'
  | 'interrupted'
  | 'failed';

export interface AgentActivity {
  agent: AgentName;
  status: AgentStage;
  message: string;
}

export interface ConversationRuntime {
  userId: string;
  conversationId: string;
  messages: ChatMessage[];
  status: AgentStage;
  activeAgent?: AgentName;
  statusMessage?: string;
  requestToken?: number;
  toolActivity?: ToolActivity;
  memoryMessage?: string;
  error?: string;
  failedQuestion?: string;
  loading?: boolean;
  unread?: boolean;
  updatedAt: number;
}

export type SseEventName =
  | 'message_start'
  | 'agent_status'
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
