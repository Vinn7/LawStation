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

export interface SkillActivity {
  skillId: string;
  status: 'selected' | 'running' | 'completed' | 'failed';
  message: string;
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
  canonical_key?: string;
  source_message_id?: string | null;
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
  runId?: string;
  lastEventSequence?: number;
  serverStatus?: AgentRunStatus;
  reconnecting?: boolean;
  toolActivity?: ToolActivity;
  skillActivity?: SkillActivity;
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
  | 'skill_status'
  | 'tool_call_start'
  | 'tool_call_result'
  | 'token'
  | 'citations'
  | 'memory_status'
  | 'message_end'
  | 'error';

export interface SseEvent<T = unknown> {
  id?: number;
  event: SseEventName;
  data: T;
}

export type AgentRunStatus = 'queued' | 'running' | 'completed' | 'interrupted' | 'failed';

export interface AgentRun {
  id: string;
  request_id: string;
  conversation_id: string;
  status: AgentRunStatus;
  current_stage: AgentStage;
  input_text: string;
  attempt: number;
  last_event_sequence: number;
  user_message_id?: string | null;
  assistant_message_id?: string | null;
  model_call_count: number;
  tool_call_count: number;
  error_type?: string;
  error_summary?: string;
}

export interface ScenarioDataset {
  id: string;
  schema_version: string;
  sha256: string;
  sample_count: number;
  synthetic: boolean;
  human_verified: boolean;
  categories: Record<string, number>;
  step_timeout_seconds?: number;
}

export type ScenarioAvailability =
  | { status: 'checking'; datasets: ScenarioDataset[]; message: string }
  | { status: 'ready'; datasets: ScenarioDataset[]; message: string }
  | { status: 'disabled'; datasets: ScenarioDataset[]; message: string }
  | { status: 'error'; datasets: ScenarioDataset[]; message: string };

export interface ScenarioSummary {
  scenario_id: string;
  title: string;
  category: string;
  description: string;
  actors: string[];
  step_count: number;
  preconditions: string[];
  fixture_types: string[];
}

export type ScenarioAction =
  | 'send_message'
  | 'switch_user'
  | 'switch_conversation'
  | 'wait_for_completion'
  | 'cancel_run'
  | 'disconnect_stream'
  | 'reconnect_stream'
  | 'inspect_messages'
  | 'inspect_memories';

export interface ScenarioStep {
  action: ScenarioAction;
  actor: string;
  conversation: string;
  content?: string;
  after_event?: string;
  expected?: Record<string, unknown>;
}

export interface DialogueScenario extends ScenarioSummary {
  schema_version: string;
  synthetic: boolean;
  steps: ScenarioStep[];
  fixture_notice: string;
  fixtures_applied: boolean;
}

export interface ScenarioRunOutcome {
  terminal_status: AgentRunStatus;
  observed_events: string[];
  retrieval_status: string;
  selected_skill_ids: string[];
  citation_count: number;
  model_call_count: number;
  tool_call_count: number;
  last_event_sequence: number;
  checkpoint_available: boolean;
}

export type ScenarioStepStatus = 'pending' | 'running' | 'passed' | 'failed' | 'inconclusive' | 'skipped';

export interface ScenarioStepResult {
  stepIndex: number;
  status: ScenarioStepStatus;
  message: string;
  runId?: string;
  expected?: Record<string, unknown>;
  actual?: Record<string, unknown>;
}

export interface ScenarioConversationBinding {
  key: string;
  actor: string;
  conversation: string;
  userId: string;
  conversationId: string;
  title: string;
}

export interface ScenarioRunObservation {
  runId: string;
  userId: string;
  conversationId: string;
  events: string[];
  sequences: number[];
  duplicateSequence: boolean;
  memoryJobIds: string[];
  subscriptions: Array<{
    epoch: number;
    connectedAfterSequence: number;
    firstReceivedSequence?: number;
    receivedSequences: number[];
  }>;
  outcome?: ScenarioRunOutcome;
}

export interface ScenarioSession {
  id: string;
  datasetId: string;
  scenario: DialogueScenario;
  stepIndex: number;
  bindings: Record<string, ScenarioConversationBinding>;
  results: ScenarioStepResult[];
  runByBinding: Record<string, string>;
  observations: Record<string, ScenarioRunObservation>;
  backgroundRunIds: string[];
  concurrentRunPairs: string[];
  executing: boolean;
  error: string;
}
