import { Beaker, BookOpenText, BrainCircuit, ChevronDown, MessageSquareText, Plus, RotateCcw, Scale, X } from 'lucide-react';
import type { AgentStage, Conversation, ScenarioAvailability, User } from '../types';

interface SidebarProps {
  users: User[];
  userId: string;
  conversations: Conversation[];
  activeConversationId: string;
  open: boolean;
  loading: boolean;
  error: string;
  disabled: boolean;
  taskStatuses: Record<string, { status: AgentStage; unread?: boolean }>;
  onUserChange: (userId: string) => void;
  onCreate: () => void;
  onOpenConversation: (id: string) => void;
  onClose: () => void;
  onOpenMemories: () => void;
  scenarioAvailability: ScenarioAvailability;
  onOpenScenarios: () => void;
  onRetryScenarios: () => void;
}

const dateFormatter = new Intl.DateTimeFormat('zh-CN', {
  month: 'numeric',
  day: 'numeric',
  hour: '2-digit',
  minute: '2-digit',
});

function formatDate(value: string): string {
  const date = new Date(value);
  return Number.isNaN(date.getTime()) ? '' : dateFormatter.format(date);
}

export function Sidebar({
  users,
  userId,
  conversations,
  activeConversationId,
  open,
  loading,
  error,
  disabled,
  taskStatuses,
  onUserChange,
  onCreate,
  onOpenConversation,
  onClose,
  onOpenMemories,
  scenarioAvailability,
  onOpenScenarios,
  onRetryScenarios,
}: SidebarProps) {
  return (
    <>
      <button
        className={`sidebar-backdrop ${open ? 'is-visible' : ''}`}
        aria-label="关闭会话侧栏"
        onClick={onClose}
        tabIndex={open ? 0 : -1}
      />
      <aside className={`sidebar ${open ? 'is-open' : ''}`} aria-label="会话工作区">
        <div className="sidebar-brand">
          <span className="brand-mark" aria-hidden="true"><Scale size={22} /></span>
          <div>
            <strong>LawStation</strong>
            <span>智能法律咨询工作台</span>
          </div>
          <button className="icon-button sidebar-close" aria-label="关闭侧栏" onClick={onClose}>
            <X size={20} />
          </button>
        </div>

        <label className="user-select-label" htmlFor="user-select">当前用户</label>
        <div className="select-wrap">
          <select
            id="user-select"
            value={userId}
            disabled={users.length === 0}
            onChange={(event) => onUserChange(event.target.value)}
          >
            {users.map((user) => <option key={user.id} value={user.id}>{user.name}</option>)}
          </select>
          <ChevronDown size={16} aria-hidden="true" />
        </div>

        <section className="scene-card" aria-labelledby="scene-title">
          <div className="scene-heading">
            <div>
              <span className="eyebrow">当前场景</span>
              <h2 id="scene-title">法律咨询</h2>
            </div>
            <BookOpenText size={20} aria-hidden="true" />
          </div>
          <div className="scene-option is-selected">
            <span className="radio-dot" aria-hidden="true" />
            <div>
              <strong>法律咨询</strong>
              <span>检索法规并结合上下文回答</span>
            </div>
          </div>
        </section>

        <button className="new-chat-button" onClick={onCreate} disabled={!userId || disabled}>
          <Plus size={18} />
          新建对话
        </button>
        <button className="memory-manage-button" onClick={onOpenMemories} disabled={!userId || disabled}>
          <BrainCircuit size={17} />
          管理我的记忆
        </button>
        {scenarioAvailability.status !== 'disabled' && (
          <button
            className={`memory-manage-button scenario-open-button is-${scenarioAvailability.status}`}
            onClick={scenarioAvailability.status === 'ready' ? onOpenScenarios : onRetryScenarios}
            disabled={!userId || disabled || scenarioAvailability.status === 'checking'}
            title={scenarioAvailability.message}
          >
            <Beaker size={17} />
            {scenarioAvailability.status === 'ready' ? '场景观察模式'
              : scenarioAvailability.status === 'checking' ? '正在检查场景模式'
                : '场景模式加载失败'}
            {scenarioAvailability.status === 'error' && <RotateCcw size={14} aria-label="重试加载场景" />}
          </button>
        )}

        <div className="conversation-section-heading">
          <span>历史对话</span>
          <span>{conversations.length}</span>
        </div>
        <nav className="conversation-list" aria-label="历史对话">
          {loading && <div className="sidebar-state">正在加载会话…</div>}
          {!loading && error && <div className="sidebar-state is-error">{error}</div>}
          {!loading && !error && conversations.length === 0 && (
            <div className="sidebar-state">
              <MessageSquareText size={24} />
              <span>还没有对话</span>
              <small>从一个法律问题开始吧</small>
            </div>
          )}
          {!loading && conversations.map((conversation) => (
            <button
              key={conversation.id}
              className={`conversation-item ${activeConversationId === conversation.id ? 'is-active' : ''}`}
              onClick={() => onOpenConversation(conversation.id)}
              aria-current={activeConversationId === conversation.id ? 'page' : undefined}
            >
              <span className="conversation-title-row">
                <span className="conversation-title">{conversation.title}</span>
                {taskStatuses[conversation.id]?.unread && <i className="unread-dot" aria-label="有新的回答" />}
              </span>
              <span className="conversation-meta">
                {taskStatuses[conversation.id] && taskStatuses[conversation.id].status !== 'idle'
                  ? ({
                    queued: '排队中', analyzing: '分析中', researching: '检索中', drafting: '生成中',
                    reviewing: '复核中', completed: '已完成', interrupted: '已停止', failed: '失败', idle: '',
                  } as Record<AgentStage, string>)[taskStatuses[conversation.id].status]
                  : formatDate(conversation.updated_at || conversation.created_at)}
              </span>
            </button>
          ))}
        </nav>
      </aside>
    </>
  );
}
