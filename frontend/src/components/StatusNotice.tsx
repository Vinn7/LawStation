import { AlertCircle, BookOpenCheck, BrainCircuit, LoaderCircle, Search, Sparkles } from 'lucide-react';
import type { AgentActivity, IndexStatus, SkillActivity, ToolActivity } from '../types';

interface StatusNoticeProps {
  index: IndexStatus;
  tool: ToolActivity | null;
  skill?: SkillActivity | null;
  agent?: AgentActivity | null;
  memoryMessage: string;
  error: string;
  failedQuestion: string;
  onRetry: () => void;
  onDismissError: () => void;
}

function toolLabel(tool: ToolActivity): string {
  const names: Record<string, string> = {
    search_laws: '检索相关法律条款',
    get_law_article: '读取指定法律条款',
  };
  const action = names[tool.name] ?? '调用法律检索工具';
  if (tool.status === 'running') return `正在${action}…`;
  if (tool.status === 'success') return `${action}完成`;
  if (tool.status === 'timeout') return `${action}超时`;
  return `${action}失败`;
}

export function StatusNotice({
  index,
  tool,
  skill = null,
  agent = null,
  memoryMessage,
  error,
  failedQuestion,
  onRetry,
  onDismissError,
}: StatusNoticeProps) {
  const rerankerDegraded = ['degraded', 'cooldown'].includes(index.reranker_status ?? '');
  const showIndex = index.status !== 'ready' || rerankerDegraded;
  if (!showIndex && !agent && !tool && !skill && !memoryMessage && !error) return null;

  return (
    <div className="status-stack" aria-live="polite">
      {showIndex && (
        <div className={`status-notice index-notice is-${rerankerDegraded ? 'degraded' : index.status}`}>
          {index.status === 'building' || index.status === 'checking'
            ? <LoaderCircle className="spin" size={17} />
            : index.status === 'degraded' || rerankerDegraded ? <BookOpenCheck size={17} /> : <AlertCircle size={17} />}
          <span>{rerankerDegraded
            ? (index.reranker_message || '法条精排暂不可用，当前使用 RRF 排序。')
            : (index.message || (index.status === 'degraded' ? 'Dense 索引不可用，当前仍可使用 BM25 检索。' : '法规索引暂不可用。'))}</span>
        </div>
      )}
      {tool && (
        <div className={`status-notice tool-notice is-${tool.status}`}>
          {tool.status === 'running' ? <LoaderCircle className="spin" size={17} /> : <Search size={17} />}
          <span>{toolLabel(tool)}</span>
        </div>
      )}
      {skill && !tool && (
        <div className={`status-notice agent-notice is-${skill.status}`}>
          {['selected', 'running'].includes(skill.status)
            ? <LoaderCircle className="spin" size={17} /> : <Sparkles size={17} />}
          <span>{skill.message}</span>
        </div>
      )}
      {agent && !tool && !skill && (
        <div className={`status-notice agent-notice is-${agent.status}`}>
          {['queued', 'analyzing', 'researching', 'drafting', 'reviewing'].includes(agent.status)
            ? <LoaderCircle className="spin" size={17} /> : <BrainCircuit size={17} />}
          <span>{agent.message}</span>
        </div>
      )}
      {memoryMessage && (
        <div className="status-notice memory-notice"><BrainCircuit size={17} /><span>{memoryMessage}</span></div>
      )}
      {error && (
        <div className="status-notice error-notice" role="alert">
          <AlertCircle size={17} />
          <span>{error}</span>
          {failedQuestion && <button onClick={onRetry}>重新发送</button>}
          <button className="notice-dismiss" onClick={onDismissError} aria-label="关闭错误提示">×</button>
        </div>
      )}
    </div>
  );
}
