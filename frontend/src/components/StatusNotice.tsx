import { AlertCircle, BookOpenCheck, BrainCircuit, LoaderCircle, Search } from 'lucide-react';
import type { IndexStatus, ToolActivity } from '../types';

interface StatusNoticeProps {
  index: IndexStatus;
  tool: ToolActivity | null;
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
  memoryMessage,
  error,
  failedQuestion,
  onRetry,
  onDismissError,
}: StatusNoticeProps) {
  const showIndex = index.status !== 'ready';
  if (!showIndex && !tool && !memoryMessage && !error) return null;

  return (
    <div className="status-stack" aria-live="polite">
      {showIndex && (
        <div className={`status-notice index-notice is-${index.status}`}>
          {index.status === 'building' || index.status === 'checking'
            ? <LoaderCircle className="spin" size={17} />
            : index.status === 'degraded' ? <BookOpenCheck size={17} /> : <AlertCircle size={17} />}
          <span>{index.message || (index.status === 'degraded' ? 'Dense 索引不可用，当前仍可使用 BM25 检索。' : '法规索引暂不可用。')}</span>
        </div>
      )}
      {tool && (
        <div className={`status-notice tool-notice is-${tool.status}`}>
          {tool.status === 'running' ? <LoaderCircle className="spin" size={17} /> : <Search size={17} />}
          <span>{toolLabel(tool)}</span>
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
