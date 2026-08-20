import { Database, Menu, Scale } from 'lucide-react';
import type { IndexStatus } from '../types';

interface ChatHeaderProps {
  conversationTitle: string;
  index: IndexStatus;
  onOpenSidebar: () => void;
}

function indexLabel(index: IndexStatus): string {
  if (index.status === 'ready') return index.dense_enabled ? '混合检索已就绪' : '法规库已就绪';
  if (index.status === 'degraded') return '仅使用 BM25';
  if (index.status === 'failed') return '法规索引异常';
  const progress = Math.round((index.progress ?? 0) * 100);
  return index.status === 'building' ? `正在构建 ${progress}%` : '正在检查法规库';
}

export function ChatHeader({ conversationTitle, index, onOpenSidebar }: ChatHeaderProps) {
  return (
    <header className="chat-header">
      <div className="header-identity">
        <button className="icon-button menu-button" aria-label="打开会话侧栏" onClick={onOpenSidebar}>
          <Menu size={21} />
        </button>
        <span className="header-mark" aria-hidden="true"><Scale size={21} /></span>
        <div>
          <strong>LawStation</strong>
          <span>法律咨询</span>
        </div>
      </div>
      <div className="header-context">
        <span className={`index-badge is-${index.status}`} title={index.message}>
          <Database size={14} />
          {indexLabel(index)}
        </span>
        <span className="current-conversation" title={conversationTitle}>
          <small>当前会话</small>
          <strong>{conversationTitle || '尚未选择'}</strong>
        </span>
      </div>
    </header>
  );
}
