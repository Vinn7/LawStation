import { BookOpen, CircleStop, Scale, Sparkles, ThumbsDown, ThumbsUp, UserRound } from 'lucide-react';
import { useEffect, useRef, useState } from 'react';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import type { ChatMessage } from '../types';

const suggestions = [
  '公司没有签劳动合同，我可以主张哪些权利？',
  '借款到期后对方不还钱，诉讼时效如何计算？',
  '租赁合同提前解除，需要承担哪些责任？',
];

interface MessageListProps {
  messages: ChatMessage[];
  loading: boolean;
  conversationSelected: boolean;
  onSuggestion: (question: string) => void;
  onFeedback: (messageId: string, score: -1 | 1, comment?: string) => Promise<void>;
}

function MessageBubble({
  message,
  onFeedback,
}: {
  message: ChatMessage;
  onFeedback: (messageId: string, score: -1 | 1, comment?: string) => Promise<void>;
}) {
  const assistant = message.role === 'assistant';
  const [submitting, setSubmitting] = useState(false);
  const [comment, setComment] = useState('');
  const [commentOpen, setCommentOpen] = useState(false);

  async function vote(score: -1 | 1, detail = '') {
    setSubmitting(true);
    try {
      await onFeedback(message.id, score, detail);
      if (score < 0) setCommentOpen(true);
    } finally {
      setSubmitting(false);
    }
  }
  return (
    <article className={`message-row is-${message.role}`}>
      <span className="message-avatar" aria-hidden="true">
        {assistant ? <Scale size={18} /> : <UserRound size={18} />}
      </span>
      <div className={`message-bubble ${message.status ? `is-${message.status}` : ''}`}>
        {assistant ? (
          message.content ? (
            <ReactMarkdown remarkPlugins={[remarkGfm]}>{message.content}</ReactMarkdown>
          ) : message.status === 'streaming' ? (
            <span className="thinking"><i /><i /><i /><span className="sr-only">正在生成回答</span></span>
          ) : null
        ) : <p>{message.content}</p>}
        {assistant && message.citations && message.citations.length > 0 && (
          <div className="message-citations" aria-label="法律依据">
            {message.citations.map((citation, index) => (
              <span key={citation.document_id ?? `${citation.law_name}-${citation.article_number}-${index}`}>
                <BookOpen size={12} />
                {[citation.law_name, citation.article_number].filter(Boolean).join(' ') || '法规依据'}
              </span>
            ))}
          </div>
        )}
        {message.status === 'interrupted' && (
          <span className="message-status"><CircleStop size={13} />回答已停止</span>
        )}
        {message.status === 'error' && (
          <span className="message-status is-error">回答生成失败</span>
        )}
        {assistant && message.status === 'complete' && !message.id.startsWith('assistant-') && (
          <div className="message-feedback" aria-label="评价回答">
            <button
              type="button"
              className={message.feedback_score === 1 ? 'is-selected' : ''}
              aria-label="回答有帮助"
              disabled={submitting}
              onClick={() => void vote(1)}
            ><ThumbsUp size={14} /></button>
            <button
              type="button"
              className={message.feedback_score === -1 ? 'is-selected' : ''}
              aria-label="回答需改进"
              disabled={submitting}
              onClick={() => void vote(-1)}
            ><ThumbsDown size={14} /></button>
            {message.feedback_score && <span>感谢反馈</span>}
            {commentOpen && message.feedback_score === -1 && (
              <form onSubmit={(event) => { event.preventDefault(); void vote(-1, comment.trim()); }}>
                <input
                  aria-label="反馈说明"
                  value={comment}
                  maxLength={1000}
                  placeholder="可选：告诉我们哪里需要改进"
                  onChange={(event) => setComment(event.target.value)}
                />
                <button type="submit" disabled={submitting || !comment.trim()}>提交说明</button>
              </form>
            )}
          </div>
        )}
      </div>
    </article>
  );
}

export function MessageList({ messages, loading, conversationSelected, onSuggestion, onFeedback }: MessageListProps) {
  const viewportRef = useRef<HTMLDivElement>(null);
  const [pinnedToBottom, setPinnedToBottom] = useState(true);
  const contentSignature = messages.map((message) => `${message.id}:${message.content.length}:${message.status}`).join('|');

  useEffect(() => {
    if (!pinnedToBottom) return;
    const frame = requestAnimationFrame(() => {
      const viewport = viewportRef.current;
      viewport?.scrollTo({ top: viewport.scrollHeight, behavior: 'smooth' });
    });
    return () => cancelAnimationFrame(frame);
  }, [contentSignature, pinnedToBottom]);

  function handleScroll() {
    const viewport = viewportRef.current;
    if (!viewport) return;
    setPinnedToBottom(viewport.scrollHeight - viewport.scrollTop - viewport.clientHeight < 96);
  }

  return (
    <div className="message-viewport" ref={viewportRef} onScroll={handleScroll}>
      <div className="message-content">
        {loading && <div className="chat-loading"><span className="skeleton-line" /><span className="skeleton-line short" /></div>}
        {!loading && messages.length === 0 && (
          <section className="welcome-state">
            <span className="welcome-mark" aria-hidden="true"><Sparkles size={28} /></span>
            <span className="eyebrow">LAWSTATION AI</span>
            <h1>{conversationSelected ? '今天想咨询什么法律问题？' : '您的智能法律咨询助手'}</h1>
            <p>我会优先检索法律法规，并结合当前会话上下文提供结构化回答。</p>
            <div className="suggestion-grid">
              {suggestions.map((question) => (
                <button key={question} onClick={() => onSuggestion(question)}>
                  <BookOpen size={17} />
                  <span>{question}</span>
                </button>
              ))}
            </div>
          </section>
        )}
        {!loading && messages.map((message) => <MessageBubble key={message.id} message={message} onFeedback={onFeedback} />)}
      </div>
    </div>
  );
}
