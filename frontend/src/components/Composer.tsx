import { ArrowUp, Square } from 'lucide-react';
import { KeyboardEvent, useEffect, useRef } from 'react';

interface ComposerProps {
  value: string;
  streaming: boolean;
  disabled: boolean;
  onChange: (value: string) => void;
  onSend: () => void;
  onStop: () => void;
}

export function Composer({ value, streaming, disabled, onChange, onSend, onStop }: ComposerProps) {
  const textareaRef = useRef<HTMLTextAreaElement>(null);

  useEffect(() => {
    const textarea = textareaRef.current;
    if (!textarea) return;
    textarea.style.height = 'auto';
    textarea.style.height = `${Math.min(textarea.scrollHeight, 160)}px`;
  }, [value]);

  function handleKeyDown(event: KeyboardEvent<HTMLTextAreaElement>) {
    if (event.key === 'Enter' && !event.shiftKey && !event.nativeEvent.isComposing) {
      event.preventDefault();
      if (!disabled && !streaming && value.trim()) onSend();
    }
  }

  return (
    <footer className="composer-shell">
      <div className="composer">
        <textarea
          ref={textareaRef}
          rows={1}
          value={value}
          disabled={disabled || streaming}
          onChange={(event) => onChange(event.target.value)}
          onKeyDown={handleKeyDown}
          placeholder="输入您的法律咨询问题…"
          aria-label="法律咨询问题"
        />
        {streaming ? (
          <button className="send-button stop-button" onClick={onStop} aria-label="停止生成">
            <Square size={16} fill="currentColor" />
          </button>
        ) : (
          <button className="send-button" onClick={onSend} disabled={disabled || !value.trim()} aria-label="发送消息">
            <ArrowUp size={20} />
          </button>
        )}
      </div>
      <p>Enter 发送 · Shift + Enter 换行 · AI 回答仅供参考，不构成正式法律意见</p>
    </footer>
  );
}
