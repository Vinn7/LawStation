import { render, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { describe, expect, it, vi } from 'vitest';
import { ChatHeader } from '../components/ChatHeader';
import { Composer } from '../components/Composer';
import { StatusNotice } from '../components/StatusNotice';
import { MessageList } from '../components/MessageList';

describe('Composer', () => {
  it('sends with Enter and keeps Shift+Enter as a newline', async () => {
    const user = userEvent.setup();
    const onSend = vi.fn();
    const onChange = vi.fn();
    const { rerender } = render(
      <Composer value="法律问题" streaming={false} disabled={false} onChange={onChange} onSend={onSend} onStop={vi.fn()} />,
    );
    const textbox = screen.getByRole('textbox', { name: '法律咨询问题' });

    await user.type(textbox, '{shift>}{enter}{/shift}');
    expect(onSend).not.toHaveBeenCalled();

    rerender(<Composer value="法律问题" streaming={false} disabled={false} onChange={onChange} onSend={onSend} onStop={vi.fn()} />);
    await user.type(screen.getByRole('textbox'), '{enter}');
    expect(onSend).toHaveBeenCalledTimes(1);
  });

  it('shows the stop action while streaming', async () => {
    const user = userEvent.setup();
    const onStop = vi.fn();
    render(<Composer value="" streaming disabled={false} onChange={vi.fn()} onSend={vi.fn()} onStop={onStop} />);
    await user.click(screen.getByRole('button', { name: '停止生成' }));
    expect(onStop).toHaveBeenCalledOnce();
  });
});

describe('status presentation', () => {
  it('shows dense degradation without disabling the chat', () => {
    render(<ChatHeader
      conversationTitle="劳动争议"
      index={{ status: 'degraded', message: '缺少密钥' }}
      onOpenSidebar={vi.fn()}
      scenarioAvailability={{ status: 'disabled', datasets: [], message: '未启用' }}
      onOpenScenarios={vi.fn()}
      onRetryScenarios={vi.fn()}
    />);
    expect(screen.getByText('仅使用 BM25')).toBeInTheDocument();
    expect(screen.getByText('劳动争议')).toBeInTheDocument();
  });

  it('does not expose tool arguments or law contents', () => {
    render(
      <StatusNotice
        index={{ status: 'ready', message: '就绪' }}
        tool={{ name: 'search_laws', status: 'running' }}
        memoryMessage=""
        error=""
        failedQuestion=""
        onRetry={vi.fn()}
        onDismissError={vi.fn()}
      />,
    );
    expect(screen.getByText('正在检索相关法律条款…')).toBeInTheDocument();
    expect(screen.queryByText(/query|top_k|正文/)).not.toBeInTheDocument();
  });

  it('shows only the safe runtime skill status message', () => {
    render(
      <StatusNotice
        index={{ status: 'ready', message: '就绪' }}
        tool={null}
        skill={{ skillId: 'evidence-audit', status: 'running', message: '正在审查证据准备情况' }}
        memoryMessage=""
        error=""
        failedQuestion=""
        onRetry={vi.fn()}
        onDismissError={vi.fn()}
      />,
    );
    expect(screen.getByText('正在审查证据准备情况')).toBeInTheDocument();
    expect(screen.queryByText(/SKILL\.md|output_schema|allowed_tools/)).not.toBeInTheDocument();
  });
});

describe('message feedback', () => {
  it('submits an owned assistant message rating', async () => {
    const user = userEvent.setup();
    const onFeedback = vi.fn().mockResolvedValue(undefined);
    render(
      <MessageList
        messages={[{ id: 'message-1', role: 'assistant', content: '法律意见', status: 'complete' }]}
        loading={false}
        conversationSelected
        onSuggestion={vi.fn()}
        onFeedback={onFeedback}
      />,
    );
    await user.click(screen.getByRole('button', { name: '回答有帮助' }));
    expect(onFeedback).toHaveBeenCalledWith('message-1', 1, '');
  });
});
