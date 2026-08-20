import { useEffect, useMemo, useRef, useState } from 'react';
import { api } from './api';
import { consumeSse } from './sse';
import { ChatHeader } from './components/ChatHeader';
import { Composer } from './components/Composer';
import { MessageList } from './components/MessageList';
import { Sidebar } from './components/Sidebar';
import { StatusNotice } from './components/StatusNotice';
import type { ChatMessage, Citation, Conversation, IndexStatus, SseEvent, ToolActivity, User } from './types';

const initialIndex: IndexStatus = { status: 'checking', message: '正在检查法律索引', progress: 0 };

interface ActiveRequest {
  token: number;
  userId: string;
  conversationId: string;
  assistantMessageId: string;
}

function errorMessage(error: unknown): string {
  if (error instanceof Error) return error.message;
  return '发生未知错误，请稍后重试';
}

export default function App() {
  const [users, setUsers] = useState<User[]>([]);
  const [userId, setUserId] = useState('');
  const [conversations, setConversations] = useState<Conversation[]>([]);
  const [conversationId, setConversationId] = useState('');
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [draft, setDraft] = useState('');
  const [index, setIndex] = useState<IndexStatus>(initialIndex);
  const [usersLoading, setUsersLoading] = useState(true);
  const [conversationsLoading, setConversationsLoading] = useState(false);
  const [messagesLoading, setMessagesLoading] = useState(false);
  const [conversationError, setConversationError] = useState('');
  const [chatError, setChatError] = useState('');
  const [failedQuestion, setFailedQuestion] = useState('');
  const [toolActivity, setToolActivity] = useState<ToolActivity | null>(null);
  const [memoryMessage, setMemoryMessage] = useState('');
  const [streaming, setStreaming] = useState(false);
  const [sidebarOpen, setSidebarOpen] = useState(false);

  const streamController = useRef<AbortController | null>(null);
  const activeRequest = useRef<ActiveRequest | null>(null);
  const requestSequence = useRef(0);
  const conversationLoadSequence = useRef(0);
  const currentUserId = useRef('');
  currentUserId.current = userId;

  const currentConversation = useMemo(
    () => conversations.find((conversation) => conversation.id === conversationId),
    [conversations, conversationId],
  );

  useEffect(() => {
    const controller = new AbortController();
    api.users(controller.signal)
      .then((data) => {
        setUsers(data);
        setUserId(data[0]?.id ?? '');
      })
      .catch((error) => {
        if (error instanceof DOMException && error.name === 'AbortError') return;
        setConversationError(`无法加载用户：${errorMessage(error)}`);
      })
      .finally(() => setUsersLoading(false));
    return () => controller.abort();
  }, []);

  useEffect(() => {
    let stopped = false;
    let controller: AbortController | null = null;
    const load = () => {
      controller?.abort();
      controller = new AbortController();
      api.indexStatus(controller.signal).then((status) => {
        if (!stopped) setIndex(status);
      }).catch(() => undefined);
    };
    load();
    const timer = window.setInterval(load, 2500);
    return () => {
      stopped = true;
      controller?.abort();
      window.clearInterval(timer);
    };
  }, []);

  useEffect(() => {
    cancelActiveStream('context-change');
    setConversationId('');
    setMessages([]);
    setConversations([]);
    setToolActivity(null);
    setMemoryMessage('');
    setChatError('');
    setFailedQuestion('');
    if (!userId) return;

    const controller = new AbortController();
    const sequence = ++conversationLoadSequence.current;
    setConversationsLoading(true);
    setConversationError('');
    api.conversations(userId, controller.signal)
      .then((data) => {
        if (sequence === conversationLoadSequence.current) setConversations(data);
      })
      .catch((error) => {
        if (error instanceof DOMException && error.name === 'AbortError') return;
        if (sequence === conversationLoadSequence.current) setConversationError(errorMessage(error));
      })
      .finally(() => {
        if (sequence === conversationLoadSequence.current) setConversationsLoading(false);
      });
    return () => controller.abort();
  }, [userId]);

  useEffect(() => () => cancelActiveStream('context-change'), []);

  function cancelActiveStream(reason: 'context-change' | 'user-stop') {
    if (streamController.current) streamController.current.abort(reason);
    streamController.current = null;
    if (reason === 'context-change') {
      activeRequest.current = null;
      setStreaming(false);
    }
  }

  async function createConversation(): Promise<Conversation | null> {
    if (!userId || streaming) return null;
    setConversationError('');
    try {
      const ownerId = userId;
      const conversation = await api.createConversation(ownerId);
      if (ownerId !== currentUserId.current) return null;
      setConversations((current) => [conversation, ...current]);
      setConversationId(conversation.id);
      setMessages([]);
      setSidebarOpen(false);
      return conversation;
    } catch (error) {
      setConversationError(errorMessage(error));
      return null;
    }
  }

  async function openConversation(id: string) {
    if (id === conversationId && !messagesLoading) {
      setSidebarOpen(false);
      return;
    }
    cancelActiveStream('context-change');
    const ownerId = userId;
    const sequence = ++conversationLoadSequence.current;
    setConversationId(id);
    setMessages([]);
    setMessagesLoading(true);
    setChatError('');
    setToolActivity(null);
    setMemoryMessage('');
    setSidebarOpen(false);
    try {
      const data = await api.messages(ownerId, id);
      if (sequence === conversationLoadSequence.current && ownerId === currentUserId.current) setMessages(data);
    } catch (error) {
      if (sequence === conversationLoadSequence.current) setChatError(errorMessage(error));
    } finally {
      if (sequence === conversationLoadSequence.current) setMessagesLoading(false);
    }
  }

  function isCurrent(request: ActiveRequest): boolean {
    const current = activeRequest.current;
    return Boolean(current && current.token === request.token && current.userId === request.userId && current.conversationId === request.conversationId);
  }

  function updateAssistant(request: ActiveRequest, update: (message: ChatMessage) => ChatMessage) {
    if (!isCurrent(request)) return;
    setMessages((current) => current.map((message) => (
      message.id === request.assistantMessageId ? update(message) : message
    )));
  }

  function handleStreamEvent(request: ActiveRequest, item: SseEvent, streamFailure: { message: string }) {
    if (!isCurrent(request)) return;
    if (item.event === 'token') {
      const token = typeof item.data === 'string' ? item.data : '';
      updateAssistant(request, (message) => ({ ...message, content: message.content + token }));
    } else if (item.event === 'tool_call_start') {
      const data = item.data as { name?: string };
      setToolActivity({ name: data.name ?? 'unknown', status: 'running' });
    } else if (item.event === 'tool_call_result') {
      const data = item.data as { name?: string; status?: string };
      const status = data.status === 'success' ? 'success' : data.status === 'timeout' ? 'timeout' : 'failed';
      setToolActivity({ name: data.name ?? 'unknown', status });
    } else if (item.event === 'memory_status') {
      const data = item.data as { compressed?: boolean };
      setMemoryMessage(data.compressed ? '本轮对话已完成摘要与记忆整理' : '本轮对话已保存');
    } else if (item.event === 'citations') {
      const raw = Array.isArray(item.data) ? item.data : [];
      const citations = raw.filter((citation): citation is Citation => Boolean(citation && typeof citation === 'object'));
      updateAssistant(request, (message) => ({ ...message, citations }));
    } else if (item.event === 'message_end') {
      updateAssistant(request, (message) => ({ ...message, status: 'complete' }));
      setToolActivity(null);
    } else if (item.event === 'error') {
      const data = item.data as { message?: string };
      streamFailure.message = data.message || '回答生成失败';
    }
  }

  async function send(questionOverride?: string) {
    const question = (questionOverride ?? draft).trim();
    if (!question || !userId || streaming || usersLoading) return;
    let targetConversationId = conversationId;
    if (!targetConversationId) {
      const created = await createConversation();
      if (!created) return;
      targetConversationId = created.id;
    }

    const token = ++requestSequence.current;
    const assistantMessageId = `assistant-${token}`;
    const request: ActiveRequest = { token, userId, conversationId: targetConversationId, assistantMessageId };
    const controller = new AbortController();
    activeRequest.current = request;
    streamController.current = controller;
    setDraft('');
    setFailedQuestion('');
    setChatError('');
    setToolActivity(null);
    setMemoryMessage('');
    setStreaming(true);
    setMessages((current) => [
      ...current,
      { id: `user-${token}`, role: 'user', content: question, status: 'complete' },
      { id: assistantMessageId, role: 'assistant', content: '', status: 'streaming' },
    ]);

    const streamFailure = { message: '' };
    try {
      const response = await api.streamMessage(userId, targetConversationId, question, controller.signal);
      await consumeSse(response, (item) => handleStreamEvent(request, item, streamFailure));
      if (!isCurrent(request)) return;
      if (streamFailure.message) throw new Error(streamFailure.message);
      updateAssistant(request, (message) => ({ ...message, status: 'complete' }));
    } catch (error) {
      if (!isCurrent(request)) return;
      const abortReason = controller.signal.reason;
      if (controller.signal.aborted && abortReason === 'user-stop') {
        updateAssistant(request, (message) => ({ ...message, status: 'interrupted' }));
      } else if (!(controller.signal.aborted && abortReason === 'context-change')) {
        setChatError(errorMessage(error));
        setFailedQuestion(question);
        setDraft(question);
        updateAssistant(request, (message) => ({ ...message, status: 'error' }));
      }
    } finally {
      if (isCurrent(request)) {
        activeRequest.current = null;
        streamController.current = null;
        setStreaming(false);
      }
    }
  }

  function stopStream() {
    cancelActiveStream('user-stop');
  }

  function chooseSuggestion(question: string) {
    setDraft(question);
  }

  return (
    <main className="app-shell">
      <Sidebar
        users={users}
        userId={userId}
        conversations={conversations}
        activeConversationId={conversationId}
        open={sidebarOpen}
        loading={conversationsLoading}
        error={conversationError}
        disabled={streaming}
        onUserChange={setUserId}
        onCreate={() => void createConversation()}
        onOpenConversation={(id) => void openConversation(id)}
        onClose={() => setSidebarOpen(false)}
      />
      <section className="chat-workspace">
        <ChatHeader
          conversationTitle={currentConversation?.title ?? ''}
          index={index}
          onOpenSidebar={() => setSidebarOpen(true)}
        />
        <MessageList
          messages={messages}
          loading={messagesLoading}
          conversationSelected={Boolean(conversationId)}
          onSuggestion={chooseSuggestion}
        />
        <div className="bottom-dock">
          <StatusNotice
            index={index}
            tool={toolActivity}
            memoryMessage={memoryMessage}
            error={chatError}
            failedQuestion={failedQuestion}
            onRetry={() => void send(failedQuestion)}
            onDismissError={() => { setChatError(''); setFailedQuestion(''); }}
          />
          <Composer
            value={draft}
            streaming={streaming}
            disabled={!userId || usersLoading || messagesLoading}
            onChange={setDraft}
            onSend={() => void send()}
            onStop={stopStream}
          />
        </div>
      </section>
    </main>
  );
}
