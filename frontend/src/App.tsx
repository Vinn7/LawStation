import { useEffect, useMemo, useRef, useState } from 'react';
import { api } from './api';
import { consumeSse } from './sse';
import { conversationKey, emptyRuntime, isActiveStage, type ConversationKey } from './chat/runtimeStore';
import { ChatHeader } from './components/ChatHeader';
import { Composer } from './components/Composer';
import { MessageList } from './components/MessageList';
import { Sidebar } from './components/Sidebar';
import { StatusNotice } from './components/StatusNotice';
import type {
  AgentActivity,
  AgentName,
  AgentStage,
  ChatMessage,
  Citation,
  Conversation,
  ConversationRuntime,
  IndexStatus,
  SseEvent,
  ToolActivity,
  User,
} from './types';

const initialIndex: IndexStatus = { status: 'checking', message: '正在检查法律索引', progress: 0 };

interface ConversationBucket {
  items: Conversation[];
  loading: boolean;
  error: string;
}

interface StreamSnapshot {
  token: number;
  key: ConversationKey;
  userId: string;
  conversationId: string;
  assistantMessageId: string;
}

function errorMessage(error: unknown): string {
  if (error instanceof Error) return error.message;
  return '发生未知错误，请稍后重试';
}

function runtimeActive(runtime?: ConversationRuntime): boolean {
  return isActiveStage(runtime?.status);
}

export default function App() {
  const [users, setUsers] = useState<User[]>([]);
  const [userId, setUserId] = useState('');
  const [conversationId, setConversationId] = useState('');
  const [conversationBuckets, setConversationBuckets] = useState<Record<string, ConversationBucket>>({});
  const [runtimes, setRuntimes] = useState<Record<string, ConversationRuntime>>({});
  const [drafts, setDrafts] = useState<Record<string, string>>({});
  const [index, setIndex] = useState<IndexStatus>(initialIndex);
  const [usersLoading, setUsersLoading] = useState(true);
  const [sidebarOpen, setSidebarOpen] = useState(false);

  const controllers = useRef(new Map<ConversationKey, AbortController>());
  const requestSequence = useRef(0);
  const loadSequences = useRef(new Map<ConversationKey, number>());
  const runtimeRef = useRef(runtimes);
  const currentSelection = useRef({ userId, conversationId });
  runtimeRef.current = runtimes;
  currentSelection.current = { userId, conversationId };

  const bucket = conversationBuckets[userId] ?? { items: [], loading: false, error: '' };
  const selectedKey = userId && conversationId ? conversationKey(userId, conversationId) : null;
  const selectedRuntime = selectedKey ? runtimes[selectedKey] : undefined;
  const draftKey = `${userId}:${conversationId || 'new'}`;
  const draft = drafts[draftKey] ?? '';
  const conversations = bucket.items;
  const messages = selectedRuntime?.messages ?? [];
  const streaming = runtimeActive(selectedRuntime);

  const currentConversation = useMemo(
    () => conversations.find((conversation) => conversation.id === conversationId),
    [conversations, conversationId],
  );

  const taskStatuses = useMemo(() => Object.fromEntries(
    conversations.flatMap((conversation) => {
      const runtime = runtimes[conversationKey(userId, conversation.id)];
      return runtime ? [[conversation.id, { status: runtime.status, unread: runtime.unread }]] : [];
    }),
  ), [conversations, runtimes, userId]);

  useEffect(() => {
    const controller = new AbortController();
    api.users(controller.signal)
      .then((data) => {
        setUsers(data);
        setUserId(data[0]?.id ?? '');
      })
      .catch((error) => {
        if (error instanceof DOMException && error.name === 'AbortError') return;
        setConversationBuckets((current) => ({
          ...current,
          '': { items: [], loading: false, error: `无法加载用户：${errorMessage(error)}` },
        }));
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
    setConversationId('');
    setSidebarOpen(false);
    if (!userId) return;
    const controller = new AbortController();
    setConversationBuckets((current) => ({
      ...current,
      [userId]: { items: current[userId]?.items ?? [], loading: true, error: '' },
    }));
    api.conversations(userId, controller.signal)
      .then((items) => setConversationBuckets((current) => ({
        ...current,
        [userId]: { items, loading: false, error: '' },
      })))
      .catch((error) => {
        if (error instanceof DOMException && error.name === 'AbortError') return;
        setConversationBuckets((current) => ({
          ...current,
          [userId]: { items: current[userId]?.items ?? [], loading: false, error: errorMessage(error) },
        }));
      });
    return () => controller.abort();
  }, [userId]);

  useEffect(() => () => {
    for (const controller of controllers.current.values()) controller.abort('page-unload');
    controllers.current.clear();
  }, []);

  function updateRuntime(
    key: ConversationKey,
    update: (runtime: ConversationRuntime) => ConversationRuntime,
    owner?: { userId: string; conversationId: string },
  ) {
    const current = runtimeRef.current;
    const base = current[key] ?? emptyRuntime(owner?.userId ?? '', owner?.conversationId ?? '');
    const next = { ...current, [key]: update(base) };
    runtimeRef.current = next;
    setRuntimes(next);
  }

  async function createConversation(ownerId = userId): Promise<Conversation | null> {
    if (!ownerId) return null;
    try {
      const conversation = await api.createConversation(ownerId);
      setConversationBuckets((current) => ({
        ...current,
        [ownerId]: {
          items: [conversation, ...(current[ownerId]?.items ?? [])],
          loading: false,
          error: '',
        },
      }));
      if (currentSelection.current.userId === ownerId) {
        setConversationId(conversation.id);
        setSidebarOpen(false);
      }
      return conversation;
    } catch (error) {
      setConversationBuckets((current) => ({
        ...current,
        [ownerId]: {
          items: current[ownerId]?.items ?? [],
          loading: false,
          error: errorMessage(error),
        },
      }));
      return null;
    }
  }

  async function openConversation(id: string) {
    const ownerId = userId;
    const key = conversationKey(ownerId, id);
    setConversationId(id);
    setSidebarOpen(false);
    updateRuntime(key, (runtime) => ({ ...runtime, unread: false, updatedAt: Date.now() }), { userId: ownerId, conversationId: id });
    const cached = runtimeRef.current[key];
    if (cached && (cached.messages.length > 0 || runtimeActive(cached))) return;

    const sequence = (loadSequences.current.get(key) ?? 0) + 1;
    loadSequences.current.set(key, sequence);
    updateRuntime(key, (runtime) => ({ ...runtime, loading: true, error: '', updatedAt: Date.now() }), { userId: ownerId, conversationId: id });
    try {
      const data = await api.messages(ownerId, id);
      const latest = runtimeRef.current[key];
      if (sequence !== loadSequences.current.get(key) || runtimeActive(latest)) return;
      updateRuntime(key, (runtime) => ({ ...runtime, messages: data, loading: false, updatedAt: Date.now() }), { userId: ownerId, conversationId: id });
    } catch (error) {
      updateRuntime(key, (runtime) => ({ ...runtime, loading: false, error: errorMessage(error), updatedAt: Date.now() }), { userId: ownerId, conversationId: id });
    }
  }

  function isCurrentStream(snapshot: StreamSnapshot): boolean {
    return runtimeRef.current[snapshot.key]?.requestToken === snapshot.token;
  }

  function updateAssistant(snapshot: StreamSnapshot, update: (message: ChatMessage) => ChatMessage) {
    if (!isCurrentStream(snapshot)) return;
    updateRuntime(snapshot.key, (runtime) => ({
      ...runtime,
      messages: runtime.messages.map((message) => (
        message.id === snapshot.assistantMessageId ? update(message) : message
      )),
      updatedAt: Date.now(),
    }), snapshot);
  }

  function handleStreamEvent(snapshot: StreamSnapshot, item: SseEvent, streamFailure: { message: string }) {
    if (!isCurrentStream(snapshot)) return;
    if (item.event === 'token') {
      const token = typeof item.data === 'string' ? item.data : '';
      updateAssistant(snapshot, (message) => ({ ...message, content: message.content + token }));
    } else if (item.event === 'agent_status') {
      const data = item.data as { agent?: AgentName; status?: AgentStage; message?: string };
      updateRuntime(snapshot.key, (runtime) => ({
        ...runtime,
        status: data.status ?? runtime.status,
        activeAgent: data.agent,
        statusMessage: data.message,
        toolActivity: undefined,
        updatedAt: Date.now(),
      }), snapshot);
    } else if (item.event === 'tool_call_start') {
      const data = item.data as { name?: string };
      updateRuntime(snapshot.key, (runtime) => ({ ...runtime, toolActivity: { name: data.name ?? 'unknown', status: 'running' }, updatedAt: Date.now() }), snapshot);
    } else if (item.event === 'tool_call_result') {
      const data = item.data as { name?: string; status?: string };
      const status: ToolActivity['status'] = data.status === 'success' ? 'success' : data.status === 'timeout' ? 'timeout' : 'failed';
      updateRuntime(snapshot.key, (runtime) => ({
        ...runtime,
        toolActivity: status === 'success' ? undefined : { name: data.name ?? 'unknown', status },
        updatedAt: Date.now(),
      }), snapshot);
    } else if (item.event === 'memory_status') {
      const data = item.data as { compressed?: boolean };
      updateRuntime(snapshot.key, (runtime) => ({ ...runtime, memoryMessage: data.compressed ? '本轮对话已完成摘要与记忆整理' : '本轮对话已保存', updatedAt: Date.now() }), snapshot);
    } else if (item.event === 'citations') {
      const raw = Array.isArray(item.data) ? item.data : [];
      const citations = raw.filter((citation): citation is Citation => Boolean(citation && typeof citation === 'object'));
      updateAssistant(snapshot, (message) => ({ ...message, citations }));
    } else if (item.event === 'message_end') {
      updateAssistant(snapshot, (message) => ({ ...message, status: 'complete' }));
      const visible = currentSelection.current.userId === snapshot.userId && currentSelection.current.conversationId === snapshot.conversationId;
      updateRuntime(snapshot.key, (runtime) => ({
        ...runtime,
        status: 'completed',
        activeAgent: undefined,
        statusMessage: undefined,
        toolActivity: undefined,
        unread: !visible,
        updatedAt: Date.now(),
      }), snapshot);
    } else if (item.event === 'error') {
      const data = item.data as { message?: string };
      streamFailure.message = data.message || '回答生成失败';
    }
  }

  async function send(questionOverride?: string) {
    const ownerId = userId;
    const question = (questionOverride ?? draft).trim();
    if (!question || !ownerId || usersLoading) return;
    let targetConversationId = conversationId;
    if (!targetConversationId) {
      const created = await createConversation(ownerId);
      if (!created) return;
      targetConversationId = created.id;
    }
    const key = conversationKey(ownerId, targetConversationId);
    if (runtimeActive(runtimeRef.current[key]) || controllers.current.has(key)) return;

    const token = ++requestSequence.current;
    const assistantMessageId = `assistant-${token}`;
    const snapshot: StreamSnapshot = { token, key, userId: ownerId, conversationId: targetConversationId, assistantMessageId };
    const controller = new AbortController();
    controllers.current.set(key, controller);
    setDrafts((current) => ({ ...current, [draftKey]: '' }));
    updateRuntime(key, (runtime) => ({
      ...runtime,
      messages: [
        ...runtime.messages,
        { id: `user-${token}`, role: 'user', content: question, status: 'complete' },
        { id: assistantMessageId, role: 'assistant', content: '', status: 'streaming' },
      ],
      status: 'analyzing',
      activeAgent: 'case_analyst',
      statusMessage: '正在启动案情分析',
      requestToken: token,
      toolActivity: undefined,
      memoryMessage: '',
      error: '',
      failedQuestion: '',
      loading: false,
      unread: false,
      updatedAt: Date.now(),
    }), snapshot);

    const streamFailure = { message: '' };
    try {
      const response = await api.streamMessage(ownerId, targetConversationId, question, controller.signal);
      await consumeSse(response, (item) => handleStreamEvent(snapshot, item, streamFailure));
      if (!isCurrentStream(snapshot)) return;
      if (streamFailure.message) throw new Error(streamFailure.message);
      updateAssistant(snapshot, (message) => ({ ...message, status: 'complete' }));
      const visible = currentSelection.current.userId === ownerId && currentSelection.current.conversationId === targetConversationId;
      updateRuntime(key, (runtime) => ({
        ...runtime,
        status: 'completed',
        activeAgent: undefined,
        statusMessage: undefined,
        toolActivity: undefined,
        unread: !visible,
        updatedAt: Date.now(),
      }), snapshot);
    } catch (error) {
      if (!isCurrentStream(snapshot)) return;
      if (controller.signal.aborted) {
        updateAssistant(snapshot, (message) => ({ ...message, status: 'interrupted' }));
        updateRuntime(key, (runtime) => ({ ...runtime, status: 'interrupted', toolActivity: undefined, updatedAt: Date.now() }), snapshot);
      } else {
        updateAssistant(snapshot, (message) => ({ ...message, status: 'error' }));
        updateRuntime(key, (runtime) => ({ ...runtime, status: 'failed', error: errorMessage(error), failedQuestion: question, toolActivity: undefined, updatedAt: Date.now() }), snapshot);
      }
    } finally {
      if (controllers.current.get(key) === controller) controllers.current.delete(key);
    }
  }

  function stopCurrentStream() {
    if (!selectedKey) return;
    controllers.current.get(selectedKey)?.abort('user-stop');
  }

  function setDraft(value: string) {
    setDrafts((current) => ({ ...current, [draftKey]: value }));
  }

  const agentActivity: AgentActivity | null = selectedRuntime?.activeAgent && selectedRuntime.statusMessage
    ? { agent: selectedRuntime.activeAgent, status: selectedRuntime.status, message: selectedRuntime.statusMessage }
    : null;

  return (
    <main className="app-shell">
      <Sidebar
        users={users}
        userId={userId}
        conversations={conversations}
        activeConversationId={conversationId}
        open={sidebarOpen}
        loading={bucket.loading}
        error={bucket.error || conversationBuckets['']?.error || ''}
        disabled={usersLoading}
        taskStatuses={taskStatuses}
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
          loading={Boolean(selectedRuntime?.loading)}
          conversationSelected={Boolean(conversationId)}
          onSuggestion={setDraft}
        />
        <div className="bottom-dock">
          <StatusNotice
            index={index}
            agent={agentActivity}
            tool={selectedRuntime?.toolActivity ?? null}
            memoryMessage={selectedRuntime?.memoryMessage ?? ''}
            error={selectedRuntime?.error ?? ''}
            failedQuestion={selectedRuntime?.failedQuestion ?? ''}
            onRetry={() => void send(selectedRuntime?.failedQuestion)}
            onDismissError={() => selectedKey && updateRuntime(selectedKey, (runtime) => ({ ...runtime, error: '', failedQuestion: '', updatedAt: Date.now() }))}
          />
          <Composer
            value={draft}
            streaming={streaming}
            disabled={!userId || usersLoading || Boolean(selectedRuntime?.loading)}
            onChange={setDraft}
            onSend={() => void send()}
            onStop={stopCurrentStream}
          />
        </div>
      </section>
    </main>
  );
}
