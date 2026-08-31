import { useEffect, useMemo, useRef, useState } from 'react';
import { api } from './api';
import { consumeSse } from './sse';
import { conversationKey, emptyRuntime, isActiveStage, type ConversationKey } from './chat/runtimeStore';
import { ChatHeader } from './components/ChatHeader';
import { Composer } from './components/Composer';
import { MessageList } from './components/MessageList';
import { MemoryPanel } from './components/MemoryPanel';
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
  SkillActivity,
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
  runId: string;
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
  const [memoryOpen, setMemoryOpen] = useState(false);

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
    setMemoryOpen(false);
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
    if (cached && runtimeActive(cached)) return;

    const sequence = (loadSequences.current.get(key) ?? 0) + 1;
    loadSequences.current.set(key, sequence);
    updateRuntime(key, (runtime) => ({ ...runtime, loading: true, error: '', updatedAt: Date.now() }), { userId: ownerId, conversationId: id });
    try {
      const [data, activeRun] = await Promise.all([
        api.messages(ownerId, id),
        api.activeRun(ownerId, id),
      ]);
      const latest = runtimeRef.current[key];
      if (sequence !== loadSequences.current.get(key) || runtimeActive(latest)) return;
      updateRuntime(key, (runtime) => ({ ...runtime, messages: data, loading: false, updatedAt: Date.now() }), { userId: ownerId, conversationId: id });
      if (activeRun && !controllers.current.has(key)) {
        const token = ++requestSequence.current;
        const assistantMessageId = `assistant-${activeRun.id}`;
        const snapshot: StreamSnapshot = {
          token, key, userId: ownerId, conversationId: id, assistantMessageId, runId: activeRun.id,
        };
        const controller = new AbortController();
        controllers.current.set(key, controller);
        updateRuntime(key, (runtime) => ({
          ...runtime,
          messages: [
            ...runtime.messages,
            ...(runtime.messages.some((message) => message.id === assistantMessageId)
              ? []
              : [{ id: assistantMessageId, role: 'assistant' as const, content: '', status: 'streaming' as const }]),
          ],
          status: activeRun.current_stage,
          runId: activeRun.id,
          serverStatus: activeRun.status,
          requestToken: token,
          lastEventSequence: 0,
          reconnecting: true,
          updatedAt: Date.now(),
        }), snapshot);
        void followRun(snapshot, controller, activeRun.input_text);
      }
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
      messages: runtime.messages.some((message) => message.id === snapshot.assistantMessageId)
        ? runtime.messages.map((message) => (
          message.id === snapshot.assistantMessageId ? update(message) : message
        ))
        : [...runtime.messages, update({ id: snapshot.assistantMessageId, role: 'assistant', content: '', status: 'streaming' })],
      updatedAt: Date.now(),
    }), snapshot);
  }

  function handleStreamEvent(snapshot: StreamSnapshot, item: SseEvent, streamFailure: { message: string }) {
    if (!isCurrentStream(snapshot)) return;
    if (item.id !== undefined) {
      updateRuntime(snapshot.key, (runtime) => ({
        ...runtime,
        lastEventSequence: Math.max(runtime.lastEventSequence ?? 0, item.id ?? 0),
        reconnecting: false,
        updatedAt: Date.now(),
      }), snapshot);
    }
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
    } else if (item.event === 'skill_status') {
      const data = item.data as { skill_id?: string; status?: SkillActivity['status']; message?: string };
      const status = data.status ?? 'running';
      updateRuntime(snapshot.key, (runtime) => ({
        ...runtime,
        skillActivity: status === 'completed'
          ? undefined
          : {
              skillId: data.skill_id ?? 'unknown',
              status,
              message: data.message ?? '正在执行领域能力',
            },
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
      const data = item.data as { status?: string; job_id?: string; compressed?: boolean };
      updateRuntime(snapshot.key, (runtime) => ({ ...runtime, memoryMessage: data.status === 'pending' ? '本轮对话已保存，正在后台整理记忆' : data.compressed ? '本轮对话已完成摘要与记忆整理' : '本轮对话已保存', updatedAt: Date.now() }), snapshot);
      if (data.job_id) void pollMemoryJob(snapshot, data.job_id);
    } else if (item.event === 'citations') {
      const raw = Array.isArray(item.data) ? item.data : [];
      const citations = raw.filter((citation): citation is Citation => Boolean(citation && typeof citation === 'object'));
      updateAssistant(snapshot, (message) => ({ ...message, citations }));
    } else if (item.event === 'message_end') {
      const data = item.data as { message_id?: string; status?: 'completed' | 'interrupted' | 'failed' };
      const messageStatus = data.status === 'interrupted' ? 'interrupted' : data.status === 'failed' ? 'error' : 'complete';
      updateAssistant(snapshot, (message) => ({ ...message, id: data.message_id ?? message.id, status: messageStatus }));
      const visible = currentSelection.current.userId === snapshot.userId && currentSelection.current.conversationId === snapshot.conversationId;
      updateRuntime(snapshot.key, (runtime) => ({
        ...runtime,
        status: data.status ?? 'completed',
        serverStatus: data.status ?? 'completed',
        activeAgent: undefined,
        statusMessage: undefined,
        toolActivity: undefined,
        skillActivity: undefined,
        unread: !visible,
        updatedAt: Date.now(),
      }), snapshot);
    } else if (item.event === 'error') {
      const data = item.data as { message?: string };
      streamFailure.message = data.message || '回答生成失败';
    }
  }

  async function pollMemoryJob(snapshot: StreamSnapshot, jobId: string) {
    for (let attempt = 0; attempt < 30; attempt += 1) {
      await new Promise((resolve) => window.setTimeout(resolve, 1000));
      if (!isCurrentStream(snapshot)) return;
      try {
        const job = await api.memoryJob(snapshot.userId, jobId);
        if (job.status === 'completed') {
          const message = job.candidate_count > 0
            ? `本轮记忆整理完成，${job.candidate_count} 条记忆已更新`
            : '本轮无需新增记忆';
          updateRuntime(snapshot.key, (runtime) => ({ ...runtime, memoryMessage: message, updatedAt: Date.now() }), snapshot);
          return;
        }
        if (job.status === 'failed') {
          updateRuntime(snapshot.key, (runtime) => ({ ...runtime, memoryMessage: '回答已保存，但本轮记忆整理失败', updatedAt: Date.now() }), snapshot);
          return;
        }
      } catch {
        return;
      }
    }
  }

  async function followRun(snapshot: StreamSnapshot, controller: AbortController, question: string) {
    const streamFailure = { message: '' };
    try {
      while (!controller.signal.aborted && isCurrentStream(snapshot)) {
        const after = runtimeRef.current[snapshot.key]?.lastEventSequence ?? 0;
        try {
          const response = await api.runEvents(
            snapshot.userId, snapshot.runId, after, controller.signal,
          );
          await consumeSse(response, (item) => handleStreamEvent(snapshot, item, streamFailure));
        } catch (error) {
          if (controller.signal.aborted || !isCurrentStream(snapshot)) return;
          updateRuntime(snapshot.key, (runtime) => ({
            ...runtime, reconnecting: true, statusMessage: '连接中断，正在恢复任务进度', updatedAt: Date.now(),
          }), snapshot);
          await new Promise((resolve) => window.setTimeout(resolve, 750));
          continue;
        }
        if (streamFailure.message) throw new Error(streamFailure.message);
        const serverRun = await api.agentRun(snapshot.userId, snapshot.runId, controller.signal);
        updateRuntime(snapshot.key, (runtime) => ({
          ...runtime, serverStatus: serverRun.status, updatedAt: Date.now(),
        }), snapshot);
        if (['completed', 'interrupted', 'failed'].includes(serverRun.status)) {
          const finalStage: AgentStage = serverRun.status === 'completed'
            ? 'completed'
            : serverRun.status === 'interrupted' ? 'interrupted' : 'failed';
          const persisted = await api.messages(snapshot.userId, snapshot.conversationId, controller.signal);
          const visible = currentSelection.current.userId === snapshot.userId
            && currentSelection.current.conversationId === snapshot.conversationId;
          updateRuntime(snapshot.key, (runtime) => ({
            ...runtime,
            messages: persisted,
            status: finalStage,
            activeAgent: undefined,
            statusMessage: undefined,
            toolActivity: undefined,
            skillActivity: undefined,
            reconnecting: false,
            unread: serverRun.status === 'completed' && !visible,
            error: serverRun.status === 'failed' ? (serverRun.error_summary || '回答生成失败') : '',
            failedQuestion: serverRun.status === 'failed' ? question : '',
            updatedAt: Date.now(),
          }), snapshot);
          return;
        }
      }
    } finally {
      if (controllers.current.get(snapshot.key) === controller) {
        controllers.current.delete(snapshot.key);
      }
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

    try {
      const run = await api.createRun(ownerId, targetConversationId, question);
      const token = ++requestSequence.current;
      const assistantMessageId = `assistant-${run.id}`;
      const snapshot: StreamSnapshot = {
        token, key, userId: ownerId, conversationId: targetConversationId, assistantMessageId, runId: run.id,
      };
      const controller = new AbortController();
      controllers.current.set(key, controller);
      setDrafts((current) => ({ ...current, [draftKey]: '' }));
      updateRuntime(key, (runtime) => ({
        ...runtime,
        messages: [
          ...runtime.messages,
          { id: `user-${run.id}`, role: 'user', content: question, status: 'complete' },
          { id: assistantMessageId, role: 'assistant', content: '', status: 'streaming' },
        ],
        status: 'queued',
        activeAgent: 'coordinator',
        statusMessage: '咨询任务正在排队',
        runId: run.id,
        serverStatus: run.status,
        lastEventSequence: 0,
        requestToken: token,
        toolActivity: undefined,
        skillActivity: undefined,
        memoryMessage: '',
        error: '',
        failedQuestion: '',
        loading: false,
        unread: false,
        updatedAt: Date.now(),
      }), snapshot);
      await followRun(snapshot, controller, question);
    } catch (error) {
      updateRuntime(key, (runtime) => ({
        ...runtime,
        status: 'failed',
        error: errorMessage(error),
        failedQuestion: question,
        toolActivity: undefined,
        skillActivity: undefined,
        updatedAt: Date.now(),
      }), { userId: ownerId, conversationId: targetConversationId });
    }
  }

  async function stopCurrentStream() {
    if (!selectedKey) return;
    const runtime = runtimeRef.current[selectedKey];
    if (runtime?.runId) {
      try {
        await api.cancelRun(runtime.userId, runtime.runId);
      } finally {
        controllers.current.get(selectedKey)?.abort('user-stop');
      }
    }
  }

  function setDraft(value: string) {
    setDrafts((current) => ({ ...current, [draftKey]: value }));
  }

  async function submitFeedback(messageId: string, score: -1 | 1, comment = '') {
    if (!userId || !selectedKey) return;
    await api.messageFeedback(userId, messageId, score, comment);
    updateRuntime(selectedKey, (runtime) => ({
      ...runtime,
      messages: runtime.messages.map((message) => (
        message.id === messageId ? { ...message, feedback_score: score } : message
      )),
      updatedAt: Date.now(),
    }), { userId, conversationId });
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
        onOpenMemories={() => { setMemoryOpen(true); setSidebarOpen(false); }}
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
          onFeedback={submitFeedback}
        />
        <div className="bottom-dock">
          <StatusNotice
            index={index}
            agent={agentActivity}
            tool={selectedRuntime?.toolActivity ?? null}
            skill={selectedRuntime?.skillActivity ?? null}
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
      <MemoryPanel
        open={memoryOpen}
        userId={userId}
        conversationId={conversationId}
        onClose={() => setMemoryOpen(false)}
      />
    </main>
  );
}
