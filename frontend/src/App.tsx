import { useEffect, useMemo, useRef, useState } from 'react';
import { api, ApiError } from './api';
import { consumeSse } from './sse';
import { conversationKey, emptyRuntime, isActiveStage, type ConversationKey } from './chat/runtimeStore';
import {
  compareScenarioOutcome,
  isolatedConversationMemoryEvidence,
  latestFactSourceEvidence,
  replayEvidence,
  reusedUserMemoryEvidence,
} from './chat/scenarioExecutor';
import { ChatHeader } from './components/ChatHeader';
import { Composer } from './components/Composer';
import { MessageList } from './components/MessageList';
import { MemoryPanel } from './components/MemoryPanel';
import { Sidebar } from './components/Sidebar';
import { ScenarioPanel } from './components/ScenarioPanel';
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
  DialogueScenario,
  MemoryJob,
  ScenarioAvailability,
  ScenarioRunOutcome,
  ScenarioSession,
  ScenarioStepResult,
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
  runId: string;
}

function scenarioBindingKey(actor: string, conversation: string): string {
  return `${actor}:${conversation}`;
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
  const [scenarioOpen, setScenarioOpen] = useState(false);
  const [scenarioAvailability, setScenarioAvailability] = useState<ScenarioAvailability>({
    status: 'checking', datasets: [], message: '正在检查场景观察模式',
  });
  const [scenarioReloadToken, setScenarioReloadToken] = useState(0);
  const [scenarioSession, setScenarioSession] = useState<ScenarioSession | null>(null);

  const controllers = useRef(new Map<ConversationKey, AbortController>());
  const requestSequence = useRef(0);
  const loadSequences = useRef(new Map<ConversationKey, number>());
  const runtimeRef = useRef(runtimes);
  const currentSelection = useRef({ userId, conversationId });
  const scenarioSessionRef = useRef<ScenarioSession | null>(scenarioSession);
  const scenarioLoadSequence = useRef(0);
  const pendingScenarioSelection = useRef<{ userId: string; conversationId: string } | null>(null);
  runtimeRef.current = runtimes;
  currentSelection.current = { userId, conversationId };
  scenarioSessionRef.current = scenarioSession;

  const bucket = conversationBuckets[userId] ?? { items: [], loading: false, error: '' };
  const selectedKey = userId && conversationId ? conversationKey(userId, conversationId) : null;
  const selectedRuntime = selectedKey ? runtimes[selectedKey] : undefined;
  const draftKey = `${userId}:${conversationId || 'new'}`;
  const draft = drafts[draftKey] ?? '';
  const conversations = bucket.items;
  const messages = selectedRuntime?.messages ?? [];
  const scenarioDatasets = scenarioAvailability.datasets;
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

  useEffect(() => {
    const sequence = scenarioLoadSequence.current + 1;
    scenarioLoadSequence.current = sequence;
    if (!userId) {
      setScenarioAvailability({ status: 'checking', datasets: [], message: '正在等待用户信息' });
      return;
    }
    const controller = new AbortController();
    setScenarioAvailability({ status: 'checking', datasets: [], message: '正在加载场景数据集' });
    api.scenarioDatasets(userId, controller.signal)
      .then((datasets) => {
        if (sequence !== scenarioLoadSequence.current) return;
        setScenarioAvailability(datasets.length
          ? { status: 'ready', datasets, message: `已加载 ${datasets.length} 个场景数据集` }
          : { status: 'error', datasets: [], message: '场景模式已启用，但没有可用数据集' });
      })
      .catch((error) => {
        if (error instanceof DOMException && error.name === 'AbortError') return;
        if (sequence !== scenarioLoadSequence.current) return;
        if (error instanceof ApiError && error.status === 404) {
          setScenarioAvailability({ status: 'disabled', datasets: [], message: '场景观察模式未启用' });
          return;
        }
        setScenarioAvailability({
          status: 'error', datasets: [], message: `场景模式加载失败：${errorMessage(error)}`,
        });
      });
    return () => controller.abort();
  }, [userId, scenarioReloadToken]);

  useEffect(() => {
    const pending = pendingScenarioSelection.current;
    if (!pending || pending.userId !== userId || bucket.loading) return;
    if (!bucket.items.some((item) => item.id === pending.conversationId)) return;
    pendingScenarioSelection.current = null;
    void openConversation(pending.conversationId);
  }, [userId, bucket.loading, bucket.items]);

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

  function updateScenario(update: (session: ScenarioSession) => ScenarioSession) {
    const current = scenarioSessionRef.current;
    if (!current) return;
    const next = update(current);
    scenarioSessionRef.current = next;
    setScenarioSession(next);
  }

  function setScenarioResult(stepIndex: number, value: ScenarioStepResult) {
    updateScenario((session) => ({
      ...session,
      results: [
        ...session.results.filter((item) => item.stepIndex !== stepIndex),
        value,
      ].sort((left, right) => left.stepIndex - right.stepIndex),
    }));
  }

  function beginScenarioSubscription(runId: string, connectedAfterSequence: number): number {
    const observation = scenarioSessionRef.current?.observations[runId];
    if (!observation) return 0;
    const latestSubscription = observation.subscriptions[observation.subscriptions.length - 1];
    const epoch = (latestSubscription?.epoch ?? 0) + 1;
    updateScenario((session) => ({
      ...session,
      observations: {
        ...session.observations,
        [runId]: {
          ...session.observations[runId],
          subscriptions: [
            ...session.observations[runId].subscriptions,
            { epoch, connectedAfterSequence, receivedSequences: [] },
          ],
        },
      },
    }));
    return epoch;
  }

  function recordScenarioEvent(runId: string, item: SseEvent) {
    const current = scenarioSessionRef.current;
    const observation = current?.observations[runId];
    if (!current || !observation) return;
    const sequence = item.id;
    const duplicate = sequence !== undefined && observation.sequences.includes(sequence);
    const data = item.data as { job_id?: string } | unknown;
    const jobId = item.event === 'memory_status' && typeof data === 'object' && data
      ? (data as { job_id?: string }).job_id : undefined;
    updateScenario((session) => ({
      ...session,
      observations: {
        ...session.observations,
        [runId]: (() => {
          const currentObservation = session.observations[runId];
          const subscriptions = currentObservation.subscriptions.map((subscription, index) => (
            index !== currentObservation.subscriptions.length - 1 || sequence === undefined
              ? subscription
              : {
                  ...subscription,
                  firstReceivedSequence: subscription.firstReceivedSequence ?? sequence,
                  receivedSequences: duplicate
                    ? subscription.receivedSequences
                    : [...subscription.receivedSequences, sequence],
                }
          ));
          return {
          ...currentObservation,
          events: duplicate
            ? currentObservation.events
            : [...currentObservation.events, item.event],
          sequences: sequence === undefined || duplicate
            ? currentObservation.sequences
            : [...currentObservation.sequences, sequence],
          duplicateSequence: currentObservation.duplicateSequence || duplicate,
          memoryJobIds: jobId && !currentObservation.memoryJobIds.includes(jobId)
            ? [...currentObservation.memoryJobIds, jobId]
            : currentObservation.memoryJobIds,
          subscriptions,
        }; })(),
      },
    }));
  }

  async function createConversation(
    ownerId = userId,
    title = '法律咨询',
    selectConversation = true,
  ): Promise<Conversation | null> {
    if (!ownerId) return null;
    try {
      const conversation = await api.createConversation(ownerId, title);
      setConversationBuckets((current) => ({
        ...current,
        [ownerId]: {
          items: [conversation, ...(current[ownerId]?.items ?? [])],
          loading: false,
          error: '',
        },
      }));
      if (selectConversation && currentSelection.current.userId === ownerId) {
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
        launchFollowRun(snapshot, controller, activeRun.input_text);
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
    recordScenarioEvent(snapshot.runId, item);
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

  async function refreshScenarioOutcome(user: string, runId: string): Promise<ScenarioRunOutcome | null> {
    const session = scenarioSessionRef.current;
    const observation = session?.observations[runId];
    if (!session || !observation) return null;
    try {
      const outcome = await api.scenarioOutcome(user, runId);
      const merged = {
        ...outcome,
        observed_events: observation.events.length ? observation.events : outcome.observed_events,
      };
      updateScenario((current) => ({
        ...current,
        observations: {
          ...current.observations,
          [runId]: { ...current.observations[runId], outcome: merged },
        },
        results: current.results.map((result) => {
          if (result.runId !== runId) return result;
          const step = current.scenario.steps[result.stepIndex];
          const comparison = compareScenarioOutcome(
            step,
            merged,
            current.scenario.fixture_types.length > 0,
          );
          return {
            ...result,
            status: comparison.status,
            message: comparison.message,
            actual: { ...merged, checks: comparison.checks },
          };
        }),
      }));
      return merged;
    } catch {
      return null;
    }
  }

  async function followRun(snapshot: StreamSnapshot, controller: AbortController, question: string) {
    const streamFailure = { message: '' };
    try {
      while (!controller.signal.aborted && isCurrentStream(snapshot)) {
        const after = runtimeRef.current[snapshot.key]?.lastEventSequence ?? 0;
        try {
          beginScenarioSubscription(snapshot.runId, after);
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
            reconnecting: false,
            unread: serverRun.status === 'completed' && !visible,
            error: serverRun.status === 'failed' ? (serverRun.error_summary || '回答生成失败') : '',
            failedQuestion: serverRun.status === 'failed' ? question : '',
            updatedAt: Date.now(),
          }), snapshot);
          await refreshScenarioOutcome(snapshot.userId, snapshot.runId);
          return;
        }
      }
    } finally {
      if (controllers.current.get(snapshot.key) === controller) {
        controllers.current.delete(snapshot.key);
      }
    }
  }

  function launchFollowRun(
    snapshot: StreamSnapshot,
    controller: AbortController,
    question: string,
  ) {
    void followRun(snapshot, controller, question).catch((error) => {
      if (controller.signal.aborted || !isCurrentStream(snapshot)) return;
      updateRuntime(snapshot.key, (runtime) => ({
        ...runtime,
        reconnecting: false,
        error: errorMessage(error),
        failedQuestion: question,
        updatedAt: Date.now(),
      }), snapshot);
    });
  }

  function attachRun(ownerId: string, targetConversationId: string, question: string, run: Awaited<ReturnType<typeof api.createRun>>) {
    const key = conversationKey(ownerId, targetConversationId);
    const token = ++requestSequence.current;
    const assistantMessageId = `assistant-${run.id}`;
    const snapshot: StreamSnapshot = {
      token, key, userId: ownerId, conversationId: targetConversationId, assistantMessageId, runId: run.id,
    };
    const controller = new AbortController();
    controllers.current.set(key, controller);
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
      memoryMessage: '',
      error: '',
      failedQuestion: '',
      loading: false,
      unread: false,
      updatedAt: Date.now(),
    }), snapshot);
    launchFollowRun(snapshot, controller, question);
    return { run, snapshot };
  }

  async function startRun(ownerId: string, targetConversationId: string, question: string) {
    const key = conversationKey(ownerId, targetConversationId);
    if (runtimeActive(runtimeRef.current[key]) || controllers.current.has(key)) {
      throw new ApiError('该会话正在生成回答，请等待完成或先停止生成。', 409);
    }
    const run = await api.createRun(ownerId, targetConversationId, question);
    return attachRun(ownerId, targetConversationId, question, run);
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

    try {
      await startRun(ownerId, targetConversationId, question);
      setDrafts((current) => ({ ...current, [draftKey]: '' }));
    } catch (error) {
      updateRuntime(key, (runtime) => ({
        ...runtime,
        status: 'failed',
        error: errorMessage(error),
        failedQuestion: question,
        toolActivity: undefined,
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

  async function showScenarioBinding(user: string, targetConversationId: string) {
    if (currentSelection.current.userId === user) {
      await openConversation(targetConversationId);
      return;
    }
    pendingScenarioSelection.current = { userId: user, conversationId: targetConversationId };
    setUserId(user);
  }

  async function startScenario(datasetId: string, scenario: DialogueScenario) {
    if (!userId) throw new Error('请先选择主测试用户');
    const secondary = users.find((item) => item.id !== userId);
    if (scenario.actors.includes('secondary') && !secondary) {
      throw new Error('该场景需要两个演示用户');
    }
    const actorUsers: Record<string, string> = {
      primary: userId,
      ...(secondary ? { secondary: secondary.id } : {}),
    };
    const pairs = [...new Set(scenario.steps.map((step) => scenarioBindingKey(step.actor, step.conversation)))];
    const bindings: ScenarioSession['bindings'] = {};
    const created: Array<{ userId: string; conversationId: string }> = [];
    try {
      for (const pair of pairs) {
        const [actor, conversationName] = pair.split(':');
        const owner = actorUsers[actor];
        if (!owner) throw new Error(`没有为 Actor ${actor} 配置用户`);
        const title = `[场景] ${scenario.scenario_id} · ${actor}/${conversationName}`;
        const conversation = await createConversation(owner, title, false);
        if (!conversation) throw new Error(`无法创建场景会话：${pair}`);
        created.push({ userId: owner, conversationId: conversation.id });
        bindings[pair] = {
          key: pair,
          actor,
          conversation: conversationName,
          userId: owner,
          conversationId: conversation.id,
          title,
        };
      }
    } catch (error) {
      for (const item of created) {
        await api.deleteScenarioConversation(item.userId, item.conversationId).catch(() => undefined);
      }
      throw error;
    }
    const next: ScenarioSession = {
      id: crypto.randomUUID(),
      datasetId,
      scenario,
      stepIndex: 0,
      bindings,
      results: [],
      runByBinding: {},
      observations: {},
      backgroundRunIds: [],
      concurrentRunPairs: [],
      executing: false,
      error: '',
    };
    scenarioSessionRef.current = next;
    setScenarioSession(next);
    const initial = bindings[scenarioBindingKey('primary', 'main')] ?? Object.values(bindings)[0];
    if (initial) await showScenarioBinding(initial.userId, initial.conversationId);
  }

  async function waitForScenarioRun(user: string, runId: string, timeoutMs: number) {
    const deadline = Date.now() + timeoutMs;
    while (Date.now() < deadline) {
      const run = await api.agentRun(user, runId);
      if (['completed', 'interrupted', 'failed'].includes(run.status)) return run;
      await new Promise((resolve) => window.setTimeout(resolve, 500));
    }
    throw new Error('等待场景任务完成超时');
  }

  async function waitForObservedEvent(runId: string, eventName: string, timeoutMs: number) {
    const deadline = Date.now() + timeoutMs;
    while (Date.now() < deadline) {
      if (scenarioSessionRef.current?.observations[runId]?.events.includes(eventName)) return;
      await new Promise((resolve) => window.setTimeout(resolve, 100));
    }
    throw new Error(`等待事件 ${eventName} 超时`);
  }

  async function waitForScenarioSequence(
    runId: string,
    subscriptionEpoch: number,
    timeoutMs: number,
  ) {
    const deadline = Date.now() + timeoutMs;
    while (Date.now() < deadline) {
      const observation = scenarioSessionRef.current?.observations[runId];
      const subscription = observation?.subscriptions.find((item) => item.epoch === subscriptionEpoch);
      if (subscription?.firstReceivedSequence !== undefined) return { observation, subscription };
      await new Promise((resolve) => window.setTimeout(resolve, 100));
    }
    throw new Error('重连后未在限定时间内收到新事件');
  }

  async function waitForServerSequence(
    user: string,
    runId: string,
    afterSequence: number,
    timeoutMs: number,
  ) {
    const deadline = Date.now() + timeoutMs;
    while (Date.now() < deadline) {
      const run = await api.agentRun(user, runId);
      if (run.last_event_sequence > afterSequence) return run.last_event_sequence;
      if (['completed', 'interrupted', 'failed'].includes(run.status)) return run.last_event_sequence;
      await new Promise((resolve) => window.setTimeout(resolve, 200));
    }
    return afterSequence;
  }

  async function waitForScenarioMemoryJobs(user: string, runId: string, timeoutMs: number) {
    const deadline = Date.now() + timeoutMs;
    while (Date.now() < deadline) {
      const jobIds = scenarioSessionRef.current?.observations[runId]?.memoryJobIds ?? [];
      if (!jobIds.length) {
        await new Promise((resolve) => window.setTimeout(resolve, 250));
        continue;
      }
      const jobs = await Promise.all(jobIds.map((jobId) => api.memoryJob(user, jobId)));
      if (jobs.every((job) => ['completed', 'failed'].includes(job.status))) return jobs;
      await new Promise((resolve) => window.setTimeout(resolve, 500));
    }
    return [];
  }

  async function scenarioMemoryEvidence(session: ScenarioSession, timeoutMs: number) {
    const latestSession = scenarioSessionRef.current ?? session;
    const observations = Object.values(latestSession.observations);
    const jobs: MemoryJob[] = [];
    for (const observation of observations) {
      jobs.push(...await waitForScenarioMemoryJobs(
        observation.userId, observation.runId, timeoutMs,
      ));
    }
    const messageEntries = await Promise.all(Object.values(latestSession.bindings).map(async (binding) => ({
      binding,
      messages: await api.messages(binding.userId, binding.conversationId),
    })));
    return {
      messageEntries,
      jobsReady: observations.length > 0
        && jobs.length >= observations.length
        && jobs.every((job) => job.status === 'completed'),
    };
  }

  async function executeScenarioNext() {
    const session = scenarioSessionRef.current;
    if (!session || session.executing || session.stepIndex >= session.scenario.steps.length) return;
    const index = session.stepIndex;
    const step = session.scenario.steps[index];
    const binding = session.bindings[scenarioBindingKey(step.actor, step.conversation)];
    if (!binding) throw new Error(`场景会话映射不存在：${step.actor}/${step.conversation}`);
    const timeoutSeconds = scenarioDatasets.find((item) => item.id === session.datasetId)?.step_timeout_seconds ?? 60;
    const timeoutMs = timeoutSeconds * 1000;
    updateScenario((current) => ({ ...current, executing: true, error: '' }));
    setScenarioResult(index, {
      stepIndex: index,
      status: 'running',
      message: '正在执行当前步骤',
      expected: step.expected,
    });
    try {
      if (step.action === 'send_message') {
        const expectsConflict = step.expected?.terminal === 'rejected_409';
        try {
          const run = await api.createRun(binding.userId, binding.conversationId, step.content ?? '');
          if (expectsConflict) {
            await api.cancelRun(binding.userId, run.id).catch(() => undefined);
            setScenarioResult(index, {
              stepIndex: index,
              status: 'failed',
              message: '预期服务端返回 409，但任务被接受。',
              expected: step.expected,
              actual: { status: run.status, run_id: run.id },
            });
          } else {
            const existingRuns = Object.entries(session.runByBinding);
            const overlapping = (await Promise.all(existingRuns.map(async ([existingKey, existingRunId]) => {
              try {
                const existingOwner = session.bindings[existingKey]?.userId;
                if (!existingOwner) return null;
                const existing = await api.agentRun(existingOwner, existingRunId);
                return ['queued', 'running'].includes(existing.status) ? existingRunId : null;
              } catch {
                return null;
              }
            }))).filter((item): item is string => Boolean(item));
            updateScenario((current) => ({
              ...current,
              runByBinding: { ...current.runByBinding, [binding.key]: run.id },
              concurrentRunPairs: [
                ...current.concurrentRunPairs,
                ...overlapping.map((existingRunId) => [existingRunId, run.id].sort().join(':')),
              ],
              observations: {
                ...current.observations,
                [run.id]: {
                  runId: run.id,
                  userId: binding.userId,
                  conversationId: binding.conversationId,
                  events: [],
                  sequences: [],
                  duplicateSequence: false,
                  memoryJobIds: [],
                  subscriptions: [],
                },
              },
            }));
            attachRun(binding.userId, binding.conversationId, step.content ?? '', run);
            setScenarioResult(index, {
              stepIndex: index,
              status: 'running',
              message: '消息已提交，等待真实 Agent 任务完成。',
              runId: run.id,
              expected: step.expected,
              actual: { run_id: run.id, accepted: true },
            });
          }
        } catch (error) {
          if (expectsConflict && error instanceof ApiError && error.status === 409) {
            setScenarioResult(index, {
              stepIndex: index,
              status: 'passed',
              message: '服务端按预期拒绝同会话并发任务（409）。',
              expected: step.expected,
              actual: { status: 409 },
            });
          } else {
            throw error;
          }
        }
      } else if (step.action === 'switch_user' || step.action === 'switch_conversation') {
        const backgroundRunIds = (await Promise.all(Object.entries(session.runByBinding).map(async ([key, runId]) => {
          const owner = session.bindings[key];
          if (!owner || owner.userId === binding.userId) return null;
          try {
            const run = await api.agentRun(owner.userId, runId);
            return ['queued', 'running'].includes(run.status) ? runId : null;
          } catch {
            return null;
          }
        }))).filter((item): item is string => Boolean(item));
        if (backgroundRunIds.length) {
          updateScenario((current) => ({
            ...current,
            backgroundRunIds: [...new Set([...current.backgroundRunIds, ...backgroundRunIds])],
          }));
        }
        await showScenarioBinding(binding.userId, binding.conversationId);
        setScenarioResult(index, {
          stepIndex: index,
          status: 'passed',
          message: `已切换到 ${step.actor}/${step.conversation}`,
          expected: step.expected,
          actual: { user_id: binding.userId, conversation_id: binding.conversationId },
        });
      } else {
        const runId = session.runByBinding[binding.key];
        if (!runId) throw new Error('当前步骤找不到对应的 AgentRun');
        if (step.action === 'wait_for_completion') {
          const run = await waitForScenarioRun(binding.userId, runId, timeoutMs);
          const outcome = await refreshScenarioOutcome(binding.userId, runId);
          const currentSession = scenarioSessionRef.current ?? session;
          const supplemental: Record<string, boolean | 'unknown'> = {};
          if ('background_continues' in (step.expected ?? {})) {
            supplemental.background_continues = currentSession.backgroundRunIds.includes(runId)
              && run.status === 'completed';
          }
          if ('different_conversation_concurrent' in (step.expected ?? {})) {
            supplemental.different_conversation_concurrent = currentSession.concurrentRunPairs.some(
              (pair) => pair.split(':').includes(runId),
            );
          }
          const comparison = outcome
            ? compareScenarioOutcome(step, outcome, false, supplemental)
            : { status: 'inconclusive' as const, message: '任务已结束，但安全结果不可用。', checks: supplemental };
          setScenarioResult(index, {
            stepIndex: index,
            status: comparison.status,
            message: comparison.message,
            runId,
            expected: step.expected,
            actual: outcome
              ? { ...outcome, checks: comparison.checks }
              : { terminal_status: run.status, checks: comparison.checks },
          });
        } else if (step.action === 'cancel_run') {
          await api.cancelRun(binding.userId, runId);
          controllers.current.get(conversationKey(binding.userId, binding.conversationId))?.abort('scenario-cancel');
          const run = await waitForScenarioRun(binding.userId, runId, timeoutMs);
          await refreshScenarioOutcome(binding.userId, runId);
          setScenarioResult(index, {
            stepIndex: index,
            status: run.status === 'interrupted' ? 'passed' : 'failed',
            message: `取消后的任务状态：${run.status}`,
            runId,
            expected: step.expected,
            actual: { terminal_status: run.status },
          });
        } else if (step.action === 'disconnect_stream') {
          if (step.after_event) await waitForObservedEvent(runId, step.after_event, timeoutMs);
          const key = conversationKey(binding.userId, binding.conversationId);
          controllers.current.get(key)?.abort('scenario-disconnect');
          updateRuntime(key, (runtime) => ({ ...runtime, reconnecting: true, statusMessage: '场景测试已断开 SSE，服务端任务继续运行', updatedAt: Date.now() }), binding);
          const afterSequence = runtimeRef.current[key]?.lastEventSequence ?? 0;
          setScenarioResult(index, {
            stepIndex: index,
            status: 'passed',
            message: 'SSE 订阅已断开，未取消服务端任务。',
            runId,
            expected: step.expected,
            actual: { disconnected_after_sequence: afterSequence },
          });
        } else if (step.action === 'reconnect_stream') {
          const key = conversationKey(binding.userId, binding.conversationId);
          const runtime = runtimeRef.current[key];
          const token = runtime?.requestToken;
          if (!token) throw new Error('缺少可恢复的请求令牌');
          const controller = new AbortController();
          controllers.current.set(key, controller);
          const snapshot: StreamSnapshot = {
            token,
            key,
            userId: binding.userId,
            conversationId: binding.conversationId,
            assistantMessageId: `assistant-${runId}`,
            runId,
          };
          const resumedAfter = runtime.lastEventSequence ?? 0;
          const serverSequenceBeforeReconnect = await waitForServerSequence(
            binding.userId, runId, resumedAfter, timeoutMs,
          );
          launchFollowRun(snapshot, controller, runtime?.failedQuestion ?? '场景问题');
          const subscriptionEpoch = scenarioSessionRef.current?.observations[runId]
            ?.subscriptions.slice(-1)[0]?.epoch ?? 0;
          setScenarioResult(index, {
            stepIndex: index,
            status: 'running',
            message: '已从保存的 sequence 重新订阅。',
            runId,
            expected: step.expected,
            actual: { resumed_after_sequence: resumedAfter },
          });
          const replay = await waitForScenarioSequence(runId, subscriptionEpoch, timeoutMs);
          const firstReceived = replay.subscription.firstReceivedSequence ?? 0;
          const replayValid = replay.subscription.connectedAfterSequence === resumedAfter
            && replayEvidence(
              resumedAfter,
              serverSequenceBeforeReconnect,
              replay.subscription.firstReceivedSequence,
              replay.observation?.duplicateSequence ?? false,
            );
          setScenarioResult(index, {
            stepIndex: index,
            status: replayValid ? 'passed' : 'failed',
            message: replayValid ? '重连后已继续接收事件且 sequence 无重复。' : '重连后发现重复 sequence。',
            runId,
            expected: step.expected,
            actual: {
              resumed_after_sequence: resumedAfter,
              server_sequence_before_reconnect: serverSequenceBeforeReconnect,
              first_received_sequence: firstReceived,
              latest_sequence: replay.observation?.sequences[replay.observation.sequences.length - 1]
                ?? resumedAfter,
              duplicate_sequence: replay.observation?.duplicateSequence ?? false,
            },
          });
        } else if (step.action === 'inspect_messages') {
          const messages = await api.messages(binding.userId, binding.conversationId);
          const otherActorMessages = session.scenario.steps
            .filter((item) => item.action === 'send_message' && item.actor !== step.actor)
            .map((item) => item.content)
            .filter(Boolean);
          const isolated = !messages.some((item) => otherActorMessages.includes(item.content));
          const run = await api.agentRun(binding.userId, runId);
          const expectedTerminal = step.expected?.terminal;
          const terminalMatches = typeof expectedTerminal !== 'string' || run.status === expectedTerminal;
          setScenarioResult(index, {
            stepIndex: index,
            status: isolated && terminalMatches ? 'passed' : 'failed',
            message: isolated && terminalMatches ? '消息检查符合预期。' : '消息检查发现不一致。',
            runId,
            expected: step.expected,
            actual: { message_count: messages.length, cross_user_isolated: isolated, terminal_status: run.status },
          });
        } else if (step.action === 'inspect_memories') {
          const memoryEvidence = await scenarioMemoryEvidence(session, timeoutMs);
          const { messageEntries } = memoryEvidence;
          const all = await api.memories(binding.userId, 'status=active');
          const current = all.filter((item) => item.conversation_id === binding.conversationId);
          const expected = step.expected ?? {};
          const checks: Record<string, boolean | 'unknown'> = {};
          const sourceIds = (actor: string, conversation: string) => {
            const entry = messageEntries.find((item) => (
              item.binding.actor === actor && item.binding.conversation === conversation
            ));
            const expectedContent = session.scenario.steps
              .filter((item) => item.action === 'send_message'
                && item.actor === actor && item.conversation === conversation)
              .map((item) => item.content);
            return (entry?.messages ?? [])
              .filter((message) => message.role === 'user' && expectedContent.includes(message.content))
              .map((message) => message.id);
          };
          if (expected.latest_fact_only) {
            const ids = sourceIds(step.actor, step.conversation);
            checks.latest_fact_only = memoryEvidence.jobsReady
              ? latestFactSourceEvidence(ids, current) : 'unknown';
          }
          if (expected.user_scope_reused) {
            const sourceIdsFromOtherConversations = new Set(Object.values(session.bindings)
              .filter((item) => item.userId === binding.userId && item.conversationId !== binding.conversationId)
              .flatMap((item) => sourceIds(item.actor, item.conversation)));
            checks.user_scope_reused = memoryEvidence.jobsReady
              ? reusedUserMemoryEvidence(sourceIdsFromOtherConversations, all)
              : 'unknown';
          }
          if (expected.conversation_scope_isolated) {
            const otherSourceIds = new Set(Object.values(session.bindings)
              .filter((item) => item.userId === binding.userId && item.conversationId !== binding.conversationId)
              .flatMap((item) => sourceIds(item.actor, item.conversation)));
            checks.conversation_scope_isolated = memoryEvidence.jobsReady
              ? isolatedConversationMemoryEvidence(otherSourceIds, current)
              : 'unknown';
          }
          const hasUnknown = Object.values(checks).includes('unknown');
          const valid = !Object.values(checks).includes(false);
          setScenarioResult(index, {
            stepIndex: index,
            status: hasUnknown ? 'inconclusive' : valid ? 'passed' : 'failed',
            message: hasUnknown ? '缺少可验证的来源消息或记忆结果。'
              : valid ? '记忆检查符合预期。' : '记忆检查发现不一致。',
            runId,
            expected,
            actual: { active_memory_count: all.length, conversation_memory_count: current.length, checks },
          });
        }
      }
      updateScenario((current) => ({
        ...current,
        stepIndex: Math.min(current.stepIndex + 1, current.scenario.steps.length),
        executing: false,
      }));
    } catch (error) {
      const failure = errorMessage(error);
      setScenarioResult(index, {
        stepIndex: index,
        status: 'failed',
        message: failure,
        expected: step.expected,
      });
      updateScenario((current) => ({ ...current, executing: false, error: failure }));
      throw error;
    }
  }

  async function stopScenarioRun() {
    const session = scenarioSessionRef.current;
    if (!session) return;
    const binding = Object.values(session.bindings).find((item) => {
      const runtime = runtimeRef.current[conversationKey(item.userId, item.conversationId)];
      return runtime?.runId && runtimeActive(runtime);
    });
    const runId = binding
      ? runtimeRef.current[conversationKey(binding.userId, binding.conversationId)]?.runId
      : undefined;
    if (!binding || !runId) throw new Error('当前没有可停止的场景任务');
    await api.cancelRun(binding.userId, runId);
    controllers.current.get(conversationKey(binding.userId, binding.conversationId))?.abort('scenario-stop');
  }

  async function cleanupScenario() {
    const session = scenarioSessionRef.current;
    if (!session) return;
    const active = await Promise.all(Object.values(session.bindings).map(async (binding) => ({
      binding,
      run: await api.activeRun(binding.userId, binding.conversationId),
    })));
    const running = active.find((item) => item.run);
    if (running) {
      throw new Error(`场景会话 ${running.binding.actor}/${running.binding.conversation} 仍有任务运行`);
    }
    for (const binding of Object.values(session.bindings)) {
      try {
        await api.deleteScenarioConversation(binding.userId, binding.conversationId);
      } catch (error) {
        if (!(error instanceof ApiError) || error.status !== 404) throw error;
      }
      controllers.current.get(conversationKey(binding.userId, binding.conversationId))?.abort('scenario-cleanup');
      controllers.current.delete(conversationKey(binding.userId, binding.conversationId));
      setConversationBuckets((current) => ({
        ...current,
        [binding.userId]: {
          items: (current[binding.userId]?.items ?? []).filter((item) => item.id !== binding.conversationId),
          loading: false,
          error: '',
        },
      }));
    }
    scenarioSessionRef.current = null;
    setScenarioSession(null);
    setConversationId('');
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
  const scenarioOwnsSelected = Boolean(scenarioSession && Object.values(scenarioSession.bindings).some(
    (binding) => binding.userId === userId && binding.conversationId === conversationId,
  ));

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
        scenarioAvailability={scenarioAvailability}
        onOpenScenarios={() => { setScenarioOpen(true); setSidebarOpen(false); }}
        onRetryScenarios={() => setScenarioReloadToken((value) => value + 1)}
      />
      <section className="chat-workspace">
        <ChatHeader
          conversationTitle={currentConversation?.title ?? ''}
          index={index}
          onOpenSidebar={() => setSidebarOpen(true)}
          scenarioAvailability={scenarioAvailability}
          onOpenScenarios={() => setScenarioOpen(true)}
          onRetryScenarios={() => setScenarioReloadToken((value) => value + 1)}
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
            memoryMessage={selectedRuntime?.memoryMessage ?? ''}
            error={selectedRuntime?.error ?? ''}
            failedQuestion={selectedRuntime?.failedQuestion ?? ''}
            onRetry={() => void send(selectedRuntime?.failedQuestion)}
            onDismissError={() => selectedKey && updateRuntime(selectedKey, (runtime) => ({ ...runtime, error: '', failedQuestion: '', updatedAt: Date.now() }))}
          />
          <Composer
            value={draft}
            streaming={streaming}
            disabled={!userId || usersLoading || Boolean(selectedRuntime?.loading) || Boolean(scenarioOwnsSelected && scenarioSession?.executing)}
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
      <ScenarioPanel
        open={scenarioOpen}
        userId={userId}
        users={users}
        datasets={scenarioDatasets}
        session={scenarioSession}
        onClose={() => setScenarioOpen(false)}
        onStart={startScenario}
        onNext={executeScenarioNext}
        onStop={stopScenarioRun}
        onCleanup={cleanupScenario}
      />
    </main>
  );
}
