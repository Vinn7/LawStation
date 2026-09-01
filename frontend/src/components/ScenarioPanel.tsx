import { Beaker, ChevronDown, Eraser, Play, Search, Square, StepForward, X } from 'lucide-react';
import { useEffect, useMemo, useState } from 'react';
import { api, ApiError } from '../api';
import type {
  DialogueScenario,
  ScenarioDataset,
  ScenarioSession,
  ScenarioSummary,
  User,
} from '../types';

interface ScenarioPanelProps {
  open: boolean;
  userId: string;
  users: User[];
  datasets: ScenarioDataset[];
  session: ScenarioSession | null;
  onClose: () => void;
  onStart: (datasetId: string, scenario: DialogueScenario) => Promise<void>;
  onNext: () => Promise<void>;
  onStop: () => Promise<void>;
  onCleanup: () => Promise<void>;
}

function message(error: unknown): string {
  return error instanceof Error ? error.message : '场景操作失败';
}

const actionLabels: Record<string, string> = {
  send_message: '发送消息',
  switch_user: '切换用户',
  switch_conversation: '切换会话',
  wait_for_completion: '等待任务完成',
  cancel_run: '取消任务',
  disconnect_stream: '断开 SSE',
  reconnect_stream: '重新连接 SSE',
  inspect_messages: '检查消息',
  inspect_memories: '检查记忆',
};

export function ScenarioPanel({
  open,
  userId,
  users,
  datasets,
  session,
  onClose,
  onStart,
  onNext,
  onStop,
  onCleanup,
}: ScenarioPanelProps) {
  const [datasetId, setDatasetId] = useState('');
  const [summaries, setSummaries] = useState<ScenarioSummary[]>([]);
  const [scenarioId, setScenarioId] = useState('');
  const [scenario, setScenario] = useState<DialogueScenario | null>(null);
  const [category, setCategory] = useState('all');
  const [query, setQuery] = useState('');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');

  useEffect(() => {
    if (!datasetId && datasets.length) setDatasetId(datasets[0].id);
  }, [datasetId, datasets]);

  useEffect(() => {
    if (!open || !userId || !datasetId) return;
    const controller = new AbortController();
    setLoading(true);
    setError('');
    api.scenarioSummaries(userId, datasetId, controller.signal)
      .then((items) => {
        setSummaries(items);
        setScenarioId((current) => items.some((item) => item.scenario_id === current)
          ? current : (items[0]?.scenario_id ?? ''));
      })
      .catch((reason) => {
        if (reason instanceof DOMException && reason.name === 'AbortError') return;
        setError(message(reason));
      })
      .finally(() => setLoading(false));
    return () => controller.abort();
  }, [open, userId, datasetId]);

  useEffect(() => {
    if (!open || !userId || !datasetId || !scenarioId) return;
    const controller = new AbortController();
    api.scenario(userId, datasetId, scenarioId, controller.signal)
      .then(setScenario)
      .catch((reason) => {
        if (reason instanceof DOMException && reason.name === 'AbortError') return;
        setError(message(reason));
      });
    return () => controller.abort();
  }, [open, userId, datasetId, scenarioId]);

  const categories = useMemo(
    () => [...new Set(summaries.map((item) => item.category))].sort(),
    [summaries],
  );
  const filtered = useMemo(() => summaries.filter((item) => (
    (category === 'all' || item.category === category)
    && (!query.trim() || `${item.title}${item.description}${item.scenario_id}`.toLowerCase().includes(query.trim().toLowerCase()))
  )), [summaries, category, query]);
  const selected = session?.scenario ?? scenario;
  const needsSecondary = selected?.actors.includes('secondary');
  const cannotStart = !selected || loading || Boolean(needsSecondary && users.length < 2);
  const completed = session ? session.stepIndex >= session.scenario.steps.length : false;

  async function invoke(action: () => Promise<void>) {
    setError('');
    try {
      await action();
    } catch (reason) {
      if (reason instanceof ApiError || reason instanceof Error) setError(reason.message);
      else setError('场景操作失败');
    }
  }

  return (
    <div className={`scenario-panel-backdrop ${open ? 'is-open' : ''}`} aria-hidden={!open}>
      <section className="scenario-panel" aria-label="场景观察模式">
        <header>
          <div><Beaker size={20} /><div><strong>场景观察模式</strong><span>每次点击只执行一个步骤</span></div></div>
          <button className="icon-button" onClick={onClose} aria-label="关闭场景观察模式"><X size={20} /></button>
        </header>

        <div className="scenario-warning">合成测试数据，不是真实用户数据，也未经过律师人工标注。</div>

        {!session && (
          <div className="scenario-selector">
            <label>数据集</label>
            <div className="select-wrap">
              <select value={datasetId} onChange={(event) => setDatasetId(event.target.value)}>
                {datasets.map((item) => <option key={item.id} value={item.id}>{item.id}（{item.sample_count}）</option>)}
              </select>
              <ChevronDown size={16} />
            </div>
            <div className="scenario-filter-row">
              <div className="select-wrap">
                <select value={category} onChange={(event) => setCategory(event.target.value)} aria-label="场景类别">
                  <option value="all">全部类别</option>
                  {categories.map((item) => <option key={item} value={item}>{item}</option>)}
                </select>
                <ChevronDown size={15} />
              </div>
              <label className="scenario-search"><Search size={15} /><input value={query} onChange={(event) => setQuery(event.target.value)} placeholder="搜索场景" /></label>
            </div>
            <div className="scenario-summary-list">
              {filtered.map((item) => (
                <button key={item.scenario_id} className={scenarioId === item.scenario_id ? 'is-selected' : ''} onClick={() => setScenarioId(item.scenario_id)}>
                  <strong>{item.title}</strong><span>{item.category} · {item.step_count} 步</span><small>{item.description}</small>
                </button>
              ))}
            </div>
          </div>
        )}

        {selected && (
          <div className="scenario-detail">
            <div className="scenario-detail-heading"><strong>{selected.title}</strong><code>{selected.scenario_id}</code></div>
            <p>{selected.description}</p>
            <div className="scenario-tags">
              {selected.actors.map((item) => <span key={item}>Actor: {item}</span>)}
              {selected.preconditions.map((item) => <span key={item} className="is-warning">{item}</span>)}
            </div>
            {selected.fixture_notice && <div className="scenario-fixture-notice">{selected.fixture_notice}</div>}
          </div>
        )}

        {session && (
          <div className="scenario-steps">
            <div className="scenario-progress" aria-label="场景进度">
              <span>当前进度</span>
              <strong>{Math.min(session.stepIndex, session.scenario.steps.length)} / {session.scenario.steps.length}</strong>
            </div>
            {session.scenario.steps.map((step, index) => {
              const result = session.results.find((item) => item.stepIndex === index);
              const status = result?.status
                ?? (index === session.stepIndex && session.executing ? 'running' : 'pending');
              return (
                <article key={`${step.action}-${index}`} className={`scenario-step is-${status}`}>
                  <div><span>{index + 1}</span><strong>{actionLabels[step.action] ?? step.action}</strong><i>{status}</i></div>
                  {step.content && <p>{step.content}</p>}
                  {result && <small>{result.message}</small>}
                  {step.expected && <details><summary>查看预期结果</summary><pre>{JSON.stringify(step.expected, null, 2)}</pre></details>}
                  {result?.actual && <details><summary>查看实际结果</summary><pre>{JSON.stringify(result.actual, null, 2)}</pre></details>}
                </article>
              );
            })}
          </div>
        )}

        {needsSecondary && users.length < 2 && <div className="scenario-error">该场景需要两个演示用户。</div>}
        {error && <div className="scenario-error" role="alert">{error}</div>}
        {session?.error && <div className="scenario-error" role="alert">{session.error}</div>}

        <footer>
          {!session ? (
            <button className="scenario-primary" disabled={cannotStart} onClick={() => selected && invoke(() => onStart(datasetId, selected))}><Play size={16} />开始场景</button>
          ) : (
            <>
              <button className="scenario-primary" disabled={session.executing || completed} onClick={() => invoke(onNext)}><StepForward size={16} />{completed ? '场景步骤已完成' : '执行下一步'}</button>
              <button onClick={() => invoke(onStop)}><Square size={14} />停止当前任务</button>
              <button onClick={() => invoke(onCleanup)}><Eraser size={14} />清理场景数据</button>
            </>
          )}
        </footer>
      </section>
    </div>
  );
}
