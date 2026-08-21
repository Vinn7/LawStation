import { useEffect, useMemo, useState } from 'react';
import { BrainCircuit, Check, Pencil, RefreshCw, Trash2, X } from 'lucide-react';
import { api } from '../api';
import type { UserMemory } from '../types';

interface MemoryPanelProps {
  open: boolean;
  userId: string;
  conversationId: string;
  onClose: () => void;
}

const typeLabels: Record<string, string> = {
  profile_preference: '表达偏好', identity_background: '稳定背景', case_fact: '案情事实',
  timeline_event: '时间线', party_relationship: '人物关系', claim_or_goal: '诉求',
  evidence_status: '证据状态', user_correction: '用户修正',
};

export function MemoryPanel({ open, userId, conversationId, onClose }: MemoryPanelProps) {
  const [items, setItems] = useState<UserMemory[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const [editing, setEditing] = useState<string | null>(null);
  const [editContent, setEditContent] = useState('');

  async function load(signal?: AbortSignal) {
    if (!userId) return;
    setLoading(true);
    setError('');
    try {
      const profileQuery = 'scope=user';
      const caseQuery = conversationId
        ? `scope=conversation&conversation_id=${encodeURIComponent(conversationId)}`
        : '';
      const [profiles, cases] = await Promise.all([
        api.memories(userId, profileQuery, signal),
        caseQuery ? api.memories(userId, caseQuery, signal) : Promise.resolve([]),
      ]);
      setItems([...profiles, ...cases]);
    } catch (caught) {
      if (caught instanceof DOMException && caught.name === 'AbortError') return;
      setError(caught instanceof Error ? caught.message : '无法加载记忆');
    } finally {
      setLoading(false);
    }
  }

  useEffect(() => {
    if (!open || !userId) return;
    setItems([]);
    setEditing(null);
    const controller = new AbortController();
    void load(controller.signal);
    return () => controller.abort();
  }, [open, userId, conversationId]);

  const visible = useMemo(() => items.filter((item) => item.status === 'active'), [items]);

  async function remove(item: UserMemory) {
    try {
      await api.deleteMemory(userId, item.id);
      await load();
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : '操作失败');
    }
  }

  async function save(item: UserMemory) {
    if (!editContent.trim()) return;
    try {
      await api.updateMemory(userId, item.id, editContent.trim(), item.version);
      setEditing(null);
      await load();
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : '修改失败');
    }
  }

  if (!open) return null;

  return (
    <div className={`memory-panel-backdrop ${open ? 'is-open' : ''}`} aria-hidden={!open}>
      <section className="memory-panel" role="dialog" aria-modal="true" aria-label="我的记忆">
        <header>
          <div><BrainCircuit size={21} /><div><strong>我的记忆</strong><span>新记忆整理后自动生效，可随时修正或删除</span></div></div>
          <button className="icon-button" onClick={onClose} aria-label="关闭记忆管理"><X size={19} /></button>
        </header>
        <div className="memory-panel-toolbar">
          <span>{conversationId ? '用户偏好与当前案件' : '用户偏好'}</span>
          <button onClick={() => void load()} disabled={loading}><RefreshCw size={14} />刷新</button>
        </div>
        {loading && <div className="memory-panel-state">正在读取记忆…</div>}
        {error && <div className="memory-panel-state is-error">{error}</div>}
        {!loading && !error && visible.length === 0 && <div className="memory-panel-state">暂无可管理的记忆</div>}
        <div className="memory-list">
          {visible.map((item) => (
            <article className={`memory-card is-${item.status}`} key={item.id}>
              <div className="memory-card-meta"><span>{typeLabels[item.memory_type] ?? item.memory_type}</span><i>已生效</i></div>
              {editing === item.id ? (
                <textarea value={editContent} onChange={(event) => setEditContent(event.target.value)} />
              ) : <p>{item.content}</p>}
              {item.source_excerpt && <small>来源：{item.source_excerpt}</small>}
              <div className="memory-card-actions">
                {editing === item.id
                  ? <button onClick={() => void save(item)}><Check size={14} />保存</button>
                  : <button onClick={() => { setEditing(item.id); setEditContent(item.content); }}><Pencil size={14} />修正</button>}
                <button onClick={() => void remove(item)}><Trash2 size={14} />删除</button>
              </div>
            </article>
          ))}
        </div>
      </section>
    </div>
  );
}
