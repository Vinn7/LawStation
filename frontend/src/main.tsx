import React, { useEffect, useRef, useState } from 'react';
import { createRoot } from 'react-dom/client';
import './style.css';

const API = '/api';

function App() {
  const [users, setUsers] = useState<any[]>([]);
  const [uid, setUid] = useState('');
  const [conversations, setConversations] = useState<any[]>([]);
  const [conversationId, setConversationId] = useState('');
  const [messages, setMessages] = useState<any[]>([]);
  const [text, setText] = useState('');
  const [index, setIndex] = useState<any>({ status: 'checking', message: '正在检查法律索引' });
  const abort = useRef<AbortController | null>(null);

  useEffect(() => {
    fetch(`${API}/users`).then(response => response.json()).then(data => {
      setUsers(data);
      setUid(data[0]?.id || '');
    });
    const loadStatus = () => fetch(`${API}/index/status`).then(response => response.json()).then(setIndex).catch(() => undefined);
    loadStatus();
    const timer = window.setInterval(loadStatus, 2000);
    return () => window.clearInterval(timer);
  }, []);

  useEffect(() => {
    abort.current?.abort();
    setConversationId('');
    setMessages([]);
    if (uid) {
      fetch(`${API}/conversations`, { headers: { 'X-User-ID': uid } })
        .then(response => response.json()).then(setConversations);
    }
  }, [uid]);

  async function createConversation() {
    const response = await fetch(`${API}/conversations`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json', 'X-User-ID': uid },
      body: JSON.stringify({ title: '法律咨询' }),
    });
    const conversation = await response.json();
    setConversations(current => [conversation, ...current]);
    setConversationId(conversation.id);
    setMessages([]);
  }

  async function openConversation(id: string) {
    setConversationId(id);
    const response = await fetch(`${API}/conversations/${id}/messages`, { headers: { 'X-User-ID': uid } });
    setMessages(await response.json());
  }

  async function send() {
    if (!text.trim() || !conversationId) return;
    const question = text;
    setText('');
    setMessages(current => [...current, { role: 'user', content: question }, { role: 'assistant', content: '' }]);
    abort.current = new AbortController();
    const response = await fetch(`${API}/conversations/${conversationId}/messages/stream`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json', 'X-User-ID': uid },
      body: JSON.stringify({ content: question }),
      signal: abort.current.signal,
    });
    if (!response.body) return;
    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    let buffer = '';
    while (true) {
      const { value, done } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });
      const blocks = buffer.split('\n\n');
      buffer = blocks.pop() || '';
      for (const block of blocks) {
        const event = block.match(/event: (.+)/)?.[1];
        const data = block.match(/data: (.+)/)?.[1];
        if (event === 'token' && data) {
          const token = JSON.parse(data);
          setMessages(current => {
            const next = [...current];
            next[next.length - 1] = { role: 'assistant', content: next[next.length - 1].content + token };
            return next;
          });
        }
      }
    }
  }

  const building = index.status === 'building' || index.status === 'checking';
  return <main>
    <aside>
      <h2>LawStation</h2>
      <select value={uid} onChange={event => setUid(event.target.value)}>
        {users.map(user => <option key={user.id} value={user.id}>{user.name}</option>)}
      </select>
      <button onClick={createConversation}>新建会话</button>
      {conversations.map(conversation => <div key={conversation.id} className={conversationId === conversation.id ? 'on' : ''} onClick={() => openConversation(conversation.id)}>{conversation.title}</div>)}
    </aside>
    <section>
      {index.status !== 'ready' && <div className={`index-status ${index.status}`}>
        {index.message}{building && `（${Math.round((index.progress || 0) * 100)}%）`}
      </div>}
      <div className="chat">{messages.map((message, i) => <article key={i} className={message.role}>{message.content}</article>)}</div>
      <footer><textarea value={text} onChange={event => setText(event.target.value)} /><button onClick={send}>发送</button></footer>
    </section>
  </main>;
}

createRoot(document.getElementById('root')!).render(<App />);
