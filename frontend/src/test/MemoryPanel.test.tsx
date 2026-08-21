import { render, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { afterEach, describe, expect, it, vi } from 'vitest';
import { MemoryPanel } from '../components/MemoryPanel';

function json(body: unknown, status = 200): Response {
  return new Response(JSON.stringify(body), { status, headers: { 'Content-Type': 'application/json' } });
}

describe('MemoryPanel', () => {
  afterEach(() => vi.restoreAllMocks());

  it('loads only the selected user and confirms a pending case memory', async () => {
    const fetchMock = vi.spyOn(globalThis, 'fetch').mockImplementation(async (input, init) => {
      const url = String(input);
      const owner = new Headers(init?.headers).get('X-User-ID');
      expect(owner).toBe('user-a');
      if (url.includes('scope=user')) return json([]);
      if (url.includes('scope=conversation')) return json([{
        id: 'memory-1', tenant_id: 'tenant', user_id: 'user-a', conversation_id: 'case-a',
        memory_type: 'case_fact', scope: 'conversation', status: 'pending', content: '月工资一万元',
        source_excerpt: '我的月工资是一万元', confidence: 0.95, importance: 80, version: 1,
        created_at: '2026-08-20T10:00:00Z', updated_at: '2026-08-20T10:00:00Z',
      }]);
      if (url.endsWith('/memories/memory-1/confirm')) return json({ ok: true });
      throw new Error(`Unexpected request: ${url}`);
    });
    const user = userEvent.setup();
    render(<MemoryPanel open userId="user-a" conversationId="case-a" onClose={() => undefined} />);

    expect(await screen.findByText('月工资一万元')).toBeInTheDocument();
    await user.click(screen.getByRole('button', { name: '确认' }));

    expect(fetchMock).toHaveBeenCalledWith(
      '/api/memories/memory-1/confirm',
      expect.objectContaining({ method: 'POST' }),
    );
  });
});
