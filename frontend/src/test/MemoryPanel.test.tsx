import { render, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { afterEach, describe, expect, it, vi } from 'vitest';
import { MemoryPanel } from '../components/MemoryPanel';

function json(body: unknown, status = 200): Response {
  return new Response(JSON.stringify(body), { status, headers: { 'Content-Type': 'application/json' } });
}

describe('MemoryPanel', () => {
  afterEach(() => vi.restoreAllMocks());

  it('loads only active memories for the selected user and deletes one', async () => {
    const fetchMock = vi.spyOn(globalThis, 'fetch').mockImplementation(async (input, init) => {
      const url = String(input);
      const owner = new Headers(init?.headers).get('X-User-ID');
      expect(owner).toBe('user-a');
      if (url.includes('scope=user')) return json([]);
      if (url.includes('scope=conversation')) return json([
        {
          id: 'memory-1', tenant_id: 'tenant', user_id: 'user-a', conversation_id: 'case-a',
          memory_type: 'case_fact', scope: 'conversation', status: 'active', content: '月工资一万元',
          source_excerpt: '我的月工资是一万元', confidence: 0.95, importance: 80, version: 1,
          created_at: '2026-08-20T10:00:00Z', updated_at: '2026-08-20T10:00:00Z',
        },
        {
          id: 'memory-old', tenant_id: 'tenant', user_id: 'user-a', conversation_id: 'case-a',
          memory_type: 'case_fact', scope: 'conversation', status: 'superseded', content: '月工资八千元',
          source_excerpt: '', confidence: 0.8, importance: 70, version: 2,
          created_at: '2026-08-19T10:00:00Z', updated_at: '2026-08-19T10:00:00Z',
        },
      ]);
      if (url.endsWith('/memories/memory-1')) return new Response(null, { status: 204 });
      throw new Error(`Unexpected request: ${url}`);
    });
    const user = userEvent.setup();
    render(<MemoryPanel open userId="user-a" conversationId="case-a" onClose={() => undefined} />);

    expect(await screen.findByText('月工资一万元')).toBeInTheDocument();
    expect(screen.queryByText('月工资八千元')).not.toBeInTheDocument();
    await user.click(screen.getByRole('button', { name: '删除' }));

    expect(fetchMock).toHaveBeenCalledWith(
      '/api/memories/memory-1',
      expect.objectContaining({ method: 'DELETE' }),
    );
  });
});
