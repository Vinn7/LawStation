import { describe, expect, it } from 'vitest';
import {
  compareScenarioOutcome,
  isolatedConversationMemoryEvidence,
  latestFactSourceEvidence,
  replayEvidence,
  reusedUserMemoryEvidence,
} from '../chat/scenarioExecutor';
import type { ScenarioRunOutcome, ScenarioStep, UserMemory } from '../types';

const outcome: ScenarioRunOutcome = {
  terminal_status: 'completed',
  observed_events: ['message_start', 'agent_status', 'agent_status', 'citations', 'message_end'],
  retrieval_status: 'matched',
  citation_count: 1,
  model_call_count: 3,
  tool_call_count: 1,
  last_event_sequence: 8,
  checkpoint_available: true,
};

const step: ScenarioStep = {
  action: 'send_message',
  actor: 'primary',
  conversation: 'main',
  expected: {
    events: ['message_start', 'agent_status', 'citations', 'message_end'],
    terminal: 'completed',
    max_model_calls: 4,
    max_tool_calls: 2,
    citations: true,
    retrieval_status: 'matched',
  },
};

function memory(id: string, scope: 'user' | 'conversation', sourceMessageId: string): UserMemory {
  return {
    id, tenant_id: 'tenant', user_id: 'user', conversation_id: 'conversation',
    memory_type: 'case_fact', scope, status: 'active', content: id,
    source_excerpt: '', confidence: 1, importance: 1, version: 1,
    source_message_id: sourceMessageId, created_at: '', updated_at: '',
  };
}

describe('scenario outcome comparison', () => {
  it('accepts ordered event subsequences and deterministic limits', () => {
    const result = compareScenarioOutcome(step, outcome);
    expect(result.status).toBe('passed');
    expect(result.checks.events).toBe(true);
  });

  it('marks fixture-dependent observations inconclusive', () => {
    expect(compareScenarioOutcome(step, outcome, true).status).toBe('inconclusive');
  });

  it('fails when citations or call limits violate the expectation', () => {
    const result = compareScenarioOutcome(step, {
      ...outcome,
      citation_count: 0,
      model_call_count: 6,
    });
    expect(result.status).toBe('failed');
    expect(result.checks.citations).toBe(false);
    expect(result.checks.modelCalls).toBe(false);
  });

  it('marks unsupported expectations inconclusive instead of silently passing', () => {
    const result = compareScenarioOutcome({
      action: 'wait_for_completion',
      actor: 'primary',
      conversation: 'main',
      expected: { background_continues: true },
    }, outcome);
    expect(result.status).toBe('inconclusive');
    expect(result.checks.background_continues).toBe('unknown');
  });

  it('accepts action evidence supplied by the scenario executor', () => {
    const result = compareScenarioOutcome({
      action: 'wait_for_completion',
      actor: 'primary',
      conversation: 'main',
      expected: { background_continues: true },
    }, outcome, false, { background_continues: true });
    expect(result.status).toBe('passed');
  });

  it('requires an actual server backlog for SSE replay evidence', () => {
    expect(replayEvidence(4, 7, 5, false)).toBe(true);
    expect(replayEvidence(4, 4, 5, false)).toBe(false);
    expect(replayEvidence(4, 7, 5, true)).toBe(false);
  });

  it('uses source message ids for latest fact and memory isolation checks', () => {
    expect(latestFactSourceEvidence(['old', 'new'], [memory('new-memory', 'conversation', 'new')])).toBe(true);
    expect(latestFactSourceEvidence(['old', 'new'], [memory('old-memory', 'conversation', 'old')])).toBe(false);
    expect(reusedUserMemoryEvidence(new Set(['preference']), [memory('preference-memory', 'user', 'preference')])).toBe(true);
    expect(isolatedConversationMemoryEvidence(new Set(['other']), [memory('current', 'conversation', 'current')])).toBe(true);
    expect(isolatedConversationMemoryEvidence(new Set(['other']), [memory('leaked', 'conversation', 'other')])).toBe(false);
  });
});
