import type { ScenarioRunOutcome, ScenarioStep, UserMemory } from '../types';

export type EvidenceCheck = boolean | 'unknown';

export function replayEvidence(
  connectedAfter: number,
  serverSequenceBeforeReconnect: number,
  firstReceivedSequence: number | undefined,
  duplicateSequence: boolean,
): boolean {
  return serverSequenceBeforeReconnect > connectedAfter
    && firstReceivedSequence !== undefined
    && firstReceivedSequence > connectedAfter
    && firstReceivedSequence <= serverSequenceBeforeReconnect
    && !duplicateSequence;
}

export function latestFactSourceEvidence(
  sourceMessageIds: string[],
  memories: UserMemory[],
): EvidenceCheck {
  const newest = sourceMessageIds[sourceMessageIds.length - 1];
  if (!newest || !memories.length) return 'unknown';
  const older = new Set(sourceMessageIds.slice(0, -1));
  return memories.some((item) => item.source_message_id === newest)
    && !memories.some((item) => item.source_message_id && older.has(item.source_message_id));
}

export function reusedUserMemoryEvidence(
  sourceMessageIds: Set<string>,
  memories: UserMemory[],
): EvidenceCheck {
  if (!sourceMessageIds.size) return 'unknown';
  return memories.some((item) => item.scope === 'user'
    && Boolean(item.source_message_id)
    && sourceMessageIds.has(item.source_message_id ?? ''));
}

export function isolatedConversationMemoryEvidence(
  otherConversationSourceIds: Set<string>,
  currentConversationMemories: UserMemory[],
): EvidenceCheck {
  if (!otherConversationSourceIds.size || !currentConversationMemories.length) return 'unknown';
  return !currentConversationMemories.some((item) => item.source_message_id
    && otherConversationSourceIds.has(item.source_message_id));
}

export interface ScenarioComparison {
  status: 'passed' | 'failed' | 'inconclusive';
  message: string;
  checks: Record<string, boolean | 'unknown'>;
}

function orderedSubsequence(expected: string[], actual: string[]): boolean {
  let cursor = 0;
  for (const value of actual) {
    if (value === expected[cursor]) cursor += 1;
    if (cursor === expected.length) return true;
  }
  return expected.length === 0;
}

export function compareScenarioOutcome(
  step: ScenarioStep,
  outcome: ScenarioRunOutcome,
  fixtureDependent = false,
  observedChecks: Record<string, boolean | 'unknown'> = {},
): ScenarioComparison {
  const expected = step.expected ?? {};
  const checks: Record<string, boolean | 'unknown'> = {};
  const handled = new Set<string>();
  if (Array.isArray(expected.events)) {
    handled.add('events');
    checks.events = orderedSubsequence(expected.events.map(String), outcome.observed_events);
  }
  if (typeof expected.terminal === 'string') {
    handled.add('terminal');
    checks.terminal = expected.terminal === outcome.terminal_status;
  }
  if (typeof expected.max_model_calls === 'number') {
    handled.add('max_model_calls');
    checks.modelCalls = outcome.model_call_count <= expected.max_model_calls;
  }
  if (typeof expected.max_tool_calls === 'number') {
    handled.add('max_tool_calls');
    checks.toolCalls = outcome.tool_call_count <= expected.max_tool_calls;
  }
  if (typeof expected.citations === 'boolean') {
    handled.add('citations');
    checks.citations = expected.citations
      ? outcome.citation_count > 0
      : outcome.citation_count === 0;
  }
  if (typeof expected.retrieval_status === 'string') {
    handled.add('retrieval_status');
    checks.retrievalStatus = outcome.retrieval_status === 'unknown'
      ? 'unknown'
      : outcome.retrieval_status === expected.retrieval_status;
  }
  for (const [key, value] of Object.entries(observedChecks)) {
    if (key in expected) {
      handled.add(key);
      checks[key] = value;
    }
  }
  for (const key of Object.keys(expected)) {
    if (!handled.has(key)) checks[key] = 'unknown';
  }
  if (fixtureDependent) {
    return {
      status: 'inconclusive',
      message: '真实服务未注入离线 Fixture，已记录实际结果但不判定通过或失败。',
      checks,
    };
  }
  if (Object.values(checks).includes('unknown')) {
    return { status: 'inconclusive', message: '部分结果不可用，无法完成全部断言。', checks };
  }
  const failed = Object.values(checks).some((value) => value === false);
  return {
    status: failed ? 'failed' : 'passed',
    message: failed ? '实际结果与预期不一致。' : '实际结果符合当前步骤预期。',
    checks,
  };
}
