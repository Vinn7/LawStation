import type { AgentStage, ConversationRuntime } from '../types';

export type ConversationKey = `${string}:${string}`;

export function conversationKey(userId: string, conversationId: string): ConversationKey {
  return `${userId}:${conversationId}`;
}

export function isActiveStage(status?: AgentStage): boolean {
  return Boolean(status && ['queued', 'analyzing', 'researching', 'drafting', 'reviewing'].includes(status));
}

export function emptyRuntime(userId: string, conversationId: string): ConversationRuntime {
  return {
    userId,
    conversationId,
    messages: [],
    status: 'idle',
    updatedAt: Date.now(),
  };
}
