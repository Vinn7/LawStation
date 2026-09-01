import { render, screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { beforeEach, describe, expect, it, vi } from 'vitest';
import { api } from '../api';
import { ScenarioPanel } from '../components/ScenarioPanel';
import type { DialogueScenario, ScenarioDataset } from '../types';

vi.mock('../api', () => ({
  ApiError: class ApiError extends Error {},
  api: {
    scenarioSummaries: vi.fn(),
    scenario: vi.fn(),
  },
}));

const dataset: ScenarioDataset = {
  id: 'dialogue-v1',
  schema_version: '1.0',
  sha256: 'sha',
  sample_count: 1,
  synthetic: true,
  human_verified: false,
  categories: { routing: 1 },
  step_timeout_seconds: 60,
};

const scenario: DialogueScenario = {
  schema_version: '1.0',
  synthetic: true,
  scenario_id: 'casual-01',
  title: '普通问候',
  category: 'routing',
  description: '验证闲聊路由',
  actors: ['primary'],
  preconditions: [],
  step_count: 1,
  fixture_types: [],
  fixture_notice: '',
  fixtures_applied: false,
  steps: [{ action: 'send_message', actor: 'primary', conversation: 'main', content: '你好' }],
};

describe('ScenarioPanel', () => {
  beforeEach(() => {
    vi.mocked(api.scenarioSummaries).mockResolvedValue([{
      scenario_id: scenario.scenario_id,
      title: scenario.title,
      category: scenario.category,
      description: scenario.description,
      actors: scenario.actors,
      step_count: 1,
      preconditions: [],
      fixture_types: [],
    }]);
    vi.mocked(api.scenario).mockResolvedValue(scenario);
  });

  it('loads a frozen scenario and starts it only after an explicit click', async () => {
    const onStart = vi.fn().mockResolvedValue(undefined);
    const user = userEvent.setup();
    render(<ScenarioPanel
      open
      userId="user-a"
      users={[{ id: 'user-a', tenant_id: 'tenant', name: 'A', created_at: '' }]}
      datasets={[dataset]}
      session={null}
      onClose={vi.fn()}
      onStart={onStart}
      onNext={vi.fn()}
      onStop={vi.fn()}
      onCleanup={vi.fn()}
    />);

    expect(await screen.findByText('普通问候')).toBeInTheDocument();
    expect(onStart).not.toHaveBeenCalled();
    await user.click(screen.getByRole('button', { name: '开始场景' }));
    await waitFor(() => expect(onStart).toHaveBeenCalledWith('dialogue-v1', scenario));
  });
});
