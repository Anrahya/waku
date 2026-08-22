import { describe, expect, test } from 'bun:test'
import {
  readComposerPreferences,
  rememberedModelTraits,
  rememberComposerSession,
  writeComposerPreferences,
} from './composer-preferences'

describe('composer preferences', () => {
  test('remembers the last model per daemon and its traits', () => {
    const storage = memoryStorage()
    const remembered = rememberComposerSession(
      readComposerPreferences(storage, 'ws://first'),
      {
        provider: 'codex',
        model: 'gpt-5.6-sol',
        reasoning_effort: 'high',
        service_tier: 'fast',
        context_window: '1m',
      },
    )
    writeComposerPreferences(storage, 'ws://first', remembered)

    expect(readComposerPreferences(storage, 'ws://first')).toMatchObject({
      lastProvider: 'codex',
      lastModel: 'gpt-5.6-sol',
      lastReasoningEffort: 'high',
      lastServiceTier: 'fast',
      lastContextWindow: '1m',
    })
    expect(rememberedModelTraits(remembered, 'codex', 'gpt-5.6-sol')).toEqual({
      reasoningEffort: 'high',
      serviceTier: 'fast',
      contextWindow: '1m',
    })
    expect(readComposerPreferences(storage, 'ws://second').lastModel).toBeNull()
  })

  test('does not erase an explicit model when a blank draft is selected', () => {
    const preferences = rememberComposerSession(
      readComposerPreferences(null, 'ws://first'),
      {
        provider: 'claude',
        model: 'claude-opus-4-1',
        reasoning_effort: null,
        service_tier: null,
      },
    )
    expect(rememberComposerSession(preferences, {
      provider: 'codex',
      model: null,
      reasoning_effort: null,
      service_tier: null,
    })).toBe(preferences)
  })

  test('restores Fx as a valid remembered provider', () => {
    const storage = memoryStorage()
    const preferences = rememberComposerSession(
      readComposerPreferences(storage, 'ws://first'),
      {
        provider: 'fx',
        model: 'openai/gpt-5.6-sol',
        reasoning_effort: null,
        service_tier: null,
      },
    )
    writeComposerPreferences(storage, 'ws://first', preferences)

    expect(readComposerPreferences(storage, 'ws://first').lastProvider).toBe('fx')
  })

  test('does not treat a hidden Renoa provider as a selectable preference', () => {
    const storage = memoryStorage()
    storage.setItem(
      'waku.composer-preferences.v1',
      JSON.stringify({
        'ws://first': {
          lastProvider: 'renoa',
          lastModel: 'ignored',
          lastReasoningEffort: null,
          lastServiceTier: null,
          lastContextWindow: null,
          modelTraits: {},
        },
      }),
    )

    expect(readComposerPreferences(storage, 'ws://first').lastProvider).toBe('codex')
  })

  test('does not remember a hidden Renoa session as the next-task provider', () => {
    const remembered = rememberComposerSession(
      readComposerPreferences(null, 'ws://first'),
      {
        provider: 'renoa',
        model: 'alpha',
        reasoning_effort: null,
        service_tier: null,
        context_window: null,
      },
    )

    expect(remembered.lastProvider).toBe('codex')
    expect(remembered.lastModel).toBeNull()
  })
})

function memoryStorage() {
  const values = new Map<string, string>()
  return {
    getItem: (key: string) => values.get(key) ?? null,
    setItem: (key: string, value: string) => { values.set(key, value) },
  }
}
