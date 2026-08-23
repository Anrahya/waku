import { describe, expect, test } from 'bun:test'
import type { AgentSession } from '@waku/client'
import {
  renoaWebRuntimeError,
  sessionRequiresDesktopRenoaReplay,
} from './renoa-web-runtime'
import { reduceRuntimeEvent } from './event-reducer'

const clock = {
  nowSeconds: () => 200,
  nowMillis: () => 200_000,
  randomUUID: () => '00000000-0000-4000-8000-000000000001',
}

describe('Renoa web runtime gate', () => {
  test('refuses a restored Renoa session before a web runtime can start', () => {
    const session = sessionWithProvider('renoa')
    expect(sessionRequiresDesktopRenoaReplay(session)).toBe(true)
    const error = renoaWebRuntimeError(session, 'en')
    expect(error).toContain('desktop app')
    expect(error).toContain('authoritative history')
  })

  test('leaves ordinary providers unchanged', () => {
    for (const provider of ['codex', 'claude', 'grok', 'cursor'] as const) {
      const session = sessionWithProvider(provider)
      expect(sessionRequiresDesktopRenoaReplay(session)).toBe(false)
      expect(renoaWebRuntimeError(session, 'en')).toBeNull()
    }
  })

  test('a Renoa replay event cannot hang the web client by advancing the cursor', () => {
    const session = runningSession('renoa')
    const result = reduceRuntimeEvent(
      session,
      {
        sessionId: session.id,
        runtimeId: 'runtime',
        epoch: 'epoch',
        sequence: 9,
        event: {
          kind: 'sessionReplay',
          payload: { replayId: 'replay', index: 0, total: 1, json: '{}' },
        } as never,
      },
      clock,
    )

    expect(result.error).toContain('desktop app')
    expect(result.removeRuntime).toBe(true)
    expect(result.session).toBe(session)
    expect(result.session.runtime_event_cursor).toBeUndefined()
    expect(result.session.status).toBe('connecting')
  })

  test('an unsupported control event cannot silently advance the web cursor', () => {
    const session = runningSession('codex')
    const result = reduceRuntimeEvent(
      session,
      {
        sessionId: session.id,
        runtimeId: 'runtime',
        epoch: 'epoch',
        sequence: 11,
        event: { kind: 'sessionReplay', payload: null } as never,
      },
      clock,
    )

    expect(result.error).toBeDefined()
    expect(result.session.runtime_event_cursor).toBeUndefined()
    expect(result.session.status).toBe('connecting')
  })

  test('ordinary provider events still advance the cursor', () => {
    const session = runningSession('codex')
    const result = reduceRuntimeEvent(
      session,
      {
        sessionId: session.id,
        runtimeId: 'runtime',
        epoch: 'epoch',
        sequence: 4,
        event: { kind: 'connected', payload: null } as never,
      },
      clock,
    )

    expect(result.error).toBeUndefined()
    expect(result.session.runtime_event_cursor).toEqual({
      runtime_id: 'runtime',
      epoch: 'epoch',
      sequence: 4,
    })
    expect(result.session.status).toBe('working')
  })
})

function sessionWithProvider(provider: AgentSession['provider']): AgentSession {
  return {
    ...runningSession(provider),
    status: 'idle',
    turns: [],
    messages: [],
  }
}

function runningSession(provider: AgentSession['provider']): AgentSession {
  return {
    id: 'session',
    title: 'New task',
    project_id: 'project',
    workspace: { kind: 'local' },
    provider,
    runtime_mode: 'fullAccess',
    interaction_mode: 'build',
    status: 'connecting',
    created_at: 100,
    updated_at: 100,
    provider_cursor: provider === 'renoa'
      ? { provider: 'renoa', sessionId: 'renoa-session' }
      : null,
    messages: [
      {
        id: 'message',
        turn_id: 'turn',
        role: 'user',
        content: 'Go',
        created_at: 100,
        streaming: false,
      },
    ],
    transcript_blocks: [],
    turns: [
      {
        id: 'turn',
        turn_count: 1,
        status: 'running',
        provider_turn_started: false,
        started_at: 100,
        completed_at: null,
        checkpoint: null,
      },
    ],
  }
}
