import { mkdtempSync, rmSync } from 'node:fs'
import { tmpdir } from 'node:os'
import { join } from 'node:path'
import { afterEach, describe, expect, it } from 'vitest'
import { Context } from '@deepseek-ai/cordis'
import { createUserMessage } from '@deepseek-ai/dsh-llm'
import type { GenerateOptions } from '@deepseek-ai/dsh-llm'
import { SessionId, type SessionEvent } from '@deepseek-ai/dsh-session'
import type { Agent } from '@deepseek-ai/dsh-agent'
import AgentLoop from '@deepseek-ai/dsh-agent-loop'
import { mountAgentLoopTestDependencies } from '@deepseek-ai/dsh-agent-loop-testkit'
import SessionProjectionRegistry from '@deepseek-ai/dsh-session-projection'
import * as ToolMemory from '@deepseek-ai/dsh-tool-memory'
import { MockAdapter, textResponse, toolCallResponse } from '../../../core/agent-loop/tests/mock-adapter.ts'

/**
 * Full-loop integration: a scripted mock model drives the REAL memory tools
 * through the agent loop, and the memory index is checked where the model
 * would see it - in the messages of its next request.
 */
const dirs: string[] = []
afterEach(() => {
  for (const d of dirs.splice(0)) rmSync(d, { recursive: true, force: true })
})

function dbPath(): string {
  const dir = mkdtempSync(join(tmpdir(), 'dsh-memory-it-'))
  dirs.push(dir)
  return join(dir, 'memory.db')
}

async function harness(adapter: MockAdapter, path: string): Promise<Context> {
  const ctx = new Context()
  await mountAgentLoopTestDependencies(ctx)
  await ctx.plugin(SessionProjectionRegistry)
  await ctx.plugin(AgentLoop, { agents: [] })
  await ctx.plugin(ToolMemory, { path })
  ctx.llm.registerAdapter(['mock'], adapter)
  return ctx
}

function waitForIdle(ctx: Context, agent: Agent): Promise<void> {
  return new Promise((resolve) => {
    const dispose = ctx.on('agent/status', ({ agent: subject, status }) => {
      if (subject === agent && status === 'idle') {
        dispose()
        resolve()
      }
    })
  })
}

async function say(ctx: Context, agent: Agent, text: string): Promise<void> {
  const idle = waitForIdle(ctx, agent)
  agent.followup(createUserMessage({ content: [{ type: 'text', text }], source: { kind: 'user' } }))
  await idle
}

function results(log: readonly SessionEvent[]): string[] {
  return log.filter((e): e is Extract<SessionEvent, { type: 'tool/result' }> => e.type === 'tool/result')
    .map(e => e.data.message.content[0].content.map(b => (b.type === 'text' ? b.text : '')).join(''))
}

function requestText(request: GenerateOptions): string {
  return JSON.stringify(request.messages)
}

const WS = process.platform === 'win32' ? 'C:\\work\\ledger' : '/work/ledger'

describe('memory tools through the agent loop', () => {
  it('saves, then the next request carries the memory index, and search finds it', async () => {
    const adapter = new MockAdapter([
      toolCallResponse('c1', 'memory_save', {
        kind: 'feedback', title: 'Use pnpm, never npm', body: 'User corrected npm twice.', scope: 'global', tags: ['tooling'],
      }),
      textResponse('Saved.'),
      toolCallResponse('c2', 'memory_search', { query: 'npm' }),
      textResponse('Found it.'),
    ])
    const ctx = await harness(adapter, dbPath())
    const agent = await ctx.agentLoop.create(SessionId('it-memory'), { provider: 'mock', model: 'mock' }, { cwd: WS })

    await say(ctx, agent, 'remember: pnpm, not npm')
    expect(results(agent.session.snapshotEvents())[0]).toContain('Saved memory #1.')
    // The request right after the save already sees the index.
    expect(requestText(adapter.requests[1]!)).toContain('Use pnpm, never npm')
    expect(requestText(adapter.requests[0]!)).not.toContain('Memory index')

    await say(ctx, agent, 'what do you remember about npm?')
    const search = results(agent.session.snapshotEvents())[1]!
    expect(search).toContain('#1 [feedback, global] Use pnpm, never npm (tooling)')
    expect(search).toContain('User corrected npm twice.')
    await ctx.fiber.dispose()
  })

  it('scopes workspace memories to their workspace and survives a restart', async () => {
    const path = dbPath()
    const first = new MockAdapter([
      toolCallResponse('c1', 'memory_save', { kind: 'project', title: 'Batch size capped at 750', body: 'Replica lag alarm.' }),
      textResponse('ok'),
    ])
    const ctx1 = await harness(first, path)
    const a1 = await ctx1.agentLoop.create(SessionId('it-ws-1'), { provider: 'mock', model: 'mock' }, { cwd: WS })
    await say(ctx1, a1, 'note the batch cap')
    await ctx1.fiber.dispose()

    // Same workspace, new process: the index lists it and memory_get reads it.
    const second = new MockAdapter([
      toolCallResponse('c2', 'memory_get', { id: 1 }),
      textResponse('ok'),
    ])
    const ctx2 = await harness(second, path)
    const a2 = await ctx2.agentLoop.create(SessionId('it-ws-2'), { provider: 'mock', model: 'mock' }, { cwd: WS })
    await say(ctx2, a2, 'what is the batch cap?')
    expect(requestText(second.requests[0]!)).toContain('[project, workspace] Batch size capped at 750')
    expect(results(a2.session.snapshotEvents())[0]).toContain('Replica lag alarm.')
    await ctx2.fiber.dispose()

    // Another workspace sees neither the index line nor the memory.
    const third = new MockAdapter([
      toolCallResponse('c3', 'memory_get', { id: 1 }),
      textResponse('ok'),
    ])
    const ctx3 = await harness(third, path)
    const other = process.platform === 'win32' ? 'C:\\work\\other' : '/work/other'
    const a3 = await ctx3.agentLoop.create(SessionId('it-ws-3'), { provider: 'mock', model: 'mock' }, { cwd: other })
    await say(ctx3, a3, 'anything about batches?')
    expect(requestText(third.requests[0]!)).not.toContain('Batch size capped')
    expect(results(a3.session.snapshotEvents())[0]).toBe('No memory #1 here.')
    await ctx3.fiber.dispose()
  })

  it('updates and forgets, and the index follows', async () => {
    const adapter = new MockAdapter([
      toolCallResponse('c1', 'memory_save', { kind: 'user', title: 'Old title', body: 'b', scope: 'global' }),
      toolCallResponse('c2', 'memory_update', { id: 1, title: 'New title', pinned: true }),
      toolCallResponse('c3', 'memory_forget', { id: 1 }),
      textResponse('done'),
    ])
    const ctx = await harness(adapter, dbPath())
    const agent = await ctx.agentLoop.create(SessionId('it-upd'), { provider: 'mock', model: 'mock' }, { cwd: WS })
    await say(ctx, agent, 'go')
    const out = results(agent.session.snapshotEvents())
    expect(out[1]).toContain('Updated memory #1.')
    expect(out[2]).toBe('Forgot memory #1.')
    expect(requestText(adapter.requests[2]!)).toContain('[user, global, pinned] New title')
    // After forgetting, the index is cleared again.
    expect(requestText(adapter.requests[3]!)).toContain('Current runtime context: none')
    await ctx.fiber.dispose()
  })
})
