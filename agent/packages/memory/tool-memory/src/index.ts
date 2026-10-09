/**
 * Durable agent memory: a SQLite store the model reads and writes through five
 * tools, plus a compact index of what it already remembers delivered as
 * runtime context (re-sent only when the index changes, so the prompt cache
 * survives).
 *
 * @module @deepseek-ai/dsh-tool-memory
 */

import { homedir } from 'node:os'
import { join } from 'node:path'
import type { Context } from '@deepseek-ai/cordis'
import z from '@deepseek-ai/schemastery'
import type {} from '@deepseek-ai/dsh-agent'
import { defineTool } from '@deepseek-ai/dsh-tools'
import type { ToolRunContext } from '@deepseek-ai/dsh-tools'
import {
  GLOBAL_SCOPE,
  MEMORY_KINDS,
  MemoryStore,
  normalizeScope,
  type Memory,
  type MemoryKind,
} from './store.ts'

export { GLOBAL_SCOPE, MEMORY_KINDS, MemoryStore, ftsQuery, normalizeScope } from './store.ts'
export type { Memory, MemoryFilter, MemoryKind, MemoryPatch, NewMemory } from './store.ts'

/** Cordis plugin name used by Loader diagnostics. */
export const name = 'tool-memory'

/** Capability services required by the plugin. */
export const inject = ['tools', 'systemPrompt']

/** Plugin configuration. */
export interface Config {
  /** Database file. Defaults to `$DSH_HOME/memory/memory.db` (`~/.dsh` when DSH_HOME is unset); `:memory:` keeps nothing. */
  path?: string
  /** Most memories listed in the runtime-context index. Defaults to 40. */
  indexMaxItems?: number
  /** Character budget of the runtime-context index. Defaults to 4000. */
  indexMaxChars?: number
  /** Default and maximum hits per `memory_search`. Defaults to 10 / 50. */
  searchLimit?: number
  maxSearchLimit?: number
}

/** Schemastery config for Loader defaults and generated docs. */
export const Config: z<Config> = z.object({
  path: z.string(),
  indexMaxItems: z.number().step(1).min(0).default(40),
  indexMaxChars: z.number().step(1).min(0).default(4000),
  searchLimit: z.number().step(1).min(1).default(10),
  maxSearchLimit: z.number().step(1).min(1).default(50),
})

const TEXT_OUTPUT = {
  schema: { type: 'string' as const },
  render: (_args: unknown, value: string) => [{ type: 'text' as const, text: value }],
}

const KIND_ENUM = [...MEMORY_KINDS] as const
const SCOPE_DESCRIPTION = '"workspace" (default: only this workspace) or "global" (every workspace).'

/** Model guidance: when to save, when not, and how. */
export const PROMPT_TEXT = [
  'You have a durable memory that survives across sessions and context compaction (memory_save, memory_search,',
  'memory_get, memory_update, memory_forget). An index of what you already remember arrives as runtime context.',
  '',
  'Save a memory when you learn something a future session would need and could not rediscover from the code:',
  '- user: who the user is, their preferences, how they like to work (usually global).',
  '- feedback: a correction or confirmed approach from the user, with why; quote their words when exact wording matters.',
  '- project: goals, decisions and constraints of this workspace that are not in the code or git history.',
  '- reference: where to find things (URLs, hosts, dashboards, file locations).',
  'Do not save secrets or credentials, things the repository already records, or details only this conversation needs.',
  '',
  'Write the title as a one-line hook that makes sense in the index on its own. Before saving, check the index or',
  'memory_search for an existing memory on the same fact and memory_update it instead of saving a duplicate. Forget',
  'memories that turn out to be wrong. A memory reflects what was true when it was written: verify file names,',
  'flags and paths still exist before relying on them.',
].join('\n')

/** Resolve the effective database path. */
function databasePath(config: Config): string {
  const configured = config.path?.trim()
  return configured === undefined || configured.length === 0
    ? join(process.env.DSH_HOME?.trim() || join(homedir(), '.dsh'), 'memory', 'memory.db')
    : configured
}

/** The two scopes visible to a caller: global plus its workspace, when it has one. */
function scopesFor(cwd: string | undefined): string[] {
  return cwd === undefined ? [GLOBAL_SCOPE] : [GLOBAL_SCOPE, normalizeScope(cwd)]
}

function callerOf(exec: ToolRunContext): { cwd: string | undefined; sessionId: string | undefined } {
  const session = exec.agent?.session
  return { cwd: session?.header.cwd, sessionId: session === undefined ? undefined : String(session.id) }
}

/** Resolve the model's scope choice against the caller's workspace. */
function scopeFor(choice: string | undefined, cwd: string | undefined): string {
  if (choice === GLOBAL_SCOPE || cwd === undefined) return GLOBAL_SCOPE
  return normalizeScope(cwd)
}

/** One index line: `#id [kind, scope] title (tags)`. */
function indexLine(memory: Memory): string {
  const scope = memory.scope === GLOBAL_SCOPE ? 'global' : 'workspace'
  const tags = memory.tags.length > 0 ? ` (${memory.tags.join(', ')})` : ''
  return `- #${memory.id} [${memory.kind}, ${scope}${memory.pinned ? ', pinned' : ''}] ${memory.title}${tags}`
}

/** Full rendering for get / save / update results. */
function renderMemory(memory: Memory): string {
  return [
    `#${memory.id} [${memory.kind}] ${memory.title}`,
    `scope: ${memory.scope}${memory.pinned ? ' (pinned)' : ''}`,
    ...memory.tags.length > 0 ? [`tags: ${memory.tags.join(', ')}`] : [],
    `updated: ${memory.updatedAt}`,
    '',
    memory.body,
  ].join('\n')
}

/**
 * Render the runtime-context index for one workspace, within budget.
 * @param store - the memory store.
 * @param cwd - the caller's workspace, if any.
 * @param maxItems - most memories to list.
 * @param maxChars - character budget.
 * @returns the index text, or '' when there is nothing to show.
 */
export function renderIndex(store: MemoryStore, cwd: string | undefined, maxItems: number, maxChars: number): string {
  if (maxItems <= 0 || maxChars <= 0) return ''
  const scopes = scopesFor(cwd)
  const total = store.count(scopes)
  if (total === 0) return ''
  const header = 'Memory index (durable memories from earlier sessions; memory_get <id> reads one in full):'
  const lines: string[] = []
  let used = header.length
  for (const memory of store.list({ scopes, limit: maxItems })) {
    const line = indexLine(memory)
    if (used + line.length + 1 > maxChars) break
    lines.push(line)
    used += line.length + 1
  }
  const hidden = total - lines.length
  if (hidden > 0) lines.push(`- ... ${hidden} more; use memory_search to find them.`)
  return [header, ...lines].join('\n')
}

/** Register the memory tools, the guidance section and the index context. */
export function apply(ctx: Context, config: Config): void {
  const store = new MemoryStore(databasePath(config))
  ctx.effect(() => () => store.close(), 'tool-memory.store')
  const indexMaxItems = config.indexMaxItems ?? 40
  const indexMaxChars = config.indexMaxChars ?? 4000
  const searchLimit = config.searchLimit ?? 10
  const maxSearchLimit = config.maxSearchLimit ?? 50

  ctx.systemPrompt.section({
    name: 'tool:memory',
    order: ctx.systemPrompt.getSectionOrder('TOOL_SESSION_QUERY') + 50,
    text: PROMPT_TEXT,
  })

  ctx.systemPrompt.context({
    name: 'memory:index',
    order: 130,
    text: (context) => {
      const session = context.agent?.session
      if (session === undefined) return ''
      try {
        return renderIndex(store, session.header.cwd, indexMaxItems, indexMaxChars)
      } catch (error) {
        ctx.logger.warn(`memory index unavailable: ${error instanceof Error ? error.message : String(error)}`)
        return ''
      }
    },
  })

  ctx.tools.register(defineTool({
    name: 'memory_save',
    description: 'Save a durable memory that later sessions can recall. Check for an existing memory on the same fact first and update it instead.',
    parameters: {
      kind: { type: 'string', enum: KIND_ENUM, required: true, description: 'user, feedback, project or reference.' },
      title: { type: 'string', required: true, description: 'One-line hook that reads well on its own in the memory index.' },
      body: { type: 'string', required: true, description: 'The fact itself; for feedback and project memories add why it matters and how to apply it.' },
      scope: { type: 'string', enum: ['workspace', 'global'], description: SCOPE_DESCRIPTION },
      tags: { type: 'array', items: { type: 'string' }, description: 'Optional search tags.' },
      pinned: { type: 'boolean', description: 'Keep it at the top of the index.' },
    },
    output: TEXT_OUTPUT,
    execute: async (args, exec) => {
      const caller = callerOf(exec)
      const saved = store.save({
        kind: args.kind as MemoryKind,
        scope: scopeFor(args.scope, caller.cwd),
        title: args.title,
        body: args.body,
        ...args.tags === undefined ? {} : { tags: args.tags },
        ...args.pinned === undefined ? {} : { pinned: args.pinned },
        ...caller.sessionId === undefined ? {} : { sessionId: caller.sessionId },
      })
      return `Saved memory #${saved.id}.\n\n${renderMemory(saved)}`
    },
    presentCall: args => ({ card: 'generic', kind: 'edit', title: 'Save memory', rawInput: args.title }),
  }))

  ctx.tools.register(defineTool({
    name: 'memory_search',
    description: 'Full-text search over memories visible here (global plus this workspace), best match first.',
    parameters: {
      query: { type: 'string', required: true, description: 'Words to look for in titles, bodies and tags.' },
      kind: { type: 'string', enum: KIND_ENUM, description: 'Only this kind.' },
      limit: { type: 'integer', description: `Most results (default ${searchLimit}, max ${maxSearchLimit}).` },
    },
    output: TEXT_OUTPUT,
    isConcurrencySafe: () => true,
    execute: async (args, exec) => {
      const limit = Math.min(Math.max(args.limit ?? searchLimit, 1), maxSearchLimit)
      const hits = store.search(args.query, {
        scopes: scopesFor(callerOf(exec).cwd),
        ...args.kind === undefined ? {} : { kind: args.kind as MemoryKind },
        limit,
      })
      if (hits.length === 0) return `No memories match "${args.query}".`
      return hits.map((memory) => {
        const snippet = memory.body.length > 240 ? `${memory.body.slice(0, 240)}…` : memory.body
        return `${indexLine(memory).slice(2)}\n  ${snippet.replace(/\n+/g, ' ')}`
      }).join('\n')
    },
    presentCall: args => ({ card: 'generic', kind: 'search', title: 'Search memory', rawInput: args.query }),
  }))

  ctx.tools.register(defineTool({
    name: 'memory_get',
    description: 'Read one memory in full by id.',
    parameters: {
      id: { type: 'integer', required: true, description: 'Memory id, as shown in the index or search results.' },
    },
    output: TEXT_OUTPUT,
    isConcurrencySafe: () => true,
    execute: async (args, exec) => {
      const memory = visible(store.get(args.id), callerOf(exec).cwd)
      return memory === undefined ? `No memory #${args.id} here.` : renderMemory(memory)
    },
    presentCall: args => ({ card: 'generic', kind: 'read', title: `Read memory #${args.id}` }),
  }))

  ctx.tools.register(defineTool({
    name: 'memory_update',
    description: 'Change an existing memory; omitted fields keep their value.',
    parameters: {
      id: { type: 'integer', required: true, description: 'Memory id.' },
      kind: { type: 'string', enum: KIND_ENUM, description: 'New kind.' },
      title: { type: 'string', description: 'New title.' },
      body: { type: 'string', description: 'New body (replaces the old one).' },
      scope: { type: 'string', enum: ['workspace', 'global'], description: SCOPE_DESCRIPTION },
      tags: { type: 'array', items: { type: 'string' }, description: 'New tags (replace the old ones).' },
      pinned: { type: 'boolean', description: 'Pin or unpin.' },
    },
    output: TEXT_OUTPUT,
    execute: async (args, exec) => {
      const caller = callerOf(exec)
      if (visible(store.get(args.id), caller.cwd) === undefined) return `No memory #${args.id} here.`
      const updated = store.update(args.id, {
        ...args.kind === undefined ? {} : { kind: args.kind as MemoryKind },
        ...args.title === undefined ? {} : { title: args.title },
        ...args.body === undefined ? {} : { body: args.body },
        ...args.scope === undefined ? {} : { scope: scopeFor(args.scope, caller.cwd) },
        ...args.tags === undefined ? {} : { tags: args.tags },
        ...args.pinned === undefined ? {} : { pinned: args.pinned },
      })
      return updated === undefined ? `No memory #${args.id} here.` : `Updated memory #${updated.id}.\n\n${renderMemory(updated)}`
    },
    presentCall: args => ({ card: 'generic', kind: 'edit', title: `Update memory #${args.id}` }),
  }))

  ctx.tools.register(defineTool({
    name: 'memory_forget',
    description: 'Forget a memory that is wrong or no longer useful.',
    parameters: {
      id: { type: 'integer', required: true, description: 'Memory id.' },
    },
    output: TEXT_OUTPUT,
    execute: async (args, exec) => {
      if (visible(store.get(args.id), callerOf(exec).cwd) === undefined) return `No memory #${args.id} here.`
      return store.forget(args.id) ? `Forgot memory #${args.id}.` : `No memory #${args.id} here.`
    },
    presentCall: args => ({ card: 'generic', kind: 'delete', title: `Forget memory #${args.id}` }),
  }))
}

/** A memory is visible when it is global or belongs to the caller's workspace. */
function visible(memory: Memory | undefined, cwd: string | undefined): Memory | undefined {
  if (memory === undefined) return undefined
  return scopesFor(cwd).includes(memory.scope) ? memory : undefined
}
