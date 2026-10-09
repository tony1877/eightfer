import { mkdtempSync, rmSync } from 'node:fs'
import { tmpdir } from 'node:os'
import { join } from 'node:path'
import { afterEach, describe, expect, it } from 'vitest'
import {
  GLOBAL_SCOPE,
  MemoryStore,
  ftsQuery,
  normalizeScope,
  renderIndex,
} from '@deepseek-ai/dsh-tool-memory'

const stores: MemoryStore[] = []
const dirs: string[] = []
function store(path = ':memory:'): MemoryStore {
  const s = new MemoryStore(path)
  stores.push(s)
  return s
}
afterEach(() => {
  for (const s of stores.splice(0)) s.close()
  for (const d of dirs.splice(0)) rmSync(d, { recursive: true, force: true })
})

const WS = process.platform === 'win32' ? 'C:\\Work\\Ledger' : '/work/ledger'
const OTHER = process.platform === 'win32' ? 'C:\\Work\\Other' : '/work/other'

describe('normalizeScope', () => {
  it('keeps global and normalizes workspace paths', () => {
    expect(normalizeScope('global')).toBe(GLOBAL_SCOPE)
    expect(normalizeScope('  GLOBAL ')).toBe(GLOBAL_SCOPE)
    expect(normalizeScope('')).toBe(GLOBAL_SCOPE)
    const a = normalizeScope(WS)
    expect(a).not.toContain('\\')
    expect(normalizeScope(`${WS}${process.platform === 'win32' ? '\\' : '/'}`)).toBe(a)
    if (process.platform === 'win32') expect(normalizeScope('c:\\work\\ledger')).toBe(a)
  })
})

describe('ftsQuery', () => {
  it('quotes each word as a prefix term and ignores punctuation', () => {
    expect(ftsQuery('pnpm, not npm!')).toBe('"pnpm"* OR "not"* OR "npm"*')
    expect(ftsQuery('  ')).toBeUndefined()
    expect(ftsQuery('"); DROP TABLE memories; --')).toBe('"drop"* OR "table"* OR "memories"*')
  })
})

describe('MemoryStore', () => {
  it('saves, reads, updates and forgets', () => {
    const s = store()
    const m = s.save({ kind: 'feedback', scope: WS, title: 'Use pnpm, not npm', body: 'User corrected this twice.', tags: ['Tooling', 'tooling', ' pnpm '], sessionId: 'S1' })
    expect(m).toMatchObject({ id: 1, kind: 'feedback', scope: normalizeScope(WS), tags: ['tooling', 'pnpm'], pinned: false, sessionId: 'S1' })
    expect(s.get(m.id)?.body).toBe('User corrected this twice.')

    const u = s.update(m.id, { body: 'Always pnpm.', pinned: true })
    expect(u).toMatchObject({ body: 'Always pnpm.', pinned: true, title: 'Use pnpm, not npm' })

    expect(s.forget(m.id)).toBe(true)
    expect(s.forget(m.id)).toBe(false)
    expect(s.get(m.id)).toBeUndefined()
    expect(s.update(m.id, { title: 'x' })).toBeUndefined()
  })

  it('searches title, body and tags with prefix matching, scoped', () => {
    const s = store()
    s.save({ kind: 'project', scope: WS, title: 'Batch size capped at 750', body: 'Replica lag alarm fires above it.' })
    s.save({ kind: 'reference', scope: GLOBAL_SCOPE, title: 'Staging host', body: 'https://staging.internal.example:8443', tags: ['urls'] })
    s.save({ kind: 'project', scope: OTHER, title: 'Other repo batch notes', body: 'unrelated batch' })
    const scopes = [GLOBAL_SCOPE, normalizeScope(WS)]

    expect(s.search('replica', { scopes }).map(m => m.title)).toEqual(['Batch size capped at 750'])
    expect(s.search('stag', { scopes }).map(m => m.title)).toEqual(['Staging host'])
    expect(s.search('urls', { scopes }).map(m => m.title)).toEqual(['Staging host'])
    expect(s.search('batch', { scopes }).map(m => m.title)).toEqual(['Batch size capped at 750'])
    expect(s.search('batch', { scopes: [normalizeScope(OTHER)] }).map(m => m.title)).toEqual(['Other repo batch notes'])
    expect(s.search('batch', { scopes, kind: 'reference' })).toEqual([])
    expect(s.search('!!!', { scopes })).toEqual([])
  })

  it('keeps search in sync after updates and forgets', () => {
    const s = store()
    const m = s.save({ kind: 'user', scope: GLOBAL_SCOPE, title: 'Prefers terse answers', body: 'no preamble' })
    s.update(m.id, { body: 'wants zstd everywhere' })
    expect(s.search('preamble')).toEqual([])
    expect(s.search('zstd').map(x => x.id)).toEqual([m.id])
    s.forget(m.id)
    expect(s.search('zstd')).toEqual([])
  })

  it('lists pinned first, then newest, and counts by scope', () => {
    const s = store()
    const a = s.save({ kind: 'user', scope: GLOBAL_SCOPE, title: 'a', body: 'a' })
    s.save({ kind: 'user', scope: GLOBAL_SCOPE, title: 'b', body: 'b' })
    s.update(a.id, { pinned: true })
    expect(s.list().map(m => m.title)).toEqual(['a', 'b'])
    expect(s.count([GLOBAL_SCOPE])).toBe(2)
    expect(s.count([normalizeScope(WS)])).toBe(0)
    expect(s.count([])).toBe(0)
  })

  it('persists across reopen on disk', () => {
    const dir = mkdtempSync(join(tmpdir(), 'dsh-memory-'))
    dirs.push(dir)
    const path = join(dir, 'nested', 'memory.db')
    const first = new MemoryStore(path)
    first.save({ kind: 'reference', scope: GLOBAL_SCOPE, title: 'kept', body: 'across restarts' })
    first.close()
    expect(store(path).search('restarts').map(m => m.title)).toEqual(['kept'])
  })
})

describe('renderIndex', () => {
  it('is empty with no memories and lists visible ones within budget', () => {
    const s = store()
    expect(renderIndex(s, WS, 40, 4000)).toBe('')
    s.save({ kind: 'user', scope: GLOBAL_SCOPE, title: 'Global fact', body: 'g', tags: ['x'] })
    s.save({ kind: 'project', scope: WS, title: 'Workspace fact', body: 'w' })
    s.save({ kind: 'project', scope: OTHER, title: 'Hidden fact', body: 'h' })
    const text = renderIndex(s, WS, 40, 4000)
    expect(text).toContain('Memory index')
    expect(text).toContain('[user, global] Global fact (x)')
    expect(text).toContain('[project, workspace] Workspace fact')
    expect(text).not.toContain('Hidden fact')
    expect(renderIndex(s, undefined, 40, 4000)).not.toContain('Workspace fact')
  })

  it('reports how many did not fit', () => {
    const s = store()
    for (let i = 0; i < 5; i++) s.save({ kind: 'user', scope: GLOBAL_SCOPE, title: `fact ${i}`, body: 'b' })
    const text = renderIndex(s, WS, 2, 4000)
    expect(text.split('\n').filter(l => l.startsWith('- #'))).toHaveLength(2)
    expect(text).toContain('... 3 more; use memory_search')
    expect(renderIndex(s, WS, 0, 4000)).toBe('')
  })
})
