/**
 * SQLite-backed durable memory store (node:sqlite, FTS5 search).
 *
 * One row per memory: a short title that works as an index line, a body, a
 * kind, a scope (`global` or one workspace path) and provenance. Forgetting is
 * a soft delete so an accidental `memory_forget` can be undone by hand.
 *
 * @module @deepseek-ai/dsh-tool-memory/store
 */

import { mkdirSync } from 'node:fs'
import { dirname, resolve as resolvePath } from 'node:path'
import type { DatabaseSync as DatabaseSyncType } from 'node:sqlite'
import { createRequire } from 'node:module'

/** The four memory kinds, mirroring how the agent should classify what it keeps. */
export const MEMORY_KINDS = ['user', 'feedback', 'project', 'reference'] as const
export type MemoryKind = typeof MEMORY_KINDS[number]

/** The scope value for memories that apply in every workspace. */
export const GLOBAL_SCOPE = 'global'

/** One stored memory as the tools see it. */
export interface Memory {
  readonly id: number
  readonly kind: MemoryKind
  readonly scope: string
  readonly title: string
  readonly body: string
  readonly tags: readonly string[]
  readonly pinned: boolean
  readonly createdAt: string
  readonly updatedAt: string
  readonly sessionId: string | null
}

/** Fields accepted when saving a new memory. */
export interface NewMemory {
  readonly kind: MemoryKind
  readonly scope: string
  readonly title: string
  readonly body: string
  readonly tags?: readonly string[]
  readonly pinned?: boolean
  readonly sessionId?: string
}

/** Fields accepted when updating; omitted fields keep their value. */
export interface MemoryPatch {
  readonly kind?: MemoryKind
  readonly scope?: string
  readonly title?: string
  readonly body?: string
  readonly tags?: readonly string[]
  readonly pinned?: boolean
}

/** Search or list filter. Scopes are matched exactly against normalized values. */
export interface MemoryFilter {
  readonly scopes?: readonly string[]
  readonly kind?: MemoryKind
  readonly limit?: number
}

const SCHEMA_VERSION = 1

const SCHEMA = `
CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS memories (
  id          INTEGER PRIMARY KEY AUTOINCREMENT,
  kind        TEXT NOT NULL CHECK (kind IN ('user', 'feedback', 'project', 'reference')),
  scope       TEXT NOT NULL,
  title       TEXT NOT NULL,
  body        TEXT NOT NULL,
  tags        TEXT NOT NULL DEFAULT '',
  pinned      INTEGER NOT NULL DEFAULT 0,
  created_at  TEXT NOT NULL,
  updated_at  TEXT NOT NULL,
  session_id  TEXT,
  deleted_at  TEXT
);
CREATE INDEX IF NOT EXISTS memories_scope ON memories (scope, deleted_at);
CREATE VIRTUAL TABLE IF NOT EXISTS memories_fts USING fts5(
  title, body, tags, content='memories', content_rowid='id', tokenize='unicode61 remove_diacritics 2'
);
CREATE TRIGGER IF NOT EXISTS memories_ai AFTER INSERT ON memories BEGIN
  INSERT INTO memories_fts (rowid, title, body, tags) VALUES (new.id, new.title, new.body, new.tags);
END;
CREATE TRIGGER IF NOT EXISTS memories_ad AFTER DELETE ON memories BEGIN
  INSERT INTO memories_fts (memories_fts, rowid, title, body, tags) VALUES ('delete', old.id, old.title, old.body, old.tags);
END;
CREATE TRIGGER IF NOT EXISTS memories_au AFTER UPDATE ON memories BEGIN
  INSERT INTO memories_fts (memories_fts, rowid, title, body, tags) VALUES ('delete', old.id, old.title, old.body, old.tags);
  INSERT INTO memories_fts (rowid, title, body, tags) VALUES (new.id, new.title, new.body, new.tags);
END;
`

interface Row {
  id: number
  kind: string
  scope: string
  title: string
  body: string
  tags: string
  pinned: number
  created_at: string
  updated_at: string
  session_id: string | null
}

/**
 * Normalize a workspace path into its scope key: absolute, forward slashes, no
 * trailing slash, and lower-case on Windows where paths are case-insensitive.
 * @param scope - `global` or a workspace path.
 * @returns the stored scope key.
 */
export function normalizeScope(scope: string): string {
  const trimmed = scope.trim()
  if (trimmed.length === 0 || trimmed.toLowerCase() === GLOBAL_SCOPE) return GLOBAL_SCOPE
  let path = resolvePath(trimmed).replace(/\\/g, '/')
  if (path.length > 1 && path.endsWith('/') && !/^[a-zA-Z]:\/$/.test(path)) path = path.slice(0, -1)
  return process.platform === 'win32' ? path.toLowerCase() : path
}

/** Normalize tags: trimmed, lower-case, unique, no empties. */
function normalizeTags(tags: readonly string[] | undefined): string[] {
  return [...new Set((tags ?? []).map(tag => tag.trim().toLowerCase()).filter(tag => tag.length > 0))]
}

/**
 * Turn free text into an FTS5 query that cannot fail to parse: each word is a
 * quoted prefix term, OR-joined so a partial match still ranks.
 * @param text - the model's search text.
 * @returns the FTS5 MATCH expression, or `undefined` when no word remains.
 */
export function ftsQuery(text: string): string | undefined {
  const words = text.toLowerCase().match(/[\p{L}\p{N}_]+/gu) ?? []
  const unique = [...new Set(words)].slice(0, 24)
  if (unique.length === 0) return undefined
  return unique.map(word => `"${word}"*`).join(' OR ')
}

/** Durable memory store over one SQLite file. */
export class MemoryStore {
  private readonly db: DatabaseSyncType

  /**
   * Open (and create when missing) the database at `path`.
   * @param path - database file, or `:memory:` for an ephemeral store.
   */
  constructor(path: string) {
    // Loaded lazily through require so importing this module never triggers
    // node:sqlite's experimental warning in processes that never open a store.
    const { DatabaseSync } = createRequire(import.meta.url)('node:sqlite') as typeof import('node:sqlite')
    if (path !== ':memory:') mkdirSync(dirname(path), { recursive: true })
    this.db = new DatabaseSync(path)
    this.db.exec('PRAGMA journal_mode = WAL; PRAGMA busy_timeout = 5000; PRAGMA foreign_keys = ON;')
    this.db.exec(SCHEMA)
    this.db.prepare('INSERT OR IGNORE INTO meta (key, value) VALUES (?, ?)').run('schema_version', String(SCHEMA_VERSION))
  }

  /** Close the underlying database handle. */
  close(): void {
    this.db.close()
  }

  /**
   * Save a new memory.
   * @param memory - the memory to store.
   * @returns the stored memory with its id.
   */
  save(memory: NewMemory): Memory {
    const now = new Date().toISOString()
    const result = this.db.prepare(`
      INSERT INTO memories (kind, scope, title, body, tags, pinned, created_at, updated_at, session_id)
      VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
    `).run(
      memory.kind,
      normalizeScope(memory.scope),
      memory.title.trim(),
      memory.body.trim(),
      normalizeTags(memory.tags).join(' '),
      memory.pinned === true ? 1 : 0,
      now,
      now,
      memory.sessionId ?? null,
    )
    const saved = this.get(Number(result.lastInsertRowid))
    if (saved === undefined) throw new Error('memory insert returned no row')
    return saved
  }

  /**
   * Read one live (not forgotten) memory.
   * @param id - memory id.
   * @returns the memory, or `undefined` when missing or forgotten.
   */
  get(id: number): Memory | undefined {
    const row = this.db.prepare('SELECT * FROM memories WHERE id = ? AND deleted_at IS NULL').get(id) as Row | undefined
    return row === undefined ? undefined : toMemory(row)
  }

  /**
   * Apply a partial update to a live memory.
   * @param id - memory id.
   * @param patch - fields to change.
   * @returns the updated memory, or `undefined` when missing or forgotten.
   */
  update(id: number, patch: MemoryPatch): Memory | undefined {
    const current = this.get(id)
    if (current === undefined) return undefined
    this.db.prepare(`
      UPDATE memories SET kind = ?, scope = ?, title = ?, body = ?, tags = ?, pinned = ?, updated_at = ?
      WHERE id = ?
    `).run(
      patch.kind ?? current.kind,
      patch.scope === undefined ? current.scope : normalizeScope(patch.scope),
      (patch.title ?? current.title).trim(),
      (patch.body ?? current.body).trim(),
      normalizeTags(patch.tags ?? current.tags).join(' '),
      (patch.pinned ?? current.pinned) ? 1 : 0,
      new Date().toISOString(),
      id,
    )
    return this.get(id)
  }

  /**
   * Soft-delete a memory; it disappears from every read but stays in the file.
   * @param id - memory id.
   * @returns whether a live memory was forgotten.
   */
  forget(id: number): boolean {
    const result = this.db.prepare('UPDATE memories SET deleted_at = ? WHERE id = ? AND deleted_at IS NULL')
      .run(new Date().toISOString(), id)
    return Number(result.changes) > 0
  }

  /**
   * Full-text search over title, body and tags, best match first.
   * @param text - free-text query.
   * @param filter - optional scopes, kind and limit.
   * @returns matching live memories.
   */
  search(text: string, filter: MemoryFilter = {}): Memory[] {
    const match = ftsQuery(text)
    if (match === undefined) return []
    const { where, params } = filterSql(filter)
    const rows = this.db.prepare(`
      SELECT m.* FROM memories_fts f JOIN memories m ON m.id = f.rowid
      WHERE memories_fts MATCH ? AND m.deleted_at IS NULL ${where}
      ORDER BY bm25(memories_fts, 10.0, 3.0, 5.0), m.pinned DESC, m.updated_at DESC
      LIMIT ?
    `).all(match, ...params, filter.limit ?? 10) as unknown as Row[]
    return rows.map(toMemory)
  }

  /**
   * List live memories, pinned first, then most recently updated.
   * @param filter - optional scopes, kind and limit.
   * @returns the listed memories.
   */
  list(filter: MemoryFilter = {}): Memory[] {
    const { where, params } = filterSql(filter)
    const rows = this.db.prepare(`
      SELECT * FROM memories m WHERE m.deleted_at IS NULL ${where}
      ORDER BY m.pinned DESC, m.updated_at DESC LIMIT ?
    `).all(...params, filter.limit ?? 50) as unknown as Row[]
    return rows.map(toMemory)
  }

  /**
   * Count live memories in the given scopes.
   * @param scopes - normalized scope keys.
   * @returns the number of live memories.
   */
  count(scopes: readonly string[]): number {
    const { where, params } = filterSql({ scopes })
    const row = this.db.prepare(`SELECT COUNT(*) AS n FROM memories m WHERE m.deleted_at IS NULL ${where}`)
      .get(...params) as { n: number }
    return Number(row.n)
  }
}

function filterSql(filter: MemoryFilter): { where: string; params: (string | number)[] } {
  const clauses: string[] = []
  const params: (string | number)[] = []
  if (filter.scopes !== undefined) {
    if (filter.scopes.length === 0) return { where: 'AND 0', params }
    clauses.push(`m.scope IN (${filter.scopes.map(() => '?').join(', ')})`)
    params.push(...filter.scopes)
  }
  if (filter.kind !== undefined) {
    clauses.push('m.kind = ?')
    params.push(filter.kind)
  }
  return { where: clauses.map(clause => `AND ${clause}`).join(' '), params }
}

function toMemory(row: Row): Memory {
  return {
    id: Number(row.id),
    kind: row.kind as MemoryKind,
    scope: row.scope,
    title: row.title,
    body: row.body,
    tags: row.tags.length === 0 ? [] : row.tags.split(' '),
    pinned: Number(row.pinned) === 1,
    createdAt: row.created_at,
    updatedAt: row.updated_at,
    sessionId: row.session_id,
  }
}
