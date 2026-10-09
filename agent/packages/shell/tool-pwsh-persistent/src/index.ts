/* jscpd:ignore-start -- deliberate mirror of tool-bash-persistent (persistent-pty note 2026-08-11-pwsh-persistent-pty):
   the PowerShell counterpart shares the session registry, polling loop, and reset contract by design. */
/**
 * Model-facing persistent `pwsh` tool over the owner-scoped PTY seam.
 * @module @deepseek-ai/dsh-tool-pwsh-persistent
 */

import { randomUUID } from 'node:crypto'
import type { Context } from '@deepseek-ai/cordis'
import z from '@deepseek-ai/schemastery'
import type { Agent } from '@deepseek-ai/dsh-agent'
import type { JobHooks, JobOutcome, JobRegistry } from '@deepseek-ai/dsh-jobs'
import type { TerminalReadResult, TerminalSendOperation, TerminalSendResult, TerminalSessionId } from '@deepseek-ai/dsh-terminal'
import { deadline, timeoutOf } from '@deepseek-ai/dsh-timeout'
import { defineTool } from '@deepseek-ai/dsh-tools'

// TODO: Replace the file-search advice; arbitrary command output need not come from a searchable file.
const TRUNCATED_MESSAGE = '<response clipped><NOTE>To save on context only part of this file has been shown to you. You should retry this tool after you have searched inside the file with Select-String in order to find the line numbers of what you are looking for.</NOTE>'
const LOST_PREFIX_MESSAGE = '<response clipped><NOTE>The beginning of this command output was dropped by the terminal scrollback limit. The following text is the earliest retained output.</NOTE>\n'
const SHELL_RESET_MESSAGE = 'The persistent pwsh shell was reset; the next pwsh call starts from the workspace with a fresh current directory and environment.'
const SHELL_PROMPT = '__DSH_PERSISTENT_PWSH_PROMPT__ '
const TIMEOUT_CODE = 'PERSISTENT_PWSH_TIMEOUT'
// One page is enough to find a just-emitted completion marker; the full
// scrollback is assembled only when a command settles or needs partial output.
const SCROLLBACK_PAGE_LINES = 1_000
const POLL_INTERVAL_MS = 25

const DEFAULT_DESCRIPTION = 'Run commands in a persistent PowerShell shell. State, including the current directory and exported environment variables, persists across calls for this agent.'
// A background watcher polls the detached shell at this interval between sends.
const BACKGROUND_POLL_MS = 250
const PROMOTED_PREFIX = '[still running after '
const LOST_HEAD_NOTICE = '[earlier output of this job is no longer retained by the terminal]\n'

interface ResolvedConfig {
  backendType: string
  timeoutMs: number
  maxOutputChars: number
  description: string
  /** Auto-background wait in ms; 0 keeps kill-and-reset on `timeoutMs`. */
  autoBackgroundAfterMs: number
  /** Upper bound on a call's own foreground wait under auto-background. */
  maxForegroundMs: number
}

/** One command's progress through its shell: markers, the send in flight, and the output seen so far. */
interface CommandRun {
  id: TerminalSessionId
  marker: CommandMarkers
  wrapped: string
  first: boolean
  fallback: string
  fallbackTruncated: boolean
  operation: TerminalSendOperation | undefined
}

/** How a command's shell interaction settled, independent of who is waiting on it. */
type Settled =
  | { kind: 'complete'; output: CapturedOutput }
  | { kind: 'exited'; status: { exitCode: number | null; signal: NodeJS.Signals | null } }
  | { kind: 'prompt' }

interface CommandMarkers {
  start: string
  end: string
}

interface RetainedOutput {
  text: string
  truncated: boolean
}

interface CapturedOutput {
  text: string
  incomplete: boolean
  exitCode?: number
}

interface PersistentShells {
  get(owner: Agent, signal: AbortSignal): Promise<TerminalSessionId>
  reset(owner: Agent, reason: string): Promise<void>
  /**
   * Hand the owner's current shell over to a background command: the shell
   * keeps running, but the owner's next `get` spawns a fresh one.
   */
  detach(owner: Agent): TerminalSessionId | undefined
  /** Close a detached shell once its background command is over (idempotent). */
  closeDetached(id: TerminalSessionId, reason: string): Promise<void>
}

declare module '@deepseek-ai/dsh-jobs' {
  interface JobKindMap {
    pwsh: 'pwsh'
  }
}

function maybeTruncate(content: string, maxOutputChars: number, incomplete = false): string {
  if (content.length <= maxOutputChars && !incomplete) return content
  return content.length <= maxOutputChars
    ? content + TRUNCATED_MESSAGE
    : content.slice(0, maxOutputChars) + TRUNCATED_MESSAGE
}

function markers(): CommandMarkers {
  const nonce = randomUUID()
  return {
    start: `__DSH_PERSISTENT_PWSH_START_${nonce}__`,
    end: `__DSH_PERSISTENT_PWSH_END_${nonce}:`,
  }
}

/**
 * Escape a command body for embedding in the wrapper's double-quoted string.
 * Backtick escapes keep every character literal: backtick first so the
 * escapes this function inserts are never re-escaped, `$` so no expansion
 * happens at wrapper construction, and `\r\n`/ESC so multi-line commands and
 * raw control bytes ride one physical input line without PSReadLine mangling.
 * @param value - the model's PowerShell command text.
 * @returns the escaped double-quoted-string body.
 */
function quoteForPwsh(value: string): string {
  return value
    .replaceAll('`', '``')
    .replaceAll('"', '`"')
    .replaceAll('$', '`$')
    .replaceAll('\r', '')
    .replaceAll('\n', '`n')
    .replaceAll('\x1b', '`e')
}

function wrapCommand(command: string, marker: CommandMarkers): string {
  // Keep the wrapper on one physical line: PSReadLine renders the echoed
  // input, and a wrapped line would split the echo the extraction strips.
  // The echoed END nonce can never fabricate completion because the status
  // regex needs digits immediately after it and the echo continues with
  // quote characters.
  const body = quoteForPwsh(command)
  return `Write-Output '${marker.start}'; $LASTEXITCODE = $null; $__s = 1; try { Invoke-Expression "${body}"; $__ok = $? } catch { $__ok = $false }; if ($null -ne $LASTEXITCODE) { $__s = [int]$LASTEXITCODE } else { $__s = if ($__ok) { 0 } else { 1 } }; Write-Output ('${marker.end}' + $__s)`
}

function stripPrompt(text: string): string {
  let result = text.replace(/\r?\n$/, '')
  while (result.endsWith(SHELL_PROMPT)) {
    result = result.slice(0, -SHELL_PROMPT.length)
  }
  return result.endsWith('\n') ? result.slice(0, -1) : result
}

function commandOutput(
  snapshot: RetainedOutput,
  marker: CommandMarkers,
  wrapper: string,
): CapturedOutput | undefined {
  const text = snapshot.text
  const end = text.lastIndexOf(marker.end)
  const status = /^(\d+)\r?\n/.exec(text.slice(end + marker.end.length))?.[1]
  if (status === undefined) return undefined
  const startMarker = text.lastIndexOf(marker.start, end)
  const start = startMarker < 0 ? 0 : startMarker + marker.start.length
  let captured = text.slice(start, end)
  // The PSReadLine echo carries the wrapper source (including both marker
  // nonces) before the real markers; anchor on the real markers excludes it,
  // and stripping the wrapper covers the rare case where the real START
  // scrolled out and extraction fell back to the echoed copy.
  captured = captured.replaceAll(wrapper, '')
  return {
    text: captured.replace(/^\r?\n/, '').replace(/\r?\n$/, ''),
    incomplete: startMarker < 0,
    exitCode: Number(status),
  }
}

function promptCompleted(result: TerminalSendResult): boolean {
  return result.viewport.endsWith(SHELL_PROMPT)
    || result.viewport.endsWith(`${SHELL_PROMPT}\r\n`)
    || result.viewport.endsWith(`${SHELL_PROMPT}\n`)
}

function partialOutput(
  snapshot: RetainedOutput,
  marker: CommandMarkers,
  wrapper: string,
  fallback: string,
  fallbackTruncated = false,
): CapturedOutput {
  const startMarker = snapshot.text.lastIndexOf(marker.start)
  if (startMarker >= 0) {
    return {
      text: stripPrompt(snapshot.text.slice(startMarker + marker.start.length).replace(/^\r?\n/, '')),
      incomplete: false,
    }
  }
  const fallbackStart = fallback.lastIndexOf(marker.start)
  const afterStart = fallbackStart < 0
    ? fallback
    : fallback.slice(fallbackStart + marker.start.length).replace(/^\r?\n/, '')
  const fallbackEnd = afterStart.lastIndexOf(marker.end)
  const beforeEnd = fallbackEnd < 0 ? afterStart : afterStart.slice(0, fallbackEnd)
  return {
    text: stripPrompt(beforeEnd.replaceAll(SHELL_PROMPT, '').replaceAll(wrapper, '')),
    incomplete: fallbackTruncated || fallbackStart < 0,
  }
}

async function pause(): Promise<void> {
  await new Promise(resolve => setTimeout(resolve, POLL_INTERVAL_MS))
}

function nextScrollbackOffset(page: TerminalReadResult, offset: number): number | undefined {
  if (page.text.length === 0 || page.lineEnd <= offset) return undefined
  return page.lineEnd
}

function retainedScrollback(
  ctx: Context,
  owner: Agent,
  id: TerminalSessionId,
  latest = ctx.terminals.read(owner, id, { offset: 0, count: SCROLLBACK_PAGE_LINES }),
): RetainedOutput {
  const pages: string[] = latest.text.length === 0 ? [] : [latest.text]
  let offset = latest.lineEnd
  let truncated = latest.truncated
  while (true) {
    if (offset >= latest.totalLines) break
    const page = ctx.terminals.read(owner, id, { offset, count: SCROLLBACK_PAGE_LINES })
    truncated ||= page.truncated
    if (page.text.length > 0) pages.unshift(page.text)
    const next = nextScrollbackOffset(page, offset)
    if (next === undefined || next >= page.totalLines) break
    offset = next
  }
  return { text: pages.join('\n'), truncated }
}

function renderCaptured(output: CapturedOutput, maxOutputChars: number): string {
  const rendered = maybeTruncate(output.text, maxOutputChars, output.incomplete)
  const withPrefix = output.incomplete && output.text.length > 0
    ? LOST_PREFIX_MESSAGE + rendered
    : rendered
  const marker = output.exitCode !== undefined && output.exitCode !== 0
    ? `[exit code: ${output.exitCode}]`
    : undefined
  return appendStatusMarker(withPrefix, marker)
}

function appendStatusMarker(content: string, marker: string | undefined): string {
  if (marker === undefined) return content
  return content.length === 0 ? marker : `${content}\n${marker}`
}

function renderShellExitStatus(
  content: string,
  exitCode: number | null,
  signal: NodeJS.Signals | null,
): string {
  const marker = signal !== null
    ? `[shell killed by signal: ${signal}]`
    : exitCode !== null
      ? `[shell exited: code ${exitCode}]`
      : '[shell exited]'
  return appendStatusMarker(content, marker)
}

/**
 * Render the exited-session result, reset the owner's shell, and reset the
 * message that tells the model the next call starts fresh.
 * @param shells - the owner-scoped registry to reset.
 * @param status - the exited session status (exit code and signal).
 * @returns the complete model-facing result.
 */
async function respondToSessionExit(
  ctx: Context,
  shells: PersistentShells,
  owner: Agent,
  id: TerminalSessionId,
  status: { exitCode: number | null; signal: NodeJS.Signals | null },
  marker: CommandMarkers,
  wrapped: string,
  fallback: string,
  fallbackTruncated: boolean,
  config: ResolvedConfig,
): Promise<string> {
  const snapshot = retainedScrollback(ctx, owner, id)
  await shells.reset(owner, 'persistent pwsh shell exited')
  return [
    renderShellExitStatus(
      renderCaptured(partialOutput(snapshot, marker, wrapped, fallback, fallbackTruncated), config.maxOutputChars),
      status.exitCode,
      status.signal,
    ),
    SHELL_RESET_MESSAGE,
  ].filter(part => part.length > 0).join('\n')
}

/**
 * The pwsh prompt function that overrides the backend bootstrap value with
 * this tool's own prompt. `[char]27`/`[char]7` build the OSC bytes at runtime
 * because raw ESC characters in submitted input are unreliable under
 * PSReadLine.
 */
const PWSH_PROMPT_SETUP =
  "function prompt { [Console]::Write([char]27 + ']133;D;' + [int]$LASTEXITCODE + [char]7); '" + SHELL_PROMPT + "' }"

function persistentShells(ctx: Context, config: ResolvedConfig): PersistentShells {
  const pending = new WeakMap<Agent, Promise<TerminalSessionId>>()
  const live = new Map<Agent, TerminalSessionId>()
  const creating = new Set<Promise<TerminalSessionId>>()
  const ownerCleanupInstalled = new WeakSet<Agent>()
  const lifecycle = new AbortController()
  // Shells running a background command, no longer any owner's current shell.
  const detached = new Map<TerminalSessionId, Agent>()

  const close = async (owner: Agent, id: TerminalSessionId, reason: string): Promise<void> => {
    if (!ctx.terminals.list(owner).some(snapshot => snapshot.sessionId === id)) return
    await ctx.terminals.kill(owner, id, reason)
  }

  ctx.effect(() => async () => {
    lifecycle.abort(new Error('tool-pwsh-persistent disposed during shell creation'))
    await Promise.allSettled([...creating])
    const closing = [...live, ...[...detached].map(([id, owner]) => [owner, id] as const)]
      .map(async ([owner, id]) => { await close(owner, id, 'tool-pwsh-persistent disposed') })
    await Promise.allSettled(closing)
    live.clear()
    detached.clear()
  }, 'tool-pwsh-persistent shell cleanup')

  const detach = (owner: Agent): TerminalSessionId | undefined => {
    pending.delete(owner)
    const id = live.get(owner)
    live.delete(owner)
    if (id !== undefined) detached.set(id, owner)
    return id
  }

  const closeDetached = async (id: TerminalSessionId, reason: string): Promise<void> => {
    const owner = detached.get(id)
    if (owner === undefined) return
    detached.delete(id)
    await close(owner, id, reason)
  }

  const reset = async (owner: Agent, reason: string): Promise<void> => {
    pending.delete(owner)
    const id = live.get(owner)
    live.delete(owner)
    if (id !== undefined) await close(owner, id, reason)
  }

  const get = (owner: Agent, signal: AbortSignal): Promise<TerminalSessionId> => {
    const existing = pending.get(owner)
    if (existing !== undefined) return existing
    const combinedSignal = AbortSignal.any([signal, lifecycle.signal])
    const creation = (async () => {
      try {
        const cwd = owner.session.header.cwd
        const spawned = await ctx.terminals.spawn(owner, {
          type: config.backendType,
          ...cwd === undefined ? {} : { cwd },
        }, combinedSignal)
        live.set(owner, spawned.sessionId)
        if (!ownerCleanupInstalled.has(owner)) {
          ownerCleanupInstalled.add(owner)
          owner.ctx.effect(() => () => {
            pending.delete(owner)
            live.delete(owner)
          }, 'tool-pwsh-persistent owner cache cleanup')
        }
        const setup = ctx.terminals.startSend(owner, spawned.sessionId, {
          text: PWSH_PROMPT_SETUP,
          submit: true,
          signal: combinedSignal,
        })
        const result = await setup.done
        if (result.sessionStatus.kind === 'exited' || result.waitReason === 'timeout') {
          throw new Error('persistent pwsh shell did not accept initialization')
        }
        return spawned.sessionId
      } catch (error: unknown) {
        await reset(owner, 'persistent pwsh initialization failed')
        throw error
      }
    })()
    const tracked = creation.finally(() => {
      creating.delete(tracked)
    })
    creating.add(tracked)
    pending.set(owner, tracked)
    return tracked
  }

  return { get, reset, detach, closeDetached }
}

async function executeCommand(
  ctx: Context,
  shells: PersistentShells,
  owner: Agent,
  command: string,
  config: ResolvedConfig,
  upstream: AbortSignal,
): Promise<string> {
  using commandDeadline = deadline(upstream, config.timeoutMs, TIMEOUT_CODE)
  const id = await shells.get(owner, commandDeadline.signal)
  const marker = markers()
  const wrapped = wrapCommand(command, marker)
  let first = true
  let fallback = ''
  let fallbackTruncated = false

  while (true) {
    // The shell may flip to exited between iterations (a fast `exit` can
    // settle the previous send while its exit event is still in flight, and
    // the echoed wrapper can then carry a marker end without status digits);
    // re-observing status before the next send closes that gap.
    const status = ctx.terminals.list(owner).find(session => session.sessionId === id)?.status
    if (status?.kind === 'exited') {
      return await respondToSessionExit(
        ctx, shells, owner, id, status, marker, wrapped, fallback, fallbackTruncated, config,
      )
    }
    let operation
    let result
    try {
      operation = ctx.terminals.startSend(owner, id, {
        text: first ? wrapped : '',
        submit: first,
        signal: commandDeadline.signal,
      })
      first = false
      result = await operation.done
    } catch (error: unknown) {
      await shells.reset(owner, 'persistent pwsh send failed')
      throw error
    }
    const incremental = operation.readOutput()
    fallback = incremental.delta.length > 0 ? fallback + incremental.delta : result.viewport
    fallbackTruncated ||= incremental.truncated || result.truncated
    const latest = ctx.terminals.read(owner, id, { offset: 0, count: SCROLLBACK_PAGE_LINES })
    const timedOut = timeoutOf(commandDeadline.signal, TIMEOUT_CODE)
    if (timedOut !== undefined) {
      const snapshot = retainedScrollback(ctx, owner, id, latest)
      const partial = renderCaptured(
        partialOutput(snapshot, marker, wrapped, fallback, fallbackTruncated),
        config.maxOutputChars,
      )
      await shells.reset(owner, 'persistent pwsh command timed out')
      return [
        // TODO: Report a timeout only; this signal does not establish an OOM.
        `Your command timed out after ${Math.round(timedOut.timeoutMs / 1000)} seconds or experienced an OOM error. Below is partial output:`,
        partial,
        SHELL_RESET_MESSAGE,
      ].join('\n')
    }
    if (commandDeadline.signal.aborted) {
      await shells.reset(owner, 'persistent pwsh command aborted')
      commandDeadline.signal.throwIfAborted()
    }
    if (latest.text.includes(marker.end)) {
      const complete = commandOutput(retainedScrollback(ctx, owner, id, latest), marker, wrapped)
      if (complete !== undefined) return renderCaptured(complete, config.maxOutputChars)
    }
    if (result.sessionStatus.kind === 'exited') {
      return await respondToSessionExit(
        ctx, shells, owner, id, result.sessionStatus, marker, wrapped, fallback, fallbackTruncated, config,
      )
    }
    if (promptCompleted(result)) {
      const snapshot = retainedScrollback(ctx, owner, id, latest)
      return renderCaptured(
        partialOutput(snapshot, marker, wrapped, fallback, fallbackTruncated),
        config.maxOutputChars,
      )
    }
    await pause()
  }
}

/**
 * Fold one settled send into the command's state and report whether the
 * command is over: its end marker arrived (`complete`), the shell itself
 * exited, or the prompt came back without a marker (`prompt`).
 */
function absorbSend(
  ctx: Context,
  owner: Agent,
  run: CommandRun,
  operation: TerminalSendOperation,
  result: TerminalSendResult,
): Settled | undefined {
  const incremental = operation.readOutput()
  run.fallback = incremental.delta.length > 0 ? run.fallback + incremental.delta : result.viewport
  run.fallbackTruncated ||= incremental.truncated || result.truncated
  const latest = ctx.terminals.read(owner, run.id, { offset: 0, count: SCROLLBACK_PAGE_LINES })
  if (latest.text.includes(run.marker.end)) {
    const complete = commandOutput(retainedScrollback(ctx, owner, run.id, latest), run.marker, run.wrapped)
    if (complete !== undefined) return { kind: 'complete', output: complete }
  }
  if (result.sessionStatus.kind === 'exited') return { kind: 'exited', status: result.sessionStatus }
  if (promptCompleted(result)) return { kind: 'prompt' }
  return undefined
}

/** The command's output so far (from its start marker), from retained scrollback. */
function runOutputSoFar(ctx: Context, owner: Agent, run: CommandRun): CapturedOutput {
  return partialOutput(retainedScrollback(ctx, owner, run.id), run.marker, run.wrapped, run.fallback, run.fallbackTruncated)
}

/** Keep the tail of an over-long job read, marking the cut. */
function clipTail(text: string, maxOutputChars: number): string {
  if (text.length <= maxOutputChars) return text
  return `[${text.length - maxOutputChars} earlier characters omitted]\n${text.slice(-maxOutputChars)}`
}

/**
 * A command that outlived its foreground wait, still running in the shell it
 * started in. The shell is detached from the owner (their next call gets a
 * fresh one) and this watcher keeps polling it until the command's end marker
 * arrives, the shell exits, or the job is cancelled - then closes the shell.
 * Output reads are consuming: each returns only what the command printed
 * since the previous read (or since the promotion notice).
 */
class BackgroundCommand {
  private emitted = ''
  private final: CapturedOutput | undefined
  private cancelled = false

  constructor(
    private readonly ctx: Context,
    private readonly shells: PersistentShells,
    private readonly owner: Agent,
    private readonly run: CommandRun,
    private readonly config: ResolvedConfig,
  ) {}

  /** Consume the output produced since the previous read. */
  readNew(): string {
    let current: string
    if (this.final !== undefined) {
      current = this.final.text
    } else {
      try {
        current = runOutputSoFar(this.ctx, this.owner, this.run).text
      } catch {
        // The shell is already closed; nothing new can be read from it.
        current = this.emitted
      }
    }
    const max = Math.min(current.length, this.emitted.length)
    let common = 0
    while (common < max && current.charCodeAt(common) === this.emitted.charCodeAt(common)) common++
    let delta: string
    if (common === this.emitted.length) {
      delta = current.slice(common)
    } else {
      // The terminal re-rendered or dropped earlier lines: resend from the
      // first changed line, and say so only when the head itself is gone.
      const lineStart = current.lastIndexOf('\n', Math.max(0, common - 1)) + 1
      delta = (common < this.emitted.length / 2 ? LOST_HEAD_NOTICE : '') + current.slice(lineStart)
    }
    this.emitted = current
    return clipTail(delta, this.config.maxOutputChars)
  }

  /** The job hooks; starting them starts the watcher. */
  hooks(): JobHooks {
    return {
      cancel: () => {
        if (this.cancelled) return
        this.cancelled = true
        void this.shells.closeDetached(this.run.id, 'background pwsh command cancelled')
      },
      done: this.watch(),
      readOutput: () => this.readNew(),
    }
  }

  private async watch(): Promise<JobOutcome> {
    const { ctx, owner, run } = this
    try {
      while (!this.cancelled) {
        const status = ctx.terminals.list(owner).find(session => session.sessionId === run.id)?.status
        if (status === undefined) break
        if (status.kind === 'exited') {
          this.final = runOutputSoFar(ctx, owner, run)
          return { status: 'completed', detail: renderShellExitStatus('', status.exitCode, status.signal) }
        }
        const operation = run.operation ?? ctx.terminals.startSend(owner, run.id, { text: '', submit: false })
        run.operation = operation
        const result = await operation.done
        run.operation = undefined
        if (this.cancelled) break
        const settled = absorbSend(ctx, owner, run, operation, result)
        if (settled?.kind === 'complete') {
          this.final = settled.output
          return { status: 'completed', detail: `exit code: ${settled.output.exitCode ?? 0}` }
        }
        if (settled?.kind === 'exited') {
          this.final = runOutputSoFar(ctx, owner, run)
          return { status: 'completed', detail: renderShellExitStatus('', settled.status.exitCode, settled.status.signal) }
        }
        if (settled?.kind === 'prompt') {
          this.final = runOutputSoFar(ctx, owner, run)
          return { status: 'completed', detail: 'exit code unknown' }
        }
        await new Promise(resolve => setTimeout(resolve, BACKGROUND_POLL_MS))
      }
      return { status: 'killed', detail: 'cancelled' }
    } catch (error: unknown) {
      if (this.cancelled) return { status: 'killed', detail: 'cancelled' }
      return { status: 'failed', detail: `persistent pwsh shell failed: ${error instanceof Error ? error.message : String(error)}` }
    } finally {
      await this.shells.closeDetached(run.id, 'background pwsh command finished')
    }
  }
}

/**
 * Run one command with auto-background: wait up to `waitMs` in the
 * foreground; a command still running then keeps running in its own shell as
 * a background job, and the owner gets a fresh shell on their next call. The
 * wait never interrupts the command - it simply stops awaiting the send. The
 * call's own cancellation interrupts the command only BEFORE promotion.
 */
async function executeCommandAuto(
  ctx: Context,
  shells: PersistentShells,
  owner: Agent,
  command: string,
  config: ResolvedConfig,
  upstream: AbortSignal,
  waitMs: number,
  jobs: JobRegistry,
): Promise<string> {
  upstream.throwIfAborted()
  const foreground = new AbortController()
  const forward = (): void => { foreground.abort(upstream.reason) }
  upstream.addEventListener('abort', forward, { once: true })
  try {
    const id = await shells.get(owner, foreground.signal)
    const marker = markers()
    const run: CommandRun = {
      id, marker, wrapped: wrapCommand(command, marker), first: true, fallback: '', fallbackTruncated: false, operation: undefined,
    }
    const promoteAt = Date.now() + waitMs

    while (true) {
      const status = ctx.terminals.list(owner).find(session => session.sessionId === id)?.status
      if (status?.kind === 'exited') {
        return await respondToSessionExit(ctx, shells, owner, id, status, marker, run.wrapped, run.fallback, run.fallbackTruncated, config)
      }
      let operation = run.operation
      if (operation === undefined) {
        try {
          operation = ctx.terminals.startSend(owner, id, {
            text: run.first ? run.wrapped : '',
            submit: run.first,
            signal: foreground.signal,
          })
        } catch (error: unknown) {
          await shells.reset(owner, 'persistent pwsh send failed')
          throw error
        }
        run.first = false
        run.operation = operation
      }
      let timer: ReturnType<typeof setTimeout> | undefined
      let result: TerminalSendResult | 'promote'
      try {
        result = await Promise.race([
          operation.done,
          new Promise<'promote'>((resolve) => { timer = setTimeout(() => resolve('promote'), Math.max(0, promoteAt - Date.now())) }),
        ])
      } catch (error: unknown) {
        await shells.reset(owner, 'persistent pwsh send failed')
        throw error
      } finally {
        if (timer !== undefined) clearTimeout(timer)
      }

      if (result === 'promote') {
        // From here on the job owns the command: the call's cancellation no
        // longer reaches the send in flight.
        upstream.removeEventListener('abort', forward)
        shells.detach(owner)
        const background = new BackgroundCommand(ctx, shells, owner, run, config)
        const soFar = background.readNew()
        let jobId: string
        try {
          jobId = jobs.start({ kind: 'pwsh', label: command, owner, run: () => background.hooks() })
        } catch (error: unknown) {
          await shells.closeDetached(id, 'background registration refused')
          const reason = error instanceof Error ? error.message : String(error)
          throw new Error(`command still running after ${Math.round(waitMs / 1000)}s could not move to the background (${reason}); it was stopped and the shell was reset`)
        }
        return [
          `${PROMOTED_PREFIX}${Math.round(waitMs / 1000)}s - moved to background job ${jobId}; it keeps running in its own shell]`,
          soFar.length > 0 ? soFar.replace(/\n+$/, '') : '(no output yet)',
          `[you are notified in-session when job ${jobId} finishes; read new output with job_output, stop it with job_kill. Your next pwsh call starts a NEW shell from the workspace - the old shell's current directory and variables stay with the job]`,
        ].join('\n')
      }

      run.operation = undefined
      if (foreground.signal.aborted) {
        await shells.reset(owner, 'persistent pwsh command aborted')
        upstream.throwIfAborted()
        foreground.signal.throwIfAborted()
      }
      const settled = absorbSend(ctx, owner, run, operation, result)
      if (settled?.kind === 'complete') return renderCaptured(settled.output, config.maxOutputChars)
      if (settled?.kind === 'exited') {
        return await respondToSessionExit(
          ctx, shells, owner, id, settled.status, marker, run.wrapped, run.fallback, run.fallbackTruncated, config,
        )
      }
      if (settled?.kind === 'prompt') {
        return renderCaptured(runOutputSoFar(ctx, owner, run), config.maxOutputChars)
      }
      await pause()
    }
  } finally {
    upstream.removeEventListener('abort', forward)
  }
}

/**
 * Register the model-facing persistent `pwsh` tool.
 * @param ctx - plugin context carrying tools and the owner-scoped PTY service.
 * @param config - selected PTY backend and command deadline.
 */
function registerPersistentPwsh(ctx: Context, config: ResolvedConfig): void {
  const shells = persistentShells(ctx, config)
  const queues = new WeakMap<Agent, Promise<void>>()

  const serialized = async <T>(owner: Agent, operation: () => Promise<T>): Promise<T> => {
    const prior = queues.get(owner) ?? Promise.resolve()
    const run = prior.then(operation, operation)
    const tail = run.then(() => undefined, () => undefined)
    queues.set(owner, tail)
    try {
      return await run
    } finally {
      if (queues.get(owner) === tail) queues.delete(owner)
    }
  }

  const autoBackground = config.autoBackgroundAfterMs > 0
  const description = autoBackground
    ? `${config.description} A command still running after ${Math.round(config.autoBackgroundAfterMs / 1000)}s (or after \`timeoutMs\`, at most ${Math.round(config.maxForegroundMs / 1000)}s) is NOT killed: it keeps running as a background job in its own shell, the call returns its job id and the output so far, you are notified when it finishes, and your next call gets a fresh shell.`
    : config.description

  ctx.tools.register(defineTool({
    name: 'pwsh',
    description,
    parameters: {
      command: {
        type: 'string',
        required: true,
        description: 'The PowerShell command to run. Relative path is preferred in the command.',
      },
      ...autoBackground ? {
        timeoutMs: {
          type: 'number' as const,
          description: `How long to wait in the foreground, in milliseconds (default ${config.autoBackgroundAfterMs}, max ${config.maxForegroundMs}). A command still running then continues as a background job instead of being killed.`,
        },
      } : {},
    },
    output: {
      schema: { type: 'string' },
      render: (_args, value) => [{ type: 'text', text: value }],
    },
    async execute(args: { command: string; timeoutMs?: number }, exec) {
      if (args.command.trim().length === 0) throw new Error('command must be a non-empty string')
      if (args.timeoutMs !== undefined && (!Number.isFinite(args.timeoutMs) || args.timeoutMs <= 0)) {
        throw new Error(`invalid timeoutMs: expected a positive number, got ${JSON.stringify(args.timeoutMs)}`)
      }
      const owner = exec.agent
      if (owner === undefined) throw new Error('pwsh requires an owning agent session')
      return serialized(owner, async () => {
        exec.signal.throwIfAborted()
        const jobs = autoBackground ? ctx.get('jobs') : undefined
        if (jobs !== undefined) {
          const waitMs = Math.min(args.timeoutMs ?? config.autoBackgroundAfterMs, config.maxForegroundMs)
          return executeCommandAuto(ctx, shells, owner, args.command, config, exec.signal, waitMs, jobs)
        }
        return executeCommand(ctx, shells, owner, args.command, config, exec.signal)
      })
    },
    presentCall: args => ({ card: 'terminal', title: args.command }),
    presentResult: (_args, result) => {
      const block = result.content.length === 1 ? result.content[0] : undefined
      if (block === undefined || block.type !== 'text' || !block.text.startsWith(PROMOTED_PREFIX)) return undefined
      return { card: 'generic', content: [{ type: 'text', text: `\`\`\`console\n${block.text}\n\`\`\`` }] }
    },
  }))
}

export const name = 'tool-pwsh-persistent'
export const inject = ['tools', 'terminals']

/** Configuration for the persistent pwsh tool. */
export interface Config {
  /** PTY backend used for each owner-isolated persistent shell (default `shell`). */
  backendType?: string
  /** Wall-clock limit for one command (default 300000). */
  timeoutMs?: number
  /** Maximum returned command-output characters before clipping (default 16000). */
  maxOutputChars?: number
  /** Model-facing tool description; deployments may describe their environment. */
  description?: string
  /**
   * Auto-background: a command still running after this many milliseconds
   * keeps running as a background job in its own shell (the owner's next call
   * gets a fresh shell) instead of being killed at `timeoutMs`. `0` (default)
   * keeps the kill-and-reset behaviour. Needs a job runtime (`ctx.jobs` plus
   * `@deepseek-ai/dsh-tool-jobs` in the composition); without one calls run
   * as before.
   */
  autoBackgroundAfterMs?: number
  /** Upper bound on the foreground wait a call's own `timeoutMs` may request (default 600000). */
  maxForegroundMs?: number
}

/** Runtime configuration schema for the persistent pwsh tool. */
export const Config: z<Config> = z.object({
  backendType: z.string().default('shell'),
  timeoutMs: z.number().default(300_000),
  maxOutputChars: z.number().default(16_000),
  description: z.string().default(DEFAULT_DESCRIPTION),
  autoBackgroundAfterMs: z.number().default(0),
  maxForegroundMs: z.number().default(600_000),
})

/** Register one owner-scoped persistent `pwsh` tool. */
export function apply(ctx: Context, config: Config): void {
  const resolved: ResolvedConfig = {
    backendType: config.backendType ?? 'shell',
    timeoutMs: config.timeoutMs ?? 300_000,
    maxOutputChars: config.maxOutputChars ?? 16_000,
    description: config.description ?? DEFAULT_DESCRIPTION,
    autoBackgroundAfterMs: config.autoBackgroundAfterMs ?? 0,
    maxForegroundMs: config.maxForegroundMs ?? 600_000,
  }
  if (!Number.isSafeInteger(resolved.autoBackgroundAfterMs) || resolved.autoBackgroundAfterMs < 0) {
    throw new Error('tool-pwsh-persistent: autoBackgroundAfterMs must be a non-negative safe integer')
  }
  if (!Number.isSafeInteger(resolved.maxForegroundMs) || resolved.maxForegroundMs <= 0) {
    throw new Error('tool-pwsh-persistent: maxForegroundMs must be a positive safe integer')
  }
  if (resolved.backendType.trim().length === 0) {
    throw new Error('tool-pwsh-persistent: backendType must be non-empty')
  }
  if (!Number.isSafeInteger(resolved.timeoutMs) || resolved.timeoutMs <= 0) {
    throw new Error('tool-pwsh-persistent: timeoutMs must be a positive safe integer')
  }
  if (!Number.isSafeInteger(resolved.maxOutputChars) || resolved.maxOutputChars <= 0) {
    throw new Error('tool-pwsh-persistent: maxOutputChars must be a positive safe integer')
  }
  if (resolved.description.trim().length === 0) {
    throw new Error('tool-pwsh-persistent: description must be non-empty')
  }
  registerPersistentPwsh(ctx, resolved)
}

/* jscpd:ignore-end */
