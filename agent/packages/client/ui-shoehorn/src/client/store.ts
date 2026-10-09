/**
 * Live state of the local shoehorn server, polled from its /stats endpoint once a second while anything is
 * subscribed, plus whether the dashboard overlay is open. Plain external stores for useSyncExternalStore.
 */

/** Where shoehorn serves its stats and dashboard: port 8090 on the machine that served this page (also over the LAN). */
export const SHOEHORN_URL = `http://${typeof location !== 'undefined' && location.hostname ? location.hostname : '127.0.0.1'}:8090`

/** What the chip and the footer button show. */
export interface ShoehornState {
  /** The server answered the last poll. */
  up: boolean
  /** A model is loaded (false: unloaded after the idle timeout, or not started yet). */
  loaded: boolean
  /** A request is running. */
  busy: boolean
  /** Writing (true) or still reading the prompt (false), while busy. */
  writing: boolean
  /** Tokens written by the running request. */
  gen: number
  /** Prompt tokens of the running request. */
  prompt: number
  /** Writing speed over the last 2 s (never over less than 1 s), tokens per second. */
  rate: number | null
  /** Share of the last finished request's prompt that came from cache, percent. */
  cached: number | null
  /** Writing speed of the last finished request, tokens per second. */
  lastRate: number | null
}

interface Recent {
  t: number
  prompt_tokens: number
  reused_tokens: number
  gen_tokens: number
  decode_s: number
}

const initial: ShoehornState = {
  up: false, loaded: false, busy: false, writing: false, gen: 0, prompt: 0, rate: null, cached: null, lastRate: null,
}

let state = initial
const listeners = new Set<() => void>()
let timer: ReturnType<typeof setInterval> | undefined
let newest = Date.now() / 1000 - 3600  // only the last hour of finished requests on the first poll
let history: Array<{ t: number; g: number }> = []
let prevGen = -1

function emit(next: ShoehornState): void {
  state = next
  for (const l of listeners) l()
}

async function poll(): Promise<void> {
  try {
    const res = await fetch(`${SHOEHORN_URL}/stats?since=${newest}`, { cache: 'no-store' })
    const d = await res.json() as { live?: { busy?: boolean; gen?: number; prompt?: number }; loaded?: string; recent?: Recent[] }
    const now = Date.now() / 1000
    let { cached, lastRate } = state
    for (const r of d.recent ?? []) {
      if (r.t <= newest) continue
      newest = r.t
      if (r.gen_tokens <= 2 && r.prompt_tokens < 300) continue  // the start script's warm-up
      cached = r.prompt_tokens > 0 ? 100 * r.reused_tokens / r.prompt_tokens : null
      lastRate = r.gen_tokens >= 16 && r.decode_s > 0 ? r.gen_tokens / r.decode_s : lastRate
    }
    const busy = !!d.live?.busy, gen = d.live?.gen ?? 0, writing = busy && gen > 0
    let rate: number | null = null
    if (writing) {
      // checks land in bursts ~270 ms apart: count from 0 just before the first burst, over >= 1 s
      if (gen < prevGen || history.length === 0) history = [{ t: now - 0.25, g: 0 }]
      history.push({ t: now, g: gen })
      while (history.length > 2 && now - (history[1]?.t ?? now) >= 2) history.shift()
      const first = history[0] ?? { t: now, g: 0 }
      rate = (gen - first.g) / Math.max(1, now - first.t)
      prevGen = gen
    } else {
      history = []
      prevGen = -1
    }
    emit({ up: true, loaded: !!d.loaded, busy, writing, gen, prompt: d.live?.prompt ?? 0, rate, cached, lastRate })
  } catch {
    if (state.up) emit({ ...state, up: false, busy: false, writing: false, rate: null })
  }
}

/** Subscribe to the server state; polling runs only while someone listens. */
export function subscribe(listener: () => void): () => void {
  listeners.add(listener)
  if (!timer) {
    void poll()
    timer = setInterval(() => void poll(), 1000)
  }
  return () => {
    listeners.delete(listener)
    if (listeners.size === 0 && timer) {
      clearInterval(timer)
      timer = undefined
    }
  }
}

/** Current server state. */
export const getState = (): ShoehornState => state

// ---- the dashboard overlay
let open = false
const openListeners = new Set<() => void>()

/** Open or close the dashboard overlay. */
export function setOpen(value: boolean): void {
  if (open === value) return
  open = value
  for (const l of openListeners) l()
}

/** Subscribe to the overlay's open state. */
export function subscribeOpen(listener: () => void): () => void {
  openListeners.add(listener)
  return () => { openListeners.delete(listener) }
}

/** Whether the dashboard overlay is open. */
export const getOpen = (): boolean => open
