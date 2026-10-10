import { useEffect, useState, useSyncExternalStore } from 'react'
import styles from './shoehorn.module.css'
import { getOpen, getState, setOpen, SHOEHORN_URL, subscribe, subscribeOpen, type ShoehornState } from './store.ts'

/** Class names of shoehorn.module.css, never undefined. */
const css = {
  backdrop: styles['backdrop'] ?? '',
  bar: styles['bar'] ?? '',
  barLink: styles['barLink'] ?? '',
  barSpacer: styles['barSpacer'] ?? '',
  barTitle: styles['barTitle'] ?? '',
  chip: styles['chip'] ?? '',
  chipDim: styles['chipDim'] ?? '',
  chipText: styles['chipText'] ?? '',
  close: styles['close'] ?? '',
  dot: styles['dot'] ?? '',
  footer: styles['footer'] ?? '',
  footerDot: styles['footerDot'] ?? '',
  footerIcon: styles['footerIcon'] ?? '',
  footerWide: styles['footerWide'] ?? '',
  frame: styles['frame'] ?? '',
  idle: styles['idle'] ?? '',
  mark: styles['mark'] ?? '',
  markHole: styles['markHole'] ?? '',
  off: styles['off'] ?? '',
  panel: styles['panel'] ?? '',
  reading: styles['reading'] ?? '',
  tab: styles['tab'] ?? '',
  tabOn: styles['tabOn'] ?? '',
  tabs: styles['tabs'] ?? '',
  writing: styles['writing'] ?? '',
}

const useShoehorn = (): ShoehornState => useSyncExternalStore(subscribe, getState)
const useOpen = (): boolean => useSyncExternalStore(subscribeOpen, getOpen)

const kTokens = (n: number): string => n >= 10000 ? `${Math.round(n / 1000)}K` : n >= 1000 ? `${(n / 1000).toFixed(1)}K` : String(n)

/** The brass shoehorn mark. */
function Mark({ size = 14 }: { size?: number }) {
  return (
    <svg width={size} height={size} viewBox="0 0 20 20" aria-hidden="true" className={css.mark}>
      <path d="M8.2 2h3.6c.7 0 1.2.5 1.2 1.2v3.4c0 1.6 2.1 3.8 2.1 7.1 0 2.8-2.5 4.3-5.1 4.3s-5.1-1.5-5.1-4.3c0-3.3 2.1-5.5 2.1-7.1V3.2C7 2.5 7.5 2 8.2 2z" />
      <circle cx="10" cy="4.6" r="1" className={css.markHole} />
    </svg>
  )
}

function describe(s: ShoehornState): { tone: string; text: string; title: string } {
  if (!s.up) return { tone: css.off, text: 'shoehorn offline', title: 'The local server is not answering on port 8090.' }
  if (s.writing) {
    return { tone: css.writing, text: `${s.rate != null ? Math.round(s.rate) : '...'} tok/s`, title: `Writing: ${s.gen} tokens so far` }
  }
  if (s.busy) return { tone: css.reading, text: `reading ${kTokens(s.prompt)}`, title: `Reading a ${s.prompt}-token prompt` }
  if (!s.loaded) return { tone: css.idle, text: 'model unloaded', title: 'Unloaded after 30 min idle; the next message loads it (~40 s).' }
  const last = s.lastRate != null ? `${Math.round(s.lastRate)} tok/s` : 'idle'
  return { tone: css.idle, text: last, title: 'Idle. Last answer speed shown.' }
}

/** Session header utility: live state and speed, opens the dashboard. */
export function ShoehornChip() {
  const s = useShoehorn()
  const d = describe(s)
  return (
    <button type="button" className={`${css.chip} ${d.tone}`} title={`${d.title}. Click to open the shoehorn dashboard.`} onClick={() => setOpen(true)}>
      <span className={css.dot} />
      <span className={css.chipText}>{d.text}</span>
      {s.up && s.cached != null && !s.busy && <span className={css.chipDim}>{`${Math.round(s.cached)}% cached`}</span>}
    </button>
  )
}

/** Sidebar footer action: opens the dashboard. */
export function ShoehornFooterButton({ wide }: { wide?: boolean }) {
  const s = useShoehorn()
  return (
    <button type="button" className={`${css.footer} ${wide ? css.footerWide : ''}`} title="shoehorn dashboard" onClick={() => setOpen(true)}>
      <span className={css.footerIcon}>
        <Mark size={16} />
        <span className={`${css.footerDot} ${describe(s).tone}`} />
      </span>
      {wide && <span>shoehorn</span>}
    </button>
  )
}

/** shoehorn's pages: the model dashboard, the hardware sensors and model management. */
const PAGES = [{ path: '/dashboard', label: 'Dashboard' }, { path: '/sensors', label: 'Sensors' }, { path: '/models', label: 'Models' }] as const

/** Frame-wide overlay with the shoehorn dashboard and sensors pages, opened from the chip or the sidebar. */
export function ShoehornOverlay() {
  const open = useOpen()
  const [page, setPage] = useState<string>(PAGES[0].path)
  // both pages load with the app and stay loaded, only hidden while the popup is closed or the other tab is shown,
  // so they keep polling and their charts and history are there when opened
  useEffect(() => {
    if (!open) return
    const onKey = (e: KeyboardEvent) => { if (e.key === 'Escape') setOpen(false) }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [open])
  return (
    <div className={css.backdrop} style={open ? undefined : { display: 'none' }} onClick={() => setOpen(false)}>
      <div className={css.panel} role="dialog" aria-label="shoehorn dashboard" onClick={e => e.stopPropagation()}>
        <div className={css.bar}>
          <span className={css.barTitle}><Mark size={15} /> shoehorn</span>
          <span className={css.tabs} role="tablist">
            {PAGES.map(p => (
              <button key={p.path} type="button" role="tab" aria-selected={page === p.path}
                className={page === p.path ? `${css.tab} ${css.tabOn}` : css.tab} onClick={() => { setPage(p.path) }}>
                {p.label}
              </button>
            ))}
          </span>
          <span className={css.barSpacer} />
          <a className={css.barLink} href={`${SHOEHORN_URL}${page}`} target="_blank" rel="noreferrer">Open in a tab</a>
          <button type="button" className={css.close} aria-label="Close" onClick={() => setOpen(false)}>x</button>
        </div>
        {PAGES.map(p => (
          <iframe key={p.path} className={css.frame} src={`${SHOEHORN_URL}${p.path}`} title={`shoehorn ${p.label.toLowerCase()}`}
            style={p.path === page ? undefined : { display: 'none' }} />
        ))}
      </div>
    </div>
  )
}
