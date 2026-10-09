/**
 * shoehorn in the Web client: a live status chip in every session header, a sidebar footer button, and the
 * shoehorn dashboard as a frame-wide overlay. Data comes straight from the local server's /stats.
 */
import type { Context as ClientContext } from '@deepseek-ai/cordis'
import type {} from '@deepseek-ai/dsh-client-ui-conversation/client'
import type {} from '@deepseek-ai/dsh-client-ui-layout/client'
import type {} from '@deepseek-ai/dsh-client-ui-renderer/client'
import type {} from '@deepseek-ai/dsh-client-ui-sidebar/client'
import { ShoehornChip, ShoehornFooterButton, ShoehornOverlay } from './Shoehorn.tsx'

/** Required service: the UI slot registry. */
export const inject = ['slots']

/**
 * Register the three entries, each when its declaring owner is mounted.
 * @param ctx - Client root context.
 */
export function apply(ctx: ClientContext): void {
  ctx.slots.inject('conversation.session.header.utilities', () => ctx.slots.register(
    { name: 'conversation.session.header.utilities', id: 'shoehorn-status', order: -10, label: 'shoehorn' },
    ShoehornChip,
  ))
  ctx.slots.inject('sidebar.footer.action', () => ctx.slots.register(
    { name: 'sidebar.footer.action', id: 'shoehorn', order: 50, label: 'shoehorn' },
    ShoehornFooterButton,
  ))
  ctx.slots.inject('shell.overlay', () => ctx.slots.register(
    { name: 'shell.overlay', id: 'shoehorn-dashboard', order: 100, label: 'shoehorn dashboard' },
    ShoehornOverlay,
  ))
}
