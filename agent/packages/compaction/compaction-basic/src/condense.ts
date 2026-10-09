/**
 * Summarizer-only condensing of tool traffic.
 *
 * Tool output (file reads, build and test logs) is most of a long agent
 * conversation, and little of it matters to a checkpoint. Condensing it before
 * summarization shrinks the prompt several-fold, which makes a small local
 * model both faster and more accurate at recalling the rest. The session is
 * never touched: only the replayed summarization input is rewritten.
 *
 * @module @deepseek-ai/dsh-compaction-basic/condense
 */

import type { ContentBlock, Message } from '@deepseek-ai/dsh-llm'

/** Lines worth keeping from the middle of a tool result. */
const SIGNAL = /error|fail|✗|exception|fatal|denied|not found|exit code|ECONN|timed? ?out|passed|\bOOM\b/i

/** Tool results at or under this many characters are kept verbatim. */
const RESULT_KEEP_CHARS = 700

/** Hard cap on one condensed tool result body. */
const RESULT_MAX_CHARS = 1400

/** String tool-call arguments longer than this are clipped. */
const ARG_MAX_CHARS = 240

/** Longest single line kept from a tool result. */
const LINE_MAX_CHARS = 200

const HEAD_LINES = 6
const TAIL_LINES = 3

/**
 * Condense one tool result's text: keep the first and last lines and every
 * signal line, marking each gap with its line count.
 * @param text - the tool result's text.
 * @returns the text itself when short, otherwise a labelled condensed form.
 */
export function condenseToolText(text: string): string {
  if (text.length <= RESULT_KEEP_CHARS) return text
  const lines = text.split('\n')
  const keep = new Set<number>()
  for (let i = 0; i < Math.min(HEAD_LINES, lines.length); i += 1) keep.add(i)
  for (let i = Math.max(0, lines.length - TAIL_LINES); i < lines.length; i += 1) keep.add(i)
  lines.forEach((line, i) => {
    if (SIGNAL.test(line)) keep.add(i)
  })
  const out: string[] = []
  let last = -1
  for (const i of [...keep].sort((a, b) => a - b)) {
    if (i > last + 1) out.push(`[… ${i - last - 1} lines]`)
    const line = lines[i] ?? ''
    out.push(line.length > LINE_MAX_CHARS ? `${line.slice(0, LINE_MAX_CHARS)}…` : line)
    last = i
  }
  let body = out.join('\n')
  // Too many signal lines: keep the start and more of the end, where a build
  // or test run's verdict usually is.
  if (body.length > RESULT_MAX_CHARS) {
    const head = Math.floor(RESULT_MAX_CHARS * 0.4)
    body = `${body.slice(0, head)}\n[…]\n${body.slice(-(RESULT_MAX_CHARS - head))}`
  }
  return `[tool output condensed for the summary: ${lines.length} lines, ${text.length} chars]\n${body}`
}

/**
 * Clip long string arguments of one tool call (file bodies, long scripts).
 * @param args - the raw JSON argument string.
 * @returns the arguments with long strings clipped, still valid JSON when the input was.
 */
export function condenseToolArguments(args: string): string {
  let parsed: unknown
  try {
    parsed = JSON.parse(args)
  } catch {
    return args.length > ARG_MAX_CHARS * 4 ? `${args.slice(0, ARG_MAX_CHARS * 4)}… [${args.length} chars]` : args
  }
  if (typeof parsed !== 'object' || parsed === null || Array.isArray(parsed)) return args
  const clipped: Record<string, unknown> = {}
  for (const [key, value] of Object.entries(parsed)) {
    clipped[key] = typeof value === 'string' && value.length > ARG_MAX_CHARS
      ? `${value.slice(0, ARG_MAX_CHARS)}… [${value.length} chars]`
      : value
  }
  return JSON.stringify(clipped)
}

/** Condense one content block; non-tool blocks pass through unchanged. */
function condenseBlock(block: ContentBlock): ContentBlock {
  switch (block.type) {
    case 'tool-call':
      return { ...block, arguments: condenseToolArguments(block.arguments) }
    case 'tool-result': {
      const text = block.content
        .map(inner => inner.type === 'text' ? inner.text : `[${inner.type}]`)
        .join('\n')
      return { ...block, content: [{ type: 'text', text: condenseToolText(text) }] }
    }
    default:
      return block
  }
}

/**
 * Rewrite tool calls and tool results in a replayed region for summarization.
 * Message identities, roles and order are kept, so call/result pairing holds.
 * @param messages - the replayed region, in surface order.
 * @returns new messages with condensed tool traffic.
 */
export function condenseToolTraffic(messages: readonly Message[]): Message[] {
  return messages.map(message => (
    message.content.some(block => block.type === 'tool-call' || block.type === 'tool-result')
      ? { ...message, content: message.content.map(condenseBlock) }
      : message
  ))
}
