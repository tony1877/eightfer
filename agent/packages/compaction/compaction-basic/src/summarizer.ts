/**
 * Default one-shot summarization and durable checkpoint framing.
 *
 * @module @deepseek-ai/dsh-compaction-basic/summarizer
 */

import type { Context } from '@deepseek-ai/cordis'
import { contentHasImage, createUserMessage, BlockAssembler, LlmError } from '@deepseek-ai/dsh-llm'
import type {
  ContentBlock, FinishReason, GenerateOptions, Message, TokenUsage, ToolSchema,
} from '@deepseek-ai/dsh-llm'
import type { Agent } from '@deepseek-ai/dsh-agent'
import { appendFileSync, writeFileSync } from 'node:fs'
import { homedir } from 'node:os'
import { join } from 'node:path'
import { condenseToolTraffic } from './condense.ts'
import type { SummaryTemplate } from './types.ts'

/**
 * Live mirror of the summarization stream. Compaction otherwise shows nothing
 * between `compaction/start` and `compaction/summary`, which for a large
 * conversation is minutes of apparent silence. Rewritten at the start of each
 * compaction; open or tail it to watch the summary being written.
 */
const LIVE_PATH = join(process.env.DSH_HOME?.trim() || join(homedir(), '.dsh'), 'compaction-live.md')

/** Observability only: a failed write must never fail the compaction. */
function live(write: () => void): void {
  try { write() } catch { /* ignore */ }
}

interface SummaryConfig {
  readonly summarizationProvider: string
  readonly summarizationModel: string
  readonly maxTokens: number
  /** Summarize from condensed tool traffic (see `condense.ts`). */
  readonly condenseToolOutput?: boolean
  /** Which checkpoint instruction to append. */
  readonly summaryTemplate?: SummaryTemplate
}

/** Tags wrapping the structured summary inside the landed checkpoint node. */
const SUMMARY_OPEN_TAG = '<compacted-summary>'
const SUMMARY_CLOSE_TAG = '</compacted-summary>'

/**
 * The summarization directive, delivered as the FINAL user message after the
 * replayed conversation rather than as a distinct summarizer system prompt.
 * Keeping the conversation's own system prompt, tools, and message prefix in
 * front of it makes the auxiliary call a genuine prefix of the last routed
 * request, so the provider's KV cache is reused instead of invalidated.
 */
const COMPACTION_INSTRUCTION = [
  'You are now acting as a compaction engine for this AI coding assistant. Condense the conversation ABOVE into a structured checkpoint that lets another model resume the work with no loss of essential context.',
  '',
  'Output EXACTLY the Markdown structure below: keep every section, in order. Use terse bullets, not prose paragraphs. Write "(none)" for an empty section — never drop a section.',
  '',
  '## Primary Request and Intent',
  "- [the user's original and evolving goals; quote verbatim where the exact wording matters]",
  '',
  '## Key Technical Concepts',
  '- [technologies, frameworks, patterns, and conventions in play]',
  '',
  '## Files and Code',
  '- [exact path: why it matters, key changes or snippets]',
  '',
  '## Errors and Fixes',
  '- [error: how it was resolved, plus any related user feedback]',
  '',
  '## Pending Jobs',
  '- [explicitly requested work not yet completed]',
  '',
  '## Current Work',
  '- [precisely what was in progress at this checkpoint]',
  '',
  '## Next Step',
  '- [the single next action, directly in line with the most recent request, or "(none)"]',
  '',
  '## Critical Context',
  '- [decisions and their rationale, constraints, user preferences, open questions, data needed to continue]',
  '',
  'Rules:',
  '- Write concise English engineering prose. Preserve exact file paths, commands, error strings, identifiers, numeric values, function signatures, and syntax fragments.',
  '- Capture user feedback and explicit instructions faithfully, especially corrections.',
  '- Do NOT mention this summarization request or that the context was compacted.',
  '- Output only the checkpoint text: do not call any tool or take any other action.',
  `- If the conversation already contains a ${SUMMARY_OPEN_TAG} block, it is a PRIOR checkpoint. Do not copy it forward verbatim: preserve still-true facts, drop stale ones, and merge newer information into a single consolidated summary under the same structure.`,
].join('\n')

/**
 * The `detailed` template: the same role as {@link COMPACTION_INSTRUCTION}, with
 * the structure spelled out for smaller models. It asks for the user's rules
 * verbatim and every user message, which smaller models otherwise drop first,
 * plus dead ends, commands and config values, and says outright that length
 * is fine.
 */
const DETAILED_COMPACTION_INSTRUCTION = [
  'STOP working on the task. Your only job now is to write a checkpoint of the conversation ABOVE. Another model will continue the work from this checkpoint alone: anything you leave out is lost forever. Be complete rather than brief; long is fine.',
  '',
  'Scan the WHOLE conversation from the very first message, not just the recent part. Then output EXACTLY the Markdown structure below, every section in order. Write "(none)" for an empty section; never drop one.',
  '',
  '## 1. User rules and corrections',
  '- Every standing rule, restriction, preference and correction the user gave, quoted VERBATIM in quotation marks. These matter most: missing one means repeating a mistake the user already corrected.',
  '',
  '## 2. Goal',
  '- The original request (ticket ids, deadlines, scope) and how it evolved.',
  '',
  '## 3. All user messages',
  '- Every user message in order, one bullet each, condensed to one or two lines; quote the exact words where they carry an instruction, number or decision.',
  '',
  '## 4. Decisions and why',
  '- Each technical decision with its reason and the numbers behind it (sizes, ratios, limits).',
  '',
  '## 5. Dead ends',
  '- Approaches that were tried and abandoned, and exactly why, so they are not retried.',
  '',
  '## 6. Errors and fixes',
  '- The exact error string, its cause, and the exact fix (setting, command or code change). Mark any error that is still unresolved as OPEN.',
  '',
  '## 7. Files',
  '- `exact/path`: created / edited / read, and what changed or why it matters (function names, settings, values).',
  '',
  '## 8. Commands',
  '- Exact commands that worked and should be reused (build, test, run), and any command the user said not to use.',
  '',
  '## 9. Environment and config',
  '- URLs, ports, hosts, environment variables, feature flags, config keys and their values.',
  '',
  '## 10. Current state',
  '- What was in progress at the very end, what is done, what is failing right now (test names, error strings).',
  '',
  '## 11. Pending work',
  '- Everything still to do, in order, including anything the user asked to be told or shown first.',
  '',
  '## 12. Next step',
  "- The single next action, and whether it needs the user's go-ahead first.",
  '',
  'Rules:',
  '- Copy exact values: paths, commands, identifiers, error strings, numbers, URLs, flag names. Never paraphrase a value.',
  `- If the conversation already contains a ${SUMMARY_OPEN_TAG} block, it is a PRIOR checkpoint: carry every still-true item forward into these sections (rules and user messages always carry forward), drop what became false, and merge in the newer information.`,
  '- Tool outputs above may be shown condensed ("[tool output condensed ...]"); summarize what they show, do not invent the omitted lines.',
  '- This request is not part of the conversation: do not list it as a user message or rule.',
  '- Do not mention this request or that the context is being compacted. Output only the checkpoint; do not call any tool.',
].join('\n')

/** Framing that makes the replacement user message established context. */
const CHECKPOINT_PREAMBLE =
  'This is an automatically generated checkpoint condensing an earlier span of the conversation to free up context. Treat the captured context as established background and build on it without restating it. Continue the task directly from the messages that follow, without acknowledging this checkpoint.'

/**
 * The replayed conversation surface the summarizer condenses. Reproducing the
 * last routed request's system prompt, tools, and leading messages verbatim
 * lets the auxiliary call reuse the provider's warm prefix cache; the trailing
 * compaction instruction is then the only novel input.
 */
export interface SummarizationInput {
  /** The conversation's own system prompt, reused for prefix-cache alignment; absent for a system-less request. */
  readonly system?: string
  /** The conversation's tool schemas, reused for prefix-cache alignment; absent when the request carried none. */
  readonly tools?: readonly ToolSchema[]
  /** The shadowed region, in surface order, that precedes the compaction instruction. */
  readonly messages: readonly Message[]
}

/** Safe summary content plus the exact auxiliary call envelope recorded with it. */
export type SummaryResult = {
  summary: ContentBlock[]
  provider: string
  model: string
  maxTokens?: number
  /** Provider-reported usage for this summarization request. */
  usage?: TokenUsage
} & (
  | {
    /** Complete provider output before the text-only summary projection. */
    rawOutput: ContentBlock[]
    /** Identifies exactly one call through this context's `ctx.llm.stream()`. */
    llmStreamCall: true
  }
  | {
    /** Optional complete output from an unmarked template, remote, or other summarizer. */
    rawOutput?: ContentBlock[]
    /** An unmarked result does not identify a call through this context's LLM seam. */
    llmStreamCall?: never
  }
)

/**
 * Run the default cache-reusing `ctx.llm.stream()` summarization call: replay
 * the conversation prefix, then append the compaction instruction as the final
 * user message so the provider's warm prefix cache is reused.
 * @param ctx - context providing the LLM service.
 * @param config - resolved backend configuration.
 * @param input - replayed conversation prefix (system, tools, and leading messages) to condense.
 * @param agent - supplies routed-model history, fallback model, and session id.
 * @param signal - optional cancellation forwarded to the adapter.
 * @returns safe text-only summary blocks and the exact call envelope and output.
 */
export async function summarizeWithLlm(
  ctx: Context,
  config: SummaryConfig,
  input: SummarizationInput,
  agent: Agent,
  signal?: AbortSignal,
): Promise<SummaryResult> {
  const latest = agent.session.requestHeader()?.config
  const configured = config.summarizationProvider.length === 0
    ? undefined
    : { provider: config.summarizationProvider, model: config.summarizationModel }
  const agentTarget = agent.options.provider !== undefined
    && agent.options.provider.length > 0
    && agent.options.model !== undefined
    && agent.options.model.length > 0
    ? { provider: agent.options.provider, model: agent.options.model }
    : undefined
  const target = configured ?? latest ?? agentTarget
  if (target === undefined) {
    throw new Error(
      'no provider/model available for summarization: set both BasicCompactionConfig summarization fields, route one request, or set both AgentOptions fields',
    )
  }

  const assembler = new BlockAssembler()
  const instruction = config.summaryTemplate === 'detailed'
    ? DETAILED_COMPACTION_INSTRUCTION
    : COMPACTION_INSTRUCTION
  const messages: Message[] = [
    ...config.condenseToolOutput === true ? condenseToolTraffic(input.messages) : input.messages,
    createUserMessage({
      content: [{ type: 'text', text: instruction }],
      source: { kind: 'plugin', plugin: 'dsh-compaction-basic' },
    }),
  ]
  const options: GenerateOptions = {
    provider: target.provider,
    model: target.model,
    messages,
    ...input.system === undefined ? {} : { system: input.system },
    ...input.tools === undefined ? {} : { tools: [...input.tools] },
    maxTokens: config.maxTokens,
    sessionId: agent.session.id,
    purpose: 'compaction',
    ...signal === undefined ? {} : { signal },
  }
  const started = Date.now()
  live(() => writeFileSync(LIVE_PATH,
    `# Compaction ${new Date(started).toISOString()}\n\n`
    + `- session: ${agent.session.id}\n`
    + `- route: ${target.provider}/${target.model}\n`
    + `- summarizing ${input.messages.length} messages, cap ${config.maxTokens} tokens`
    + `, template ${config.summaryTemplate ?? 'standard'}`
    + `${config.condenseToolOutput === true ? ', tool output condensed' : ''}\n\n---\n`))
  let lastKind: string | undefined
  for await (const chunk of ctx.llm.stream(options)) {
    assembler.push(chunk)
    if (chunk.type === 'text-delta' || chunk.type === 'reasoning-delta') {
      const kind = chunk.type
      const text = chunk.text
      live(() => {
        if (kind !== lastKind) {
          appendFileSync(LIVE_PATH, kind === 'reasoning-delta' ? '\n\n[reasoning]\n' : '\n\n[summary]\n')
          lastKind = kind
        }
        appendFileSync(LIVE_PATH, text)
      })
    }
  }
  live(() => appendFileSync(LIVE_PATH,
    `\n\n---\nfinished ${new Date().toISOString()} after ${Math.round((Date.now() - started) / 1000)}s`
    + ` - finish: ${assembler.finish.kind}\n`))
  const error = finishError(assembler.finish)
  if (error !== undefined) throw error

  const rawOutput = assembler.blocks()
  const summary = summaryText(rawOutput)
  if (!summary.some(block => block.text.trim().length > 0)) {
    throw new Error('summarization produced no text summary content')
  }
  return {
    summary,
    rawOutput,
    llmStreamCall: true,
    provider: options.provider,
    model: options.model,
    maxTokens: config.maxTokens,
    ...(assembler.usage === undefined ? {} : { usage: assembler.usage }),
  }
}

/**
 * Wrap raw summary blocks in the durable checkpoint framing.
 * @param summary - safe text-only model output.
 * @returns content for the synthesized replacement user message.
 */
export function frameSummary(summary: readonly ContentBlock[]): ContentBlock[] {
  return [
    { type: 'text', text: `${CHECKPOINT_PREAMBLE}\n\n${SUMMARY_OPEN_TAG}` },
    ...summary,
    { type: 'text', text: SUMMARY_CLOSE_TAG },
  ]
}

/** Map a terminal summarization finish to its fail-closed error. */
function finishError(finish: FinishReason): Error | undefined {
  switch (finish.kind) {
    case 'error':
    case 'aborted': {
      const error = new Error(finish.failure.message) as Error & { code?: string }
      error.code = finish.failure.code
      return error
    }
    case 'max-tokens': {
      const error = new Error('summarization truncated at the token cap (incomplete checkpoint)') as Error & { code?: string }
      error.code = 'MAX_TOKENS'
      return error
    }
    default:
      return undefined
  }
}

/** Reject visual output and keep only text before synthesizing a user message. */
function summaryText(
  blocks: readonly ContentBlock[],
): Array<Extract<ContentBlock, { type: 'text' }>> {
  if (contentHasImage(blocks)) {
    throw new LlmError('compaction summary cannot contain image output', 'UNSUPPORTED_CONTENT')
  }
  return blocks.filter((block): block is Extract<ContentBlock, { type: 'text' }> => block.type === 'text')
}
