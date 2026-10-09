import { describe, expect, it } from 'vitest'
import { createMessage, createToolResultMessage, createUserMessage, ToolCallId } from '@deepseek-ai/dsh-llm'
import {
  condenseToolArguments,
  condenseToolText,
  condenseToolTraffic,
} from '@deepseek-ai/dsh-compaction-basic/src/condense.ts'
import { resolveConfig } from '@deepseek-ai/dsh-compaction-basic/src/config.ts'

describe('condenseToolText', () => {
  it('keeps short output verbatim', () => {
    expect(condenseToolText('ok\nexit 0')).toBe('ok\nexit 0')
  })

  it('keeps head, tail and signal lines and counts the gaps', () => {
    const lines = Array.from({ length: 200 }, (_, i) => `line ${i}`)
    lines[100] = "TypeError: Cannot read properties of undefined (reading 'checksum')"
    const out = condenseToolText(lines.join('\n'))
    expect(out.split('\n')[0]).toBe('[tool output condensed for the summary: 200 lines, '
      + `${lines.join('\n').length} chars]`)
    expect(out).toContain('line 0\nline 1\nline 2\nline 3\nline 4\nline 5\n[… 94 lines]')
    expect(out).toContain("(reading 'checksum')\n[… 96 lines]\nline 197\nline 198\nline 199")
    expect(out).not.toContain('line 50')
  })

  it('caps a result with many signal lines, keeping more of the end', () => {
    const lines = Array.from({ length: 400 }, (_, i) => `test ${i} failed`)
    lines.push('Tests  400 failed')
    const out = condenseToolText(lines.join('\n'))
    expect(out.length).toBeLessThan(1600)
    expect(out).toContain('\n[…]\n')
    expect(out.endsWith('Tests  400 failed')).toBe(true)
  })
})

describe('condenseToolArguments', () => {
  it('clips long string arguments and keeps valid JSON', () => {
    const out = condenseToolArguments(JSON.stringify({ path: 'a.ts', file_text: 'y'.repeat(1000), n: 3 }))
    expect(JSON.parse(out)).toEqual({ path: 'a.ts', file_text: `${'y'.repeat(240)}… [1000 chars]`, n: 3 })
  })

  it('leaves short or non-object arguments alone', () => {
    expect(condenseToolArguments('{"command":"pnpm test"}')).toBe('{"command":"pnpm test"}')
    expect(condenseToolArguments('[1,2]')).toBe('[1,2]')
    expect(condenseToolArguments('not json')).toBe('not json')
  })
})

describe('condenseToolTraffic', () => {
  it('rewrites only tool blocks and keeps message identity and order', () => {
    const callId = ToolCallId('c1')
    const user = createUserMessage({ content: [{ type: 'text', text: 'x'.repeat(5000) }], source: { kind: 'user' } })
    const call = createMessage({
      role: 'assistant',
      content: [{ type: 'tool-call', id: callId, name: 'read', arguments: JSON.stringify({ path: 'p'.repeat(300) }) }],
      source: { kind: 'model', provider: 'p', model: 'm' },
    })
    const result = createToolResultMessage({ callId, content: [{ type: 'text', text: 'r\n'.repeat(1000) }], isError: true })
    const out = condenseToolTraffic([user, call, result])
    expect(out[0]).toBe(user)
    expect(out.map(m => m.id)).toEqual([user.id, call.id, result.id])
    const block = out[2]?.content[0]
    expect(block?.type).toBe('tool-result')
    if (block?.type !== 'tool-result') return
    expect(block.isError).toBe(true)
    expect(block.toolCallId).toBe(callId)
    expect(block.content).toHaveLength(1)
    expect(block.content[0]?.type === 'text' && block.content[0].text.length).toBeLessThan(200)
  })
})

describe('config', () => {
  it('defaults both options off / standard and validates them', () => {
    expect(resolveConfig({})).toMatchObject({ condenseToolOutput: false, summaryTemplate: 'standard' })
    expect(resolveConfig({ condenseToolOutput: true, summaryTemplate: 'detailed' }))
      .toMatchObject({ condenseToolOutput: true, summaryTemplate: 'detailed' })
    expect(() => resolveConfig({ summaryTemplate: 'long' as never })).toThrow(/summaryTemplate/)
    expect(() => resolveConfig({ condenseToolOutput: 'yes' as never })).toThrow(/condenseToolOutput/)
  })
})
