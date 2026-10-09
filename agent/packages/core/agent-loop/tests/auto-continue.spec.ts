import { describe, expect, it } from 'vitest'
import { announcesAction } from '../src/agent.ts'

describe('announcesAction', () => {
  it('catches replies that stop right before the action they announce', () => {
    expect(announcesAction('Now update the actual logic — total_value skips qty 0, and low_stock excludes qty 0:')).toBe(true)
    expect(announcesAction('Let me do these edits. I already edited the docstring. Next I\'ll edit the sum line.')).toBe(true)
    expect(announcesAction('The tests failed on precedence.\n\nI\'ll fix the parser now.')).toBe(true)
    expect(announcesAction('Now the first chunk of calc.py:')).toBe(true)
  })

  it('leaves final answers and questions alone', () => {
    expect(announcesAction('All tests pass. Files created: calc.py, test_calc.py.')).toBe(false)
    expect(announcesAction('Should I also add exact mode now?')).toBe(false)
    expect(announcesAction('')).toBe(false)
    expect(announcesAction('first reply')).toBe(false)
    expect(announcesAction('**Summary:**')).toBe(false)
  })
})
