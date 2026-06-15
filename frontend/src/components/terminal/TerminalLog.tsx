import { useEffect, useRef } from 'react'
import type { SSEMessage } from '../../types'

function ts() {
  return new Date().toTimeString().slice(0, 8)
}

interface TerminalLogProps {
  messages: SSEMessage[]
  height?: number
}

export default function TerminalLog({ messages, height = 200 }: TerminalLogProps) {
  const bottomRef = useRef<HTMLDivElement>(null)

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'auto' })
  }, [messages])

  if (messages.length === 0) return null

  return (
    <div
      className="w-full bg-black border border-border overflow-y-auto text-2xs leading-relaxed p-2"
      style={{ height, fontVariantNumeric: 'tabular-nums' }}
      aria-live="polite"
      aria-label="Analysis log"
    >
      {messages.map((m, i) => (
        <div key={i} className={m.error ? 'text-down' : 'text-up'}>
          <span className="text-dim select-none">[{ts()}] </span>
          {m.error ? `ERROR: ${m.error}` : m.message}
        </div>
      ))}
      <div ref={bottomRef} />
    </div>
  )
}
