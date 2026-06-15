import { useState, useEffect, useRef } from 'react'
import { useNavigate } from 'react-router-dom'

const COMMANDS = [
  { label: 'Home',               shortcut: 'Alt+1', action: '/' },
  { label: 'Markets',            shortcut: 'Alt+2', action: '/market' },
  { label: 'Quant Trading',      shortcut: 'Alt+3', action: '/system' },
  { label: 'Technical Ranking',  shortcut: 'Alt+4', action: '/technical' },
  { label: 'Daily Lineup',       shortcut: 'Alt+5', action: '/daily-lineup' },
  { label: 'Portfolio',          shortcut: 'Alt+6', action: '/portfolio' },
  { label: 'Documentation',      shortcut: 'Alt+7', action: '/docs' },
  { label: 'Settings',           shortcut: 'Alt+8', action: '/settings' },
]

interface CommandPaletteProps {
  open: boolean
  onClose: () => void
}

export default function CommandPalette({ open, onClose }: CommandPaletteProps) {
  const [query, setQuery] = useState('')
  const [cursor, setCursor] = useState(0)
  const navigate = useNavigate()
  const inputRef = useRef<HTMLInputElement>(null)

  const filtered = COMMANDS.filter(c =>
    c.label.toLowerCase().includes(query.toLowerCase())
  )

  useEffect(() => {
    if (open) {
      setQuery('')
      setCursor(0)
      setTimeout(() => inputRef.current?.focus(), 0)
    }
  }, [open])

  function go(path: string) {
    navigate(path)
    onClose()
  }

  function onKey(e: React.KeyboardEvent) {
    if (e.key === 'Escape') { onClose(); return }
    if (e.key === 'ArrowDown') { e.preventDefault(); setCursor(c => Math.min(c + 1, filtered.length - 1)) }
    if (e.key === 'ArrowUp')   { e.preventDefault(); setCursor(c => Math.max(c - 1, 0)) }
    if (e.key === 'Enter' && filtered[cursor]) go(filtered[cursor].action)
  }

  if (!open) return null

  return (
    <>
      {/* Scrim */}
      <div
        className="fixed inset-0 z-40"
        style={{ background: 'rgba(0,0,0,0.5)' }}
        onClick={onClose}
        aria-hidden="true"
      />

      {/* Panel */}
      <div
        className="fixed left-1/2 top-24 z-50 w-full max-w-md -translate-x-1/2 bg-s1 border border-border-strong"
        role="dialog"
        aria-label="Command palette"
        aria-modal="true"
      >
        {/* Input */}
        <div className="flex items-center gap-2 px-3 py-2 border-b border-border">
          <span className="text-muted text-sm select-none">:</span>
          <input
            ref={inputRef}
            value={query}
            onChange={e => { setQuery(e.target.value); setCursor(0) }}
            onKeyDown={onKey}
            placeholder="command or navigation..."
            className="flex-1 bg-transparent text-text text-sm outline-none placeholder:text-dim"
            aria-autocomplete="list"
            aria-controls="cmd-list"
          />
          <kbd className="text-2xs text-dim">Esc</kbd>
        </div>

        {/* Results */}
        <ul id="cmd-list" role="listbox" className="max-h-64 overflow-y-auto">
          {filtered.length === 0 && (
            <li className="px-3 py-2 text-muted text-sm">No commands match.</li>
          )}
          {filtered.map((cmd, i) => (
            <li
              key={cmd.label}
              role="option"
              aria-selected={i === cursor}
              onClick={() => go(cmd.action)}
              onMouseEnter={() => setCursor(i)}
              className={[
                'flex justify-between items-center px-3 py-1.5 cursor-pointer text-sm',
                i === cursor ? 'bg-s2 text-text' : 'text-muted hover:bg-s2',
              ].join(' ')}
            >
              <span>{cmd.label}</span>
              <kbd className="text-2xs text-dim">{cmd.shortcut}</kbd>
            </li>
          ))}
        </ul>
      </div>
    </>
  )
}
