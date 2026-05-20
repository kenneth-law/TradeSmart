import { NavLink } from 'react-router-dom'
import { useAppStore } from '../../store/useAppStore'

const NAV = [
  { to: '/',           label: 'HOME',       hint: '1' },
  { to: '/technical',  label: 'TECHNICAL',  hint: '2' },
  { to: '/system',     label: 'SYSTEM',     hint: '3' },
  { to: '/market',     label: 'MARKET',     hint: '4' },
  { to: '/portfolio',  label: 'PORTFOLIO',  hint: '5' },
  { to: '/docs',       label: 'DOCS',       hint: '6' },
  { to: '/settings',   label: 'SETTINGS',   hint: '7' },
]

export default function TopBar({ onCommandPalette }: { onCommandPalette: () => void }) {
  const context = useAppStore(s => s.tickerContext)

  return (
    <header
      className="fixed top-0 left-0 right-0 z-50 flex items-center border-b border-border bg-s1"
      style={{ height: 'var(--topbar-height)' }}
    >
      {/* Brand */}
      <div className="flex items-center gap-3 px-3 border-r border-border h-full shrink-0">
        <span className="text-2xs text-accent font-medium tracking-widest">TRADESMART</span>
      </div>

      {/* Ticker context strip */}
      {context && (
        <div className="flex items-center gap-3 px-3 border-r border-border h-full text-2xs tabnum shrink-0">
          <span className="text-text font-medium">{context.ticker}</span>
          <span className="text-text">{context.price.toFixed(2)}</span>
          <span className={context.change >= 0 ? 'text-up' : 'text-down'}>
            {context.change >= 0 ? '+' : ''}{context.change.toFixed(2)}%
          </span>
          <span className="text-dim">score</span>
          <span className="text-accent">{context.score.toFixed(1)}</span>
        </div>
      )}

      {/* Nav */}
      <nav className="flex items-center h-full ml-auto" role="navigation" aria-label="Main navigation">
        {NAV.map(({ to, label, hint }) => (
          <NavLink
            key={to}
            to={to}
            end={to === '/'}
            title={`Alt+${hint}`}
            className={({ isActive }) =>
              [
                'flex items-center h-full px-3 text-2xs tracking-widest border-r border-border',
                'hover:text-text transition-colors',
                isActive
                  ? 'text-accent border-b-2 border-b-accent'
                  : 'text-dim',
              ].join(' ')
            }
          >
            {label}
          </NavLink>
        ))}

        {/* Command palette trigger */}
        <button
          onClick={onCommandPalette}
          className="flex items-center h-full px-3 text-dim text-2xs hover:text-text"
          title="Ctrl+K — command palette"
          aria-label="Open command palette"
        >
          ⌘K
        </button>
      </nav>
    </header>
  )
}
