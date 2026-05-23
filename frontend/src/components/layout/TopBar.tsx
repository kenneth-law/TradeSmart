import { Link, NavLink } from 'react-router-dom'
import { useAppStore } from '../../store/useAppStore'
import logoImage from '../../assets/TradeSmartLogowhite.png'

const NAV = [
  { to: '/',           label: 'HOME',       hint: '1' },
  { to: '/market',     label: 'MARKETS',    hint: '2' },
  { to: '/system',     label: 'QUANT',      hint: '3' },
  { to: '/technical',  label: 'TECHNICAL',  hint: '4' },
  { to: '/daily-lineup', label: 'LINEUP',    hint: '5' },
  { to: '/portfolio',  label: 'PORTFOLIO',  hint: '6' },
  { to: '/docs',       label: 'DOCS',       hint: '7' },
  { to: '/settings',   label: 'SETTINGS',   hint: '8' },
]

export default function TopBar({ onCommandPalette }: { onCommandPalette: () => void }) {
  const context = useAppStore(s => s.tickerContext)

  return (
    <header
      className="app-chrome fixed top-0 left-0 right-0 z-50 flex min-w-0 items-center overflow-x-auto overflow-y-hidden border-b"
      style={{ height: 'var(--topbar-height)' }}
    >
      {/* Brand */}
      <Link
        to="/"
        className="app-chrome-border-r app-chrome-hover flex h-full w-40 shrink-0 items-center border-r px-4"
        aria-label="TradeSmart home"
      >
        <img
          src={logoImage}
          alt="TradeSmart"
          className="brand-logo h-auto max-h-6 w-28 object-contain"
        />
      </Link>

      {/* Ticker context strip */}
      {context && (
        <div className="app-chrome-border-r flex h-full shrink-0 items-center gap-3 border-r px-3 text-2xs tabnum">
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
      <nav className="ml-auto flex h-full min-w-max items-center" role="navigation" aria-label="Main navigation">
        {NAV.map(({ to, label, hint }) => (
          <NavLink
            key={to}
            to={to}
            end={to === '/'}
            title={`Alt+${hint}`}
            className={({ isActive }) =>
              [
                'flex items-center h-full px-3 text-2xs tracking-widest border-r border-white/[0.08]',
                'app-chrome-border-r app-chrome-hover hover:text-text transition-colors',
                isActive
                  ? 'text-accent bg-accent/10 border-b-2 border-b-accent'
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
          title="Ctrl+K - command palette"
          aria-label="Open command palette"
        >
          Cmd+K
        </button>
      </nav>
    </header>
  )
}
