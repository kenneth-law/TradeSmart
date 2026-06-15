import type { StockResult } from '../../types'

const CONFIG: Record<StockResult['day_trading_strategy'], { label: string; className: string }> = {
  'Strong Buy':     { label: 'STRONG BUY',  className: 'text-up font-medium' },
  'Buy':            { label: 'BUY',          className: 'text-up' },
  'Neutral/Watch':  { label: 'WATCH',        className: 'text-warn' },
  'Sell':           { label: 'SELL',         className: 'text-down' },
  'Strong Sell':    { label: 'STRONG SELL',  className: 'text-down font-medium' },
}

export default function SignalBadge({ signal }: { signal: StockResult['day_trading_strategy'] }) {
  const { label, className } = CONFIG[signal] ?? { label: signal, className: 'text-muted' }
  return (
    <span className={`text-2xs tabnum ${className}`} aria-label={`Signal: ${signal}`}>
      {label}
    </span>
  )
}
