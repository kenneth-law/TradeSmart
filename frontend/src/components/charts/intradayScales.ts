import type { PriceHistory } from '../../types'

export type IntradayScale = 5 | 15 | 30 | 60 | 300

export const INTRADAY_SCALES: Array<{ seconds: IntradayScale; label: string }> = [
  { seconds: 5, label: '5s' },
  { seconds: 15, label: '15s' },
  { seconds: 30, label: '30s' },
  { seconds: 60, label: '1m' },
  { seconds: 300, label: '5m' },
]

export function floorToScale(second: number, scale: IntradayScale) {
  return Math.floor(second / scale) * scale
}

export function systemTimeZone() {
  return Intl.DateTimeFormat().resolvedOptions().timeZone
}

export function formatMarketTime(value: unknown, scale: IntradayScale = 60) {
  const seconds = typeof value === 'number' ? value : null
  if (seconds == null || !Number.isFinite(seconds)) return String(value ?? '')
  return new Intl.DateTimeFormat('en-US', {
    timeZone: systemTimeZone(),
    hour: '2-digit',
    minute: '2-digit',
    second: scale < 60 ? '2-digit' : undefined,
    hour12: false,
  }).format(new Date(seconds * 1000))
}

export function formatChartTime(value: unknown, intraday: boolean, scale: IntradayScale = 60) {
  if (intraday) return formatMarketTime(value, scale)
  if (typeof value === 'string') return value
  if (typeof value === 'number') return new Date(value * 1000).toISOString().slice(0, 10)
  if (value && typeof value === 'object' && 'year' in value && 'month' in value && 'day' in value) {
    const day = value as { year: number; month: number; day: number }
    return `${day.year}-${String(day.month).padStart(2, '0')}-${String(day.day).padStart(2, '0')}`
  }
  return String(value ?? '')
}

function finite(value: unknown): value is number {
  return typeof value === 'number' && Number.isFinite(value)
}

export function aggregateIntradayHistory(data: PriceHistory | undefined, scale: IntradayScale): PriceHistory | undefined {
  if (!data || scale <= 60 || !data.dates.every(d => typeof d === 'number')) return data

  const buckets = new Map<number, {
    open: number
    high: number
    low: number
    close: number
    volume: number
  }>()
  const volumes = data.volumes ?? data.volume ?? []

  data.dates.forEach((rawTime, i) => {
    const time = rawTime as number
    const open = data.open?.[i]
    const high = data.high?.[i]
    const low = data.low?.[i]
    const close = data.close?.[i]
    if (!finite(open) || !finite(high) || !finite(low) || !finite(close)) return
    const bucketTime = floorToScale(time, scale)
    const existing = buckets.get(bucketTime)
    if (existing) {
      existing.high = Math.max(existing.high, high)
      existing.low = Math.min(existing.low, low)
      existing.close = close
      existing.volume += finite(volumes[i]) ? volumes[i] : 0
    } else {
      buckets.set(bucketTime, {
        open,
        high,
        low,
        close,
        volume: finite(volumes[i]) ? volumes[i] : 0,
      })
    }
  })

  const dates = Array.from(buckets.keys()).sort((a, b) => a - b)
  const rows = dates.map(time => buckets.get(time)!)
  const close = rows.map(row => row.close)
  return {
    ...data,
    dates,
    open: rows.map(row => row.open),
    high: rows.map(row => row.high),
    low: rows.map(row => row.low),
    close,
    prices: close,
    volumes: rows.map(row => row.volume),
    ma5: undefined,
    ma20: undefined,
    interval: `${scale}s`,
    stats: close.length ? {
      last: close[close.length - 1],
      high: Math.max(...rows.map(row => row.high)),
      low: Math.min(...rows.map(row => row.low)),
      avg: close.reduce((sum, value) => sum + value, 0) / close.length,
    } : data.stats,
  }
}
