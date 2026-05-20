import { useEffect, useRef, useState, useCallback, useLayoutEffect } from 'react'
import { createChart, CandlestickSeries, LineSeries, HistogramSeries } from 'lightweight-charts'
import type { IChartApi, Time } from 'lightweight-charts'
import { useQuery, useQueries } from '@tanstack/react-query'
import { api } from '../../lib/api'
import type { PriceHistory } from '../../types'

type ChartType = 'candlestick' | 'line'

const PERIODS: { days: number; label: string }[] = [
  { days: 30,  label: '1M' },
  { days: 90,  label: '3M' },
  { days: 180, label: '6M' },
  { days: 365, label: '1Y' },
]

const OVERLAY_COLORS = ['#A78BFA', '#14B8A6', '#F472B6', '#FACC15', '#94A3B8', '#3B82F6', '#E89B2C']

const CHART_H = 340

interface Props {
  defaultTicker?: string
  overlayPeers?: string[]
}

function isFiniteNumber(value: unknown): value is number {
  return typeof value === 'number' && Number.isFinite(value)
}

export default function TickerSearchChart({ defaultTicker = 'SPY', overlayPeers = [] }: Props) {
  const [inputValue,        setInputValue]        = useState(defaultTicker)
  const [activeTicker,      setActiveTicker]      = useState(defaultTicker)
  const [chartType,         setChartType]         = useState<ChartType>('candlestick')
  const [days,              setDays]              = useState(90)
  const [selectedOverlays,  setSelectedOverlays]  = useState<string[]>([])
  const [overlayInput,      setOverlayInput]      = useState('')

  const containerRef = useRef<HTMLDivElement>(null)
  const chartRef     = useRef<IChartApi | null>(null)

  const { data, isLoading, isError } = useQuery<PriceHistory>({
    queryKey:  ['tickerChart', activeTicker, days],
    queryFn:   () => api.getPriceHistory(activeTicker, days),
    staleTime: 60_000,
    retry: 1,
  })

  const overlayQueries = useQueries({
    queries: selectedOverlays.map(t => ({
      queryKey:  ['tickerOverlay', t, days],
      queryFn:   () => api.getPriceHistory(t, days),
      staleTime: 60_000,
      retry: 1,
    })),
  })

  const overlayQueriesRef = useRef(overlayQueries)
  useLayoutEffect(() => { overlayQueriesRef.current = overlayQueries })

  const overlayDataKey = selectedOverlays
    .map((t, i) => `${t}:${overlayQueries[i]?.data?.dates?.length ?? 0}`)
    .join('|')

  const load = useCallback(() => {
    const t = inputValue.trim().toUpperCase()
    if (t) setActiveTicker(t)
  }, [inputValue])

  function addOverlay(raw: string) {
    const t = raw.trim().toUpperCase()
    if (!t || selectedOverlays.includes(t)) return
    setSelectedOverlays(o => [...o, t])
    setOverlayInput('')
  }

  function toggleOverlay(t: string) {
    setSelectedOverlays(o => o.includes(t) ? o.filter(x => x !== t) : [...o, t])
  }

  useEffect(() => {
    const el = containerRef.current
    if (chartRef.current) { chartRef.current.remove(); chartRef.current = null }
    if (!el || !data?.dates?.length) return

    const hasOverlays = selectedOverlays.length > 0

    const chart = createChart(el, {
      autoSize: true,
      height:   CHART_H,
      layout: { background: { color: '#0A0A0B' }, textColor: '#8A8A93' },
      grid:   { vertLines: { color: '#1A1A1F' }, horzLines: { color: '#1A1A1F' } },
      crosshair: {
        vertLine: { color: '#3A3A40', labelBackgroundColor: '#16161A' },
        horzLine: { color: '#3A3A40', labelBackgroundColor: '#16161A' },
      },
      rightPriceScale: { borderColor: '#1F1F23' },
      leftPriceScale:  { borderColor: '#1F1F23', visible: hasOverlays },
      timeScale:       { borderColor: '#1F1F23', timeVisible: true },
    })
    chartRef.current = chart

    const times = data.dates as Time[]
    const vols  = data.volumes ?? data.volume ?? []

    if (vols.length) {
      const volSeries = chart.addSeries(HistogramSeries, {
        color: '#2A4A6A',
        priceFormat:  { type: 'volume' },
        priceScaleId: 'volume',
      })
      chart.priceScale('volume').applyOptions({ scaleMargins: { top: 0.76, bottom: 0 }, visible: false })
      volSeries.setData(
        times
          .map((t, i) => ({
            time:  t,
            value: vols[i],
            color: i > 0 && (data.close[i] ?? 0) >= (data.close[i - 1] ?? 0) ? '#1F4A35' : '#4A1F24',
          }))
          .filter(r => isFiniteNumber(r.value))
      )
    }

    chart.priceScale('right').applyOptions({
      scaleMargins: { top: 0.04, bottom: vols.length ? 0.26 : 0.04 },
    })

    if (chartType === 'candlestick') {
      chart.addSeries(CandlestickSeries, {
        upColor: '#1FBF75', downColor: '#E5484D',
        borderUpColor: '#1FBF75', borderDownColor: '#E5484D',
        wickUpColor: '#1FBF75', wickDownColor: '#E5484D',
      }).setData(
        times
          .map((t, i) => ({ time: t, open: data.open?.[i], high: data.high?.[i], low: data.low?.[i], close: data.close?.[i] }))
          .filter(r => isFiniteNumber(r.open) && isFiniteNumber(r.high) && isFiniteNumber(r.low) && isFiniteNumber(r.close))
      )
    } else {
      chart.addSeries(LineSeries, { color: '#E89B2C', lineWidth: 1 })
        .setData(times.map((t, i) => ({ time: t, value: data.close?.[i] })).filter(r => isFiniteNumber(r.value) && (r.value as number) > 0))
    }

    if (data.ma5?.length) {
      chart.addSeries(LineSeries, { color: '#C47A0A', lineWidth: 1, title: 'MA5' })
        .setData(times.map((t, i) => ({ time: t, value: data.ma5![i] })).filter(r => isFiniteNumber(r.value) && (r.value as number) > 0))
    }
    if (data.ma20?.length) {
      chart.addSeries(LineSeries, { color: '#3B82F6', lineWidth: 1, title: 'MA20' })
        .setData(times.map((t, i) => ({ time: t, value: data.ma20![i] })).filter(r => isFiniteNumber(r.value) && (r.value as number) > 0))
    }

    const currentOverlayQueries = overlayQueriesRef.current
    selectedOverlays.forEach((overlayTicker, i) => {
      const od = currentOverlayQueries[i]?.data
      if (!od?.dates?.length) return
      const color = OVERLAY_COLORS[i % OVERLAY_COLORS.length]
      chart.addSeries(LineSeries, {
        color,
        lineWidth: 1,
        priceScaleId: 'left',
        priceFormat: { type: 'price', precision: 2, minMove: 0.01 },
        title: overlayTicker,
      }).setData(
        (od.dates as Time[])
          .map((t, j) => ({ time: t, value: od.close?.[j] }))
          .filter(r => isFiniteNumber(r.value) && (r.value as number) > 0)
      )
    })

    chart.timeScale().fitContent()

    return () => { chart.remove(); chartRef.current = null }
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [data, chartType, selectedOverlays, overlayDataKey])

  const stats            = data?.stats
  const lastClose        = data?.close?.at(-1) ?? 0
  const prevClose        = data?.close?.at(-2) ?? lastClose
  const last             = stats?.last ?? lastClose
  const isUp             = last >= prevClose
  const overlayLoading   = overlayQueries.some(q => q.isLoading || q.isFetching)
  const manualOverlays   = selectedOverlays.filter(t => !overlayPeers.includes(t))

  return (
    <div className="border border-border bg-bg">
      {/* Row 1: ticker input + type + period + stats */}
      <div className="flex flex-wrap items-center gap-x-2 gap-y-1 px-3 py-1.5 border-b border-border bg-s1">
        <div className="flex items-center gap-1 border border-border-strong bg-bg px-2 py-0.5">
          <span className="text-muted text-xs select-none">{'>'}</span>
          <input
            value={inputValue}
            onChange={e => setInputValue(e.target.value.toUpperCase())}
            onKeyDown={e => e.key === 'Enter' && load()}
            placeholder="TICKER"
            className="w-20 bg-transparent text-text text-xs outline-none placeholder:text-dim tabnum"
            spellCheck={false}
          />
        </div>
        <button onClick={load} className="text-2xs text-bg bg-accent px-2 py-0.5 hover:opacity-90 font-medium">
          GO
        </button>

        <span className="text-border">|</span>

        {(['candlestick', 'line'] as ChartType[]).map(t => (
          <button
            key={t}
            onClick={() => setChartType(t)}
            className={`text-2xs px-2 py-0.5 border transition-colors ${chartType === t ? 'border-accent text-accent' : 'border-border text-muted hover:text-text'}`}
          >
            {t === 'candlestick' ? 'CANDLE' : 'LINE'}
          </button>
        ))}

        <span className="text-border">|</span>

        {PERIODS.map(p => (
          <button
            key={p.days}
            onClick={() => setDays(p.days)}
            className={`text-2xs px-2 py-0.5 border transition-colors ${days === p.days ? 'border-accent text-accent' : 'border-border text-muted hover:text-text'}`}
          >
            {p.label}
          </button>
        ))}

        <div className="ml-auto flex items-center gap-3">
          {isLoading && <span className="text-2xs text-dim animate-pulse">Loading...</span>}
          {isError   && <span className="text-2xs text-down">Error: {activeTicker}</span>}
          {stats && !isLoading && (
            <>
              <span className={`text-sm tabnum font-medium ${isUp ? 'text-up' : 'text-down'}`}>{last.toFixed(2)}</span>
              <span className="text-2xs text-muted tabnum">H&nbsp;{stats.high.toFixed(2)}</span>
              <span className="text-2xs text-muted tabnum">L&nbsp;{stats.low.toFixed(2)}</span>
              <span className="text-2xs text-dim   tabnum">AVG&nbsp;{stats.avg.toFixed(2)}</span>
            </>
          )}
        </div>
      </div>

      {/* Row 2: compare / overlay */}
      <div className="flex flex-wrap items-center gap-x-2 gap-y-1 px-3 py-1.5 border-b border-border bg-s1">
        <span className="text-2xs text-dim">Compare</span>

        {overlayPeers.map((peer, i) => {
          const selected  = selectedOverlays.includes(peer)
          const colorIdx  = selectedOverlays.indexOf(peer)
          const color     = colorIdx >= 0 ? OVERLAY_COLORS[colorIdx % OVERLAY_COLORS.length] : OVERLAY_COLORS[i % OVERLAY_COLORS.length]
          return (
            <button
              key={peer}
              onClick={() => toggleOverlay(peer)}
              className={['text-2xs tabnum border px-2 py-0.5 transition-colors', selected ? 'border-border-strong bg-s2' : 'border-border text-muted hover:text-text'].join(' ')}
              style={selected ? { color, borderColor: color } : undefined}
              aria-pressed={selected}
            >
              {peer}
            </button>
          )
        })}

        {manualOverlays.map(t => {
          const colorIdx = selectedOverlays.indexOf(t)
          const color    = OVERLAY_COLORS[colorIdx % OVERLAY_COLORS.length]
          return (
            <button
              key={t}
              onClick={() => toggleOverlay(t)}
              className="text-2xs tabnum border px-2 py-0.5 transition-colors"
              style={{ color, borderColor: color }}
              title="Click to remove"
            >
              {t} ×
            </button>
          )
        })}

        <div className="flex items-center gap-1">
          <div className="flex items-center border border-border-strong bg-bg px-1.5 py-0.5">
            <input
              value={overlayInput}
              onChange={e => setOverlayInput(e.target.value.toUpperCase())}
              onKeyDown={e => e.key === 'Enter' && addOverlay(overlayInput)}
              placeholder="+TICKER"
              className="w-16 bg-transparent text-2xs text-text outline-none placeholder:text-dim tabnum"
              spellCheck={false}
            />
          </div>
          <button
            onClick={() => addOverlay(overlayInput)}
            disabled={!overlayInput.trim()}
            className="text-2xs text-bg bg-accent/80 px-1.5 py-0.5 hover:opacity-90 disabled:opacity-40"
          >
            ADD
          </button>
        </div>

        {overlayLoading && <span className="text-2xs text-dim animate-pulse">loading...</span>}
      </div>

      <div ref={containerRef} className="relative" style={{ height: CHART_H }}>
        {!isLoading && !data?.dates?.length && (
          <div className="absolute inset-0 flex items-center justify-center text-sm text-dim">
            {isError ? `No chart data for ${activeTicker}` : 'No chart data'}
          </div>
        )}
      </div>
    </div>
  )
}
