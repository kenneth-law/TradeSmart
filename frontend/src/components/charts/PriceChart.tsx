import { useState, useEffect, useMemo, useRef, useLayoutEffect } from 'react'
import { useQuery, useQueries } from '@tanstack/react-query'
import { createChart, CandlestickSeries, LineSeries, HistogramSeries } from 'lightweight-charts'
import type { IChartApi, Time } from 'lightweight-charts'
import { api } from '../../lib/api'
import type { PriceHistory } from '../../types'
import { quotePrice, useLiveQuotes } from '../../hooks/useLiveQuotes'
import { aggregateIntradayHistory, floorToScale, formatChartTime, INTRADAY_SCALES, type IntradayScale } from './intradayScales'

type ChartType = 'candlestick' | 'line'

const PERIODS = [
  { days: 1,   label: '1D' },
  { days: 30,  label: '1M' },
  { days: 90,  label: '3M' },
  { days: 180, label: '6M' },
  { days: 365, label: '1Y' },
]

const OVERLAY_COLORS = ['#E89B2C', '#3B82F6', '#A78BFA', '#14B8A6', '#F472B6', '#FACC15', '#94A3B8']

interface PriceChartProps {
  ticker: string
  overlayPeers?: string[]
}

function isFiniteNumber(v: unknown): v is number {
  return typeof v === 'number' && Number.isFinite(v)
}

export default function PriceChart({ ticker, overlayPeers = [] }: PriceChartProps) {
  const containerRef       = useRef<HTMLDivElement>(null)
  const chartRef           = useRef<IChartApi | null>(null)
  const candleSeriesRef    = useRef<any>(null)
  const volumeSeriesRef    = useRef<any>(null)
  const liveCandleRef      = useRef<{ time: number; open: number; high: number; low: number; close: number } | null>(null)
  const latestSeriesTimeRef = useRef<number | null>(null)
  const [chartType, setChartType]           = useState<ChartType>('candlestick')
  const [days, setDays]                     = useState(90)
  const [intradayScale, setIntradayScale]   = useState<IntradayScale>(5)
  const [selectedOverlays, setSelectedOverlays] = useState<string[]>([])
  const [overlayInput, setOverlayInput]     = useState('')
  const isIntraday = days === 1
  const live = useLiveQuotes([ticker], isIntraday && chartType === 'candlestick' && !!ticker)

  const { data, isLoading } = useQuery<PriceHistory>({
    queryKey:  ['priceChart', ticker, days],
    queryFn:   () => api.getPriceHistory(ticker, days),
    enabled:   !!ticker,
    staleTime: 60_000,
  })
  const chartData = useMemo(
    () => isIntraday ? aggregateIntradayHistory(data, intradayScale) : data,
    [data, intradayScale, isIntraday]
  )

  const overlayQueries = useQueries({
    queries: selectedOverlays.map(t => ({
      queryKey: ['priceOverlay', t, days],
      queryFn:  () => api.getPriceHistory(t, days),
      staleTime: 60_000,
      retry: 1,
    })),
  })

  // Keep a ref so the chart effect can read current overlay data without it being a dep
  const overlayQueriesRef = useRef(overlayQueries)
  useLayoutEffect(() => { overlayQueriesRef.current = overlayQueries })

  // Stable key: only changes when overlay data actually loads (date count changes)
  const overlayDataKey = selectedOverlays
    .map((t, i) => `${t}:${overlayQueries[i]?.data?.dates?.length ?? 0}`)
    .join('|')

  useEffect(() => {
    const el = containerRef.current
    if (chartRef.current) {
      chartRef.current.remove()
      chartRef.current = null
      candleSeriesRef.current = null
      volumeSeriesRef.current = null
      liveCandleRef.current = null
      latestSeriesTimeRef.current = null
    }
    if (!el || !chartData?.dates?.length) return

    const hasOverlays = selectedOverlays.length > 0

    const chart = createChart(el, {
      autoSize: true,
      layout: { background: { color: '#0A0A0B' }, textColor: '#8A8A93' },
      grid:   { vertLines: { color: '#1F1F23' }, horzLines: { color: '#1F1F23' } },
      crosshair: {
        vertLine: { color: '#2A2A30', labelBackgroundColor: '#16161A' },
        horzLine: { color: '#2A2A30', labelBackgroundColor: '#16161A' },
      },
      rightPriceScale: { borderColor: '#1F1F23' },
      leftPriceScale:  { borderColor: '#1F1F23', visible: hasOverlays },
      localization: {
        timeFormatter: time => formatChartTime(time, isIntraday, intradayScale),
      },
      timeScale: {
        borderColor: '#1F1F23',
        timeVisible: isIntraday,
        secondsVisible: isIntraday && intradayScale < 60,
        tickMarkFormatter: time => formatChartTime(time, isIntraday, intradayScale),
      },
    })
    chartRef.current = chart

    const times = chartData.dates as Time[]
    latestSeriesTimeRef.current = typeof chartData.dates.at(-1) === 'number' ? chartData.dates.at(-1) as number : null
    const vols  = chartData.volumes ?? chartData.volume ?? []

    // Volume histogram on bottom 25%
    if (vols.length) {
      const volSeries = chart.addSeries(HistogramSeries, {
        color: '#2A4A6A',
        priceFormat:  { type: 'volume' },
        priceScaleId: 'volume',
      })
      volumeSeriesRef.current = volSeries
      chart.priceScale('volume').applyOptions({ scaleMargins: { top: 0.76, bottom: 0 }, visible: false })
      volSeries.setData(
        times
          .map((t, i) => ({
            time:  t,
            value: vols[i],
            color: i > 0 && (chartData.close[i] ?? 0) >= (chartData.close[i - 1] ?? 0) ? '#1F4A35' : '#4A1F24',
          }))
          .filter(r => isFiniteNumber(r.value))
      )
    }

    chart.priceScale('right').applyOptions({
      scaleMargins: { top: 0.04, bottom: vols.length ? 0.26 : 0.04 },
    })

    // Main price series
    if (chartType === 'candlestick') {
      const cs = chart.addSeries(CandlestickSeries, {
        upColor:         '#1FBF75', downColor:       '#E5484D',
        borderUpColor:   '#1FBF75', borderDownColor: '#E5484D',
        wickUpColor:     '#1FBF75', wickDownColor:   '#E5484D',
      })
      candleSeriesRef.current = cs
      cs.setData(
        times
          .map((t, i) => ({ time: t, open: chartData.open?.[i], high: chartData.high?.[i], low: chartData.low?.[i], close: chartData.close?.[i] }))
          .filter(r => isFiniteNumber(r.open) && isFiniteNumber(r.high) && isFiniteNumber(r.low) && isFiniteNumber(r.close))
      )
    } else {
      chart.addSeries(LineSeries, { color: '#E89B2C', lineWidth: 1 })
        .setData(times.map((t, i) => ({ time: t, value: chartData.close?.[i] })).filter(r => isFiniteNumber(r.value) && (r.value as number) > 0))
    }

    // Moving averages
    if (chartData.ma5?.length) {
      chart.addSeries(LineSeries, { color: '#C47A0A', lineWidth: 1, title: 'MA5' })
        .setData(times.map((t, i) => ({ time: t, value: chartData.ma5![i] })).filter(r => isFiniteNumber(r.value) && (r.value as number) > 0))
    }
    if (chartData.ma20?.length) {
      chart.addSeries(LineSeries, { color: '#3B82F6', lineWidth: 1, title: 'MA20' })
        .setData(times.map((t, i) => ({ time: t, value: chartData.ma20![i] })).filter(r => isFiniteNumber(r.value) && (r.value as number) > 0))
    }

    // Overlays on secondary scale
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

    return () => {
      chart.remove()
      chartRef.current = null
      candleSeriesRef.current = null
      volumeSeriesRef.current = null
      liveCandleRef.current = null
      latestSeriesTimeRef.current = null
    }
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [chartData, chartType, selectedOverlays, overlayDataKey])

  useEffect(() => {
    liveCandleRef.current = null
  }, [ticker, days, chartData?.dates?.at(-1), intradayScale])

  useEffect(() => {
    if (!isIntraday || chartType !== 'candlestick' || !candleSeriesRef.current) return
    if (!chartData?.dates?.length || !chartData.dates.every(time => typeof time === 'number')) return
    const quote = live.quotes[ticker.toUpperCase()]
    const price = quotePrice(quote)
    if (price == null) return
    const parsedTime = quote?.timestamp ? Date.parse(quote.timestamp) : NaN
    const second = Number.isFinite(parsedTime) ? Math.floor(parsedTime / 1000) : Math.floor(Date.now() / 1000)
    const candleTime = floorToScale(second, intradayScale)
    const lastHistoricalTime = typeof chartData?.dates?.at(-1) === 'number' ? chartData.dates.at(-1) as number : 0
    const latestSeriesTime = latestSeriesTimeRef.current ?? lastHistoricalTime
    if (latestSeriesTime && candleTime < latestSeriesTime) return

    const previous = liveCandleRef.current
    const lastIndex = chartData.dates.length - 1
    const sameAsLoadedBar = !previous && candleTime === lastHistoricalTime
    const baseOpen = sameAsLoadedBar && isFiniteNumber(chartData.open?.[lastIndex]) ? chartData.open[lastIndex] : price
    const baseHigh = sameAsLoadedBar && isFiniteNumber(chartData.high?.[lastIndex]) ? chartData.high[lastIndex] : price
    const baseLow = sameAsLoadedBar && isFiniteNumber(chartData.low?.[lastIndex]) ? chartData.low[lastIndex] : price
    const candle = previous && previous.time === candleTime
      ? {
          ...previous,
          high: Math.max(previous.high, price),
          low: Math.min(previous.low, price),
          close: price,
        }
      : {
          time: candleTime,
          open: baseOpen,
          high: Math.max(baseHigh, price),
          low: Math.min(baseLow, price),
          close: price,
        }
    liveCandleRef.current = candle
    try {
      candleSeriesRef.current.update(candle)
      latestSeriesTimeRef.current = Math.max(latestSeriesTime ?? candleTime, candleTime)
    } catch {
      return
    }
    if (volumeSeriesRef.current) {
      volumeSeriesRef.current.update({
        time: candleTime,
        value: quote?.last_size ?? 0,
        color: candle.close >= candle.open ? '#1F4A35' : '#4A1F24',
      })
    }
  }, [chartData?.dates, chartType, intradayScale, isIntraday, live.quotes, ticker])

  function addOverlay(raw: string) {
    const t = raw.trim().toUpperCase()
    if (!t || selectedOverlays.includes(t)) return
    setSelectedOverlays(o => [...o, t])
    setOverlayInput('')
  }

  function toggleOverlay(t: string) {
    setSelectedOverlays(o => o.includes(t) ? o.filter(x => x !== t) : [...o, t])
  }

  const overlayLoading = overlayQueries.some(q => q.isLoading || q.isFetching)
  const manualOverlays = selectedOverlays.filter(t => !overlayPeers.includes(t))

  return (
    <div className="flex flex-col h-full">
      {/* Toolbar */}
      <div className="flex flex-wrap items-center gap-x-1.5 gap-y-1 px-3 py-1.5 border-b border-border bg-s1 shrink-0">
        {/* Chart type */}
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

        {/* Period */}
        {PERIODS.map(p => (
          <button
            key={p.days}
            onClick={() => {
              setDays(p.days)
              if (p.days === 1) setChartType('candlestick')
            }}
            className={`text-2xs px-2 py-0.5 border transition-colors ${days === p.days ? 'border-accent text-accent' : 'border-border text-muted hover:text-text'}`}
          >
            {p.label}
          </button>
        ))}

        {isIntraday && (
          <>
            <span className="text-border">|</span>
            {INTRADAY_SCALES.map(scale => (
              <button
                key={scale.seconds}
                onClick={() => setIntradayScale(scale.seconds)}
                className={`text-2xs px-2 py-0.5 border transition-colors ${intradayScale === scale.seconds ? 'border-accent text-accent' : 'border-border text-muted hover:text-text'}`}
                aria-pressed={intradayScale === scale.seconds}
              >
                {scale.label}
              </button>
            ))}
          </>
        )}

        <span className="text-border">|</span>
        <span className="text-2xs text-dim">Compare</span>

        {/* Peer quick-picks */}
        {overlayPeers.map((peer, i) => {
          const selected = selectedOverlays.includes(peer)
          const colorIdx = selectedOverlays.indexOf(peer)
          const color = colorIdx >= 0 ? OVERLAY_COLORS[colorIdx % OVERLAY_COLORS.length] : OVERLAY_COLORS[i % OVERLAY_COLORS.length]
          return (
            <button
              key={peer}
              onClick={() => toggleOverlay(peer)}
              className={[
                'text-2xs tabnum border px-2 py-0.5 transition-colors',
                selected ? 'border-border-strong bg-s2' : 'border-border text-muted hover:text-text',
              ].join(' ')}
              style={selected ? { color, borderColor: color } : undefined}
              aria-pressed={selected}
            >
              {peer}
            </button>
          )
        })}

        {/* Manual overlays (non-peer selections) */}
        {manualOverlays.map(t => {
          const colorIdx = selectedOverlays.indexOf(t)
          const color = OVERLAY_COLORS[colorIdx % OVERLAY_COLORS.length]
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

        {/* Manual input */}
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
        {isIntraday && (
          <span className={`text-2xs tabnum ${live.configured ? 'text-accent' : 'text-muted'}`}>
            {live.configured ? `${INTRADAY_SCALES.find(scale => scale.seconds === intradayScale)?.label ?? '5s'} live` : '1m base'}
          </span>
        )}
        {isLoading && <span className="ml-auto text-2xs text-dim animate-pulse">Loading...</span>}
      </div>

      {/* Chart fills remaining height */}
      <div ref={containerRef} className="flex-1 min-h-0" />
    </div>
  )
}
