import { useEffect, useRef, useState, useCallback } from 'react'
import { createChart, CandlestickSeries, LineSeries, HistogramSeries } from 'lightweight-charts'
import type { IChartApi, Time } from 'lightweight-charts'
import { useQuery } from '@tanstack/react-query'
import { api } from '../../lib/api'
import type { PriceHistory } from '../../types'

type ChartType = 'candlestick' | 'line'

const PERIODS: { days: number; label: string }[] = [
  { days: 30,  label: '1M' },
  { days: 90,  label: '3M' },
  { days: 180, label: '6M' },
  { days: 365, label: '1Y' },
]

const CHART_H = 340

interface Props {
  defaultTicker?: string
}

function isFiniteNumber(value: unknown): value is number {
  return typeof value === 'number' && Number.isFinite(value)
}

export default function TickerSearchChart({ defaultTicker = 'SPY' }: Props) {
  const [inputValue,   setInputValue]   = useState(defaultTicker)
  const [activeTicker, setActiveTicker] = useState(defaultTicker)
  const [chartType,    setChartType]    = useState<ChartType>('candlestick')
  const [days,         setDays]         = useState(90)

  const containerRef = useRef<HTMLDivElement>(null)
  const chartRef     = useRef<IChartApi | null>(null)

  const { data, isLoading, isError } = useQuery<PriceHistory>({
    queryKey: ['tickerChart', activeTicker, days],
    queryFn:  () => api.getPriceHistory(activeTicker, days),
    staleTime: 60_000,
    retry: 1,
  })

  const load = useCallback(() => {
    const t = inputValue.trim().toUpperCase()
    if (t) setActiveTicker(t)
  }, [inputValue])

  useEffect(() => {
    const el = containerRef.current
    if (chartRef.current) {
      chartRef.current.remove()
      chartRef.current = null
    }

    if (!el || !data?.dates?.length) return

    const chart = createChart(el, {
      autoSize: true,
      height:   CHART_H,
      layout: {
        background: { color: '#0A0A0B' },
        textColor:  '#8A8A93',
      },
      grid: {
        vertLines: { color: '#1A1A1F' },
        horzLines: { color: '#1A1A1F' },
      },
      crosshair: {
        vertLine: { color: '#3A3A40', labelBackgroundColor: '#16161A' },
        horzLine: { color: '#3A3A40', labelBackgroundColor: '#16161A' },
      },
      rightPriceScale: { borderColor: '#1F1F23' },
      timeScale:       { borderColor: '#1F1F23', timeVisible: true },
    })
    chartRef.current = chart

    const times = data.dates as Time[]
    const vols  = data.volumes ?? data.volume ?? []

    // Volume histogram on bottom 25%
    if (vols.length) {
      const volSeries = chart.addSeries(HistogramSeries, {
        color: '#2A4A6A',
        priceFormat:  { type: 'volume' },
        priceScaleId: 'volume',
      })
      chart.priceScale('volume').applyOptions({
        scaleMargins: { top: 0.76, bottom: 0 },
        visible: false,
      })
      volSeries.setData(
        times
          .map((t, i) => ({
            time:  t,
            value: vols[i],
            color: i > 0 && (data.close[i] ?? 0) >= (data.close[i - 1] ?? 0)
              ? '#1F4A35'
              : '#4A1F24',
          }))
          .filter(r => isFiniteNumber(r.value))
      )
    }

    chart.priceScale('right').applyOptions({
      scaleMargins: { top: 0.04, bottom: vols.length ? 0.26 : 0.04 },
    })

    // Price series
    if (chartType === 'candlestick') {
      const cs = chart.addSeries(CandlestickSeries, {
        upColor:         '#1FBF75',
        downColor:       '#E5484D',
        borderUpColor:   '#1FBF75',
        borderDownColor: '#E5484D',
        wickUpColor:     '#1FBF75',
        wickDownColor:   '#E5484D',
      })
      cs.setData(
        times
          .map((t, i) => ({
            time:  t,
            open:  data.open?.[i],
            high:  data.high?.[i],
            low:   data.low?.[i],
            close: data.close?.[i],
          }))
          .filter(r => (
            isFiniteNumber(r.open) &&
            isFiniteNumber(r.high) &&
            isFiniteNumber(r.low) &&
            isFiniteNumber(r.close)
          ))
      )
    } else {
      const ls = chart.addSeries(LineSeries, { color: '#E89B2C', lineWidth: 1 })
      ls.setData(
        times
          .map((t, i) => ({ time: t, value: data.close?.[i] }))
          .filter(r => isFiniteNumber(r.value) && r.value > 0)
      )
    }

    if (data.ma5?.length) {
      const ma5 = chart.addSeries(LineSeries, { color: '#C47A0A', lineWidth: 1, title: 'MA5' })
      ma5.setData(
        times
          .map((t, i) => ({ time: t, value: data.ma5![i] }))
          .filter(r => isFiniteNumber(r.value) && r.value > 0)
      )
    }

    if (data.ma20?.length) {
      const ma20 = chart.addSeries(LineSeries, { color: '#3B82F6', lineWidth: 1, title: 'MA20' })
      ma20.setData(
        times
          .map((t, i) => ({ time: t, value: data.ma20![i] }))
          .filter(r => isFiniteNumber(r.value) && r.value > 0)
      )
    }

    chart.timeScale().fitContent()

    return () => {
      chart.remove()
      chartRef.current = null
    }
  }, [data, chartType])

  const stats     = data?.stats
  const lastClose = data?.close?.at(-1) ?? 0
  const prevClose = data?.close?.at(-2) ?? lastClose
  const last      = stats?.last ?? lastClose
  const isUp      = last >= prevClose

  return (
    <div className="border border-border bg-bg">
      {/* Toolbar */}
      <div className="flex items-center gap-2 px-3 py-1.5 border-b border-border bg-s1 flex-wrap">
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
        <button
          onClick={load}
          className="text-2xs text-bg bg-accent px-2 py-0.5 hover:opacity-90 font-medium"
        >
          GO
        </button>

        <span className="text-border">|</span>

        {(['candlestick', 'line'] as ChartType[]).map(t => (
          <button
            key={t}
            onClick={() => setChartType(t)}
            className={`text-2xs px-2 py-0.5 border ${
              chartType === t
                ? 'border-accent text-accent'
                : 'border-border text-muted hover:text-text'
            }`}
          >
            {t === 'candlestick' ? 'CANDLE' : 'LINE'}
          </button>
        ))}

        <span className="text-border">|</span>

        {PERIODS.map(p => (
          <button
            key={p.days}
            onClick={() => setDays(p.days)}
            className={`text-2xs px-2 py-0.5 border ${
              days === p.days
                ? 'border-accent text-accent'
                : 'border-border text-muted hover:text-text'
            }`}
          >
            {p.label}
          </button>
        ))}

        <div className="ml-auto flex items-center gap-3">
          {isLoading && <span className="text-2xs text-dim animate-pulse">Loading...</span>}
          {isError   && <span className="text-2xs text-down">Error: {activeTicker}</span>}
          {stats && !isLoading && (
            <>
              <span className={`text-sm tabnum font-medium ${isUp ? 'text-up' : 'text-down'}`}>
                {last.toFixed(2)}
              </span>
              <span className="text-2xs text-muted tabnum">H&nbsp;{stats.high.toFixed(2)}</span>
              <span className="text-2xs text-muted tabnum">L&nbsp;{stats.low.toFixed(2)}</span>
              <span className="text-2xs text-dim   tabnum">AVG&nbsp;{stats.avg.toFixed(2)}</span>
            </>
          )}
        </div>
      </div>

      <div className="px-3 py-0.5 bg-s1 border-b border-border">
        <span className="text-2xs text-dim">
          {activeTicker} {'>'} Price / Volume
        </span>
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
