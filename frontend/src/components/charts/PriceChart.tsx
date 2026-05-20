import { useEffect, useRef } from 'react'
import { createChart, CandlestickSeries, LineSeries } from 'lightweight-charts'
import type { Time } from 'lightweight-charts'
import type { PriceHistory } from '../../types'

interface PriceChartProps {
  data: PriceHistory | null
  height?: number
}

function isFiniteNumber(value: unknown): value is number {
  return typeof value === 'number' && Number.isFinite(value)
}

export default function PriceChart({ data, height = 360 }: PriceChartProps) {
  const containerRef = useRef<HTMLDivElement>(null)

  useEffect(() => {
    const el = containerRef.current
    if (!el || !data?.dates?.length) {
      return
    }

    const chart = createChart(el, {
      autoSize: true,
      height,
      layout: {
        background: { color: '#0A0A0B' },
        textColor: '#8A8A93',
      },
      grid: {
        vertLines: { color: '#1F1F23' },
        horzLines: { color: '#1F1F23' },
      },
      crosshair: {
        vertLine: { color: '#2A2A30', labelBackgroundColor: '#16161A' },
        horzLine: { color: '#2A2A30', labelBackgroundColor: '#16161A' },
      },
      rightPriceScale: { borderColor: '#1F1F23' },
      timeScale:       { borderColor: '#1F1F23', timeVisible: true },
    })

    const times = data.dates as Time[]

    const candleSeries = chart.addSeries(CandlestickSeries, {
      upColor:         '#1FBF75',
      downColor:       '#E5484D',
      borderUpColor:   '#1FBF75',
      borderDownColor: '#E5484D',
      wickUpColor:     '#1FBF75',
      wickDownColor:   '#E5484D',
    })

    candleSeries.setData(
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

    if (data.ma5?.length) {
      const ma5 = chart.addSeries(LineSeries, { color: '#E89B2C', lineWidth: 1, title: 'MA5' })
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
    }
  }, [data, height])

  if (!data?.dates?.length) {
    return (
      <div
        className="w-full flex items-center justify-center bg-bg border-b border-border"
        style={{ height }}
      >
        <span className="text-dim text-sm">No chart data</span>
      </div>
    )
  }

  return <div ref={containerRef} className="w-full" style={{ height }} />
}
