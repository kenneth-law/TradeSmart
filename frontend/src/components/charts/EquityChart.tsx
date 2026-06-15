import { useEffect, useRef } from 'react'
import { createChart, LineSeries, createSeriesMarkers } from 'lightweight-charts'
import type { IChartApi, SeriesMarker, Time } from 'lightweight-charts'
import type { Trade } from '../../types'

interface EquityPoint {
  date: string
  value: number
}

export interface PriceOverlay {
  ticker: string
  color: string
  points: EquityPoint[]
}

interface Props {
  equityCurve: EquityPoint[]
  trades?: Trade[]
  overlays?: PriceOverlay[]
  height?: number
}

export default function EquityChart({ equityCurve, trades = [], overlays = [], height = 260 }: Props) {
  const containerRef = useRef<HTMLDivElement>(null)
  const chartRef = useRef<IChartApi | null>(null)

  useEffect(() => {
    const el = containerRef.current
    if (!el || !equityCurve.length) return

    if (chartRef.current) {
      chartRef.current.remove()
      chartRef.current = null
    }

    const chart = createChart(el, {
      autoSize: true,
      height,
      layout: {
        background: { color: '#0A0A0B' },
        textColor: '#8A8A93',
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
      leftPriceScale: {
        borderColor: '#1F1F23',
        visible: overlays.length > 0,
      },
      timeScale: { borderColor: '#1F1F23', timeVisible: false },
    })
    chartRef.current = chart

    const startValue = equityCurve[0]?.value ?? 1
    const endValue   = equityCurve.at(-1)?.value ?? startValue
    const lineColor  = endValue >= startValue ? '#1FBF75' : '#E5484D'

    const lineSeries = chart.addSeries(LineSeries, {
      color:     lineColor,
      lineWidth: 1,
      priceFormat: { type: 'price', precision: 0, minMove: 1 },
    })

    lineSeries.setData(
      equityCurve.map(p => ({
        time:  p.date.slice(0, 10) as Time,
        value: p.value,
      }))
    )

    overlays.forEach(overlay => {
      const overlaySeries = chart.addSeries(LineSeries, {
        color: overlay.color,
        lineWidth: 1,
        priceScaleId: 'left',
        priceFormat: { type: 'price', precision: 2, minMove: 0.01 },
        title: overlay.ticker,
      })

      overlaySeries.setData(
        overlay.points.map(p => ({
          time: p.date.slice(0, 10) as Time,
          value: p.value,
        }))
      )
    })

    // Trade markers via the v5 plugin API
    if (trades.length) {
      const dateSet = new Set(equityCurve.map(p => p.date.slice(0, 10)))

      // Collapse multiple trades on the same date into one marker
      const byDate = new Map<string, SeriesMarker<Time>>()
      trades
        .filter(t => dateSet.has(t.date.slice(0, 10)))
        .forEach(t => {
          const d = t.date.slice(0, 10) as Time
          if (!byDate.has(d as string)) {
            byDate.set(d as string, {
              time:     d,
              position: t.type === 'BUY' ? 'belowBar' : 'aboveBar',
              color:    t.type === 'BUY' ? '#1FBF75' : '#E5484D',
              shape:    t.type === 'BUY' ? 'arrowUp' : 'arrowDown',
              text:     t.ticker,
              size:     1,
            })
          }
        })

      if (byDate.size) {
        const markersPlugin = createSeriesMarkers(lineSeries, [...byDate.values()])
        void markersPlugin // prevent unused warning
      }
    }

    chart.timeScale().fitContent()

    return () => {
      chart.remove()
      chartRef.current = null
    }
  }, [equityCurve, trades, overlays, height])

  if (!equityCurve.length) return null

  return <div ref={containerRef} style={{ height }} />
}
