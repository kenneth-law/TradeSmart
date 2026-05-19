import { useEffect, useRef } from 'react'
import { createChart, CandlestickSeries, LineSeries } from 'lightweight-charts'
import type { PriceHistory } from '../../types'

interface PriceChartProps {
  data: PriceHistory | null
  height?: number
}

export default function PriceChart({ data, height = 360 }: PriceChartProps) {
  const containerRef = useRef<HTMLDivElement>(null)

  useEffect(() => {
    if (!containerRef.current || !data || !data.dates.length) return

    const chart = createChart(containerRef.current, {
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
      rightPriceScale: {
        borderColor: '#1F1F23',
      },
      timeScale: {
        borderColor: '#1F1F23',
        timeVisible: true,
      },
      handleScale: { axisPressedMouseMove: true },
    })

    const candleSeries = chart.addSeries(CandlestickSeries, {
      upColor:         '#1FBF75',
      downColor:       '#E5484D',
      borderUpColor:   '#1FBF75',
      borderDownColor: '#E5484D',
      wickUpColor:     '#1FBF75',
      wickDownColor:   '#E5484D',
    })

    const candleData = data!.dates.map((d, i) => ({
      time: d as import('lightweight-charts').Time,
      open:  data!.open[i],
      high:  data!.high[i],
      low:   data!.low[i],
      close: data!.close[i],
    })).filter(r => r.open && r.high && r.low && r.close)

    candleSeries.setData(candleData)

    if (data!.ma5?.length) {
      const ma5 = chart.addSeries(LineSeries, {
        color: '#E89B2C',
        lineWidth: 1,
        title: 'MA5',
      })
      ma5.setData(
        data!.dates.map((d, i) => ({ time: d as import('lightweight-charts').Time, value: data!.ma5![i] }))
          .filter(r => r.value)
      )
    }

    if (data!.ma20?.length) {
      const ma20 = chart.addSeries(LineSeries, {
        color: '#3B82F6',
        lineWidth: 1,
        title: 'MA20',
      })
      ma20.setData(
        data!.dates.map((d, i) => ({ time: d as import('lightweight-charts').Time, value: data!.ma20![i] }))
          .filter(r => r.value)
      )
    }

    chart.timeScale().fitContent()

    const ro = new ResizeObserver(() => {
      if (containerRef.current) {
        chart.applyOptions({ width: containerRef.current.clientWidth })
      }
    })
    ro.observe(containerRef.current)

    return () => {
      ro.disconnect()
      chart.remove()
    }
  }, [data, height])

  if (!data) {
    return (
      <div className="w-full flex items-center justify-center bg-bg border-b border-border" style={{ height }}>
        <span className="text-dim text-sm">No chart data</span>
      </div>
    )
  }

  return <div ref={containerRef} className="w-full" style={{ height }} />
}
