import { BarChart, Bar, XAxis, YAxis, Tooltip, Cell, ResponsiveContainer } from 'recharts'

interface DistributionChartProps {
  data: Array<{ name: string; value: number }>
  height?: number
}

const SIGNAL_COLORS: Record<string, string> = {
  'Strong Buy':    '#1FBF75',
  'Buy':           '#1FBF75',
  'Neutral/Watch': '#D4A017',
  'Sell':          '#E5484D',
  'Strong Sell':   '#E5484D',
}

export default function DistributionChart({ data, height = 160 }: DistributionChartProps) {
  return (
    <div style={{ width: '100%', minWidth: 0 }}>
    <ResponsiveContainer width="99%" height={height}>
      <BarChart data={data} margin={{ top: 4, right: 0, left: -16, bottom: 0 }} barSize={20}>
        <XAxis
          dataKey="name"
          tick={{ fill: '#8A8A93', fontSize: 11, fontFamily: 'Segoe UI' }}
          axisLine={{ stroke: '#1F1F23' }}
          tickLine={false}
        />
        <YAxis
          tick={{ fill: '#8A8A93', fontSize: 11, fontFamily: 'Segoe UI' }}
          axisLine={false}
          tickLine={false}
        />
        <Tooltip
          contentStyle={{
            background: '#111114',
            border: '1px solid #1F1F23',
            borderRadius: 0,
            fontFamily: 'Segoe UI',
            fontSize: 12,
            color: '#E6E6E6',
          }}
          cursor={{ fill: '#16161A' }}
        />
        <Bar dataKey="value" radius={0}>
          {data.map((entry) => (
            <Cell key={entry.name} fill={SIGNAL_COLORS[entry.name] ?? '#5A5A63'} />
          ))}
        </Bar>
      </BarChart>
    </ResponsiveContainer>
    </div>
  )
}
