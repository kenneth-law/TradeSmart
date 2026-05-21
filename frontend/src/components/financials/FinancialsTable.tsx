import type { FinancialsStatement } from '../../types'

interface Props {
  title: string
  data: FinancialsStatement | undefined
}

function formatNumber(v: number | null | undefined): string {
  if (v == null || !Number.isFinite(v)) return '-'
  const abs = Math.abs(v)
  if (abs >= 1e12) return `${(v / 1e12).toFixed(2)}T`
  if (abs >= 1e9) return `${(v / 1e9).toFixed(2)}B`
  if (abs >= 1e6) return `${(v / 1e6).toFixed(2)}M`
  if (abs >= 1e3) return `${(v / 1e3).toFixed(2)}K`
  return v.toFixed(2)
}

function formatPeriod(p: string): string {
  // ISO date "2024-12-31" -> "Dec '24"
  const m = p.match(/^(\d{4})-(\d{2})-(\d{2})$/)
  if (!m) return p
  const year = m[1].slice(2)
  const monthIdx = parseInt(m[2], 10) - 1
  const months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
  return `${months[monthIdx] ?? m[2]} '${year}`
}

export default function FinancialsTable({ title, data }: Props) {
  if (!data || data.periods.length === 0 || Object.keys(data.rows).length === 0) {
    return (
      <div className="flex-1 min-w-0 border-r border-border last:border-r-0">
        <div className="text-2xs uppercase tracking-[0.2em] text-dim px-3 pt-3 pb-2 border-b border-border bg-s1/40 sticky top-0">
          {title}
        </div>
        <div className="px-3 py-4 text-2xs text-dim italic">No data available.</div>
      </div>
    )
  }

  const rowNames = Object.keys(data.rows)

  return (
    <div className="flex-1 min-w-0 border-r border-border last:border-r-0">
      <div className="text-2xs uppercase tracking-[0.2em] text-dim px-3 pt-3 pb-2 border-b border-border bg-s1/40 sticky top-0 z-10">
        {title}
      </div>
      <table className="w-full text-2xs tabnum">
        <thead className="sticky top-7 bg-s1/80 z-10">
          <tr className="border-b border-border">
            <th className="text-left font-medium text-dim px-3 py-1.5 w-1/2">Line item</th>
            {data.periods.map(p => (
              <th key={p} className="text-right font-medium text-dim px-2 py-1.5 whitespace-nowrap">
                {formatPeriod(p)}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {rowNames.map(name => {
            const row = data.rows[name]
            return (
              <tr key={name} className="border-b border-border/40 hover:bg-s2/40">
                <td className="px-3 py-1 text-muted truncate" title={name}>{name}</td>
                {data.periods.map(p => {
                  const v = row[p]
                  return (
                    <td key={p} className={`px-2 py-1 text-right ${v != null && v < 0 ? 'text-down' : 'text-text'}`}>
                      {formatNumber(v)}
                    </td>
                  )
                })}
              </tr>
            )
          })}
        </tbody>
      </table>
    </div>
  )
}
