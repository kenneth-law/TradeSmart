import { useState, useRef, useMemo } from 'react'
import { useVirtualizer } from '@tanstack/react-virtual'

export interface Column<T> {
  key: string
  label: string
  width?: number
  align?: 'left' | 'right'
  sortable?: boolean
  render?: (row: T) => React.ReactNode
}

interface DataTableProps<T extends Record<string, unknown>> {
  columns: Column<T>[]
  data: T[]
  rowKey: (row: T) => string
  onRowClick?: (row: T) => void
  selectedKey?: string
  emptyMessage?: string
  virtualRows?: boolean
  rowHeight?: number
  fitColumns?: boolean
}

export default function DataTable<T extends Record<string, unknown>>({
  columns,
  data,
  rowKey,
  onRowClick,
  selectedKey,
  emptyMessage = 'No data.',
  virtualRows = false,
  rowHeight = 26,
  fitColumns = false,
}: DataTableProps<T>) {
  const [sortKey, setSortKey] = useState<string | null>(null)
  const [sortDir, setSortDir] = useState<'asc' | 'desc'>('desc')

  const sorted = useMemo(() => {
    if (!sortKey) return data
    return [...data].sort((a, b) => {
      const av = a[sortKey]
      const bv = b[sortKey]
      const diff = typeof av === 'number' && typeof bv === 'number'
        ? av - bv
        : String(av ?? '').localeCompare(String(bv ?? ''))
      return sortDir === 'asc' ? diff : -diff
    })
  }, [data, sortKey, sortDir])

  function toggleSort(key: string) {
    if (sortKey === key) setSortDir(d => d === 'asc' ? 'desc' : 'asc')
    else { setSortKey(key); setSortDir('desc') }
  }

  const parentRef = useRef<HTMLDivElement>(null)
  const tableWidth = columns.reduce((sum, col) => sum + (col.width ?? 120), 0)
  const tableStyle: React.CSSProperties = fitColumns
    ? { width: tableWidth, minWidth: tableWidth }
    : { minWidth: tableWidth }
  const virtualizer = useVirtualizer({
    count: sorted.length,
    getScrollElement: () => parentRef.current,
    estimateSize: () => rowHeight,
    enabled: virtualRows,
  })

  function handleKey(e: React.KeyboardEvent, row: T) {
    if (e.key === 'Enter' || e.key === ' ') onRowClick?.(row)
  }

  const header = (
    <thead>
      <tr>
        {columns.map(col => (
          <th
            key={col.key}
            onClick={col.sortable !== false ? () => toggleSort(col.key) : undefined}
            style={{ width: col.width }}
            className={[
              'px-4 py-1.5 text-2xs text-muted font-medium',
              'sticky top-0 z-10 border-b border-border bg-s2 select-none',
              col.align === 'right' ? 'text-right' : 'text-left',
              col.sortable !== false ? 'cursor-pointer hover:text-text' : '',
            ].join(' ')}
            aria-sort={sortKey === col.key ? (sortDir === 'asc' ? 'ascending' : 'descending') : undefined}
          >
            {col.label}
            {col.sortable !== false && sortKey === col.key && (
              <span className="ml-1 opacity-60">{sortDir === 'asc' ? '↑' : '↓'}</span>
            )}
          </th>
        ))}
      </tr>
    </thead>
  )

  if (sorted.length === 0) {
    return (
      <div>
        <table className={fitColumns ? 'table-fixed' : 'w-full table-fixed'} style={tableStyle}>{header}</table>
        <p className="text-muted text-sm p-4">{emptyMessage}</p>
      </div>
    )
  }

  function renderRow(row: T, style?: React.CSSProperties) {
    const key = rowKey(row)
    const isSelected = key === selectedKey
    return (
      <tr
        key={key}
        style={style}
        onClick={() => onRowClick?.(row)}
        onKeyDown={(e) => handleKey(e, row)}
        tabIndex={onRowClick ? 0 : undefined}
        role={onRowClick ? 'button' : undefined}
        aria-selected={isSelected}
        className={[
          'border-b border-border transition-row',
          isSelected ? 'bg-s2 border-l-2 border-l-accent' : 'hover:bg-s2',
          onRowClick ? 'cursor-pointer' : '',
        ].join(' ')}
      >
        {columns.map(col => {
          const val = row[col.key]
          return (
            <td
              key={col.key}
              style={{ width: col.width }}
              className={[
                'px-4 py-1.5 text-sm tabnum',
                col.align === 'right' ? 'text-right' : 'text-left',
              ].join(' ')}
            >
              {col.render ? col.render(row) : (val !== undefined && val !== null && val !== '') ? String(val) : <span className="text-dim">—</span>}
            </td>
          )
        })}
      </tr>
    )
  }

  if (!virtualRows) {
    return (
      <div className="overflow-auto w-full">
        <table className={fitColumns ? 'table-fixed' : 'w-full table-fixed'} style={tableStyle}>
          {header}
          <tbody>{sorted.map(row => renderRow(row))}</tbody>
        </table>
      </div>
    )
  }

  const vItems = virtualizer.getVirtualItems()
  return (
    <div ref={parentRef} className="overflow-auto w-full" style={{ maxHeight: 600 }}>
      <table className={fitColumns ? 'table-fixed' : 'w-full table-fixed'} style={tableStyle}>
        {header}
        <tbody style={{ height: virtualizer.getTotalSize(), position: 'relative' }}>
          {vItems.map(vRow => {
            const row = sorted[vRow.index]
            return renderRow(row, {
              position: 'absolute',
              top: vRow.start,
              left: 0,
              width: '100%',
              height: rowHeight,
            })
          })}
        </tbody>
      </table>
    </div>
  )
}
