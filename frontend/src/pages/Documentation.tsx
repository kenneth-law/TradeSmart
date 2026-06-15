import { useParams, useNavigate } from 'react-router-dom'
import { useQuery } from '@tanstack/react-query'
import { api } from '../lib/api'

const DOC_TYPES = [
  { key: 'readme',      label: 'README' },
  { key: 'architecture', label: 'Architecture' },
  { key: 'logic-flow',  label: 'Logic Flow' },
  { key: 'strategy',    label: 'Strategy' },
  { key: 'backtest',    label: 'Backtest' },
  { key: 'api',         label: 'API' },
  { key: 'about',       label: 'About' },
]

export default function Documentation() {
  const { type = 'readme' } = useParams<{ type?: string }>()
  const navigate = useNavigate()

  const { data, isLoading, error } = useQuery({
    queryKey: ['docs', type],
    queryFn: () => api.getDocumentation(type),
  })

  return (
    <div className="flex flex-col h-full">
      {/* Tab bar */}
      <div className="flex items-center border-b border-border bg-s1 shrink-0" style={{ height: 34 }}>
        {DOC_TYPES.map(doc => (
          <button
            key={doc.key}
            onClick={() => navigate(`/docs/${doc.key}`)}
            className={[
              'flex items-center h-full px-3 text-2xs tracking-widest border-r border-border',
              'hover:text-text transition-colors',
              type === doc.key
                ? 'text-accent border-b-2 border-b-accent'
                : 'text-dim',
            ].join(' ')}
          >
            {doc.label.toUpperCase()}
          </button>
        ))}
      </div>

      {/* Content */}
      <div className="flex-1 overflow-auto p-6 max-w-4xl">
        {isLoading && (
          <div className="text-muted text-sm tabnum">Loading documentation…</div>
        )}
        {error && (
          <p className="text-down text-sm">Failed to load documentation.</p>
        )}
        {data && (
          <>
            {data.title && (
              <h1 className="text-base text-text font-medium mb-4 border-b border-border pb-2">
                {data.title}
              </h1>
            )}
            <div
              className="prose-terminal"
              dangerouslySetInnerHTML={{ __html: data.html_content }}
            />
          </>
        )}
      </div>

      <style>{`
        .prose-terminal h1, .prose-terminal h2, .prose-terminal h3 {
          color: #E6E6E6; font-size: 13px; 
          letter-spacing: 0.05em; margin: 1.5em 0 0.5em; border-bottom: 1px solid #1F1F23;
          padding-bottom: 4px;
        }
        .prose-terminal h1 { font-size: 14px; }
        .prose-terminal p { color: #8A8A93; font-size: 12px; line-height: 1.7; margin: 0.5em 0; }
        .prose-terminal code {
          font-family: 'Segoe UI', system-ui, sans-serif; font-size: 11px;
          background: #111114; color: #E89B2C; padding: 1px 4px;
          border: 1px solid #1F1F23;
        }
        .prose-terminal pre {
          background: #111114; border: 1px solid #1F1F23;
          padding: 12px; overflow-x: auto; margin: 0.75em 0;
        }
        .prose-terminal pre code {
          background: none; border: none; padding: 0; color: #E6E6E6;
        }
        .prose-terminal a { color: #E89B2C; text-decoration: none; }
        .prose-terminal a:hover { text-decoration: underline; }
        .prose-terminal ul, .prose-terminal ol {
          color: #8A8A93; font-size: 12px; padding-left: 1.5em; margin: 0.5em 0;
        }
        .prose-terminal li { margin: 0.25em 0; }
        .prose-terminal table {
          border-collapse: collapse; width: 100%; font-size: 12px; margin: 0.75em 0;
        }
        .prose-terminal th {
          background: #16161A; color: #5A5A63; 
          font-size: 11px; padding: 6px 8px; border: 1px solid #1F1F23; text-align: left;
        }
        .prose-terminal td {
          color: #8A8A93; padding: 4px 8px; border: 1px solid #1F1F23;
        }
        .prose-terminal blockquote {
          border-left: 2px solid #E89B2C; margin: 0.5em 0; padding-left: 12px;
          color: #5A5A63; font-style: normal;
        }
        .prose-terminal hr { border: none; border-top: 1px solid #1F1F23; margin: 1.5em 0; }
        .prose-terminal .doc-flowchart {
          margin: 1rem 0 1.4rem;
          overflow-x: auto;
          border: 1px solid #2A2A30;
          background: #09090B;
          padding: 14px;
        }
        .prose-terminal .doc-flowchart svg {
          min-width: 760px;
          width: 100%;
          height: auto;
          color: #E89B2C;
          display: block;
        }
        .prose-terminal .flow-lane rect,
        .prose-terminal .flow-step circle,
        .prose-terminal .flow-note {
          fill: #141418;
          stroke: #E89B2C;
          stroke-width: 1.5;
        }
        .prose-terminal .flow-note {
          fill: #101014;
          stroke: #2A2A30;
        }
        .prose-terminal .flow-edge {
          stroke: #E89B2C;
          stroke-width: 1.6;
          fill: none;
        }
        .prose-terminal .flow-edge.muted {
          stroke: #5A5A63;
        }
        .prose-terminal .flow-lane text,
        .prose-terminal .flow-step text,
        .prose-terminal .flow-note-text {
          fill: #E6E6E6;
          font-family: 'Segoe UI', system-ui, sans-serif;
          font-size: 13px;
          text-anchor: middle;
          dominant-baseline: middle;
        }
        .prose-terminal .flow-lane text + text,
        .prose-terminal .flow-note-text + .flow-note-text {
          fill: #A1A1AA;
          font-size: 11px;
        }
        .prose-terminal .flow-step text:first-of-type {
          fill: #E89B2C;
          font-size: 15px;
          font-weight: 600;
        }
      `}</style>
    </div>
  )
}
