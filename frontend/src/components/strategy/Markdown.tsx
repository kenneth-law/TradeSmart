import ReactMarkdown from 'react-markdown'
import remarkGfm from 'remark-gfm'

interface Props {
  text: string
}

export default function Markdown({ text }: Props) {
  return (
    <div className="markdown-strategy text-2xs leading-relaxed text-muted break-words">
      <ReactMarkdown
        remarkPlugins={[remarkGfm]}
        components={{
          p: ({ children }) => <p className="mb-2 last:mb-0">{children}</p>,
          strong: ({ children }) => <strong className="font-semibold text-text">{children}</strong>,
          em: ({ children }) => <em className="italic">{children}</em>,
          ul: ({ children }) => <ul className="list-disc pl-4 mb-2 space-y-0.5">{children}</ul>,
          ol: ({ children }) => <ol className="list-decimal pl-4 mb-2 space-y-0.5">{children}</ol>,
          li: ({ children }) => <li className="leading-relaxed">{children}</li>,
          h1: ({ children }) => <h3 className="text-xs font-semibold text-text mb-1.5 mt-2">{children}</h3>,
          h2: ({ children }) => <h3 className="text-xs font-semibold text-text mb-1.5 mt-2">{children}</h3>,
          h3: ({ children }) => <h4 className="text-2xs font-semibold text-text mb-1 mt-2 uppercase tracking-wider">{children}</h4>,
          h4: ({ children }) => <h4 className="text-2xs font-semibold text-text mb-1 mt-2">{children}</h4>,
          h5: ({ children }) => <h5 className="text-2xs font-semibold text-text mb-1 mt-2">{children}</h5>,
          h6: ({ children }) => <h6 className="text-2xs font-semibold text-text mb-1 mt-2">{children}</h6>,
          a: ({ children, href }) => (
            <a
              href={href}
              target="_blank"
              rel="noopener noreferrer"
              className="text-accent hover:underline break-words"
            >
              {children}
            </a>
          ),
          code: ({ children, className }) => {
            const isBlock = className?.includes('language-')
            if (isBlock) return <>{children}</>
            return (
              <code className="px-1 py-px bg-s2 text-text border border-border text-[10px] tabnum break-words">
                {children}
              </code>
            )
          },
          pre: ({ children }) => (
            <pre className="overflow-x-auto bg-s2 border border-border p-2 mb-2 text-[10px] text-text tabnum">
              {children}
            </pre>
          ),
          blockquote: ({ children }) => (
            <blockquote className="border-l-2 border-accent pl-2 my-2 text-dim italic">
              {children}
            </blockquote>
          ),
          hr: () => <hr className="border-border my-2" />,
          table: ({ children }) => (
            <div className="overflow-x-auto mb-2">
              <table className="min-w-full text-2xs border-collapse">{children}</table>
            </div>
          ),
          thead: ({ children }) => <thead className="border-b border-border">{children}</thead>,
          th: ({ children }) => <th className="text-left px-1.5 py-1 text-dim font-medium uppercase tracking-wider text-[10px]">{children}</th>,
          td: ({ children }) => <td className="px-1.5 py-1 border-b border-border/40 align-top">{children}</td>,
        }}
      >
        {text}
      </ReactMarkdown>
    </div>
  )
}
