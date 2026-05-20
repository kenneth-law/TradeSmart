import { useState, useEffect } from 'react'
import { useNavigate } from 'react-router-dom'
import { Search, Bookmark, Clock, X } from 'lucide-react'
import {
  getRecentSearches, addRecentSearch,
  getWatchlist, removeFromWatchlist,
} from '../lib/userPrefs'

export default function Research() {
  const navigate = useNavigate()
  const [input, setInput] = useState('')
  const [recent, setRecent] = useState<string[]>([])
  const [watchlist, setWatchlist] = useState<string[]>([])

  useEffect(() => {
    setRecent(getRecentSearches())
    setWatchlist(getWatchlist())
  }, [])

  function go(ticker: string) {
    const t = ticker.trim().toUpperCase()
    if (!t) return
    addRecentSearch(t)
    navigate(`/stock/${t}`)
  }

  function submit() { go(input) }

  function removeWatch(t: string) {
    removeFromWatchlist(t)
    setWatchlist(getWatchlist())
  }

  const hasWatchlist = watchlist.length > 0
  const hasRecent = recent.length > 0

  return (
    <div className="h-full overflow-auto px-6 py-10">
      <div className="mx-auto w-full max-w-xl">

        <p className="mb-1 text-2xs font-medium uppercase tracking-[0.28em] text-accent">Research</p>
        <h2 className="mb-7 text-xl font-medium text-text">Look up a stock</h2>

        {/* Search bar */}
        <div className="flex items-center border border-border-strong bg-s1 focus-within:border-accent/80 transition-colors">
          <Search size={14} className="ml-3 shrink-0 text-dim" />
          <input
            autoFocus
            value={input}
            onChange={e => setInput(e.target.value.toUpperCase())}
            onKeyDown={e => e.key === 'Enter' && submit()}
            placeholder="Enter ticker symbol..."
            className="flex-1 bg-transparent px-3 py-2.5 text-sm text-text outline-none placeholder:text-dim tabnum"
            spellCheck={false}
          />
          <button
            onClick={submit}
            disabled={!input.trim()}
            className="shrink-0 px-4 py-2.5 text-2xs font-medium text-bg bg-accent hover:opacity-90 disabled:opacity-40 transition-opacity"
          >
            GO
          </button>
        </div>

        <p className="mt-2 text-2xs text-dim">
          ASX-listed stocks use the{' '}
          <span className="tabnum text-muted">XXX.AX</span> suffix — e.g.{' '}
          <span className="tabnum text-muted">BHP.AX</span>,{' '}
          <span className="tabnum text-muted">CBA.AX</span>
        </p>

        {/* Watchlist */}
        {hasWatchlist && (
          <section className="mt-9">
            <div className="mb-3 flex items-center gap-2">
              <Bookmark size={12} className="text-accent" />
              <span className="text-2xs font-medium uppercase tracking-[0.22em] text-accent">Watchlist</span>
            </div>
            <div className="flex flex-wrap gap-2">
              {watchlist.map(t => (
                <div key={t} className="flex items-center border border-border hover:border-accent/50 transition-colors">
                  <button
                    onClick={() => go(t)}
                    className="px-3 py-1.5 text-xs tabnum text-text"
                  >
                    {t}
                  </button>
                  <button
                    onClick={() => removeWatch(t)}
                    className="pr-2 text-dim hover:text-down transition-colors"
                    aria-label={`Remove ${t} from watchlist`}
                  >
                    <X size={11} />
                  </button>
                </div>
              ))}
            </div>
          </section>
        )}

        {/* Recent searches */}
        {hasRecent && (
          <section className="mt-8">
            <div className="mb-3 flex items-center gap-2">
              <Clock size={12} className="text-dim" />
              <span className="text-2xs font-medium uppercase tracking-[0.22em] text-dim">Recent</span>
            </div>
            <div className="flex flex-wrap gap-2">
              {recent.map(t => (
                <button
                  key={t}
                  onClick={() => go(t)}
                  className="border border-border px-3 py-1.5 text-xs tabnum text-muted hover:border-accent/50 hover:text-text transition-colors"
                >
                  {t}
                </button>
              ))}
            </div>
          </section>
        )}

        {!hasWatchlist && !hasRecent && (
          <p className="mt-9 text-2xs text-dim">
            Your watchlist and recent searches will appear here.
          </p>
        )}

      </div>
    </div>
  )
}
