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
  const hasSections = hasWatchlist || hasRecent

  return (
    <div className="flex h-full flex-col items-center overflow-auto px-6 py-16">

      {/* Header */}
      <div className="w-full max-w-lg text-center">
        <p className="mb-3 text-2xs font-medium uppercase tracking-[0.32em] text-accent">Markets</p>
        <h2 className="mb-1 text-3xl font-medium text-text">Market lookup</h2>
        <p className="mb-10 text-sm text-dim">
          Search any ticker to inspect market data, analysis, charts, and signals.
        </p>

        {/* Search bar */}
        <div className="flex items-stretch border border-border-strong bg-s1">
          <Search size={15} className="ml-4 shrink-0 self-center text-dim" />
          <input
            autoFocus
            value={input}
            onChange={e => setInput(e.target.value.toUpperCase())}
            onKeyDown={e => e.key === 'Enter' && submit()}
            placeholder="Enter ticker symbol..."
            className="flex-1 bg-transparent px-4 py-3 text-base text-text outline-none placeholder:text-dim tabnum"
            spellCheck={false}
          />
          <button
            onClick={submit}
            disabled={!input.trim()}
            className="shrink-0 self-stretch px-6 text-2xs font-semibold tracking-widest text-bg bg-accent hover:opacity-90 disabled:opacity-40 transition-opacity"
          >
            GO
          </button>
        </div>

        <p className="mt-3 text-left text-2xs text-dim">
          ASX-listed stocks use the{' '}
          <span className="tabnum text-muted">.AX</span> suffix &mdash;{' '}
          <span className="tabnum text-muted">BHP.AX</span>,{' '}
          <span className="tabnum text-muted">CBA.AX</span>,{' '}
          <span className="tabnum text-muted">NAB.AX</span>
        </p>
      </div>

      {/* Lists */}
      {hasSections ? (
        <div className="mt-12 w-full max-w-lg space-y-10">

          {hasWatchlist && (
            <section>
              <div className="mb-4 flex items-center gap-2">
                <Bookmark size={12} className="text-accent" />
                <span className="text-2xs font-medium uppercase tracking-[0.24em] text-accent">Market Watchlist</span>
              </div>
              <div className="flex flex-wrap gap-2">
                {watchlist.map(t => (
                  <div key={t} className="flex items-center border border-border hover:border-accent/50 transition-colors">
                    <button
                      onClick={() => go(t)}
                      className="px-3 py-2 text-xs tabnum text-text"
                    >
                      {t}
                    </button>
                    <button
                      onClick={() => removeWatch(t)}
                      className="pr-2.5 text-dim hover:text-down transition-colors"
                      aria-label={`Remove ${t} from watchlist`}
                    >
                      <X size={11} />
                    </button>
                  </div>
                ))}
              </div>
            </section>
          )}

          {hasRecent && (
            <section>
              <div className="mb-4 flex items-center gap-2">
                <Clock size={12} className="text-dim" />
                <span className="text-2xs font-medium uppercase tracking-[0.24em] text-dim">Recent</span>
              </div>
              <div className="flex flex-wrap gap-2">
                {recent.map(t => (
                  <button
                    key={t}
                    onClick={() => go(t)}
                    className="border border-border px-3 py-2 text-xs tabnum text-muted hover:border-accent/50 hover:text-text transition-colors"
                  >
                    {t}
                  </button>
                ))}
              </div>
            </section>
          )}

        </div>
      ) : (
        <p className="mt-12 text-2xs text-dim">
          Your market watchlist and recent ticker searches will appear here.
        </p>
      )}

    </div>
  )
}
