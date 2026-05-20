const RECENT_KEY = 'ts_recent'
const WATCHLIST_KEY = 'ts_watchlist'
const MAX_RECENT = 12

function readJSON<T>(key: string, fallback: T): T {
  try { return JSON.parse(localStorage.getItem(key) ?? 'null') ?? fallback } catch { return fallback }
}

export function getRecentSearches(): string[] {
  return readJSON<string[]>(RECENT_KEY, [])
}

export function addRecentSearch(ticker: string) {
  const list = getRecentSearches().filter(t => t !== ticker)
  localStorage.setItem(RECENT_KEY, JSON.stringify([ticker, ...list].slice(0, MAX_RECENT)))
}

export function getWatchlist(): string[] {
  return readJSON<string[]>(WATCHLIST_KEY, [])
}

export function isInWatchlist(ticker: string): boolean {
  return getWatchlist().includes(ticker)
}

export function toggleWatchlist(ticker: string): boolean {
  const list = getWatchlist()
  const next = list.includes(ticker) ? list.filter(t => t !== ticker) : [...list, ticker]
  localStorage.setItem(WATCHLIST_KEY, JSON.stringify(next))
  return next.includes(ticker)
}

export function removeFromWatchlist(ticker: string) {
  localStorage.setItem(WATCHLIST_KEY, JSON.stringify(getWatchlist().filter(t => t !== ticker)))
}
