import { useState, useRef } from 'react'
import { useNavigate } from 'react-router-dom'
import { useAppStore } from '../store/useAppStore'
import { useSSE } from '../hooks/useSSE'
import { api } from '../lib/api'
import TerminalLog from '../components/terminal/TerminalLog'
import ProgressBar from '../components/terminal/ProgressBar'

const ASX200 = 'CBA.AX, BHP.AX, CSL.AX, NAB.AX, WBC.AX, WES.AX, ANZ.AX, MQG.AX, GMG.AX, TLS.AX, FMG.AX, RIO.AX, TCL.AX, WDS.AX, ALL.AX, WOW.AX, QBE.AX, REA.AX, WTC.AX, BXB.AX, PME.AX, COL.AX, XRO.AX, NST.AX, CPU.AX, RMD.AX, SUN.AX, STO.AX, IAG.AX, COH.AX, JHX.AX, EVN.AX, QAN.AX, CAR.AX, S32.AX, SGP.AX, MPL.AX, SHL.AX, JBH.AX, TNE.AX, APA.AX, REH.AX, BSL.AX, TPG.AX, AMC.AX, ORI.AX, GPT.AX, LYC.AX, AGL.AX, TWE.AX, WOR.AX, BEN.AX, GQG.AX, HVN.AX, HUB.AX, ALD.AX, CWY.AX, A2M.AX'

const QUICK_NAV = [
  { label: 'Markets',        to: '/market',       key: '2' },
  { label: 'Quant Trading',  to: '/system',       key: '3' },
  { label: 'Daily Lineup',   to: '/daily-lineup', key: '5' },
  { label: 'Portfolio',      to: '/portfolio',    key: '6' },
  { label: 'Docs',           to: '/docs',         key: '7' },
]

export default function Technical() {
  const navigate = useNavigate()
  const setAnalysisResult = useAppStore(s => s.setAnalysisResult)
  const setLastUpdated = useAppStore(s => s.setLastUpdated)

  const [tickerInput, setTickerInput] = useState('')
  const [sseUrl, setSseUrl] = useState<string | null>(null)
  const [fetchingResults, setFetchingResults] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const inputRef = useRef<HTMLTextAreaElement>(null)

  const { messages, progress, status, lastMessage } = useSSE(sseUrl)

  const isRunning = status === 'connecting' || status === 'streaming'

  // When SSE completes, fetch results and navigate
  if (status === 'complete' && lastMessage?.session_id && !fetchingResults) {
    setFetchingResults(true)
    api.getAnalysisResults(lastMessage.session_id).then(result => {
      setAnalysisResult(result)
      setLastUpdated(new Date())
      navigate('/results')
    }).catch(e => {
      setError(`Failed to load results: ${e.message}`)
      setFetchingResults(false)
    })
  }

  function startAnalysis() {
    const tickers = tickerInput.trim()
    if (!tickers) { inputRef.current?.focus(); return }
    setError(null)
    setSseUrl(null)
    setTimeout(() => {
      setSseUrl(`/analysis_progress?tickers=${encodeURIComponent(tickers)}`)
    }, 0)
  }

  function onKeyDown(e: React.KeyboardEvent) {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      startAnalysis()
    }
  }

  return (
    <div className="p-4 max-w-3xl mx-auto">

      {/* Section: Analysis */}
      <section aria-labelledby="analysis-heading">
        <div className="flex items-center justify-between mb-2">
          <h1 id="analysis-heading" className="text-sm text-text font-medium">
            Technical Analysis
          </h1>
          <span className="text-2xs text-dim">Shift+Enter for newline · Enter to run</span>
        </div>

        <div className="border border-border bg-s1">
          {/* Prefix */}
          <div className="flex items-start gap-2 p-2 border-b border-border">
            <span className="text-muted text-sm pt-px select-none shrink-0">{'>'}</span>
            <textarea
              ref={inputRef}
              value={tickerInput}
              onChange={e => setTickerInput(e.target.value)}
              onKeyDown={onKeyDown}
              placeholder="AAPL, MSFT, GOOGL  —  or load ASX 200 below"
              rows={3}
              disabled={isRunning}
              className="flex-1 bg-transparent text-text text-sm placeholder:text-dim resize-none outline-none disabled:opacity-50"
              aria-label="Ticker symbols"
            />
          </div>

          {/* Toolbar */}
          <div className="flex items-center gap-2 px-2 py-1.5">
            <button
              onClick={() => setTickerInput(ASX200)}
              disabled={isRunning}
              className="text-2xs text-muted border border-border-strong px-2 py-1 hover:text-text disabled:opacity-40"
            >
              Load ASX 200
            </button>
            <button
              onClick={() => setTickerInput('')}
              disabled={isRunning || !tickerInput}
              className="text-2xs text-muted border border-border-strong px-2 py-1 hover:text-text disabled:opacity-40"
            >
              Clear
            </button>
            <div className="ml-auto flex items-center gap-2">
              {tickerInput && !isRunning && (
                <span className="text-2xs text-dim tabnum">
                  {tickerInput.split(',').filter(t => t.trim()).length} tickers
                </span>
              )}
              <button
                onClick={startAnalysis}
                disabled={isRunning || !tickerInput.trim()}
                className="text-sm text-bg bg-accent px-3 py-1 hover:opacity-90 disabled:opacity-40 font-medium"
                aria-label="Run analysis [Enter]"
              >
                Run [↵]
              </button>
            </div>
          </div>
        </div>

        {/* Error */}
        {error && (
          <p className="mt-2 text-2xs text-down" role="alert">
            {error}
          </p>
        )}

        {/* Progress */}
        {isRunning && (
          <div className="mt-3">
            <div className="flex justify-between items-center mb-1">
              <span className="text-2xs text-muted">
                {status === 'connecting' ? 'Connecting…' : 'Analysing…'}
              </span>
              <span className="text-2xs text-dim tabnum">{progress}%</span>
            </div>
            <ProgressBar value={progress} />
          </div>
        )}

        {(isRunning || messages.length > 0) && (
          <div className="mt-2">
            <TerminalLog messages={messages} height={180} />
          </div>
        )}
      </section>

      {/* Divider */}
      <div className="my-6 border-t border-border" />

      {/* Section: Quick navigation */}
      <section aria-labelledby="nav-heading">
        <h2 id="nav-heading" className="text-2xs text-dim mb-2">
          Tools
        </h2>
        <div className="grid grid-cols-2 gap-px border border-border bg-border">
          {QUICK_NAV.map(({ label, to, key }) => (
            <button
              key={to}
              onClick={() => navigate(to)}
              className="bg-s1 text-left px-3 py-2 hover:bg-s2 flex items-center justify-between group"
              title={`Alt+${key}`}
            >
              <span className="text-sm text-muted group-hover:text-text">{label}</span>
              <span className="text-2xs text-dim">Alt+{key}</span>
            </button>
          ))}
        </div>
      </section>
    </div>
  )
}
