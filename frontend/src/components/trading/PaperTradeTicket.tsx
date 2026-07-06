import { useEffect, useMemo, useRef, useState } from 'react'
import { Activity, ChevronDown, ChevronUp, SlidersHorizontal, X } from 'lucide-react'
import type { StockResult } from '../../types'
import { useAppStore, type PaperOptionType } from '../../store/useAppStore'
import {
  buildExecutionPlan,
  DEFAULT_EXECUTION_PARAMETERS,
  orderIsTriggered,
  type ExecutionParameters,
  type ExecutionPlan,
} from '../../lib/paperExecution'

type Side = 'buy' | 'sell'
type TicketMode = 'stock' | 'option'
type Panel = 'ticket' | 'model' | 'blotter'

interface LiveExecution {
  status: 'waiting' | 'working' | 'filled' | 'cancelled' | 'rejected'
  side: Side
  quantity: number
  referencePrice: number
  plan: ExecutionPlan
  filledQuantity: number
  filledNotional: number
  completedSlices: number
  message?: string
}

const EXECUTION_SETTINGS_STORAGE = 'tradesmart_execution_model'

function loadExecutionParameters(): ExecutionParameters {
  if (typeof window === 'undefined') return { ...DEFAULT_EXECUTION_PARAMETERS }
  try {
    const saved = JSON.parse(window.localStorage.getItem(EXECUTION_SETTINGS_STORAGE) ?? '{}')
    return { ...DEFAULT_EXECUTION_PARAMETERS, ...saved }
  } catch {
    return { ...DEFAULT_EXECUTION_PARAMETERS }
  }
}

function nextMonthlyExpiry() {
  const d = new Date()
  d.setMonth(d.getMonth() + 1)
  d.setDate(15)
  return d.toISOString().slice(0, 10)
}

function NumericField({ label, value, onChange, min = 0, step = 1, suffix }: {
  label: string
  value: number
  onChange: (value: number) => void
  min?: number
  step?: number
  suffix?: string
}) {
  return (
    <label className="block min-w-0">
      <span className="mb-1 block text-2xs text-dim">{label}</span>
      <div className="flex border border-border bg-bg focus-within:border-accent">
        <input
          type="number"
          min={min}
          step={step}
          value={value}
          onChange={event => onChange(Number(event.target.value))}
          className="h-7 min-w-0 flex-1 bg-transparent px-2 text-xs text-text outline-none tabnum"
        />
        {suffix && <span className="flex items-center pr-2 text-[10px] text-dim">{suffix}</span>}
      </div>
    </label>
  )
}

function statusColor(status: LiveExecution['status']) {
  if (status === 'filled') return 'text-up'
  if (status === 'rejected' || status === 'cancelled') return 'text-down'
  return 'text-accent'
}

export default function PaperTradeTicket({ stock }: { stock: StockResult }) {
  const account = useAppStore(s => s.paperAccount)
  const buyPaperStock = useAppStore(s => s.buyPaperStock)
  const sellPaperStock = useAppStore(s => s.sellPaperStock)
  const buyPaperOption = useAppStore(s => s.buyPaperOption)
  const sellPaperOption = useAppStore(s => s.sellPaperOption)
  const [mode, setMode] = useState<TicketMode>('stock')
  const [panel, setPanel] = useState<Panel>('ticket')
  const [side, setSide] = useState<Side>('buy')
  const [quantity, setQuantity] = useState(100)
  const [params, setParams] = useState<ExecutionParameters>(loadExecutionParameters)
  const [advancedOpen, setAdvancedOpen] = useState(false)
  const [execution, setExecution] = useState<LiveExecution | null>(null)
  const [message, setMessage] = useState<{ text: string; tone: 'ok' | 'error' } | null>(null)
  const timers = useRef<number[]>([])

  const [optionAction, setOptionAction] = useState<Side>('buy')
  const [optionType, setOptionType] = useState<PaperOptionType>('call')
  const [strike, setStrike] = useState(Math.round(stock.current_price))
  const [expiry, setExpiry] = useState(nextMonthlyExpiry())
  const [contracts, setContracts] = useState(1)
  const [premium, setPremium] = useState(Math.max(0.05, stock.current_price * 0.03))
  const [closePositionId, setClosePositionId] = useState('')

  const stockPosition = account.positions.find(p => p.kind === 'stock' && p.ticker === stock.ticker)
  const optionPositions = account.positions.filter(p => p.kind === 'option' && p.ticker === stock.ticker)
  const selectedOption = optionPositions.find(p => p.id === closePositionId) ?? optionPositions[0]
  const recentOrders = account.orders.filter(order => order.ticker === stock.ticker).slice(0, 8)

  const preview = useMemo(
    () => buildExecutionPlan(side, Math.max(1, quantity || 1), stock.current_price, params),
    [params, quantity, side, stock.current_price],
  )

  function clearTimers() {
    timers.current.forEach(timer => window.clearTimeout(timer))
    timers.current = []
  }

  useEffect(() => () => clearTimers(), [])

  useEffect(() => {
    try {
      window.localStorage.setItem(EXECUTION_SETTINGS_STORAGE, JSON.stringify(params))
    } catch {
      // The simulator still works when browser storage is unavailable.
    }
  }, [params])

  useEffect(() => {
    setStrike(Math.round(stock.current_price))
    setPremium(Math.max(0.05, stock.current_price * 0.03))
    setExecution(null)
    setMessage(null)
    clearTimers()
  }, [stock.ticker])

  function patchParams(patch: Partial<ExecutionParameters>) {
    setParams(current => ({ ...current, ...patch }))
  }

  function runPlan(plan: ExecutionPlan, referencePrice: number, orderQuantity: number, executionSide: Side) {
    clearTimers()
    setExecution({
      status: 'working', side: executionSide, quantity: orderQuantity, referencePrice, plan,
      filledQuantity: 0, filledNotional: 0, completedSlices: 0,
    })

    plan.slices.forEach((slice, index) => {
      const timer = window.setTimeout(() => {
        const executionMeta = {
          executionAlgo: params.algo,
          orderType: params.orderType,
          timeInForce: params.timeInForce,
          latencyMs: slice.delayMs,
          sliceCount: plan.slices.length,
        } as const
        const result = executionSide === 'buy'
          ? buyPaperStock({
              ticker: stock.ticker,
              companyName: stock.company_name,
              sector: stock.sector,
              quantity: slice.quantity,
              price: referencePrice,
              fillPrice: slice.expectedPrice,
              execution: executionMeta,
            })
          : sellPaperStock({
              ticker: stock.ticker,
              quantity: slice.quantity,
              price: referencePrice,
              fillPrice: slice.expectedPrice,
              execution: executionMeta,
            })

        if ('error' in result) {
          clearTimers()
          setExecution(current => current ? { ...current, status: 'rejected', message: result.error } : current)
          setMessage({ text: result.error, tone: 'error' })
          return
        }

        setExecution(current => {
          if (!current || current.status !== 'working') return current
          const filledQuantity = current.filledQuantity + slice.quantity
          const filledNotional = current.filledNotional + slice.quantity * slice.expectedPrice
          const finished = index === plan.slices.length - 1 || filledQuantity >= orderQuantity
          if (finished) {
            setMessage({
              text: `${executionSide === 'buy' ? 'Bought' : 'Sold'} ${filledQuantity} ${stock.ticker} at avg $${(filledNotional / filledQuantity).toFixed(2)}`,
              tone: 'ok',
            })
          }
          return {
            ...current,
            status: finished ? 'filled' : 'working',
            filledQuantity,
            filledNotional,
            completedSlices: current.completedSlices + 1,
          }
        })
      }, slice.delayMs)
      timers.current.push(timer)
    })
  }

  function submitStock() {
    setMessage(null)
    const qty = Math.floor(quantity)
    if (!Number.isFinite(qty) || qty <= 0) {
      setMessage({ text: 'Enter a valid share quantity.', tone: 'error' })
      return
    }
    if (side === 'sell' && (stockPosition?.quantity ?? 0) < qty) {
      setMessage({ text: 'Not enough shares to sell.', tone: 'error' })
      return
    }
    const plan = buildExecutionPlan(side, qty, stock.current_price, params)
    const estimatedCost = plan.expectedAveragePrice * qty + account.costModel.stockPerShare * qty
    if (side === 'buy' && params.timeInForce === 'FOK' && estimatedCost > account.cash) {
      setExecution({ status: 'rejected', side, quantity: qty, referencePrice: stock.current_price, plan, filledQuantity: 0, filledNotional: 0, completedSlices: 0, message: 'FOK rejected: insufficient buying power for the full parent order.' })
      return
    }
    if (params.volatilityPct > params.volatilityPausePct) {
      setExecution({ status: 'waiting', side, quantity: qty, referencePrice: stock.current_price, plan, filledQuantity: 0, filledNotional: 0, completedSlices: 0, message: 'Risk pause: simulated volatility exceeds the configured threshold.' })
      return
    }
    if (!orderIsTriggered(side, stock.current_price, params)) {
      if (params.timeInForce !== 'DAY') {
        setExecution({ status: 'rejected', side, quantity: qty, referencePrice: stock.current_price, plan, filledQuantity: 0, filledNotional: 0, completedSlices: 0, message: `${params.timeInForce} order was not immediately marketable.` })
        return
      }
      setExecution({ status: 'waiting', side, quantity: qty, referencePrice: stock.current_price, plan, filledQuantity: 0, filledNotional: 0, completedSlices: 0, message: 'Waiting for the trigger price.' })
      return
    }
    runPlan(plan, stock.current_price, qty, side)
  }

  useEffect(() => {
    if (execution?.status !== 'waiting') return
    if (params.volatilityPct > params.volatilityPausePct) return
    if (!orderIsTriggered(execution.side, stock.current_price, params)) return
    const plan = buildExecutionPlan(execution.side, execution.quantity, stock.current_price, params)
    runPlan(plan, stock.current_price, execution.quantity, execution.side)
  }, [stock.current_price, execution?.status])

  function cancelExecution() {
    clearTimers()
    setExecution(current => current && (current.status === 'waiting' || current.status === 'working')
      ? { ...current, status: 'cancelled', message: 'Remaining quantity cancelled.' }
      : current)
  }

  function submitOption() {
    const result = optionAction === 'buy'
      ? buyPaperOption({ ticker: stock.ticker, companyName: stock.company_name, sector: stock.sector, optionType, strike, expiry, contracts, premium })
      : selectedOption
        ? sellPaperOption({ positionId: selectedOption.id, contracts, premium })
        : { ok: false as const, error: 'No open option contracts for this ticker.' }
    setMessage(!('error' in result)
      ? { text: `${optionAction === 'buy' ? 'Opened' : 'Closed'} ${Math.floor(contracts)} option contract${contracts === 1 ? '' : 's'}.`, tone: 'ok' }
      : { text: result.error, tone: 'error' })
  }

  const expectedNotional = preview.expectedAveragePrice * Math.max(0, quantity)
  const progress = execution ? execution.filledQuantity / Math.max(1, execution.quantity) * 100 : 0

  return (
    <div className="border-b border-border bg-s1/20">
      <div className="flex items-center justify-between gap-2 border-b border-border px-3 py-2">
        <div className="flex items-center gap-2">
          <Activity size={13} className="text-accent" />
          <span className="text-2xs uppercase tracking-[0.2em] text-text">Paper Execution</span>
        </div>
        <div className="text-2xs text-muted tabnum">BP ${Math.round(account.cash).toLocaleString()}</div>
      </div>

      <div className="grid grid-cols-3 border-b border-border bg-bg/40 px-2 pt-1">
        {(['ticket', 'model', 'blotter'] as const).map(value => (
          <button key={value} onClick={() => setPanel(value)} className={`border-b-2 px-2 py-1.5 text-2xs uppercase tracking-wider ${panel === value ? 'border-accent text-text' : 'border-transparent text-dim hover:text-text'}`}>
            {value}
          </button>
        ))}
      </div>

      {panel === 'ticket' && (
        <div className="space-y-3 p-3">
          <div className="grid grid-cols-2 gap-px bg-border">
            {(['stock', 'option'] as const).map(value => (
              <button key={value} onClick={() => setMode(value)} className={`bg-s1 px-2 py-1.5 text-2xs uppercase ${mode === value ? 'text-accent' : 'text-muted'}`}>{value}</button>
            ))}
          </div>

          {mode === 'stock' ? (
            <>
              <div className="grid grid-cols-2 gap-2">
                <div className="grid grid-cols-2 gap-px bg-border">
                  {(['buy', 'sell'] as const).map(value => (
                    <button key={value} onClick={() => setSide(value)} className={`bg-s1 px-2 py-2 text-2xs font-semibold uppercase ${side === value ? (value === 'buy' ? 'text-up' : 'text-down') : 'text-dim'}`}>{value}</button>
                  ))}
                </div>
                <select value={params.orderType} onChange={event => patchParams({ orderType: event.target.value as ExecutionParameters['orderType'] })} className="h-8 border border-border bg-bg px-2 text-xs text-text outline-none focus:border-accent">
                  <option value="market">Market</option><option value="limit">Limit</option><option value="stop">Stop</option><option value="stop-limit">Stop Limit</option>
                </select>
              </div>

              <div className="grid grid-cols-2 gap-2">
                <NumericField label="Shares" value={quantity} min={1} onChange={setQuantity} />
                <label className="block"><span className="mb-1 block text-2xs text-dim">Time in force</span><select value={params.timeInForce} onChange={event => patchParams({ timeInForce: event.target.value as ExecutionParameters['timeInForce'] })} className="h-7 w-full border border-border bg-bg px-2 text-xs text-text outline-none"><option>DAY</option><option>IOC</option><option>FOK</option></select></label>
              </div>

              {(params.orderType === 'limit' || params.orderType === 'stop-limit') && <NumericField label="Limit price" value={params.limitPrice ?? stock.current_price} min={0.01} step={0.01} onChange={value => patchParams({ limitPrice: value })} suffix="USD" />}
              {(params.orderType === 'stop' || params.orderType === 'stop-limit') && <NumericField label="Stop price" value={params.stopPrice ?? stock.current_price} min={0.01} step={0.01} onChange={value => patchParams({ stopPrice: value })} suffix="USD" />}

              <div className="grid grid-cols-[1fr_auto] gap-2">
                <label className="block"><span className="mb-1 block text-2xs text-dim">Execution algo</span><select value={params.algo} onChange={event => patchParams({ algo: event.target.value as ExecutionParameters['algo'] })} className="h-8 w-full border border-border bg-bg px-2 text-xs text-text outline-none focus:border-accent"><option value="smart">Smart Router</option><option value="arrival">Arrival / IS</option><option value="pov">POV</option><option value="twap">TWAP</option><option value="vwap">VWAP</option><option value="open">Opening Auction</option><option value="close">Closing Auction</option><option value="iceberg">Iceberg</option></select></label>
                <button type="button" onClick={() => setAdvancedOpen(value => !value)} className="mt-4 flex h-8 items-center gap-1 border border-border bg-s2 px-2 text-2xs text-muted hover:text-text"><SlidersHorizontal size={12} />Tune {advancedOpen ? <ChevronUp size={11} /> : <ChevronDown size={11} />}</button>
              </div>

              {advancedOpen && (
                <div className="grid grid-cols-2 gap-2 border border-border bg-bg/50 p-2">
                  <NumericField label="Duration" value={params.durationSeconds} onChange={value => patchParams({ durationSeconds: value })} suffix="sec" />
                  <NumericField label="Start delay" value={params.startDelaySeconds} onChange={value => patchParams({ startDelaySeconds: value })} suffix="sec" />
                  <NumericField label="Child slices" value={params.slices} min={1} onChange={value => patchParams({ slices: value })} />
                  {params.algo === 'pov' && <NumericField label="POV target" value={params.participationRate} min={0.1} step={0.5} onChange={value => patchParams({ participationRate: value })} suffix="%" />}
                  {params.algo === 'iceberg' && <NumericField label="Display qty" value={params.displayQuantity} min={1} onChange={value => patchParams({ displayQuantity: value })} />}
                  <NumericField label="Max participation" value={params.maxParticipationRate} min={0.1} step={0.5} onChange={value => patchParams({ maxParticipationRate: value })} suffix="%" />
                  <NumericField label="Max child" value={params.maxChildQuantity} min={1} onChange={value => patchParams({ maxChildQuantity: value })} />
                </div>
              )}

              <div className="grid grid-cols-3 gap-2 border-y border-border py-2 text-2xs">
                <div><span className="block text-dim">Est. avg</span><span className="tabnum text-text">${preview.expectedAveragePrice.toFixed(2)}</span></div>
                <div><span className="block text-dim">Slippage</span><span className="tabnum text-accent">{preview.expectedSlippageBps.toFixed(1)} bp</span></div>
                <div><span className="block text-dim">Notional</span><span className="tabnum text-text">${Math.round(expectedNotional).toLocaleString()}</span></div>
              </div>

              <button type="button" disabled={execution?.status === 'working' || execution?.status === 'waiting'} onClick={submitStock} className={`w-full px-3 py-2.5 text-2xs font-semibold uppercase text-bg disabled:cursor-not-allowed disabled:opacity-40 ${side === 'buy' ? 'bg-up' : 'bg-down'}`}>Route {side} order · {params.algo.toUpperCase()}</button>
              <div className="flex justify-between text-[10px] text-dim"><span>Held {stockPosition?.quantity ?? 0}</span><span>{preview.slices.length} fills over {(preview.expectedDurationMs / 1000).toFixed(1)}s</span></div>
            </>
          ) : (
            <div className="space-y-2">
              <div className="grid grid-cols-2 gap-px bg-border">{(['buy', 'sell'] as const).map(value => <button key={value} onClick={() => setOptionAction(value)} className={`bg-s1 px-2 py-2 text-2xs ${optionAction === value ? 'text-accent' : 'text-dim'}`}>{value === 'buy' ? 'Buy to Open' : 'Sell to Close'}</button>)}</div>
              {optionAction === 'sell' && optionPositions.length > 0 && <select value={selectedOption?.id ?? ''} onChange={event => setClosePositionId(event.target.value)} className="h-8 w-full border border-border bg-bg px-2 text-xs text-text">{optionPositions.map(position => <option key={position.id} value={position.id}>{position.expiry} {position.strike?.toFixed(2)} {position.optionType?.toUpperCase()} x{position.quantity}</option>)}</select>}
              {optionAction === 'buy' && <div className="grid grid-cols-2 gap-2"><label><span className="mb-1 block text-2xs text-dim">Type</span><select value={optionType} onChange={event => setOptionType(event.target.value as PaperOptionType)} className="h-7 w-full border border-border bg-bg px-2 text-xs text-text"><option value="call">Call</option><option value="put">Put</option></select></label><NumericField label="Strike" value={strike} min={0.01} step={0.01} onChange={setStrike} /></div>}
              {optionAction === 'buy' && <label className="block"><span className="mb-1 block text-2xs text-dim">Expiry</span><input type="date" value={expiry} onChange={event => setExpiry(event.target.value)} className="h-7 w-full border border-border bg-bg px-2 text-xs text-text" /></label>}
              <div className="grid grid-cols-2 gap-2"><NumericField label="Contracts" value={contracts} min={1} onChange={setContracts} /><NumericField label="Premium" value={premium} min={0.01} step={0.01} onChange={setPremium} /></div>
              <button onClick={submitOption} className={`w-full px-3 py-2.5 text-2xs font-semibold text-bg ${optionAction === 'buy' ? 'bg-up' : 'bg-down'}`}>{optionAction === 'buy' ? 'Buy Option' : 'Sell Option'}</button>
            </div>
          )}
        </div>
      )}

      {panel === 'model' && (
        <div className="space-y-3 p-3">
          <p className="text-2xs leading-relaxed text-muted">Tune the synthetic venue model. Spread, square-root impact, latency drift, and volatility noise are applied independently to every child order.</p>
          <div className="grid grid-cols-2 gap-2">
            <label className="block"><span className="mb-1 block text-2xs text-dim">Urgency</span><select value={params.urgency} onChange={event => patchParams({ urgency: Number(event.target.value) as ExecutionParameters['urgency'] })} className="h-7 w-full border border-border bg-bg px-2 text-xs text-text"><option value={1}>1 · Passive</option><option value={2}>2 · Patient</option><option value={3}>3 · Neutral</option><option value={4}>4 · Aggressive</option><option value={5}>5 · Get done</option></select></label>
            <label className="block"><span className="mb-1 block text-2xs text-dim">Completion</span><select value={params.completion} onChange={event => patchParams({ completion: event.target.value as ExecutionParameters['completion'] })} className="h-7 w-full border border-border bg-bg px-2 text-xs text-text"><option value="must-complete">Must complete</option><option value="price-sensitive">Do not chase</option><option value="liquidity-only">Liquidity only</option></select></label>
          </div>
          <div className="grid grid-cols-2 gap-2">
            <NumericField label="Spread" value={params.spreadBps} step={0.1} onChange={value => patchParams({ spreadBps: value })} suffix="bp" />
            <NumericField label="Base slip" value={params.baseSlippageBps} step={0.1} onChange={value => patchParams({ baseSlippageBps: value })} suffix="bp" />
            <NumericField label="Impact coeff." value={params.impactCoefficient} step={0.5} onChange={value => patchParams({ impactCoefficient: value })} />
            <NumericField label="Daily vol" value={params.volatilityPct} step={0.1} onChange={value => patchParams({ volatilityPct: value })} suffix="%" />
            <NumericField label="Min latency" value={params.latencyMinMs} onChange={value => patchParams({ latencyMinMs: value })} suffix="ms" />
            <NumericField label="Max latency" value={params.latencyMaxMs} onChange={value => patchParams({ latencyMaxMs: value })} suffix="ms" />
            <NumericField label="Max slippage" value={params.maxSlippageBps} step={0.5} onChange={value => patchParams({ maxSlippageBps: value })} suffix="bp" />
            <NumericField label="Volatility pause" value={params.volatilityPausePct} step={0.1} onChange={value => patchParams({ volatilityPausePct: value })} suffix="%" />
            <NumericField label="Min child" value={params.minChildQuantity} min={1} onChange={value => patchParams({ minChildQuantity: value })} />
            <NumericField label="Min dark fill" value={params.minDarkFill} min={1} onChange={value => patchParams({ minDarkFill: value })} />
            <NumericField label="Dark fill chance" value={params.darkFillProbabilityPct} step={5} onChange={value => patchParams({ darkFillProbabilityPct: value })} suffix="%" />
            <NumericField label="Dark-first window" value={params.darkExposureSeconds} onChange={value => patchParams({ darkExposureSeconds: value })} suffix="sec" />
            <NumericField label="Stealth" value={params.stealthPct} step={5} onChange={value => patchParams({ stealthPct: value })} suffix="%" />
            <div className="col-span-2"><NumericField label="Simulated interval volume" value={params.availableVolume} min={1} onChange={value => patchParams({ availableVolume: value })} suffix="shares" /></div>
          </div>
          <div className="grid grid-cols-2 gap-px bg-border">
            {[['Lit exchanges', 'useLitVenues'], ['Dark pools', 'useDarkPools'], ['Midpoint books', 'useMidpoint'], ['Open/close auction', 'includeAuction']].map(([label, key]) => <label key={key} className="flex items-center justify-between bg-bg px-2 py-2 text-2xs text-muted"><span>{label}</span><input type="checkbox" checked={Boolean(params[key as keyof ExecutionParameters])} onChange={event => patchParams({ [key]: event.target.checked })} /></label>)}
          </div>
          <label className="flex items-center justify-between border border-border bg-bg px-2 py-2 text-2xs text-muted"><span>Randomized venue outcomes</span><input type="checkbox" checked={params.randomize} onChange={event => patchParams({ randomize: event.target.checked })} /></label>
          <button onClick={() => setParams({ ...DEFAULT_EXECUTION_PARAMETERS })} className="w-full border border-border bg-s2 px-2 py-2 text-2xs text-muted hover:text-text">Reset execution model</button>
        </div>
      )}

      {panel === 'blotter' && (
        <div className="p-3">
          {execution && (
            <div className="mb-3 border border-border bg-bg p-2">
              <div className="mb-2 flex items-center justify-between"><span className={`text-2xs font-semibold uppercase ${statusColor(execution.status)}`}>{execution.status}</span>{(execution.status === 'working' || execution.status === 'waiting') && <button onClick={cancelExecution} title="Cancel remaining"><X size={13} className="text-down" /></button>}</div>
              <div className="h-1 bg-border"><div className="h-full bg-accent transition-all" style={{ width: `${Math.min(100, progress)}%` }} /></div>
              <div className="mt-2 flex justify-between text-[10px] text-dim"><span>{execution.filledQuantity}/{execution.quantity} shares</span><span>{execution.completedSlices}/{execution.plan.slices.length} fills</span></div>
              {execution.filledQuantity > 0 && <div className="mt-1 text-[10px] text-muted tabnum">VWAP ${(execution.filledNotional / execution.filledQuantity).toFixed(3)}</div>}
              {execution.message && <p className="mt-1 text-[10px] text-muted">{execution.message}</p>}
              <div className="mt-2 max-h-32 overflow-y-auto border-t border-border pt-1">
                {execution.plan.slices.map((slice, index) => <div key={`${slice.delayMs}-${index}`} className="grid grid-cols-[20px_1fr_auto] gap-1 py-0.5 text-[9px] text-dim"><span>{index + 1}</span><span>{slice.venue} · {slice.instruction} · {slice.quantity.toLocaleString()} sh</span><span className="tabnum">+{(slice.delayMs / 1000).toFixed(1)}s</span></div>)}
              </div>
            </div>
          )}
          <div className="space-y-px">
            {recentOrders.length === 0 && <p className="py-4 text-center text-2xs text-dim">No fills for {stock.ticker}.</p>}
            {recentOrders.map(order => (
              <div key={order.id} className="grid grid-cols-[auto_1fr_auto] items-center gap-2 border-b border-border py-2 text-[10px]">
                <span className={order.action.startsWith('BUY') ? 'text-up' : 'text-down'}>{order.action.replace('_', ' ')}</span>
                <div><span className="block text-text tabnum">{order.quantity} @ ${order.price.toFixed(3)}</span><span className="text-dim">{order.executionAlgo?.toUpperCase() ?? 'DIRECT'} · {order.latencyMs != null ? `${order.latencyMs}ms` : 'instant'}</span></div>
                <span className="text-dim tabnum">{new Date(order.timestamp).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' })}</span>
              </div>
            ))}
          </div>
        </div>
      )}

      {message && <div className={`border-t border-border px-3 py-2 text-2xs ${message.tone === 'ok' ? 'text-up' : 'text-down'}`}>{message.text}</div>}
    </div>
  )
}
