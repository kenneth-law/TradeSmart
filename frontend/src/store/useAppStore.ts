import { create } from 'zustand'
import type { AnalysisResult, TickerContext } from '../types'

export type ThemeMode = 'dark' | 'light'
export type FontFamily = 'lato' | 'ibm-plex' | 'inter' | 'system' | 'mono'
export type AccentColor = 'amber' | 'blue' | 'green' | 'rose' | 'custom'
export type InterfaceDensity = 'standard' | 'compact'
export type OpenAIModel = 'gpt-5.4-nano' | 'gpt-5.4-mini' | 'gpt-4.1' | 'gpt-5.5' | 'gpt-5.3-codex-spark'
export type PaperInstrumentKind = 'stock' | 'option'
export type PaperOptionType = 'call' | 'put'
export type PaperOrderAction = 'BUY' | 'SELL' | 'BUY_OPTION' | 'SELL_OPTION'
export type FinancialsDepth = 'concise' | 'full'
export type BrokerPreset = 'ibkr' | 'retail' | 'conservative' | 'custom'

export interface PaperCostModel {
  stockPerShare: number
  optionPerContract: number
  slippageBps: number
}

export interface SystemSettings {
  theme: ThemeMode
  fontFamily: FontFamily
  fontSizePx: number
  accentColor: AccentColor
  highContrast: boolean
  reduceMotion: boolean
  density: InterfaceDensity
  marketRefreshSeconds: number
  openaiModel: OpenAIModel
  briefWebSearch: boolean
  fastPaperRefresh: boolean
  openaiTemperature: number
  openaiSystemPrompt: string
  financialsDepth: FinancialsDepth
  accentCustomColor: string
  backtestBrokerPreset: BrokerPreset
  backtestTxCost: string
  backtestTxCostType: 'percent' | 'fixed'
}

export interface PaperPosition {
  id: string
  kind: PaperInstrumentKind
  ticker: string
  companyName?: string
  sector?: string
  quantity: number
  costBasis: number
  openedAt: string
  optionType?: PaperOptionType
  strike?: number
  expiry?: string
}

export interface PaperOrder {
  id: string
  timestamp: string
  action: PaperOrderAction
  ticker: string
  quantity: number
  price: number
  requestedPrice?: number
  notional: number
  fee?: number
  slippage?: number
  realizedPnl?: number
  optionType?: PaperOptionType
  strike?: number
  expiry?: string
}

export interface PaperAccount {
  cash: number
  initialCash: number
  positions: PaperPosition[]
  orders: PaperOrder[]
  costModel: PaperCostModel
}

export type PaperTradeResult = { ok: true } | { ok: false; error: string }

export function isReasoningModel(model: string): boolean {
  return model === 'gpt-5.5'
}

export function maxOutputTokensFor(model: string): number {
  if (model === 'gpt-5.3-codex-spark') return 16000
  return 128000
}

export const OPENAI_MODEL_OPTIONS: Array<{ value: OpenAIModel; label: string }> = [
  { value: 'gpt-5.4-nano', label: 'gpt-5.4-nano (fastest)' },
  { value: 'gpt-5.4-mini', label: 'gpt-5.4-mini (fast, cheap)' },
  { value: 'gpt-4.1', label: 'gpt-4.1 (stronger)' },
  { value: 'gpt-5.5', label: 'gpt-5.5 (reasoning)' },
  { value: 'gpt-5.3-codex-spark', label: 'gpt-5.3-codex-spark (code)' },
]

const LEGACY_MODEL_MIGRATION: Record<string, OpenAIModel> = {
  'gpt-4.1-mini': 'gpt-5.4-mini',
  'gpt-4o-mini': 'gpt-5.4-mini',
  'gpt-4o': 'gpt-4.1',
  'gpt-4-turbo': 'gpt-4.1',
}

export const FONT_OPTIONS: Array<{ value: FontFamily; label: string; css: string }> = [
  { value: 'lato', label: 'Lato', css: "'Lato', sans-serif" },
  { value: 'ibm-plex', label: 'IBM Plex Sans', css: "'IBM Plex Sans', 'Lato', sans-serif" },
  { value: 'inter', label: 'Inter', css: "'Inter', 'Lato', sans-serif" },
  { value: 'system', label: 'System UI', css: "system-ui, -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif" },
  { value: 'mono', label: 'Mono', css: "'IBM Plex Mono', 'SFMono-Regular', Consolas, monospace" },
]

export const ACCENT_OPTIONS: Array<{ value: AccentColor; label: string; color: string; rgb: string }> = [
  { value: 'amber', label: 'Amber', color: '#E89B2C', rgb: '232, 155, 44' },
  { value: 'blue', label: 'Blue', color: '#3B82F6', rgb: '59, 130, 246' },
  { value: 'green', label: 'Green', color: '#10B981', rgb: '16, 185, 129' },
  { value: 'rose', label: 'Rose', color: '#F43F5E', rgb: '244, 63, 94' },
]

export const DEFAULT_SYSTEM_SETTINGS: SystemSettings = {
  theme: 'dark',
  fontFamily: 'lato',
  fontSizePx: 17,
  accentColor: 'amber',
  highContrast: false,
  reduceMotion: false,
  density: 'standard',
  marketRefreshSeconds: 300,
  openaiModel: 'gpt-5.4-nano',
  briefWebSearch: false,
  fastPaperRefresh: false,
  openaiTemperature: 0.4,
  openaiSystemPrompt: '',
  financialsDepth: 'concise',
  accentCustomColor: '#E89B2C',
  backtestBrokerPreset: 'retail',
  backtestTxCost: '0.02',
  backtestTxCostType: 'percent',
}

export const SETTINGS_COOKIE = 'tradesmart_system_settings'
const OPENAI_KEY_STORAGE = 'tradesmart_openai_key'
const PRODIA_KEY_STORAGE = 'tradesmart_prodia_key'
const PAPER_ACCOUNT_STORAGE = 'tradesmart_paper_account'
const DEFAULT_PAPER_CASH = 100000
export const DEFAULT_PAPER_COST_MODEL: PaperCostModel = {
  stockPerShare: 0,
  optionPerContract: 0,
  slippageBps: 0,
}

function isThemeMode(value: unknown): value is ThemeMode {
  return value === 'dark' || value === 'light'
}

function isFontFamily(value: unknown): value is FontFamily {
  return FONT_OPTIONS.some(option => option.value === value)
}

function isAccentColor(value: unknown): value is AccentColor {
  return value === 'custom' || ACCENT_OPTIONS.some(option => option.value === value)
}

function isDensity(value: unknown): value is InterfaceDensity {
  return value === 'standard' || value === 'compact'
}

function isOpenAIModel(value: unknown): value is OpenAIModel {
  return OPENAI_MODEL_OPTIONS.some(option => option.value === value)
}

function coerceOpenAIModel(value: unknown): OpenAIModel {
  if (isOpenAIModel(value)) return value
  if (typeof value === 'string' && value in LEGACY_MODEL_MIGRATION) {
    return LEGACY_MODEL_MIGRATION[value]
  }
  return DEFAULT_SYSTEM_SETTINGS.openaiModel
}

function readCookie(name: string) {
  if (typeof document === 'undefined') return null
  const found = document.cookie
    .split('; ')
    .find(cookie => cookie.startsWith(`${name}=`))
  return found ? decodeURIComponent(found.split('=').slice(1).join('=')) : null
}

export function hasSystemSettingsCookie() {
  return Boolean(readCookie(SETTINGS_COOKIE))
}

function writeCookie(name: string, value: string) {
  if (typeof document === 'undefined') return
  const maxAge = 60 * 60 * 24 * 365
  document.cookie = `${name}=${encodeURIComponent(value)}; path=/; max-age=${maxAge}; SameSite=Lax`
}

function clamp(value: unknown, min: number, max: number, fallback: number) {
  const numberValue = typeof value === 'number' ? value : Number(value)
  if (!Number.isFinite(numberValue)) return fallback
  return Math.min(max, Math.max(min, Math.round(numberValue)))
}

function clampFloat(value: unknown, min: number, max: number, fallback: number) {
  const numberValue = typeof value === 'number' ? value : Number(value)
  if (!Number.isFinite(numberValue)) return fallback
  return Math.min(max, Math.max(min, numberValue))
}

function normaliseSettings(value: unknown): SystemSettings {
  const raw = value && typeof value === 'object'
    ? value as Partial<SystemSettings>
    : {}

  return {
    theme: isThemeMode(raw.theme) ? raw.theme : DEFAULT_SYSTEM_SETTINGS.theme,
    fontFamily: isFontFamily(raw.fontFamily) ? raw.fontFamily : DEFAULT_SYSTEM_SETTINGS.fontFamily,
    fontSizePx: clamp(raw.fontSizePx, 14, 22, DEFAULT_SYSTEM_SETTINGS.fontSizePx),
    accentColor: isAccentColor(raw.accentColor) ? raw.accentColor : DEFAULT_SYSTEM_SETTINGS.accentColor,
    highContrast: typeof raw.highContrast === 'boolean' ? raw.highContrast : DEFAULT_SYSTEM_SETTINGS.highContrast,
    reduceMotion: typeof raw.reduceMotion === 'boolean' ? raw.reduceMotion : DEFAULT_SYSTEM_SETTINGS.reduceMotion,
    density: isDensity(raw.density) ? raw.density : DEFAULT_SYSTEM_SETTINGS.density,
    marketRefreshSeconds: clamp(raw.marketRefreshSeconds, 5, 900, DEFAULT_SYSTEM_SETTINGS.marketRefreshSeconds),
    openaiModel: coerceOpenAIModel(raw.openaiModel),
    briefWebSearch: typeof raw.briefWebSearch === 'boolean' ? raw.briefWebSearch : DEFAULT_SYSTEM_SETTINGS.briefWebSearch,
    fastPaperRefresh: typeof raw.fastPaperRefresh === 'boolean' ? raw.fastPaperRefresh : DEFAULT_SYSTEM_SETTINGS.fastPaperRefresh,
    openaiTemperature: clampFloat(raw.openaiTemperature, 0, 1.5, DEFAULT_SYSTEM_SETTINGS.openaiTemperature),
    openaiSystemPrompt: typeof raw.openaiSystemPrompt === 'string' ? raw.openaiSystemPrompt : DEFAULT_SYSTEM_SETTINGS.openaiSystemPrompt,
    financialsDepth: raw.financialsDepth === 'full' || raw.financialsDepth === 'concise' ? raw.financialsDepth : DEFAULT_SYSTEM_SETTINGS.financialsDepth,
    accentCustomColor: typeof raw.accentCustomColor === 'string' && /^#[0-9a-fA-F]{6}$/.test(raw.accentCustomColor) ? raw.accentCustomColor : DEFAULT_SYSTEM_SETTINGS.accentCustomColor,
    backtestBrokerPreset: (['ibkr', 'retail', 'conservative', 'custom'] as BrokerPreset[]).includes(raw.backtestBrokerPreset as BrokerPreset) ? raw.backtestBrokerPreset as BrokerPreset : DEFAULT_SYSTEM_SETTINGS.backtestBrokerPreset,
    backtestTxCost: typeof raw.backtestTxCost === 'string' ? raw.backtestTxCost : DEFAULT_SYSTEM_SETTINGS.backtestTxCost,
    backtestTxCostType: (raw.backtestTxCostType === 'percent' || raw.backtestTxCostType === 'fixed') ? raw.backtestTxCostType : DEFAULT_SYSTEM_SETTINGS.backtestTxCostType,
  }
}

export function loadOpenAIKey(): string {
  if (typeof window === 'undefined') return ''
  try {
    return window.localStorage.getItem(OPENAI_KEY_STORAGE) ?? ''
  } catch {
    return ''
  }
}

export function saveOpenAIKey(key: string) {
  if (typeof window === 'undefined') return
  try {
    if (key) window.localStorage.setItem(OPENAI_KEY_STORAGE, key)
    else window.localStorage.removeItem(OPENAI_KEY_STORAGE)
  } catch {
    // ignore quota / disabled storage
  }
}

export function loadProdiaKey(): string {
  if (typeof window === 'undefined') return ''
  try {
    return window.localStorage.getItem(PRODIA_KEY_STORAGE) ?? ''
  } catch {
    return ''
  }
}

export function saveProdiaKey(key: string) {
  if (typeof window === 'undefined') return
  try {
    if (key) window.localStorage.setItem(PRODIA_KEY_STORAGE, key)
    else window.localStorage.removeItem(PRODIA_KEY_STORAGE)
  } catch {
    // ignore quota / disabled storage
  }
}

function loadSettings() {
  const cookieValue = readCookie(SETTINGS_COOKIE)
  if (!cookieValue) return DEFAULT_SYSTEM_SETTINGS
  try {
    return normaliseSettings(JSON.parse(cookieValue))
  } catch {
    return DEFAULT_SYSTEM_SETTINGS
  }
}

function persistSettings(settings: SystemSettings) {
  writeCookie(SETTINGS_COOKIE, JSON.stringify(settings))
}

function newId(prefix: string) {
  if (typeof crypto !== 'undefined' && 'randomUUID' in crypto) {
    return `${prefix}_${crypto.randomUUID()}`
  }
  return `${prefix}_${Date.now()}_${Math.random().toString(36).slice(2)}`
}

function normalizePaperPosition(value: unknown): PaperPosition | null {
  if (!value || typeof value !== 'object') return null
  const raw = value as Partial<PaperPosition>
  const ticker = typeof raw.ticker === 'string' ? raw.ticker.trim().toUpperCase() : ''
  const quantity = Number(raw.quantity)
  const costBasis = Number(raw.costBasis)
  if (!ticker || !Number.isFinite(quantity) || quantity <= 0 || !Number.isFinite(costBasis) || costBasis < 0) {
    return null
  }
  const kind: PaperInstrumentKind = raw.kind === 'option' ? 'option' : 'stock'
  return {
    id: typeof raw.id === 'string' && raw.id ? raw.id : newId('pos'),
    kind,
    ticker,
    companyName: typeof raw.companyName === 'string' ? raw.companyName : undefined,
    sector: typeof raw.sector === 'string' ? raw.sector : undefined,
    quantity,
    costBasis,
    openedAt: typeof raw.openedAt === 'string' ? raw.openedAt : new Date().toISOString(),
    optionType: raw.optionType === 'put' ? 'put' : raw.optionType === 'call' ? 'call' : undefined,
    strike: Number.isFinite(Number(raw.strike)) ? Number(raw.strike) : undefined,
    expiry: typeof raw.expiry === 'string' ? raw.expiry : undefined,
  }
}

function normalizePaperOrder(value: unknown): PaperOrder | null {
  if (!value || typeof value !== 'object') return null
  const raw = value as Partial<PaperOrder>
  const ticker = typeof raw.ticker === 'string' ? raw.ticker.trim().toUpperCase() : ''
  const quantity = Number(raw.quantity)
  const price = Number(raw.price)
  const notional = Number(raw.notional)
  if (!ticker || !Number.isFinite(quantity) || quantity <= 0 || !Number.isFinite(price) || price < 0) {
    return null
  }
  const actions: PaperOrderAction[] = ['BUY', 'SELL', 'BUY_OPTION', 'SELL_OPTION']
  return {
    id: typeof raw.id === 'string' && raw.id ? raw.id : newId('ord'),
    timestamp: typeof raw.timestamp === 'string' ? raw.timestamp : new Date().toISOString(),
    action: actions.includes(raw.action as PaperOrderAction) ? raw.action as PaperOrderAction : 'BUY',
    ticker,
    quantity,
    price,
    requestedPrice: Number.isFinite(Number(raw.requestedPrice)) ? Number(raw.requestedPrice) : undefined,
    notional: Number.isFinite(notional) ? notional : quantity * price,
    fee: Number.isFinite(Number(raw.fee)) ? Math.max(0, Number(raw.fee)) : undefined,
    slippage: Number.isFinite(Number(raw.slippage)) ? Number(raw.slippage) : undefined,
    realizedPnl: Number.isFinite(Number(raw.realizedPnl)) ? Number(raw.realizedPnl) : undefined,
    optionType: raw.optionType === 'put' ? 'put' : raw.optionType === 'call' ? 'call' : undefined,
    strike: Number.isFinite(Number(raw.strike)) ? Number(raw.strike) : undefined,
    expiry: typeof raw.expiry === 'string' ? raw.expiry : undefined,
  }
}

function normalizePaperCostModel(value: unknown): PaperCostModel {
  const raw = value && typeof value === 'object'
    ? value as Partial<PaperCostModel>
    : {}
  return {
    stockPerShare: clampFloat(raw.stockPerShare, 0, 100, DEFAULT_PAPER_COST_MODEL.stockPerShare),
    optionPerContract: clampFloat(raw.optionPerContract, 0, 100, DEFAULT_PAPER_COST_MODEL.optionPerContract),
    slippageBps: clampFloat(raw.slippageBps, 0, 1000, DEFAULT_PAPER_COST_MODEL.slippageBps),
  }
}

function normalizePaperAccount(value: unknown): PaperAccount {
  const raw = value && typeof value === 'object'
    ? value as Partial<PaperAccount>
    : {}
  const initialCash = clampFloat(raw.initialCash, 0, 1000000000, DEFAULT_PAPER_CASH)
  const cash = clampFloat(raw.cash, 0, 1000000000, initialCash)
  const positions = Array.isArray(raw.positions)
    ? raw.positions.map(normalizePaperPosition).filter((p): p is PaperPosition => Boolean(p))
    : []
  const orders = Array.isArray(raw.orders)
    ? raw.orders.map(normalizePaperOrder).filter((o): o is PaperOrder => Boolean(o)).slice(0, 200)
    : []
  const costModel = normalizePaperCostModel(raw.costModel)
  return { cash, initialCash, positions, orders, costModel }
}

function defaultPaperAccount(cash = DEFAULT_PAPER_CASH): PaperAccount {
  return { cash, initialCash: cash, positions: [], orders: [], costModel: DEFAULT_PAPER_COST_MODEL }
}

function loadPaperAccount(): PaperAccount {
  if (typeof window === 'undefined') {
    return defaultPaperAccount()
  }
  try {
    const raw = window.localStorage.getItem(PAPER_ACCOUNT_STORAGE)
    return raw ? normalizePaperAccount(JSON.parse(raw)) : defaultPaperAccount()
  } catch {
    return defaultPaperAccount()
  }
}

function persistPaperAccount(account: PaperAccount) {
  if (typeof window === 'undefined') return
  try {
    window.localStorage.setItem(PAPER_ACCOUNT_STORAGE, JSON.stringify(account))
  } catch {
    // ignore quota / disabled storage
  }
}

function stockFill(costModel: PaperCostModel, side: 'buy' | 'sell', quantity: number, price: number) {
  const slippageRate = costModel.slippageBps / 10000
  const fillPrice = price * (side === 'buy' ? 1 + slippageRate : 1 - slippageRate)
  const notional = fillPrice * quantity
  const fee = costModel.stockPerShare * quantity
  return {
    fillPrice,
    notional,
    fee,
    slippage: (fillPrice - price) * quantity,
  }
}

function optionFill(costModel: PaperCostModel, side: 'buy' | 'sell', contracts: number, premium: number) {
  const slippageRate = costModel.slippageBps / 10000
  const fillPrice = premium * (side === 'buy' ? 1 + slippageRate : 1 - slippageRate)
  const notional = fillPrice * contracts * 100
  const fee = costModel.optionPerContract * contracts
  return {
    fillPrice,
    notional,
    fee,
    slippage: (fillPrice - premium) * contracts * 100,
  }
}

function hexToRgb(hex: string): string {
  const r = parseInt(hex.slice(1, 3), 16)
  const g = parseInt(hex.slice(3, 5), 16)
  const b = parseInt(hex.slice(5, 7), 16)
  return `${r}, ${g}, ${b}`
}

export function applySystemSettings(settings: SystemSettings) {
  if (typeof document === 'undefined') return

  const root = document.documentElement
  const font = FONT_OPTIONS.find(option => option.value === settings.fontFamily) ?? FONT_OPTIONS[0]

  let accentHex: string
  let accentRgb: string
  if (settings.accentColor === 'custom') {
    accentHex = settings.accentCustomColor
    accentRgb = hexToRgb(accentHex)
  } else {
    const accent = ACCENT_OPTIONS.find(option => option.value === settings.accentColor) ?? ACCENT_OPTIONS[0]
    accentHex = accent.color
    accentRgb = accent.rgb
  }

  root.dataset.theme = settings.theme
  root.dataset.contrast = settings.highContrast ? 'high' : 'standard'
  root.dataset.motion = settings.reduceMotion ? 'reduced' : 'standard'
  root.dataset.density = settings.density
  root.style.setProperty('--app-font-family', font.css)
  root.style.setProperty('--app-font-size', `${settings.fontSizePx}px`)
  root.style.setProperty('--color-accent', accentHex)
  root.style.setProperty('--selection-rgb', accentRgb)
}

const initialSettings = loadSettings()
applySystemSettings(initialSettings)
const initialOpenAIKey = loadOpenAIKey()
const initialProdiaKey = loadProdiaKey()
const initialPaperAccount = loadPaperAccount()

interface AppStore {
  analysisResult: AnalysisResult | null
  setAnalysisResult: (r: AnalysisResult) => void

  tickerContext: TickerContext | null
  setTickerContext: (ctx: TickerContext) => void

  lastUpdated: Date | null
  setLastUpdated: (d: Date) => void

  connection: 'connected' | 'degraded' | 'offline'
  setConnection: (s: 'connected' | 'degraded' | 'offline') => void

  settings: SystemSettings
  setSettings: (patch: Partial<SystemSettings>) => void
  resetSettings: () => void

  openaiKey: string
  setOpenAIKey: (key: string) => void

  prodiaKey: string
  setProdiaKey: (key: string) => void

  paperAccount: PaperAccount
  setPaperCash: (cash: number) => void
  setPaperCostModel: (patch: Partial<PaperCostModel>) => void
  resetPaperAccount: (cash?: number) => void
  buyPaperStock: (order: {
    ticker: string
    companyName?: string
    sector?: string
    quantity: number
    price: number
  }) => PaperTradeResult
  sellPaperStock: (order: { ticker: string; quantity: number; price: number }) => PaperTradeResult
  buyPaperOption: (order: {
    ticker: string
    companyName?: string
    sector?: string
    optionType: PaperOptionType
    strike: number
    expiry: string
    contracts: number
    premium: number
  }) => PaperTradeResult
  sellPaperOption: (order: { positionId: string; contracts: number; premium: number }) => PaperTradeResult
}

export const useAppStore = create<AppStore>((set) => ({
  analysisResult: null,
  setAnalysisResult: (r) => set({ analysisResult: r, lastUpdated: new Date() }),

  tickerContext: null,
  setTickerContext: (ctx) => set({ tickerContext: ctx }),

  lastUpdated: null,
  setLastUpdated: (d) => set({ lastUpdated: d }),

  connection: 'connected',
  setConnection: (s) => set({ connection: s }),

  settings: initialSettings,
  setSettings: (patch) => set((state) => {
    const settings = normaliseSettings({ ...state.settings, ...patch })
    applySystemSettings(settings)
    persistSettings(settings)
    return { settings }
  }),
  resetSettings: () => {
    const settings = DEFAULT_SYSTEM_SETTINGS
    applySystemSettings(settings)
    persistSettings(settings)
    set({ settings })
  },

  openaiKey: initialOpenAIKey,
  setOpenAIKey: (key) => {
    const trimmed = key.trim()
    saveOpenAIKey(trimmed)
    set({ openaiKey: trimmed })
  },

  prodiaKey: initialProdiaKey,
  setProdiaKey: (key) => {
    const trimmed = key.trim()
    saveProdiaKey(trimmed)
    set({ prodiaKey: trimmed })
  },

  paperAccount: initialPaperAccount,
  setPaperCash: (cash) => set((state) => {
    const value = clampFloat(cash, 0, 1000000000, state.paperAccount.cash)
    const cashDelta = value - state.paperAccount.cash
    const initialCash = state.paperAccount.positions.length === 0
      ? value
      : Math.max(0, state.paperAccount.initialCash + cashDelta)
    const account = { ...state.paperAccount, cash: value, initialCash }
    persistPaperAccount(account)
    return { paperAccount: account }
  }),
  setPaperCostModel: (patch) => set((state) => {
    const account = {
      ...state.paperAccount,
      costModel: normalizePaperCostModel({ ...state.paperAccount.costModel, ...patch }),
    }
    persistPaperAccount(account)
    return { paperAccount: account }
  }),
  resetPaperAccount: (cash = DEFAULT_PAPER_CASH) => set((state) => {
    const value = clampFloat(cash, 0, 1000000000, DEFAULT_PAPER_CASH)
    const account = { ...defaultPaperAccount(value), costModel: state.paperAccount.costModel }
    persistPaperAccount(account)
    return { paperAccount: account }
  }),
  buyPaperStock: ({ ticker, companyName, sector, quantity, price }) => {
    const t = ticker.trim().toUpperCase()
    const qty = Math.floor(Number(quantity))
    const px = Number(price)
    if (!t || qty <= 0 || !Number.isFinite(px) || px <= 0) return { ok: false, error: 'Enter a valid stock order.' }
    let result: PaperTradeResult = { ok: true }
    set((state) => {
      const fill = stockFill(state.paperAccount.costModel, 'buy', qty, px)
      const totalCost = fill.notional + fill.fee
      if (totalCost > state.paperAccount.cash) {
        result = { ok: false, error: 'Insufficient paper cash.' }
        return state
      }
      const now = new Date().toISOString()
      const positions = [...state.paperAccount.positions]
      const idx = positions.findIndex(p => p.kind === 'stock' && p.ticker === t)
      if (idx >= 0) {
        const existing = positions[idx]
        const nextQty = existing.quantity + qty
        positions[idx] = {
          ...existing,
          companyName: companyName ?? existing.companyName,
          sector: sector ?? existing.sector,
          quantity: nextQty,
          costBasis: ((existing.costBasis * existing.quantity) + totalCost) / nextQty,
        }
      } else {
        positions.push({
          id: newId('pos'),
          kind: 'stock',
          ticker: t,
          companyName,
          sector,
          quantity: qty,
          costBasis: totalCost / qty,
          openedAt: now,
        })
      }
      const account = {
        ...state.paperAccount,
        cash: state.paperAccount.cash - totalCost,
        positions,
        orders: [{
          id: newId('ord'),
          timestamp: now,
          action: 'BUY' as const,
          ticker: t,
          quantity: qty,
          price: fill.fillPrice,
          requestedPrice: px,
          notional: fill.notional,
          fee: fill.fee,
          slippage: fill.slippage,
        }, ...state.paperAccount.orders].slice(0, 200),
      }
      persistPaperAccount(account)
      return { paperAccount: account }
    })
    return result
  },
  sellPaperStock: ({ ticker, quantity, price }) => {
    const t = ticker.trim().toUpperCase()
    const qty = Math.floor(Number(quantity))
    const px = Number(price)
    if (!t || qty <= 0 || !Number.isFinite(px) || px <= 0) return { ok: false, error: 'Enter a valid stock order.' }
    let result: PaperTradeResult = { ok: true }
    set((state) => {
      const existing = state.paperAccount.positions.find(p => p.kind === 'stock' && p.ticker === t)
      if (!existing || existing.quantity < qty) {
        result = { ok: false, error: 'Not enough shares to sell.' }
        return state
      }
      const fill = stockFill(state.paperAccount.costModel, 'sell', qty, px)
      const proceeds = fill.notional - fill.fee
      const realizedPnl = proceeds - (existing.costBasis * qty)
      const positions = state.paperAccount.positions
        .map(p => p.id === existing.id ? { ...p, quantity: p.quantity - qty } : p)
        .filter(p => p.quantity > 0)
      const account = {
        ...state.paperAccount,
        cash: state.paperAccount.cash + proceeds,
        positions,
        orders: [{
          id: newId('ord'),
          timestamp: new Date().toISOString(),
          action: 'SELL' as const,
          ticker: t,
          quantity: qty,
          price: fill.fillPrice,
          requestedPrice: px,
          notional: fill.notional,
          fee: fill.fee,
          slippage: fill.slippage,
          realizedPnl,
        }, ...state.paperAccount.orders].slice(0, 200),
      }
      persistPaperAccount(account)
      return { paperAccount: account }
    })
    return result
  },
  buyPaperOption: ({ ticker, companyName, sector, optionType, strike, expiry, contracts, premium }) => {
    const t = ticker.trim().toUpperCase()
    const qty = Math.floor(Number(contracts))
    const px = Number(premium)
    const k = Number(strike)
    if (!t || qty <= 0 || !Number.isFinite(px) || px <= 0 || !Number.isFinite(k) || k <= 0 || !expiry) {
      return { ok: false, error: 'Enter valid option contract details.' }
    }
    let result: PaperTradeResult = { ok: true }
    set((state) => {
      const fill = optionFill(state.paperAccount.costModel, 'buy', qty, px)
      const totalCost = fill.notional + fill.fee
      if (totalCost > state.paperAccount.cash) {
        result = { ok: false, error: 'Insufficient paper cash.' }
        return state
      }
      const now = new Date().toISOString()
      const positions = [...state.paperAccount.positions]
      const idx = positions.findIndex(p =>
        p.kind === 'option' &&
        p.ticker === t &&
        p.optionType === optionType &&
        p.strike === k &&
        p.expiry === expiry
      )
      if (idx >= 0) {
        const existing = positions[idx]
        const nextQty = existing.quantity + qty
        positions[idx] = {
          ...existing,
          companyName: companyName ?? existing.companyName,
          sector: sector ?? existing.sector,
          quantity: nextQty,
          costBasis: ((existing.costBasis * existing.quantity) + (totalCost / 100)) / nextQty,
        }
      } else {
        positions.push({
          id: newId('pos'),
          kind: 'option',
          ticker: t,
          companyName,
          sector,
          quantity: qty,
          costBasis: totalCost / qty / 100,
          openedAt: now,
          optionType,
          strike: k,
          expiry,
        })
      }
      const account = {
        ...state.paperAccount,
        cash: state.paperAccount.cash - totalCost,
        positions,
        orders: [{
          id: newId('ord'),
          timestamp: now,
          action: 'BUY_OPTION' as const,
          ticker: t,
          quantity: qty,
          price: fill.fillPrice,
          requestedPrice: px,
          notional: fill.notional,
          fee: fill.fee,
          slippage: fill.slippage,
          optionType,
          strike: k,
          expiry,
        }, ...state.paperAccount.orders].slice(0, 200),
      }
      persistPaperAccount(account)
      return { paperAccount: account }
    })
    return result
  },
  sellPaperOption: ({ positionId, contracts, premium }) => {
    const qty = Math.floor(Number(contracts))
    const px = Number(premium)
    if (!positionId || qty <= 0 || !Number.isFinite(px) || px <= 0) return { ok: false, error: 'Enter a valid option close order.' }
    let result: PaperTradeResult = { ok: true }
    set((state) => {
      const existing = state.paperAccount.positions.find(p => p.id === positionId && p.kind === 'option')
      if (!existing || existing.quantity < qty) {
        result = { ok: false, error: 'Not enough contracts to close.' }
        return state
      }
      const fill = optionFill(state.paperAccount.costModel, 'sell', qty, px)
      const proceeds = fill.notional - fill.fee
      const realizedPnl = proceeds - (existing.costBasis * qty * 100)
      const positions = state.paperAccount.positions
        .map(p => p.id === existing.id ? { ...p, quantity: p.quantity - qty } : p)
        .filter(p => p.quantity > 0)
      const account = {
        ...state.paperAccount,
        cash: state.paperAccount.cash + proceeds,
        positions,
        orders: [{
          id: newId('ord'),
          timestamp: new Date().toISOString(),
          action: 'SELL_OPTION' as const,
          ticker: existing.ticker,
          quantity: qty,
          price: fill.fillPrice,
          requestedPrice: px,
          notional: fill.notional,
          fee: fill.fee,
          slippage: fill.slippage,
          realizedPnl,
          optionType: existing.optionType,
          strike: existing.strike,
          expiry: existing.expiry,
        }, ...state.paperAccount.orders].slice(0, 200),
      }
      persistPaperAccount(account)
      return { paperAccount: account }
    })
    return result
  },
}))
