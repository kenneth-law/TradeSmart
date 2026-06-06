import { create } from 'zustand'
import type { AnalysisResult, TickerContext } from '../types'

export type ThemeMode = 'dark' | 'light'
export type FontFamily = 'lato' | 'ibm-plex' | 'inter' | 'system' | 'mono'
export type AccentColor = 'amber' | 'blue' | 'green' | 'rose'
export type InterfaceDensity = 'standard' | 'compact'

export interface SystemSettings {
  theme: ThemeMode
  fontFamily: FontFamily
  fontSizePx: number
  accentColor: AccentColor
  highContrast: boolean
  reduceMotion: boolean
  density: InterfaceDensity
  marketRefreshSeconds: number
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
}

const SETTINGS_COOKIE = 'tradesmart_system_settings'

function isThemeMode(value: unknown): value is ThemeMode {
  return value === 'dark' || value === 'light'
}

function isFontFamily(value: unknown): value is FontFamily {
  return FONT_OPTIONS.some(option => option.value === value)
}

function isAccentColor(value: unknown): value is AccentColor {
  return ACCENT_OPTIONS.some(option => option.value === value)
}

function isDensity(value: unknown): value is InterfaceDensity {
  return value === 'standard' || value === 'compact'
}

function readCookie(name: string) {
  if (typeof document === 'undefined') return null
  const found = document.cookie
    .split('; ')
    .find(cookie => cookie.startsWith(`${name}=`))
  return found ? decodeURIComponent(found.split('=').slice(1).join('=')) : null
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
    marketRefreshSeconds: clamp(raw.marketRefreshSeconds, 30, 900, DEFAULT_SYSTEM_SETTINGS.marketRefreshSeconds),
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

export function applySystemSettings(settings: SystemSettings) {
  if (typeof document === 'undefined') return

  const root = document.documentElement
  const font = FONT_OPTIONS.find(option => option.value === settings.fontFamily) ?? FONT_OPTIONS[0]
  const accent = ACCENT_OPTIONS.find(option => option.value === settings.accentColor) ?? ACCENT_OPTIONS[0]

  root.dataset.theme = settings.theme
  root.dataset.contrast = settings.highContrast ? 'high' : 'standard'
  root.dataset.motion = settings.reduceMotion ? 'reduced' : 'standard'
  root.dataset.density = settings.density
  root.style.setProperty('--app-font-family', font.css)
  root.style.setProperty('--app-font-size', `${settings.fontSizePx}px`)
  root.style.setProperty('--color-accent', accent.color)
  root.style.setProperty('--selection-rgb', accent.rgb)
}

const initialSettings = loadSettings()
applySystemSettings(initialSettings)

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
}))
