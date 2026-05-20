import type { ReactNode } from 'react'
import {
  Check,
  Eye,
  Gauge,
  Moon,
  Palette,
  RefreshCw,
  RotateCcw,
  Settings as SettingsIcon,
  Sun,
  Type,
  Zap,
} from 'lucide-react'
import {
  ACCENT_OPTIONS,
  FONT_OPTIONS,
  useAppStore,
} from '../store/useAppStore'
import type {
  AccentColor,
  FontFamily,
  InterfaceDensity,
  SystemSettings,
  ThemeMode,
} from '../store/useAppStore'

const REFRESH_OPTIONS = [
  { value: 60, label: '1 min' },
  { value: 180, label: '3 min' },
  { value: 300, label: '5 min' },
  { value: 900, label: '15 min' },
]

function SettingRow({
  icon,
  title,
  caption,
  children,
}: {
  icon: ReactNode
  title: string
  caption: string
  children: ReactNode
}) {
  return (
    <section className="grid grid-cols-[minmax(0,1fr)_minmax(280px,420px)] gap-4 border-b border-border px-4 py-4 max-lg:grid-cols-1">
      <div className="flex gap-3">
        <div className="mt-0.5 flex h-8 w-8 shrink-0 items-center justify-center border border-border bg-s2 text-accent">
          {icon}
        </div>
        <div>
          <h2 className="text-sm font-medium text-text">{title}</h2>
          <p className="mt-1 max-w-xl text-2xs text-muted">{caption}</p>
        </div>
      </div>
      <div className="flex items-center justify-start gap-2 max-sm:flex-wrap">
        {children}
      </div>
    </section>
  )
}

function SegmentButton<T extends string>({
  value,
  selected,
  onSelect,
  children,
}: {
  value: T
  selected: boolean
  onSelect: (value: T) => void
  children: ReactNode
}) {
  return (
    <button
      type="button"
      onClick={() => onSelect(value)}
      className={[
        'flex min-h-11 items-center gap-2 border px-3 text-2xs font-medium transition-colors',
        selected
          ? 'border-accent bg-s2 text-text'
          : 'border-border bg-s1 text-muted hover:border-border-strong hover:text-text',
      ].join(' ')}
      aria-pressed={selected}
    >
      {children}
    </button>
  )
}

function Toggle({
  checked,
  onChange,
  label,
}: {
  checked: boolean
  onChange: (checked: boolean) => void
  label: string
}) {
  return (
    <button
      type="button"
      onClick={() => onChange(!checked)}
      className={[
        'relative h-6 w-11 border transition-colors',
        checked ? 'border-accent bg-accent' : 'border-border-strong bg-s2',
      ].join(' ')}
      role="switch"
      aria-checked={checked}
      aria-label={label}
      title={label}
    >
      <span
        className={[
          'absolute top-0.5 h-5 w-5 bg-text transition-transform',
          checked ? 'translate-x-5' : 'translate-x-0.5',
        ].join(' ')}
      />
    </button>
  )
}

export default function Settings() {
  const settings = useAppStore(s => s.settings)
  const setSettings = useAppStore(s => s.setSettings)
  const resetSettings = useAppStore(s => s.resetSettings)

  function update(patch: Partial<SystemSettings>) {
    setSettings(patch)
  }

  return (
    <div className="flex h-full flex-col bg-bg">
      <div className="flex shrink-0 items-center gap-3 border-b border-border bg-s1 px-4 py-3">
        <SettingsIcon size={17} className="text-accent" aria-hidden="true" />
        <div>
          <h1 className="text-sm font-medium text-text">System Settings</h1>
          <p className="text-2xs text-muted">Global preferences are saved in this browser.</p>
        </div>
        <button
          type="button"
          onClick={resetSettings}
          className="ml-auto flex min-h-11 items-center gap-2 border border-border bg-s2 px-3 text-2xs text-muted hover:border-border-strong hover:text-text"
        >
          <RotateCcw size={15} aria-hidden="true" />
          Reset
        </button>
      </div>

      <div className="flex-1 overflow-auto">
        <SettingRow
          icon={settings.theme === 'dark' ? <Moon size={16} aria-hidden="true" /> : <Sun size={16} aria-hidden="true" />}
          title="Theme"
          caption="Switch the platform shell between dark and light mode."
        >
          <SegmentButton<ThemeMode>
            value="dark"
            selected={settings.theme === 'dark'}
            onSelect={theme => update({ theme })}
          >
            <Moon size={15} aria-hidden="true" />
            Dark
          </SegmentButton>
          <SegmentButton<ThemeMode>
            value="light"
            selected={settings.theme === 'light'}
            onSelect={theme => update({ theme })}
          >
            <Sun size={15} aria-hidden="true" />
            Light
          </SegmentButton>
        </SettingRow>

        <SettingRow
          icon={<Type size={16} aria-hidden="true" />}
          title="Font"
          caption="Choose the system-wide interface typeface."
        >
          <select
            value={settings.fontFamily}
            onChange={event => update({ fontFamily: event.target.value as FontFamily })}
            className="min-h-11 w-full border border-border bg-s1 px-3 text-sm text-text outline-none focus-visible:border-accent"
            aria-label="Font family"
          >
            {FONT_OPTIONS.map(option => (
              <option key={option.value} value={option.value}>
                {option.label}
              </option>
            ))}
          </select>
        </SettingRow>

        <SettingRow
          icon={<Type size={16} aria-hidden="true" />}
          title="Font Size"
          caption="Set the base font size used across the app."
        >
          <div className="flex w-full items-center gap-3">
            <input
              type="range"
              min={14}
              max={22}
              step={1}
              value={settings.fontSizePx}
              onChange={event => update({ fontSizePx: Number(event.target.value) })}
              className="w-full accent-[var(--color-accent)]"
              aria-label="System font size"
            />
            <span className="w-14 border border-border bg-s2 px-2 py-2 text-center text-2xs text-text tabnum">
              {settings.fontSizePx}px
            </span>
          </div>
        </SettingRow>

        <SettingRow
          icon={<Palette size={16} aria-hidden="true" />}
          title="Accent"
          caption="Select the highlight color for focus, active navigation, and controls."
        >
          {ACCENT_OPTIONS.map(option => (
            <button
              key={option.value}
              type="button"
              onClick={() => update({ accentColor: option.value as AccentColor })}
              className={[
                'flex min-h-11 items-center gap-2 border px-3 text-2xs font-medium',
                settings.accentColor === option.value
                  ? 'border-accent bg-s2 text-text'
                  : 'border-border bg-s1 text-muted hover:border-border-strong hover:text-text',
              ].join(' ')}
              aria-pressed={settings.accentColor === option.value}
            >
              <span
                className="h-3 w-3 border border-border-strong"
                style={{ background: option.color }}
                aria-hidden="true"
              />
              {option.label}
              {settings.accentColor === option.value && <Check size={14} aria-hidden="true" />}
            </button>
          ))}
        </SettingRow>

        <SettingRow
          icon={<Gauge size={16} aria-hidden="true" />}
          title="Density"
          caption="Tune the shell spacing for scanning or relaxed viewing."
        >
          <SegmentButton<InterfaceDensity>
            value="standard"
            selected={settings.density === 'standard'}
            onSelect={density => update({ density })}
          >
            Standard
          </SegmentButton>
          <SegmentButton<InterfaceDensity>
            value="compact"
            selected={settings.density === 'compact'}
            onSelect={density => update({ density })}
          >
            Compact
          </SegmentButton>
        </SettingRow>

        <SettingRow
          icon={<Eye size={16} aria-hidden="true" />}
          title="Contrast"
          caption="Increase text and border contrast across the interface."
        >
          <Toggle
            checked={settings.highContrast}
            onChange={highContrast => update({ highContrast })}
            label="High contrast"
          />
        </SettingRow>

        <SettingRow
          icon={<Zap size={16} aria-hidden="true" />}
          title="Motion"
          caption="Reduce interface animations and transitions."
        >
          <Toggle
            checked={settings.reduceMotion}
            onChange={reduceMotion => update({ reduceMotion })}
            label="Reduced motion"
          />
        </SettingRow>

        <SettingRow
          icon={<RefreshCw size={16} aria-hidden="true" />}
          title="Market Refresh"
          caption="Set the global market refresh preference saved with the app."
        >
          <select
            value={settings.marketRefreshSeconds}
            onChange={event => update({ marketRefreshSeconds: Number(event.target.value) })}
            className="min-h-11 w-full border border-border bg-s1 px-3 text-sm text-text outline-none focus-visible:border-accent"
            aria-label="Market refresh interval"
          >
            {REFRESH_OPTIONS.map(option => (
              <option key={option.value} value={option.value}>
                {option.label}
              </option>
            ))}
          </select>
        </SettingRow>

        <section className="px-4 py-4">
          <div className="border border-border bg-s1 p-4">
            <p className="text-2xs uppercase tracking-widest text-dim">Preview</p>
            <div className="mt-3 grid grid-cols-3 gap-2 max-sm:grid-cols-1">
              <div className="border border-border bg-bg p-3">
                <p className="text-2xs text-muted">Signal</p>
                <p className="mt-1 text-md font-medium text-text">BUY</p>
              </div>
              <div className="border border-border bg-bg p-3">
                <p className="text-2xs text-muted">Confidence</p>
                <p className="mt-1 text-md font-medium text-accent tabnum">82.4</p>
              </div>
              <div className="border border-border bg-bg p-3">
                <p className="text-2xs text-muted">Refresh</p>
                <p className="mt-1 text-md font-medium text-text tabnum">
                  {settings.marketRefreshSeconds / 60}m
                </p>
              </div>
            </div>
          </div>
        </section>

      </div>
    </div>
  )
}
