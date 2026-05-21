import { useState, type ReactNode } from 'react'
import {
  ACCENT_OPTIONS,
  FONT_OPTIONS,
  isReasoningModel,
  OPENAI_MODEL_OPTIONS,
  useAppStore,
} from '../store/useAppStore'
import type {
  AccentColor,
  FinancialsDepth,
  FontFamily,
  InterfaceDensity,
  OpenAIModel,
  SystemSettings,
  ThemeMode,
} from '../store/useAppStore'

const REFRESH_OPTIONS = [
  { value: 5, label: '5 sec' },
  { value: 60, label: '1 min' },
  { value: 180, label: '3 min' },
  { value: 300, label: '5 min' },
  { value: 900, label: '15 min' },
]

const SETTINGS_NAV: Array<{
  id: string
  title: string
  description: string
}> = [
  {
    id: 'appearance',
    title: 'Appearance',
    description: 'Theme, type, color, and density',
  },
  {
    id: 'accessibility',
    title: 'Accessibility',
    description: 'Contrast and motion preferences',
  },
  {
    id: 'market-data',
    title: 'Market Data',
    description: 'Refresh behavior and live quotes',
  },
  {
    id: 'ai-assistant',
    title: 'AI Assistant',
    description: 'Model, search, prompt, and context',
  },
  {
    id: 'api-key',
    title: 'API Key',
    description: 'Local OpenAI credential',
  },
]

type SettingsCategoryId = typeof SETTINGS_NAV[number]['id']

const inputClassName = 'min-h-[44px] w-full rounded-[8px] border border-border bg-bg px-4 py-3 text-sm text-text outline-none transition-colors placeholder:text-dim focus-visible:border-accent'

function CategoryNav({
  activeCategory,
  onSelect,
}: {
  activeCategory: SettingsCategoryId
  onSelect: (category: SettingsCategoryId) => void
}) {
  return (
    <aside className="lg:w-60 lg:shrink-0">
      <nav
        className="sticky top-5 flex gap-1 overflow-x-auto border border-border bg-s1 p-2 lg:flex-col lg:overflow-visible"
        aria-label="Settings categories"
      >
        {SETTINGS_NAV.map(({ id, title, description }) => (
          <button
            key={id}
            type="button"
            onClick={() => onSelect(id)}
            className={[
              'block min-w-[208px] px-4 py-3 text-left transition-colors lg:min-w-0',
              activeCategory === id
                ? 'bg-s2 text-text'
                : 'text-muted hover:bg-s2 hover:text-text focus-visible:bg-s2',
            ].join(' ')}
            aria-current={activeCategory === id ? 'page' : undefined}
          >
            <span className="block text-sm font-medium text-text">{title}</span>
            <span className="mt-1 block truncate text-2xs text-muted">{description}</span>
          </button>
        ))}
      </nav>
    </aside>
  )
}

function SettingsSection({
  id,
  title,
  caption,
  children,
}: {
  id: string
  title: string
  caption: string
  children: ReactNode
}) {
  return (
    <section
      id={id}
      className="scroll-mt-6 border-x border-t border-border bg-s1 last:border-b"
      aria-labelledby={`${id}-heading`}
    >
      <header className="border-b border-border px-6 py-4">
        <h2 id={`${id}-heading`} className="text-base font-semibold text-text">{title}</h2>
        <p className="mt-1 max-w-2xl text-2xs leading-5 text-muted">{caption}</p>
      </header>
      <div className="divide-y divide-border">
        {children}
      </div>
    </section>
  )
}

function FieldRow({
  title,
  caption,
  children,
}: {
  title: string
  caption: string
  children: ReactNode
}) {
  return (
    <div className="grid grid-cols-[minmax(0,1fr)_minmax(260px,380px)] items-center gap-6 px-6 py-4 max-md:grid-cols-1 max-md:gap-3">
      <div className="min-w-0">
        <h3 className="text-sm font-medium text-text">{title}</h3>
        <p className="mt-1 max-w-2xl text-2xs leading-5 text-muted">{caption}</p>
      </div>
      <div className="min-w-0">
        {children}
      </div>
    </div>
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
        'flex min-h-[36px] items-center rounded-[8px] border px-4 py-1.5 text-2xs font-medium transition-colors',
        selected
          ? 'border-accent bg-s2 text-text'
          : 'border-border bg-bg text-muted hover:border-border-strong hover:bg-s2 hover:text-text',
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
    <div className="flex items-center gap-3">
      <button
        type="button"
        onClick={() => onChange(!checked)}
        className={[
          'relative h-[28px] w-[52px] shrink-0 rounded-full transition-colors',
          checked ? 'bg-accent' : 'bg-border-strong',
        ].join(' ')}
        role="switch"
        aria-checked={checked}
        aria-label={label}
        title={label}
      >
        <span
          className={[
            'absolute left-[2px] top-[2px] h-[24px] w-[24px] rounded-full bg-white shadow-sm transition-transform',
            checked ? 'translate-x-[24px]' : 'translate-x-0',
          ].join(' ')}
        />
      </button>
      <span className="min-w-[48px] text-2xs font-medium text-text">
        {checked ? 'On' : 'Off'}
      </span>
    </div>
  )
}

export default function Settings() {
  const settings = useAppStore(s => s.settings)
  const setSettings = useAppStore(s => s.setSettings)
  const resetSettings = useAppStore(s => s.resetSettings)
  const openaiKey = useAppStore(s => s.openaiKey)
  const setOpenAIKey = useAppStore(s => s.setOpenAIKey)
  const [keyDraft, setKeyDraft] = useState(openaiKey)
  const [showKey, setShowKey] = useState(false)
  const [activeCategory, setActiveCategory] = useState<SettingsCategoryId>('appearance')

  function update(patch: Partial<SystemSettings>) {
    setSettings(patch)
  }

  function saveKey() {
    setOpenAIKey(keyDraft)
  }

  function clearKey() {
    setKeyDraft('')
    setOpenAIKey('')
  }

  return (
    <div className="h-full overflow-auto" style={{ colorScheme: settings.theme }}>
      <div className="border-b border-border bg-s1 px-6 py-4">
        <div className="mx-auto flex max-w-6xl items-center gap-4">
          <div className="min-w-0">
            <h1 className="text-md font-semibold text-text">System Settings</h1>
            <p className="mt-1 text-2xs text-muted">Preferences are saved in this browser and apply across TradeSmart.</p>
          </div>
          <button
            type="button"
            onClick={resetSettings}
            className="ml-auto min-h-[40px] shrink-0 rounded-[8px] border border-border bg-bg px-4 py-2 text-2xs font-medium text-muted transition-colors hover:border-border-strong hover:bg-s2 hover:text-text"
          >
            Reset
          </button>
        </div>
      </div>

      <div className="mx-auto flex max-w-6xl gap-5 px-5 py-6 max-lg:flex-col">
        <CategoryNav activeCategory={activeCategory} onSelect={setActiveCategory} />

        <div className="flex min-w-0 flex-1 flex-col">
          {activeCategory === 'appearance' && (
            <SettingsSection
              id="appearance"
              title="Appearance"
              caption="Set the visual defaults for the TradeSmart interface."
            >
            <FieldRow
              title="Theme"
              caption="Switch the platform shell between dark and light mode."
            >
              <div className="flex items-center gap-2">
                <SegmentButton<ThemeMode>
                  value="dark"
                  selected={settings.theme === 'dark'}
                  onSelect={theme => update({ theme })}
                >
                  Dark
                </SegmentButton>
                <SegmentButton<ThemeMode>
                  value="light"
                  selected={settings.theme === 'light'}
                  onSelect={theme => update({ theme })}
                >
                  Light
                </SegmentButton>
              </div>
            </FieldRow>

            <FieldRow
              title="Font"
              caption="Choose the system-wide interface typeface."
            >
              <select
                value={settings.fontFamily}
                onChange={event => update({ fontFamily: event.target.value as FontFamily })}
                className={inputClassName}
                aria-label="Font family"
              >
                {FONT_OPTIONS.map(option => (
                  <option key={option.value} value={option.value}>
                    {option.label}
                  </option>
                ))}
              </select>
            </FieldRow>

            <FieldRow
              title="Font size"
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
                <span className="w-[56px] rounded-[8px] border border-border bg-bg px-2 py-2 text-center text-2xs text-text tabnum">
                  {settings.fontSizePx}px
                </span>
              </div>
            </FieldRow>

            <FieldRow
              title="Accent"
              caption="Select the highlight color for focus, active navigation, and controls."
            >
              <div className="flex flex-wrap items-center gap-2">
              {ACCENT_OPTIONS.map(option => (
                <button
                  key={option.value}
                  type="button"
                  onClick={() => update({ accentColor: option.value as AccentColor })}
                  className={[
                    'flex min-h-[36px] items-center gap-2 rounded-[8px] border px-3 py-1.5 text-2xs font-medium transition-colors',
                    settings.accentColor === option.value
                      ? 'border-accent bg-s2 text-text'
                      : 'border-border bg-bg text-muted hover:border-border-strong hover:bg-s2 hover:text-text',
                  ].join(' ')}
                  aria-pressed={settings.accentColor === option.value}
                >
                  <span
                    className="h-3.5 w-3.5 rounded-full border border-border-strong"
                    style={{ background: option.color }}
                    aria-hidden="true"
                  />
                  {option.label}
                </button>
              ))}
              </div>
            </FieldRow>

            <FieldRow
              title="Density"
              caption="Tune the shell spacing for scanning or relaxed viewing."
            >
              <div className="flex items-center gap-2">
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
              </div>
            </FieldRow>
          </SettingsSection>
          )}

          {activeCategory === 'accessibility' && (
            <SettingsSection
              id="accessibility"
              title="Accessibility"
              caption="Adjust contrast and motion so the app is easier to read and navigate."
            >
            <FieldRow
              title="High contrast"
              caption="Increase text and border contrast across the interface."
            >
              <Toggle
                checked={settings.highContrast}
                onChange={highContrast => update({ highContrast })}
                label="High contrast"
              />
            </FieldRow>

            <FieldRow
              title="Reduced motion"
              caption="Reduce interface animations and transitions."
            >
              <Toggle
                checked={settings.reduceMotion}
                onChange={reduceMotion => update({ reduceMotion })}
                label="Reduced motion"
              />
            </FieldRow>
          </SettingsSection>
          )}

          {activeCategory === 'market-data' && (
            <SettingsSection
              id="market-data"
              title="Market Data"
              caption="Control how frequently market context updates while you work."
            >
            <FieldRow
              title="Market refresh"
              caption="Set the global market refresh preference saved with the app."
            >
              <select
                value={settings.marketRefreshSeconds}
                onChange={event => update({ marketRefreshSeconds: Number(event.target.value) })}
                className={inputClassName}
                aria-label="Market refresh interval"
              >
                {REFRESH_OPTIONS.map(option => (
                  <option key={option.value} value={option.value}>
                    {option.label}
                  </option>
                ))}
              </select>
            </FieldRow>

            <FieldRow
              title="Paper trading live refresh"
              caption="When on, Stock Detail and paper portfolio positions use Alpaca live quotes when credentials are configured."
            >
              <Toggle
                checked={settings.fastPaperRefresh}
                onChange={fastPaperRefresh => update({ fastPaperRefresh })}
                label="Paper trading live refresh"
              />
            </FieldRow>
          </SettingsSection>
          )}

          {activeCategory === 'ai-assistant' && (
            <SettingsSection
              id="ai-assistant"
              title="AI Assistant"
              caption="Configure strategy brief and chat behavior for stock pages."
            >
            <FieldRow
              title="Chat model"
              caption="Model used by the Strategy chatbot on stock pages."
            >
              <select
                value={settings.openaiModel}
                onChange={event => update({ openaiModel: event.target.value as OpenAIModel })}
                className={inputClassName}
                aria-label="OpenAI chat model"
              >
                {OPENAI_MODEL_OPTIONS.map(option => (
                  <option key={option.value} value={option.value}>
                    {option.label}
                  </option>
                ))}
              </select>
            </FieldRow>

            <FieldRow
              title="Brief web search"
              caption="When on, the auto-generated Strategy Brief uses web search for recent earnings, news, and filings. Adds cost per stock load."
            >
              <Toggle
                checked={settings.briefWebSearch}
                onChange={briefWebSearch => update({ briefWebSearch })}
                label="Brief web search"
              />
            </FieldRow>

            <FieldRow
              title="Temperature"
              caption={
                isReasoningModel(settings.openaiModel)
                  ? `${settings.openaiModel} is a reasoning model and ignores temperature. This slider only applies to other models.`
                  : 'Sampling temperature for chat and brief. Lower is more focused; higher is more varied.'
              }
            >
              <div className="flex w-full items-center gap-3">
                <input
                  type="range"
                  min={0}
                  max={1.5}
                  step={0.1}
                  value={settings.openaiTemperature}
                  onChange={event => update({ openaiTemperature: Number(event.target.value) })}
                  disabled={isReasoningModel(settings.openaiModel)}
                  className="w-full accent-[var(--color-accent)] disabled:opacity-40"
                  aria-label="OpenAI temperature"
                />
                <span className="w-[56px] rounded-[8px] border border-border bg-bg px-2 py-2 text-center text-2xs text-text tabnum">
                  {settings.openaiTemperature.toFixed(1)}
                </span>
              </div>
            </FieldRow>

            <FieldRow
              title="Custom system prompt"
              caption="Optional extra instructions appended to the built-in chat and brief prompts."
            >
              <div className="w-full">
                <div className="mb-2 text-2xs text-muted">
                  Applies to chat and strategy briefs
                </div>
                <textarea
                  value={settings.openaiSystemPrompt}
                  onChange={event => update({ openaiSystemPrompt: event.target.value })}
                  placeholder="e.g. Prefer conservative position sizing. Always note if RSI < 30 is a divergence vs continuation."
                  rows={4}
                  spellCheck={false}
                  className="min-h-[112px] w-full resize-y rounded-[8px] border border-border bg-bg px-4 py-3 text-sm leading-5 text-text outline-none transition-colors placeholder:text-dim focus-visible:border-accent"
                  aria-label="Custom system prompt"
                />
              </div>
            </FieldRow>

            <FieldRow
              title="Financials in AI"
              caption="Choose how much 3-statement financial data is injected into AI context."
            >
              <div className="flex items-center gap-2">
                <SegmentButton<FinancialsDepth>
                  value="concise"
                  selected={settings.financialsDepth === 'concise'}
                  onSelect={financialsDepth => update({ financialsDepth })}
                >
                  Concise
                </SegmentButton>
                <SegmentButton<FinancialsDepth>
                  value="full"
                  selected={settings.financialsDepth === 'full'}
                  onSelect={financialsDepth => update({ financialsDepth })}
                >
                  Full
                </SegmentButton>
              </div>
            </FieldRow>
          </SettingsSection>
          )}

          {activeCategory === 'api-key' && (
            <SettingsSection
              id="api-key"
              title="API Key"
              caption="Manage the OpenAI key stored locally in this browser."
            >
            <FieldRow
              title="OpenAI API key"
              caption="Stored only in localStorage. Used by Strategy chat and AI-generated strategy briefs."
            >
              <div className="flex w-full flex-col gap-3">
                <div className="flex w-full items-center gap-2">
                  <input
                    type={showKey ? 'text' : 'password'}
                    value={keyDraft}
                    onChange={event => setKeyDraft(event.target.value)}
                    placeholder="sk-..."
                    spellCheck={false}
                    autoComplete="off"
                    className={`${inputClassName} flex-1 text-xs tabnum`}
                    aria-label="OpenAI API key"
                  />
                  <button
                    type="button"
                    onClick={() => setShowKey(v => !v)}
                    className="min-h-[44px] shrink-0 rounded-[8px] border border-border bg-bg px-4 py-2 text-2xs font-medium text-muted transition-colors hover:border-border-strong hover:bg-s2 hover:text-text"
                  >
                    {showKey ? 'Hide' : 'Show'}
                  </button>
                </div>
                <div className="flex items-center gap-2 max-sm:flex-wrap">
                  <button
                    type="button"
                    onClick={saveKey}
                    disabled={keyDraft === openaiKey}
                    className="min-h-[44px] rounded-[8px] border border-accent bg-accent px-4 py-2 text-2xs font-medium text-bg transition-opacity hover:opacity-90 disabled:cursor-not-allowed disabled:opacity-40"
                  >
                    {openaiKey && keyDraft === openaiKey ? 'Saved' : 'Save key'}
                  </button>
                  {openaiKey && (
                    <button
                      type="button"
                      onClick={clearKey}
                      className="min-h-[44px] rounded-[8px] border border-border bg-bg px-4 py-2 text-2xs font-medium text-muted transition-colors hover:border-border-strong hover:bg-s2 hover:text-down"
                    >
                      Clear
                    </button>
                  )}
                  <span className="text-2xs text-dim">
                    {openaiKey ? 'Key saved' : 'No key saved'}
                  </span>
                </div>
              </div>
            </FieldRow>
          </SettingsSection>
          )}
        </div>
      </div>
    </div>
  )
}
