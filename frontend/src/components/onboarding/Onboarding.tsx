import { useEffect, useMemo, useRef, useState, type ReactNode } from 'react'
import { Eye, EyeOff } from 'lucide-react'
import {
  ACCENT_OPTIONS,
  DEFAULT_SYSTEM_SETTINGS,
  isReasoningModel,
  OPENAI_MODEL_OPTIONS,
  useAppStore,
} from '../../store/useAppStore'
import type { AccentColor, InterfaceDensity, OpenAIModel, SystemSettings } from '../../store/useAppStore'
import backgroundVideo from '../../assets/background.mp4'
import demoImage from '../../assets/demo1.png'
import landingHeroImage from '../../assets/landing-hero.jpg'

type StepId = 'overview' | 'account' | 'keys' | 'preferences' | 'review'

const STEPS: Array<{ id: StepId; label: string }> = [
  { id: 'overview', label: 'Overview' },
  { id: 'account', label: 'Sign in' },
  { id: 'keys', label: 'Keys' },
  { id: 'preferences', label: 'Preferences' },
  { id: 'review', label: 'Review' },
]

const SYSTEM_POINTS = [
  'Screen equities and sector moves from a private research terminal.',
  'Inspect stock detail pages with price action, technicals, peers, and financial context.',
  'Run backtests and integrated strategy workflows without leaving the app shell.',
  'Use an optional OpenAI key for Daily Lineup, strategy briefs, and stock chat.',
]

const inputClassName = 'min-h-[44px] w-full rounded-[8px] border border-border bg-black px-4 py-3 text-sm text-text outline-none transition-colors placeholder:text-dim focus-visible:border-white'
const fieldLabelClassName = 'block text-2xs font-medium uppercase leading-none tracking-[0.2em] text-muted'
const formGroupClassName = 'space-y-3'
const ASCII_VIDEO_PLAYBACK_RATE = 0.45
const ASCII_CELL_WIDTH_MOBILE = 8
const ASCII_CELL_WIDTH_DESKTOP = 16
const ASCII_MAX_DPR = 1

function StepRail({ activeIndex }: { activeIndex: number }) {
  return (
    <nav className="flex gap-1 overflow-x-auto border-b border-white/10 px-5 py-3 md:flex-col md:border-b-0 md:border-r md:px-0 md:py-0" aria-label="Onboarding steps">
      {STEPS.map((step, index) => (
        <div
          key={step.id}
          className={[
            'flex min-w-28 items-center gap-3 px-4 py-3 text-left text-2xs transition-colors',
            activeIndex === index ? 'bg-white/[0.08] text-white' : index < activeIndex ? 'text-text' : 'text-muted',
          ].join(' ')}
        >
          <span className="flex h-5 w-5 shrink-0 items-center justify-center text-[0.68rem] tabnum text-current">
            {index < activeIndex ? '✓' : index + 1}
          </span>
          <span className="font-medium">{step.label}</span>
        </div>
      ))}
    </nav>
  )
}

function TextField({
  id,
  label,
  type = 'text',
  value,
  onChange,
  placeholder,
  autoComplete,
}: {
  id: string
  label: string
  type?: string
  value: string
  onChange: (value: string) => void
  placeholder?: string
  autoComplete?: string
}) {
  return (
    <div className={formGroupClassName}>
      <label htmlFor={id} className={fieldLabelClassName}>
        {label}
      </label>
      <input
        id={id}
        type={type}
        value={value}
        onChange={event => onChange(event.target.value)}
        placeholder={placeholder}
        autoComplete={autoComplete}
        className={inputClassName}
      />
    </div>
  )
}

function BooleanRow({
  title,
  caption,
  checked,
  onChange,
}: {
  title: string
  caption: string
  checked: boolean
  onChange: (checked: boolean) => void
}) {
  return (
    <div className="flex items-center justify-between gap-5 border-b border-white/10 py-4 last:border-b-0">
      <span className="min-w-0">
        <span className="block text-sm font-medium text-text">{title}</span>
        <span className="mt-1 block text-2xs leading-5 text-muted">{caption}</span>
      </span>
      <button
        type="button"
        onClick={() => onChange(!checked)}
        className={[
          'relative h-[28px] w-[52px] shrink-0 rounded-full border transition-colors',
          checked ? 'border-white bg-white' : 'border-white/30 bg-black',
        ].join(' ')}
        role="switch"
        aria-checked={checked}
        aria-label={title}
      >
        <span
          className={[
            'absolute left-[2px] top-[2px] h-[22px] w-[22px] rounded-full transition-transform',
            checked ? 'translate-x-[24px] bg-black' : 'translate-x-0 bg-white',
          ].join(' ')}
        />
      </button>
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
        'min-h-[40px] rounded-[8px] border px-4 py-2 text-2xs font-medium transition-colors',
        selected ? 'border-white bg-white text-black' : 'border-white/20 bg-black text-muted hover:border-white/45 hover:text-text',
      ].join(' ')}
      aria-pressed={selected}
    >
      {children}
    </button>
  )
}

function AsciiVideoBackground({ ariaLabel }: { ariaLabel?: string }) {
  const canvasRef = useRef<HTMLCanvasElement | null>(null)
  const videoRef = useRef<HTMLVideoElement | null>(null)
  const fallbackImageRef = useRef<HTMLImageElement | null>(null)

  useEffect(() => {
    const canvas = canvasRef.current
    const video = videoRef.current
    const ctx = canvas?.getContext('2d')
    if (!canvas || !video || !ctx) return

    const sample = document.createElement('canvas')
    const sampleCtx = sample.getContext('2d', { willReadFrequently: true })
    if (!sampleCtx) return

    const fallbackImage = new Image()
    fallbackImage.src = landingHeroImage
    fallbackImageRef.current = fallbackImage

    let stopped = false
    let animationFrameId: number | null = null
    let fadeActive = false
    let smoothedPixels: Float32Array | null = null
    let hasSmoothedFrame = false
    const fadeVideo = document.createElement('video')
    const chars = '.,:;irsXA253hMHGS#9B&@'

    video.playbackRate = ASCII_VIDEO_PLAYBACK_RATE
    fadeVideo.src = backgroundVideo
    fadeVideo.muted = true
    fadeVideo.playsInline = true
    fadeVideo.preload = 'auto'
    fadeVideo.playbackRate = ASCII_VIDEO_PLAYBACK_RATE

    const getVideoSource = (sourceVideo: HTMLVideoElement) => {
      if (sourceVideo.readyState >= HTMLMediaElement.HAVE_CURRENT_DATA && sourceVideo.videoWidth && sourceVideo.videoHeight) {
        return {
          source: sourceVideo,
          width: sourceVideo.videoWidth,
          height: sourceVideo.videoHeight,
        }
      }

      return null
    }

    const getSource = () => {
      const videoSource = getVideoSource(video)
      if (videoSource) return videoSource

      if (fallbackImage.complete && fallbackImage.naturalWidth && fallbackImage.naturalHeight) {
        return {
          source: fallbackImage,
          width: fallbackImage.naturalWidth,
          height: fallbackImage.naturalHeight,
        }
      }

      return null
    }

    const drawSampleSource = (
      drawable: { source: CanvasImageSource; width: number; height: number },
      cols: number,
      rows: number,
      cellWidth: number,
      cellHeight: number,
      alpha = 1,
    ) => {
      const videoAspect = drawable.width / drawable.height
      const targetAspect = (cols * cellWidth) / (rows * cellHeight)
      let sx = 0
      let sy = 0
      let sw = drawable.width
      let sh = drawable.height

      if (videoAspect > targetAspect) {
        sw = drawable.height * targetAspect
        sx = (drawable.width - sw) / 2
      } else {
        sh = drawable.width / targetAspect
        sy = (drawable.height - sh) / 2
      }

      sampleCtx.globalAlpha = alpha
      sampleCtx.drawImage(drawable.source, sx, sy, sw, sh, 0, 0, cols, rows)
      sampleCtx.globalAlpha = 1
    }

    const syncLoopFade = () => {
      const duration = video.duration
      if (!Number.isFinite(duration) || duration <= 0) return { nextSource: null, mix: 0 }

      const fadeSeconds = Math.min(1.2, duration * 0.18)
      const remaining = duration - video.currentTime
      const mix = Math.max(0, Math.min(1, (fadeSeconds - remaining) / fadeSeconds))

      if (mix > 0 && !fadeActive) {
        fadeActive = true
        fadeVideo.currentTime = 0
        fadeVideo.playbackRate = ASCII_VIDEO_PLAYBACK_RATE
        void fadeVideo.play().catch(() => undefined)
      }

      if (mix === 0 && fadeActive && video.currentTime < fadeSeconds) {
        fadeActive = false
        fadeVideo.pause()
        fadeVideo.currentTime = 0
      }

      if (remaining <= 0.05 || video.ended) {
        const handoffTime = Math.min(Math.max(fadeVideo.currentTime || 0, 0), duration - 0.05)
        video.currentTime = handoffTime
        fadeVideo.pause()
        fadeVideo.currentTime = 0
        fadeActive = false
        video.playbackRate = ASCII_VIDEO_PLAYBACK_RATE
        void video.play().catch(() => undefined)
        return { nextSource: null, mix: 0 }
      }

      return {
        nextSource: fadeActive ? getVideoSource(fadeVideo) : null,
        mix,
      }
    }

    const render = () => {
      if (stopped) return

      const rect = canvas.getBoundingClientRect()
      const width = Math.round(rect.width || window.innerWidth)
      const height = Math.round(rect.height || window.innerHeight)
      const dpr = Math.min(window.devicePixelRatio || 1, ASCII_MAX_DPR)
      if (width <= 0 || height <= 0) {
        return
      }

      if (canvas.width !== Math.floor(width * dpr) || canvas.height !== Math.floor(height * dpr)) {
        canvas.width = Math.floor(width * dpr)
        canvas.height = Math.floor(height * dpr)
      }

      ctx.setTransform(dpr, 0, 0, dpr, 0, 0)
      ctx.globalCompositeOperation = 'source-over'
      ctx.fillStyle = 'rgba(0, 0, 0, 0.92)'
      ctx.fillRect(0, 0, width, height)

      const cellWidth = width < 760 ? ASCII_CELL_WIDTH_MOBILE : ASCII_CELL_WIDTH_DESKTOP
      const cellHeight = Math.round(cellWidth * 1.45)
      const cols = Math.max(1, Math.ceil(width / cellWidth))
      const rows = Math.max(1, Math.ceil(height / cellHeight))

      if (sample.width !== cols || sample.height !== rows) {
        sample.width = cols
        sample.height = rows
        smoothedPixels = null
        hasSmoothedFrame = false
      }

      const drawable = getSource()

      if (drawable) {
        const { nextSource, mix } = syncLoopFade()
        const easedMix = nextSource && mix > 0 ? mix * mix * (3 - (2 * mix)) : 0
        sampleCtx.clearRect(0, 0, cols, rows)
        drawSampleSource(drawable, cols, rows, cellWidth, cellHeight, 1 - easedMix)
        if (nextSource && easedMix > 0) {
          drawSampleSource(nextSource, cols, rows, cellWidth, cellHeight, easedMix)
        }
        const pixels = sampleCtx.getImageData(0, 0, cols, rows).data
        if (!smoothedPixels || smoothedPixels.length !== cols * rows * 3) {
          smoothedPixels = new Float32Array(cols * rows * 3)
          hasSmoothedFrame = false
        }

        ctx.font = `600 ${cellHeight}px "IBM Plex Mono", "SFMono-Regular", Consolas, monospace`
        ctx.textBaseline = 'top'
        ctx.textAlign = 'left'
        ctx.shadowBlur = 1

        for (let row = 0; row < rows; row += 1) {
          for (let col = 0; col < cols; col += 1) {
            const index = (row * cols + col) * 4
            const smoothIndex = (row * cols + col) * 3
            const currentWeight = hasSmoothedFrame ? 0.34 : 1
            smoothedPixels[smoothIndex] = (smoothedPixels[smoothIndex] * (1 - currentWeight)) + (pixels[index] * currentWeight)
            smoothedPixels[smoothIndex + 1] = (smoothedPixels[smoothIndex + 1] * (1 - currentWeight)) + (pixels[index + 1] * currentWeight)
            smoothedPixels[smoothIndex + 2] = (smoothedPixels[smoothIndex + 2] * (1 - currentWeight)) + (pixels[index + 2] * currentWeight)

            const r = smoothedPixels[smoothIndex]
            const g = smoothedPixels[smoothIndex + 1]
            const b = smoothedPixels[smoothIndex + 2]
            const brightness = (0.2126 * r) + (0.7152 * g) + (0.0722 * b)
            const contrast = Math.min(255, Math.max(0, (brightness - 6) * 2.65))
            const char = chars[Math.min(chars.length - 1, Math.floor((contrast / 256) * chars.length))]
            const lift = 36
            const saturation = 0.42
            const cr = Math.min(205, Math.round((brightness + (r - brightness) * saturation) * 1.05 + lift))
            const cg = Math.min(205, Math.round((brightness + (g - brightness) * saturation) * 1.05 + lift))
            const cb = Math.min(205, Math.round((brightness + (b - brightness) * saturation) * 1.05 + lift))
            const alpha = Math.max(0.2, Math.min(0.72, (contrast / 255) * 0.44 + 0.2))

            ctx.fillStyle = `rgba(${cr}, ${cg}, ${cb}, ${alpha})`
            ctx.shadowColor = `rgba(${cr}, ${cg}, ${cb}, 0.18)`
            ctx.fillText(char, col * cellWidth, row * cellHeight)
          }
        }
        hasSmoothedFrame = true
        ctx.shadowBlur = 0
      }
    }

    const renderLoop = () => {
      render()
      if (!stopped && !document.hidden) {
        animationFrameId = window.requestAnimationFrame(renderLoop)
      }
    }

    const start = () => {
      video.playbackRate = ASCII_VIDEO_PLAYBACK_RATE
      void video.play().catch(() => undefined)
      if (animationFrameId == null) {
        animationFrameId = window.requestAnimationFrame(renderLoop)
      }
    }

    const handleVisibilityChange = () => {
      if (document.hidden) {
        if (animationFrameId != null) {
          window.cancelAnimationFrame(animationFrameId)
          animationFrameId = null
        }
        video.pause()
        fadeVideo.pause()
        return
      }

      video.playbackRate = ASCII_VIDEO_PLAYBACK_RATE
      void video.play().catch(() => undefined)
      if (animationFrameId == null) {
        animationFrameId = window.requestAnimationFrame(renderLoop)
      }
    }

    fallbackImage.addEventListener('load', start)
    video.addEventListener('loadedmetadata', start)
    video.addEventListener('loadeddata', start)
    video.addEventListener('canplay', start)
    document.addEventListener('visibilitychange', handleVisibilityChange)
    start()

    return () => {
      stopped = true
      fadeVideo.pause()
      fadeVideo.removeAttribute('src')
      fadeVideo.load()
      fallbackImage.removeEventListener('load', start)
      video.removeEventListener('loadedmetadata', start)
      video.removeEventListener('loadeddata', start)
      video.removeEventListener('canplay', start)
      document.removeEventListener('visibilitychange', handleVisibilityChange)
      if (animationFrameId != null) window.cancelAnimationFrame(animationFrameId)
    }
  }, [])

  return (
    <div className="absolute inset-0 overflow-hidden bg-black" role={ariaLabel ? 'img' : undefined} aria-label={ariaLabel}>
      <video
        ref={videoRef}
        className="pointer-events-none absolute inset-0 h-full w-full object-cover opacity-10 saturate-75"
        autoPlay
        muted
        playsInline
        preload="auto"
        aria-hidden="true"
      >
        <source src={backgroundVideo} type="video/mp4" />
      </video>
      <canvas
        ref={canvasRef}
        className="absolute inset-0 h-full w-full opacity-100 [image-rendering:pixelated]"
        aria-hidden="true"
      />
    </div>
  )
}

function AsciiBackground() {
  return (
    <div className="pointer-events-none absolute inset-0 z-[1] overflow-hidden bg-black" aria-hidden="true">
      <AsciiVideoBackground />
    </div>
  )
}

export default function Onboarding({ onComplete }: { onComplete: () => void }) {
  const storedSettings = useAppStore(s => s.settings)
  const openaiKey = useAppStore(s => s.openaiKey)
  const setSettings = useAppStore(s => s.setSettings)
  const setOpenAIKey = useAppStore(s => s.setOpenAIKey)

  const [showSplash, setShowSplash] = useState(true)
  const [activeIndex, setActiveIndex] = useState(0)
  const [draft, setDraft] = useState<SystemSettings>({ ...DEFAULT_SYSTEM_SETTINGS, ...storedSettings, theme: 'dark' })
  const [userIdDraft, setUserIdDraft] = useState('')
  const [passwordDraft, setPasswordDraft] = useState('')
  const [demoMode, setDemoMode] = useState(false)
  const [keyDraft, setKeyDraft] = useState(openaiKey)
  const [showKey, setShowKey] = useState(false)
  const [showPassword, setShowPassword] = useState(false)

  useEffect(() => {
    const timer = window.setTimeout(() => setShowSplash(false), 2000)
    return () => window.clearTimeout(timer)
  }, [])

  const activeStep = STEPS[activeIndex].id
  const savedSummary = useMemo(() => [
    ['Account', demoMode ? 'Demo mode selected' : userIdDraft.trim() ? userIdDraft.trim() : 'Not connected yet'],
    ['Theme', 'Dark'],
    ['Density', draft.density === 'compact' ? 'Compact' : 'Standard'],
    ['Refresh', `${draft.marketRefreshSeconds} sec`],
    ['AI model', draft.openaiModel],
    ['OpenAI key', keyDraft.trim() ? 'Will be saved locally' : 'Skipped for now'],
  ], [demoMode, draft, keyDraft, userIdDraft])

  function update(patch: Partial<SystemSettings>) {
    setDraft(current => ({ ...current, ...patch }))
  }

  function goNext() {
    setActiveIndex(index => Math.min(index + 1, STEPS.length - 1))
  }

  function goBack() {
    setActiveIndex(index => Math.max(index - 1, 0))
  }

  function finish() {
    setSettings(draft)
    setOpenAIKey(keyDraft)
    onComplete()
  }

  return (
    <div className="fixed inset-0 isolate z-[90] overflow-auto bg-black text-text">
      <AsciiBackground />

      <main
        className={[
          'relative z-[2] mx-auto flex min-h-full w-full max-w-6xl items-center px-4 py-6 transition-opacity duration-700 sm:px-6',
          showSplash ? 'pointer-events-none opacity-0' : 'opacity-100',
        ].join(' ')}
        role="dialog"
        aria-modal="true"
        aria-labelledby="onboarding-title"
        aria-hidden={showSplash}
      >
        <section className="grid w-full overflow-hidden border border-white/15 bg-black/68 shadow-2xl shadow-black/35 backdrop-blur-xl md:grid-cols-[220px_minmax(0,1fr)]">
          <StepRail activeIndex={activeIndex} />

          <div className="flex min-h-[620px] flex-col p-5 sm:p-8">
            <div className="mb-8 flex items-start justify-between gap-6 border-b border-white/10 pb-5">
              <div>
                <p className="text-2xs font-medium uppercase tracking-[0.28em] text-muted">First run</p>
                <h1 id="onboarding-title" className="mt-2 text-xl font-semibold text-white">TradeSmart setup</h1>
              </div>
              <div className="hidden px-2 py-1 text-2xs text-muted tabnum sm:block">
                {activeIndex + 1}/{STEPS.length}
              </div>
            </div>

            <div className="flex-1">
              {activeStep === 'overview' && (
                <div className="grid gap-9 lg:grid-cols-[minmax(0,1.05fr)_minmax(260px,0.95fr)]">
                <div>
                  <h2 className="text-base font-semibold text-white">A private research terminal for market work.</h2>
                  <p className="mt-3 max-w-2xl text-xs leading-6 text-muted">
                    TradeSmart brings market overview, stock analysis, strategy testing, paper portfolio context, and optional AI assistance into one browser-based workspace.
                  </p>
                  <div className="mt-8 divide-y divide-white/10 border-y border-white/10">
                    {SYSTEM_POINTS.map(point => (
                      <div key={point} className="py-4 text-sm leading-6 text-text">
                        <span>{point}</span>
                      </div>
                    ))}
                  </div>
                </div>
                <figure className="min-h-72 overflow-hidden bg-black">
                  <img
                    src={demoImage}
                    alt="ASCII visual demo"
                    className="h-full min-h-72 w-full object-cover"
                  />
                </figure>
              </div>
              )}

              {activeStep === 'account' && (
                <div className="max-w-3xl">
                  <h2 className="text-base font-semibold text-white">Enter your TradeSmart account details.</h2>
                  <p className="mt-3 text-xs leading-6 text-muted">
                  </p>

                <div className="mt-8 grid gap-5">
                    <TextField
                      id="onboarding-user-id"
                      label="User ID"
                      value={userIdDraft}
                      onChange={setUserIdDraft}
                      placeholder="Your user ID"
                      autoComplete="username"
                    />

                    <div className={formGroupClassName}>
                      <label htmlFor="onboarding-password" className={fieldLabelClassName}>
                        Password
                      </label>
                      <div className="flex gap-2">
                        <input
                          id="onboarding-password"
                          type={showPassword ? 'text' : 'password'}
                          value={passwordDraft}
                          onChange={event => setPasswordDraft(event.target.value)}
                          placeholder="Your password"
                          autoComplete="current-password"
                          className={`${inputClassName} flex-1`}
                        />
                        <button
                          type="button"
                          onClick={() => setShowPassword(value => !value)}
                          className="flex min-h-[44px] w-12 shrink-0 items-center justify-center rounded-[8px] bg-white/5 text-muted transition-colors hover:bg-white/10 hover:text-text focus-visible:outline focus-visible:outline-2 focus-visible:outline-white/60"
                          aria-label={showPassword ? 'Hide password' : 'Show password'}
                          title={showPassword ? 'Hide password' : 'Show password'}
                        >
                          {showPassword ? <EyeOff size={16} aria-hidden="true" /> : <Eye size={16} aria-hidden="true" />}
                        </button>
                      </div>
                    </div>

                    <BooleanRow
                      title="Demo mode"
                      caption="No Account? Use demo mode to explore the app using local storage."
                      checked={demoMode}
                      onChange={setDemoMode}
                    />
                  </div>
                </div>
              )}

              {activeStep === 'keys' && (
                <div className="max-w-3xl">
                <h2 className="text-base font-semibold text-white">Connect the key used by browser AI features.</h2>
                <p className="mt-3 text-xs leading-6 text-muted">
                  The OpenAI key is stored in localStorage on this browser. It is used for Daily Lineup, strategy briefs, and stock chat.
                </p>
                <a
                  href="https://platform.openai.com/api-keys"
                  target="_blank"
                  rel="noreferrer"
                  className="mt-4 inline-flex text-xs font-medium text-white underline-offset-4 hover:underline focus-visible:outline focus-visible:outline-2 focus-visible:outline-white/60"
                >
                  Open OpenAI dashboard to create an API key
                </a>

                <div className="mt-8 space-y-3">
                  <label htmlFor="onboarding-openai-key" className={fieldLabelClassName}>
                    OpenAI API key
                  </label>
                  <div className="flex gap-2">
                    <input
                      id="onboarding-openai-key"
                      type={showKey ? 'text' : 'password'}
                      value={keyDraft}
                      onChange={event => setKeyDraft(event.target.value)}
                      placeholder="sk-..."
                      spellCheck={false}
                      autoComplete="off"
                      className={`${inputClassName} flex-1 text-xs tabnum`}
                    />
                    <button
                      type="button"
                      onClick={() => setShowKey(value => !value)}
                      className="flex min-h-[44px] w-12 shrink-0 items-center justify-center rounded-[8px] bg-white/5 text-muted transition-colors hover:bg-white/10 hover:text-text focus-visible:outline focus-visible:outline-2 focus-visible:outline-white/60"
                      aria-label={showKey ? 'Hide OpenAI key' : 'Show OpenAI key'}
                      title={showKey ? 'Hide key' : 'Show key'}
                    >
                      {showKey ? <EyeOff size={16} aria-hidden="true" /> : <Eye size={16} aria-hidden="true" />}
                    </button>
                  </div>
                </div>

                <div className="mt-8 border-y border-white/10 bg-white/[0.035] py-4">
                  <p className="text-sm font-medium text-white">Live market feed keys</p>
                  <p className="mt-2 text-2xs leading-5 text-muted">
                    Alpaca live data is configured on the server with environment variables. This setup keeps browser credentials limited to the OpenAI key the frontend actually uses.
                  </p>
                </div>
              </div>
              )}

              {activeStep === 'preferences' && (
                <div className="grid gap-10 lg:grid-cols-[minmax(0,1fr)_minmax(260px,340px)]">
                <div>
                  <h2 className="text-base font-semibold text-white">Set the defaults before the workspace opens.</h2>
                  <p className="mt-3 text-xs leading-6 text-muted">These preferences are saved in the settings cookie and can be changed later.</p>

                  <div className="mt-9 grid gap-8">
                    <div className={formGroupClassName}>
                      <p className={fieldLabelClassName}>Accent</p>
                      <div className="flex flex-wrap gap-2">
                        {ACCENT_OPTIONS.map(option => (
                          <button
                            key={option.value}
                            type="button"
                            onClick={() => update({ accentColor: option.value as AccentColor })}
                            className={[
                              'flex min-h-[40px] items-center gap-2 rounded-[8px] border px-3 py-2 text-2xs font-medium transition-colors',
                              draft.accentColor === option.value ? 'border-white bg-white text-black' : 'border-white/20 bg-black text-muted hover:border-white/45 hover:text-text',
                            ].join(' ')}
                            aria-pressed={draft.accentColor === option.value}
                          >
                            <span className="h-3 w-3 rounded-full border border-white/30" style={{ background: option.color }} aria-hidden="true" />
                            {option.label}
                          </button>
                        ))}
                      </div>
                    </div>

                    <div className={formGroupClassName}>
                      <p className={fieldLabelClassName}>Density</p>
                      <div className="flex flex-wrap gap-2">
                        <SegmentButton<InterfaceDensity> value="standard" selected={draft.density === 'standard'} onSelect={density => update({ density })}>
                          Standard
                        </SegmentButton>
                        <SegmentButton<InterfaceDensity> value="compact" selected={draft.density === 'compact'} onSelect={density => update({ density })}>
                          Compact
                        </SegmentButton>
                      </div>
                    </div>

                    <div className={formGroupClassName}>
                      <label htmlFor="onboarding-model" className={fieldLabelClassName}>
                        AI model
                      </label>
                      <select
                        id="onboarding-model"
                        value={draft.openaiModel}
                        onChange={event => update({ openaiModel: event.target.value as OpenAIModel })}
                        className={inputClassName}
                      >
                        {OPENAI_MODEL_OPTIONS.map(option => (
                          <option key={option.value} value={option.value}>
                            {option.label}
                          </option>
                        ))}
                      </select>
                      {isReasoningModel(draft.openaiModel) && (
                        <p className="mt-2 text-2xs text-muted">Reasoning models ignore temperature. The existing slider remains available in Settings.</p>
                      )}
                    </div>

                    <div className={formGroupClassName}>
                      <label htmlFor="onboarding-refresh" className={fieldLabelClassName}>
                        Market refresh
                      </label>
                      <select
                        id="onboarding-refresh"
                        value={draft.marketRefreshSeconds}
                        onChange={event => update({ marketRefreshSeconds: Number(event.target.value) })}
                        className={inputClassName}
                      >
                        <option value={60}>1 min</option>
                        <option value={180}>3 min</option>
                        <option value={300}>5 min</option>
                        <option value={900}>15 min</option>
                      </select>
                    </div>
                  </div>
                </div>

                <div className="self-start border-y border-white/10 pt-1 lg:mt-[73px]">
                  <BooleanRow
                    title="High contrast"
                    caption="Increase text and border contrast."
                    checked={draft.highContrast}
                    onChange={highContrast => update({ highContrast })}
                  />
                  <BooleanRow
                    title="Reduced motion"
                    caption="Limit interface transitions."
                    checked={draft.reduceMotion}
                    onChange={reduceMotion => update({ reduceMotion })}
                  />
                  <BooleanRow
                    title="Brief web search"
                    caption="Let AI briefs use recent web context."
                    checked={draft.briefWebSearch}
                    onChange={briefWebSearch => update({ briefWebSearch })}
                  />
                  <BooleanRow
                    title="Paper live refresh"
                    caption="Use live quotes when backend credentials exist."
                    checked={draft.fastPaperRefresh}
                    onChange={fastPaperRefresh => update({ fastPaperRefresh })}
                  />
                </div>
              </div>
              )}

              {activeStep === 'review' && (
                <div className="max-w-3xl">
                <h2 className="text-base font-semibold text-white">Ready to open the workspace.</h2>
                <p className="mt-3 text-xs leading-6 text-muted">
                  Finishing writes the preferences cookie. On the next launch, TradeSmart opens directly unless that cookie is cleared.
                </p>
                <dl className="mt-8 divide-y divide-white/10 border-y border-white/10">
                  {savedSummary.map(([label, value]) => (
                    <div key={label} className="grid grid-cols-[150px_minmax(0,1fr)] gap-4 py-4 text-sm max-sm:grid-cols-1 max-sm:gap-1">
                      <dt className="text-muted">{label}</dt>
                      <dd className="font-medium text-text">{value}</dd>
                    </div>
                  ))}
                </dl>
              </div>
              )}
            </div>

            <div className="mt-8 flex items-center justify-between gap-3 border-t border-white/10 pt-5">
              <button
                type="button"
                onClick={goBack}
                disabled={activeIndex === 0}
                className="flex h-[44px] min-w-[96px] items-center justify-center rounded-[8px] bg-white/5 px-5 text-xs font-medium text-muted transition-colors hover:bg-white/10 hover:text-text disabled:cursor-not-allowed disabled:opacity-35"
              >
                Back
              </button>
              {activeIndex === STEPS.length - 1 ? (
                <button
                  type="button"
                  onClick={finish}
                  className="flex h-[44px] min-w-[132px] items-center justify-center rounded-[8px] bg-white px-5 text-xs font-semibold text-black transition-opacity hover:opacity-90"
                >
                  Finish setup
                </button>
              ) : (
                <button
                  type="button"
                  onClick={goNext}
                  className="flex h-[44px] min-w-[120px] items-center justify-center rounded-[8px] bg-white px-5 text-xs font-semibold text-black transition-opacity hover:opacity-90"
                >
                  Continue
                </button>
              )}
            </div>
          </div>
        </section>
      </main>
    </div>
  )
}
