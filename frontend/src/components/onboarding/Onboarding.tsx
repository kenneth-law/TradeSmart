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
  'Use optional OpenAI and Prodia keys for AI assistance and VibeOS image generation.',
]

const inputClassName = 'min-h-[44px] w-full rounded-[8px] border border-border bg-black px-4 py-3 text-sm text-text outline-none transition-colors placeholder:text-dim focus-visible:border-white'
const fieldLabelClassName = 'block text-2xs font-medium uppercase leading-none tracking-[0.2em] text-muted'
const formGroupClassName = 'space-y-3'
const ASCII_VIDEO_PLAYBACK_RATE = 0.45
const ASCII_CELL_WIDTH_MOBILE = 4
const ASCII_CELL_WIDTH_DESKTOP = 8
const ASCII_MAX_DPR = 2
const SPLASH_DURATION_MS = 4000
const SETUP_FADE_MS = 700
const TRANSITION_VIDEO_MS = 4000
const HOMEPAGE_FADE_MS = 700
const TRANSITION_VIDEO_SRC = '/transition1.mp4'
const TRANSITION_REVERSE_VIDEO_SRC = '/transition1-reverse.mp4'
const ASCII_LOOP_FADE_SECONDS = 0.9
const ASCII_CHARS = '.,:;irsXA253hMHGS#9B&@'
const ASCII_GLYPH_WIDTH = 24
const ASCII_GLYPH_HEIGHT = 36

type ExitPhase = 'idle' | 'transition-crossfade' | 'homepage-fade'

function StepRail({ activeIndex }: { activeIndex: number }) {
  return (
    <nav className="flex min-h-0 gap-1 overflow-x-auto border-b border-white/10 px-5 py-3 md:flex-col md:overflow-y-auto md:border-b-0 md:border-r md:px-0 md:py-0" aria-label="Onboarding steps">
      {STEPS.map((step, index) => (
        <div
          key={step.id}
          className={[
            'flex min-w-28 items-center gap-3 px-4 py-3 text-left text-2xs transition-colors md:min-w-0',
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

function compileShader(gl: WebGLRenderingContext, type: number, source: string) {
  const shader = gl.createShader(type)
  if (!shader) return null
  gl.shaderSource(shader, source)
  gl.compileShader(shader)
  if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS)) {
    gl.deleteShader(shader)
    return null
  }
  return shader
}

function createProgram(gl: WebGLRenderingContext, vertexSource: string, fragmentSource: string) {
  const vertex = compileShader(gl, gl.VERTEX_SHADER, vertexSource)
  const fragment = compileShader(gl, gl.FRAGMENT_SHADER, fragmentSource)
  if (!vertex || !fragment) return null

  const program = gl.createProgram()
  if (!program) return null
  gl.attachShader(program, vertex)
  gl.attachShader(program, fragment)
  gl.linkProgram(program)
  gl.deleteShader(vertex)
  gl.deleteShader(fragment)
  if (!gl.getProgramParameter(program, gl.LINK_STATUS)) {
    gl.deleteProgram(program)
    return null
  }
  return program
}

function createGlyphAtlas() {
  const canvas = document.createElement('canvas')
  canvas.width = ASCII_GLYPH_WIDTH * ASCII_CHARS.length
  canvas.height = ASCII_GLYPH_HEIGHT

  const ctx = canvas.getContext('2d')
  if (!ctx) return null
  ctx.clearRect(0, 0, canvas.width, canvas.height)
  ctx.fillStyle = '#fff'
  ctx.font = `600 ${Math.round(ASCII_GLYPH_HEIGHT * 0.78)}px "IBM Plex Mono", "SFMono-Regular", Consolas, monospace`
  ctx.textAlign = 'center'
  ctx.textBaseline = 'middle'

  for (let index = 0; index < ASCII_CHARS.length; index += 1) {
    ctx.fillText(
      ASCII_CHARS[index],
      (index * ASCII_GLYPH_WIDTH) + (ASCII_GLYPH_WIDTH / 2),
      ASCII_GLYPH_HEIGHT / 2,
    )
  }

  return canvas
}

function AsciiVideoBackground({
  ariaLabel,
  videoSrc = backgroundVideo,
  playbackRate = ASCII_VIDEO_PLAYBACK_RATE,
  loopVideo = true,
  fallbackImageSrc = landingHeroImage,
  onVideoError,
}: {
  ariaLabel?: string
  videoSrc?: string
  playbackRate?: number
  loopVideo?: boolean
  fallbackImageSrc?: string | null
  onVideoError?: () => void
}) {
  const canvasRef = useRef<HTMLCanvasElement | null>(null)
  const videoRef = useRef<HTMLVideoElement | null>(null)
  const fallbackImageRef = useRef<HTMLImageElement | null>(null)
  const onVideoErrorRef = useRef(onVideoError)

  useEffect(() => {
    onVideoErrorRef.current = onVideoError
  }, [onVideoError])

  useEffect(() => {
    const canvas = canvasRef.current
    const video = videoRef.current
    if (!canvas || !video) return

    const gl = canvas.getContext('webgl', {
      alpha: false,
      antialias: false,
      depth: false,
      powerPreference: 'high-performance',
      premultipliedAlpha: false,
      stencil: false,
    })
    const glyphAtlas = createGlyphAtlas()
    if (!gl || !glyphAtlas) return

    const vertexSource = `
      attribute vec2 a_position;
      varying vec2 v_uv;

      void main() {
        v_uv = a_position * 0.5 + 0.5;
        gl_Position = vec4(a_position, 0.0, 1.0);
      }
    `

    const fragmentSource = `
      precision mediump float;

      varying vec2 v_uv;
      uniform sampler2D u_video;
      uniform sampler2D u_nextVideo;
      uniform sampler2D u_glyphs;
      uniform vec2 u_resolution;
      uniform vec2 u_videoSize;
      uniform vec2 u_nextVideoSize;
      uniform vec2 u_cellSize;
      uniform float u_charCount;
      uniform float u_mix;

      vec2 coverUv(vec2 uv, vec2 sourceSize) {
        float videoAspect = sourceSize.x / max(sourceSize.y, 1.0);
        float canvasAspect = u_resolution.x / max(u_resolution.y, 1.0);
        vec2 scale = vec2(1.0);

        if (videoAspect > canvasAspect) {
          scale.x = canvasAspect / videoAspect;
        } else {
          scale.y = videoAspect / canvasAspect;
        }

        return ((uv - 0.5) * scale) + 0.5;
      }

      void main() {
        vec2 cell = floor(gl_FragCoord.xy / u_cellSize);
        vec2 cellCenter = ((cell + 0.5) * u_cellSize) / u_resolution;
        vec4 videoColor = mix(
          texture2D(u_video, coverUv(cellCenter, u_videoSize)),
          texture2D(u_nextVideo, coverUv(cellCenter, u_nextVideoSize)),
          u_mix
        );
        float brightness = dot(videoColor.rgb, vec3(0.2126, 0.7152, 0.0722));
        float contrast = clamp((brightness - 0.024) * 2.65, 0.0, 1.0);
        float charIndex = floor(contrast * (u_charCount - 1.0));
        vec2 local = clamp(fract(gl_FragCoord.xy / u_cellSize), vec2(0.02), vec2(0.98));
        vec2 glyphUv = vec2((charIndex + local.x) / u_charCount, local.y);
        float glyphAlpha = texture2D(u_glyphs, glyphUv).a;
        vec3 chroma = vec3(brightness) + ((videoColor.rgb - vec3(brightness)) * 0.42);
        vec3 lifted = min(vec3(0.82), (chroma * 1.05) + vec3(0.141));
        float alpha = clamp((contrast * 0.44) + 0.2, 0.2, 0.72);

        gl_FragColor = vec4(lifted * glyphAlpha * alpha, 1.0);
      }
    `

    const program = createProgram(gl, vertexSource, fragmentSource)
    if (!program) return

    const positionBuffer = gl.createBuffer()
    const videoTexture = gl.createTexture()
    const nextVideoTexture = gl.createTexture()
    const glyphTexture = gl.createTexture()
    if (!positionBuffer || !videoTexture || !nextVideoTexture || !glyphTexture) return

    const fallbackImage = fallbackImageSrc ? new Image() : null
    if (fallbackImage) fallbackImage.src = fallbackImageSrc
    fallbackImageRef.current = fallbackImage

    let stopped = false
    let animationFrameId: number | null = null
    let loopFadeActive = false
    let lastWidth = 0
    let lastHeight = 0
    const nextVideo = document.createElement('video')
    let activeVideo = video
    let standbyVideo = nextVideo

    const aPosition = gl.getAttribLocation(program, 'a_position')
    const uVideo = gl.getUniformLocation(program, 'u_video')
    const uNextVideo = gl.getUniformLocation(program, 'u_nextVideo')
    const uGlyphs = gl.getUniformLocation(program, 'u_glyphs')
    const uResolution = gl.getUniformLocation(program, 'u_resolution')
    const uVideoSize = gl.getUniformLocation(program, 'u_videoSize')
    const uNextVideoSize = gl.getUniformLocation(program, 'u_nextVideoSize')
    const uCellSize = gl.getUniformLocation(program, 'u_cellSize')
    const uCharCount = gl.getUniformLocation(program, 'u_charCount')
    const uMix = gl.getUniformLocation(program, 'u_mix')

    gl.bindBuffer(gl.ARRAY_BUFFER, positionBuffer)
    gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([
      -1, -1,
      1, -1,
      -1, 1,
      1, 1,
    ]), gl.STATIC_DRAW)

    gl.activeTexture(gl.TEXTURE1)
    gl.bindTexture(gl.TEXTURE_2D, glyphTexture)
    gl.pixelStorei(gl.UNPACK_FLIP_Y_WEBGL, true)
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, gl.RGBA, gl.UNSIGNED_BYTE, glyphAtlas)
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE)
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE)
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR)
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR)

    gl.activeTexture(gl.TEXTURE0)
    gl.bindTexture(gl.TEXTURE_2D, videoTexture)
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE)
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE)
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR)
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR)

    gl.activeTexture(gl.TEXTURE2)
    gl.bindTexture(gl.TEXTURE_2D, nextVideoTexture)
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE)
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE)
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR)
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR)

    video.src = videoSrc
    video.loop = false
    video.muted = true
    video.playsInline = true
    video.preload = 'auto'
    video.playbackRate = playbackRate
    video.load()

    nextVideo.src = videoSrc
    nextVideo.loop = false
    nextVideo.muted = true
    nextVideo.playsInline = true
    nextVideo.preload = 'auto'
    nextVideo.playbackRate = playbackRate
    nextVideo.load()

    const getDrawable = () => {
      if (activeVideo.readyState >= HTMLMediaElement.HAVE_CURRENT_DATA && activeVideo.videoWidth && activeVideo.videoHeight) {
        return {
          source: activeVideo,
          width: activeVideo.videoWidth,
          height: activeVideo.videoHeight,
        }
      }

      if (fallbackImage?.complete && fallbackImage.naturalWidth && fallbackImage.naturalHeight) {
        return {
          source: fallbackImage,
          width: fallbackImage.naturalWidth,
          height: fallbackImage.naturalHeight,
        }
      }

      return null
    }

    const getNextDrawable = () => {
      if (standbyVideo.readyState >= HTMLMediaElement.HAVE_CURRENT_DATA && standbyVideo.videoWidth && standbyVideo.videoHeight) {
        return {
          source: standbyVideo,
          width: standbyVideo.videoWidth,
          height: standbyVideo.videoHeight,
        }
      }

      return null
    }

    const getLoopMix = () => {
      if (!loopVideo) return 0

      const duration = activeVideo.duration
      if (!Number.isFinite(duration) || duration <= 0) return 0

      const remaining = duration - activeVideo.currentTime
      const fadeSeconds = Math.min(ASCII_LOOP_FADE_SECONDS, duration * 0.4)

      if (remaining <= fadeSeconds && !loopFadeActive) {
        loopFadeActive = true
        standbyVideo.currentTime = 0
        standbyVideo.playbackRate = playbackRate
        void standbyVideo.play().catch(() => undefined)
      }

      if (activeVideo.ended || remaining <= 0.035) {
        const previousActive = activeVideo
        activeVideo = standbyVideo
        standbyVideo = previousActive
        activeVideo.playbackRate = playbackRate
        void activeVideo.play().catch(() => undefined)
        standbyVideo.pause()
        standbyVideo.currentTime = 0
        loopFadeActive = false
        return 0
      }

      if (!loopFadeActive) return 0

      const nextDrawable = getNextDrawable()
      if (!nextDrawable) return 0

      const rawMix = Math.max(0, Math.min(1, (fadeSeconds - remaining) / fadeSeconds))
      return rawMix * rawMix * (3 - (2 * rawMix))
    }

    const resize = () => {
      const rect = canvas.getBoundingClientRect()
      const dpr = Math.min(window.devicePixelRatio || 1, ASCII_MAX_DPR)
      const width = Math.max(1, Math.floor((rect.width || window.innerWidth) * dpr))
      const height = Math.max(1, Math.floor((rect.height || window.innerHeight) * dpr))

      if (width !== lastWidth || height !== lastHeight) {
        lastWidth = width
        lastHeight = height
        canvas.width = width
        canvas.height = height
        gl.viewport(0, 0, width, height)
      }

      return { dpr, width, height }
    }

    const render = () => {
      if (stopped) return

      const drawable = getDrawable()
      if (!drawable) return
      const mix = getLoopMix()
      const nextDrawable = mix > 0 ? getNextDrawable() : null
      const { dpr, width, height } = resize()
      const cellWidth = (width / dpr) < 760 ? ASCII_CELL_WIDTH_MOBILE : ASCII_CELL_WIDTH_DESKTOP
      const cellHeight = Math.round(cellWidth * 1.45)

      gl.activeTexture(gl.TEXTURE0)
      gl.bindTexture(gl.TEXTURE_2D, videoTexture)
      gl.pixelStorei(gl.UNPACK_FLIP_Y_WEBGL, true)
      try {
        gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, gl.RGBA, gl.UNSIGNED_BYTE, drawable.source)
      } catch {
        return
      }

      gl.activeTexture(gl.TEXTURE2)
      gl.bindTexture(gl.TEXTURE_2D, nextVideoTexture)
      gl.pixelStorei(gl.UNPACK_FLIP_Y_WEBGL, true)
      try {
        gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, gl.RGBA, gl.UNSIGNED_BYTE, nextDrawable?.source ?? drawable.source)
      } catch {
        return
      }

      gl.useProgram(program)
      gl.bindBuffer(gl.ARRAY_BUFFER, positionBuffer)
      gl.enableVertexAttribArray(aPosition)
      gl.vertexAttribPointer(aPosition, 2, gl.FLOAT, false, 0, 0)
      gl.uniform1i(uVideo, 0)
      gl.uniform1i(uGlyphs, 1)
      gl.uniform1i(uNextVideo, 2)
      gl.uniform2f(uResolution, width, height)
      gl.uniform2f(uVideoSize, drawable.width, drawable.height)
      gl.uniform2f(uNextVideoSize, nextDrawable?.width ?? drawable.width, nextDrawable?.height ?? drawable.height)
      gl.uniform2f(uCellSize, cellWidth * dpr, cellHeight * dpr)
      gl.uniform1f(uCharCount, ASCII_CHARS.length)
      gl.uniform1f(uMix, nextDrawable ? mix : 0)
      gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4)
    }

    const renderLoop = () => {
      render()
      if (!stopped && !document.hidden) {
        animationFrameId = window.requestAnimationFrame(renderLoop)
      }
    }

    const start = () => {
      activeVideo.playbackRate = playbackRate
      void activeVideo.play().catch(() => undefined)
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
        activeVideo.pause()
        standbyVideo.pause()
        return
      }

      void activeVideo.play().catch(() => undefined)
      if (loopFadeActive) void standbyVideo.play().catch(() => undefined)
      if (animationFrameId == null) {
        animationFrameId = window.requestAnimationFrame(renderLoop)
      }
    }

    const handleVideoError = () => onVideoErrorRef.current?.()

    fallbackImage?.addEventListener('load', start)
    video.addEventListener('loadedmetadata', start)
    video.addEventListener('loadeddata', start)
    video.addEventListener('canplay', start)
    video.addEventListener('error', handleVideoError)
    document.addEventListener('visibilitychange', handleVisibilityChange)
    start()

    return () => {
      stopped = true
      fallbackImage?.removeEventListener('load', start)
      video.removeEventListener('loadedmetadata', start)
      video.removeEventListener('loadeddata', start)
      video.removeEventListener('canplay', start)
      video.removeEventListener('error', handleVideoError)
      document.removeEventListener('visibilitychange', handleVisibilityChange)
      if (animationFrameId != null) window.cancelAnimationFrame(animationFrameId)
      gl.deleteTexture(videoTexture)
      gl.deleteTexture(nextVideoTexture)
      gl.deleteTexture(glyphTexture)
      gl.deleteBuffer(positionBuffer)
      gl.deleteProgram(program)
    }
  }, [fallbackImageSrc, loopVideo, playbackRate, videoSrc])

  return (
    <div className="absolute inset-0 overflow-hidden bg-black" role={ariaLabel ? 'img' : undefined} aria-label={ariaLabel}>
      <video
        ref={videoRef}
        className="pointer-events-none absolute h-px w-px opacity-0"
        muted
        playsInline
        preload="auto"
        aria-hidden="true"
      />
      <canvas
        ref={canvasRef}
        className="absolute inset-0 h-full w-full opacity-100 [image-rendering:pixelated]"
        aria-hidden="true"
      />
    </div>
  )
}

function AsciiBackground({
  introMounted,
  showIntro,
  introVideoSrc,
  onIntroVideoError,
}: {
  introMounted: boolean
  showIntro: boolean
  introVideoSrc: string
  onIntroVideoError: () => void
}) {
  return (
    <div className="pointer-events-none absolute inset-0 z-[1] overflow-hidden bg-black" aria-hidden="true">
      <div
        className={[
          'absolute inset-0 transition-opacity duration-700',
          showIntro ? 'opacity-0' : 'opacity-100',
        ].join(' ')}
      >
        <AsciiVideoBackground />
      </div>
      {introMounted && (
        <div
          className={[
            'absolute inset-0 transition-opacity duration-700',
            showIntro ? 'opacity-100' : 'opacity-0',
          ].join(' ')}
        >
          <AsciiVideoBackground
            videoSrc={introVideoSrc}
            playbackRate={1}
            loopVideo={false}
            fallbackImageSrc={null}
            onVideoError={onIntroVideoError}
          />
        </div>
      )}
    </div>
  )
}

export function AppIntroOverlay({ onComplete }: { onComplete: () => void }) {
  const [fading, setFading] = useState(false)
  const [introVideoSrc, setIntroVideoSrc] = useState(TRANSITION_VIDEO_SRC)

  useEffect(() => {
    const fadeTimer = window.setTimeout(() => setFading(true), TRANSITION_VIDEO_MS)
    const completeTimer = window.setTimeout(onComplete, TRANSITION_VIDEO_MS + HOMEPAGE_FADE_MS)

    return () => {
      window.clearTimeout(fadeTimer)
      window.clearTimeout(completeTimer)
    }
  }, [onComplete])

  return (
    <div
      className={[
        'fixed inset-0 z-[95] overflow-hidden bg-black transition-opacity duration-700',
        fading ? 'pointer-events-none opacity-0' : 'opacity-100',
      ].join(' ')}
      aria-hidden="true"
    >
      <AsciiVideoBackground
        videoSrc={introVideoSrc}
        playbackRate={1}
        loopVideo={false}
        fallbackImageSrc={null}
        onVideoError={() => {
          if (introVideoSrc !== backgroundVideo) setIntroVideoSrc(backgroundVideo)
        }}
      />
    </div>
  )
}

export default function Onboarding({ onComplete }: { onComplete: () => void }) {
  const storedSettings = useAppStore(s => s.settings)
  const openaiKey = useAppStore(s => s.openaiKey)
  const prodiaKey = useAppStore(s => s.prodiaKey)
  const setSettings = useAppStore(s => s.setSettings)
  const setOpenAIKey = useAppStore(s => s.setOpenAIKey)
  const setProdiaKey = useAppStore(s => s.setProdiaKey)

  const [showSplash, setShowSplash] = useState(true)
  const [introMounted, setIntroMounted] = useState(true)
  const [exitPhase, setExitPhase] = useState<ExitPhase>('idle')
  const [transitionLayerVisible, setTransitionLayerVisible] = useState(false)
  const [introVideoSrc, setIntroVideoSrc] = useState(TRANSITION_REVERSE_VIDEO_SRC)
  const [transitionVideoSrc, setTransitionVideoSrc] = useState(TRANSITION_VIDEO_SRC)
  const [activeIndex, setActiveIndex] = useState(0)
  const [draft, setDraft] = useState<SystemSettings>({ ...DEFAULT_SYSTEM_SETTINGS, ...storedSettings, theme: 'dark' })
  const [userIdDraft, setUserIdDraft] = useState('')
  const [passwordDraft, setPasswordDraft] = useState('')
  const [demoMode, setDemoMode] = useState(false)
  const [keyDraft, setKeyDraft] = useState(openaiKey)
  const [prodiaKeyDraft, setProdiaKeyDraft] = useState(prodiaKey)
  const [showKey, setShowKey] = useState(false)
  const [showProdiaKey, setShowProdiaKey] = useState(false)
  const [showPassword, setShowPassword] = useState(false)

  useEffect(() => {
    const timer = window.setTimeout(() => setShowSplash(false), SPLASH_DURATION_MS)
    return () => window.clearTimeout(timer)
  }, [])

  useEffect(() => {
    if (showSplash) return

    const timer = window.setTimeout(() => setIntroMounted(false), SETUP_FADE_MS)
    return () => window.clearTimeout(timer)
  }, [showSplash])

  useEffect(() => {
    if (exitPhase !== 'transition-crossfade') return

    const fadeTimer = window.setTimeout(() => setExitPhase('homepage-fade'), TRANSITION_VIDEO_MS)
    const completeTimer = window.setTimeout(onComplete, TRANSITION_VIDEO_MS + HOMEPAGE_FADE_MS)

    return () => {
      window.clearTimeout(fadeTimer)
      window.clearTimeout(completeTimer)
    }
  }, [exitPhase, onComplete])

  const activeStep = STEPS[activeIndex].id
  const savedSummary = useMemo(() => [
    ['Account', demoMode ? 'Demo mode selected' : userIdDraft.trim() ? userIdDraft.trim() : 'Not connected yet'],
    ['Theme', 'Dark'],
    ['Density', draft.density === 'compact' ? 'Compact' : 'Standard'],
    ['Refresh', `${draft.marketRefreshSeconds} sec`],
    ['AI model', draft.openaiModel],
    ['OpenAI key', keyDraft.trim() ? 'Will be saved locally' : 'Skipped for now'],
    ['Prodia key', prodiaKeyDraft.trim() ? 'Will be saved locally' : 'Skipped for now'],
  ], [demoMode, draft, keyDraft, prodiaKeyDraft, userIdDraft])

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
    setProdiaKey(prodiaKeyDraft)
    setTransitionLayerVisible(false)
    setExitPhase('transition-crossfade')
    window.requestAnimationFrame(() => setTransitionLayerVisible(true))
  }

  const isExiting = exitPhase !== 'idle'
  const showTransitionVideo = isExiting

  return (
    <div
      className={[
        'fixed inset-0 isolate z-[90] overflow-y-auto bg-black text-text transition-opacity duration-700',
        exitPhase === 'homepage-fade' ? 'pointer-events-none opacity-0' : 'opacity-100',
      ].join(' ')}
    >
      <AsciiBackground
        introMounted={introMounted}
        showIntro={showSplash}
        introVideoSrc={introVideoSrc}
        onIntroVideoError={() => {
          if (introVideoSrc !== backgroundVideo) setIntroVideoSrc(backgroundVideo)
        }}
      />

      <main
        className={[
          'relative z-[2] mx-auto flex min-h-dvh w-full max-w-6xl items-start px-4 py-4 transition-opacity duration-700 sm:px-6 md:items-center md:py-6',
          showSplash || isExiting ? 'pointer-events-none opacity-0' : 'opacity-100',
        ].join(' ')}
        role="dialog"
        aria-modal="true"
        aria-labelledby="onboarding-title"
        aria-hidden={showSplash || isExiting}
      >
        <section className="grid max-h-[calc(100dvh-2rem)] w-full min-h-0 overflow-hidden border border-white/15 bg-black/68 shadow-2xl shadow-black/35 backdrop-blur-xl md:max-h-[calc(100dvh-3rem)] md:grid-cols-[minmax(160px,220px)_minmax(0,1fr)]">
          <StepRail activeIndex={activeIndex} />

          <div className="flex min-h-0 flex-col p-5 sm:p-8">
            <div className="mb-5 flex shrink-0 items-start justify-between gap-6 border-b border-white/10 pb-5">
              <div>
                <p className="text-2xs font-medium uppercase tracking-[0.28em] text-muted">First run</p>
                <h1 id="onboarding-title" className="mt-2 text-xl font-semibold text-white">TradeSmart setup</h1>
              </div>
              <div className="hidden px-2 py-1 text-2xs text-muted tabnum sm:block">
                {activeIndex + 1}/{STEPS.length}
              </div>
            </div>

            <div className="min-h-0 flex-1 overflow-y-auto pr-1">
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
                <h2 className="text-base font-semibold text-white">Connect the keys used by browser AI features.</h2>
                <p className="mt-3 text-xs leading-6 text-muted">
                  OpenAI and Prodia keys are stored in localStorage on this browser. OpenAI powers Daily Lineup, strategy briefs, and stock chat. Prodia enables VibeOS image generation.
                </p>
                <div className="mt-4 flex flex-wrap gap-4">
                  <a
                    href="https://platform.openai.com/api-keys"
                    target="_blank"
                    rel="noreferrer"
                    className="inline-flex text-xs font-medium text-white underline-offset-4 hover:underline focus-visible:outline focus-visible:outline-2 focus-visible:outline-white/60"
                  >
                    Open OpenAI dashboard to create an API key
                  </a>
                  <a
                    href="https://app.prodia.com/api"
                    target="_blank"
                    rel="noreferrer"
                    className="inline-flex text-xs font-medium text-white underline-offset-4 hover:underline focus-visible:outline focus-visible:outline-2 focus-visible:outline-white/60"
                  >
                    Open Prodia dashboard to create an API key
                  </a>
                </div>

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

                <div className="mt-6 space-y-3">
                  <label htmlFor="onboarding-prodia-key" className={fieldLabelClassName}>
                    Prodia API key
                  </label>
                  <div className="flex gap-2">
                    <input
                      id="onboarding-prodia-key"
                      type={showProdiaKey ? 'text' : 'password'}
                      value={prodiaKeyDraft}
                      onChange={event => setProdiaKeyDraft(event.target.value)}
                      placeholder="prodia-..."
                      spellCheck={false}
                      autoComplete="off"
                      className={`${inputClassName} flex-1 text-xs tabnum`}
                    />
                    <button
                      type="button"
                      onClick={() => setShowProdiaKey(value => !value)}
                      className="flex min-h-[44px] w-12 shrink-0 items-center justify-center rounded-[8px] bg-white/5 text-muted transition-colors hover:bg-white/10 hover:text-text focus-visible:outline focus-visible:outline-2 focus-visible:outline-white/60"
                      aria-label={showProdiaKey ? 'Hide Prodia key' : 'Show Prodia key'}
                      title={showProdiaKey ? 'Hide key' : 'Show key'}
                    >
                      {showProdiaKey ? <EyeOff size={16} aria-hidden="true" /> : <Eye size={16} aria-hidden="true" />}
                    </button>
                  </div>
                </div>

                <div className="mt-8 border-y border-white/10 bg-white/[0.035] py-4">
                  <p className="text-sm font-medium text-white">Live market feed keys</p>
                  <p className="mt-2 text-2xs leading-5 text-muted">
                    Alpaca live data is configured on the server with environment variables. This setup keeps browser credentials limited to the OpenAI and Prodia keys the frontend actually uses.
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
                <div className="mt-6 border-y border-white/10 bg-white/[0.035] py-4">
                  <p className="text-sm font-medium text-white">Work in progress</p>
                  <p className="mt-2 text-2xs leading-5 text-muted">
                    TradeSmart is made by Kenneth Law, a Monash student, and is still a work in progress.
                  </p>
                </div>
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

            <div className="sticky bottom-0 z-10 mt-5 flex shrink-0 items-center justify-between gap-3 border-t border-white/10 bg-black/90 pt-5 backdrop-blur-xl">
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
                  disabled={isExiting}
                  className="flex h-[44px] min-w-[132px] items-center justify-center rounded-[8px] bg-white px-5 text-xs font-semibold text-black transition-opacity hover:opacity-90"
                >
                  Continue
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

      {showTransitionVideo && (
        <div
          className={[
            'fixed inset-0 z-[4] bg-black transition-opacity duration-700',
            transitionLayerVisible && exitPhase !== 'homepage-fade' ? 'opacity-100' : 'opacity-0',
          ].join(' ')}
          aria-hidden="true"
        >
          <AsciiVideoBackground
            videoSrc={transitionVideoSrc}
            playbackRate={1}
            loopVideo={false}
            fallbackImageSrc={null}
            onVideoError={() => {
              if (transitionVideoSrc !== backgroundVideo) setTransitionVideoSrc(backgroundVideo)
            }}
          />
        </div>
      )}
    </div>
  )
}
