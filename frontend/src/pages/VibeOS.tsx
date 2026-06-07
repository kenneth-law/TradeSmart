import { useEffect, useMemo, useRef, useState } from 'react'
import { Link } from 'react-router-dom'
import {
  AppWindow,
  FileText,
  Globe2,
  Layers,
  Monitor,
  RefreshCw,
  Sparkles,
  X,
} from 'lucide-react'
import { isReasoningModel, maxOutputTokensFor, useAppStore } from '../store/useAppStore'
import vibeOsLogo from '../assets/generated/vibeos-logo.png'
import xpAiWallpaper from '../assets/generated/xp-ai-wallpaper.png'

type VibeAppKind = 'terminal' | 'calculator' | 'browser' | 'notes' | 'money' | 'paint' | 'encyclopedia' | 'programs' | 'generic'
type DesktopWallpaper = 'teal' | 'rolling-hills'

interface VibeApp {
  id: string
  title: string
  prompt: string
  kind: VibeAppKind
  version?: number
  savedId?: string
  accent: string
  x: number
  y: number
  w: number
  h: number
}

interface WindowLayout {
  x: number
  y: number
  width: number
  height: number
}

interface DesktopIconLayout {
  x: number
  y: number
}

interface ContextMenuItem {
  label?: string
  onSelect?: () => void
  disabled?: boolean
  separator?: boolean
}

interface ContextMenuState {
  x: number
  y: number
  title?: string
  items: ContextMenuItem[]
}

interface GeneratedWidget {
  label: string
  value: string
  tone?: 'up' | 'down' | 'warn' | 'info' | 'neutral'
}

interface GeneratedEntry {
  title: string
  body: string
  meta?: string
}

type VibeNodeType = 'panel' | 'text' | 'metric' | 'button' | 'progress' | 'list' | 'terminal' | 'canvas' | 'split' | 'grid' | 'input' | 'badge'
type VibeTone = 'up' | 'down' | 'warn' | 'info' | 'neutral' | 'accent'
type VibeSkin = 'plain' | 'terminal' | 'win95' | 'neon' | 'paper' | 'danger' | 'success'

interface VibeNode {
  id: string
  type: VibeNodeType
  title?: string
  text?: string
  label?: string
  value?: string
  meta?: string
  tone?: VibeTone
  skin?: VibeSkin
  progress?: number
  columns?: number
  action?: string
  items?: string[]
  children?: VibeNode[]
}

interface GeneratedVibeSurface {
  headline: string
  subtitle: string
  status: string
  skin: VibeSkin
  state: Record<string, string | number | boolean>
  documentHtml: string
  nodes: VibeNode[]
  widgets: GeneratedWidget[]
  entries: GeneratedEntry[]
  terminalLines: string[]
  buttons: string[]
  caution: string
}

interface VibeSurfaceState {
  loading: boolean
  error?: string
  source: 'ai' | 'local'
  generatedAt?: string
  events?: string[]
  data?: GeneratedVibeSurface
}

interface AiBrowserPage {
  title: string
  url: string
  status: string
  documentHtml: string
}

type BrowserRequest =
  | { kind: 'address'; value: string }
  | { kind: 'search'; value: string }
  | { kind: 'link'; value: string; text?: string }
  | { kind: 'form'; value: string; fields: Record<string, string> }

interface SavedProgram {
  id: string
  promptKey: string
  prompt: string
  title: string
  kind: VibeAppKind
  version: number
  createdAt: string
  updatedAt: string
  app: VibeApp
  surface: GeneratedVibeSurface
}

interface CachedGeneratedImage {
  id: string
  prompt: string
  category: VibeImageCategory
  title: string
  dataUrl: string
  createdAt: string
  updatedAt: string
}

const ACCENTS = ['#E89B2C', '#3B82F6', '#10B981', '#F43F5E', '#A78BFA', '#22D3EE', '#F97316']
const SAVED_PROGRAMS_STORAGE = 'system_desktop_program_registry_v1'
const VIBEOS_DB_NAME = 'tradesmart_vibeos'
const VIBEOS_DB_VERSION = 2
const SAVED_PROGRAMS_STORE = 'saved_programs'
const IMAGE_CACHE_STORE = 'image_cache'
const MAX_SAVED_PROGRAMS = 200
const DESKTOP_WALLPAPER_STORAGE = 'system_desktop_wallpaper_xp_v2'
const PRODIA_IMAGE_URL = 'https://inference.prodia.com/v2/job'
const PRODIA_TOKEN = '***REDACTED_PRODIA_KEY***'
type VibeImageCategory = 'nature' | 'city' | 'technology' | 'food' | 'still_life' | 'abstract' | 'wildlife'

const STARTERS: Array<Pick<VibeApp, 'title' | 'prompt' | 'kind'>> = [
  { title: 'Program Manager', prompt: 'installed program manager', kind: 'programs' },
  { title: 'Market Notes', prompt: 'Hallucination Explorer XP', kind: 'notes' },
  { title: 'Calculator', prompt: 'Windows 95 calculator with one quietly unreliable operator', kind: 'calculator' },
  { title: 'Internet Console', prompt: 'Hallucination Explorer XP', kind: 'browser' },
  { title: 'Command Prompt', prompt: 'NT style command prompt with dry administrative messages', kind: 'terminal' },
]

function SystemLogo({ size = 16, className = '' }: { size?: number; className?: string }) {
  const [imageFailed, setImageFailed] = useState(false)
  const tileGap = Math.max(1, Math.round(size * 0.06))
  const tileRadius = Math.max(1, Math.round(size * 0.08))
  if (!imageFailed) {
    return (
      <img
        src={vibeOsLogo}
        alt=""
        className={className}
        onError={() => setImageFailed(true)}
        style={{ width: size, height: size, maxWidth: size, maxHeight: size, objectFit: 'contain', display: 'block' }}
        aria-hidden="true"
      />
    )
  }

  return (
    <span
      className={className}
      style={{
        width: size,
        height: size,
        maxWidth: size,
        maxHeight: size,
        display: 'inline-grid',
        gridTemplateColumns: '1fr 1fr',
        gap: tileGap,
        position: 'relative',
        filter: 'drop-shadow(1px 1px 1px rgba(0,0,0,0.35))',
      }}
      aria-hidden="true"
    >
      <span style={{ borderRadius: tileRadius, background: 'linear-gradient(135deg,#ff5f3f,#c71f17)' }} />
      <span style={{ borderRadius: tileRadius, background: 'linear-gradient(135deg,#9ca56a,#343a23)' }} />
      <span style={{ borderRadius: tileRadius, background: 'linear-gradient(135deg,#1aa0ff,#064fb7)' }} />
      <span style={{ borderRadius: tileRadius, background: 'linear-gradient(135deg,#ffd94a,#c68f00)' }} />
    </span>
  )
}

function hashString(value: string) {
  let hash = 2166136261
  for (let i = 0; i < value.length; i++) {
    hash ^= value.charCodeAt(i)
    hash = Math.imul(hash, 16777619)
  }
  return hash >>> 0
}

function mulberry32(seed: number) {
  return function next() {
    seed |= 0
    seed = seed + 0x6D2B79F5 | 0
    let t = Math.imul(seed ^ seed >>> 15, 1 | seed)
    t = t + Math.imul(t ^ t >>> 7, 61 | t) ^ t
    return ((t ^ t >>> 14) >>> 0) / 4294967296
  }
}

function pick<T>(items: T[], rand: () => number) {
  return items[Math.floor(rand() * items.length) % items.length]
}

function titleCase(input: string) {
  return input
    .trim()
    .replace(/\s+/g, ' ')
    .split(' ')
    .map(word => word.charAt(0).toUpperCase() + word.slice(1))
    .join(' ')
    .slice(0, 48)
}

function promptKey(prompt: string) {
  return prompt.trim().toLowerCase().replace(/\s+/g, ' ')
}

function versionLabel(version?: number) {
  return `1.${Math.max(0, version ?? 0)}`
}

function displayTitle(title: string, version?: number) {
  return typeof version === 'number' ? `${title} ${versionLabel(version)}` : title
}

function appKindForPrompt(prompt: string): VibeAppKind {
  const text = prompt.toLowerCase()
  if (/(program manager|installed program|saved app|applications)/.test(text)) return 'programs'
  if (/^(https?:\/\/)?[a-z0-9.-]+\.[a-z]{2,}([/?#].*)?$/i.test(prompt.trim())) return 'browser'
  if (/(terminal|bash|shell|cmd|ssh|unix)/.test(text)) return 'terminal'
  if (/(calc|math|money|budget|account|tax|invoice)/.test(text)) return text.includes('calc') ? 'calculator' : 'money'
  if (/(browser|internet|web|google|search|wiki)/.test(text)) return text.includes('wiki') ? 'encyclopedia' : 'browser'
  if (/(paint|draw|image|canvas|photo|art)/.test(text)) return 'paint'
  if (/(note|doc|write|memo|txt)/.test(text)) return 'notes'
  if (/(encyclopedia|encarta|facts|article)/.test(text)) return 'encyclopedia'
  return 'generic'
}

function makeApp(prompt: string, seed: number, index: number): VibeApp {
  const rand = mulberry32(hashString(`${prompt}:${seed}:${index}`))
  const kind = appKindForPrompt(prompt)
  const names = {
    terminal: ['Command Whisperer', 'Bash With Opinions', 'Terminal XCE', 'Shell of Theseus'],
    calculator: ['Calculator Eventually', 'Number Oracle', 'Equals Finder', 'Fiscal Abacus 98'],
    browser: ['Internet Probably', 'Search Adjacent', 'WebView of Dreams', 'Citation Theatre'],
    notes: ['Notepad Deluxe', 'Loose Thoughts', 'Memo Pad 2095', 'Meeting Notes?'],
    money: ['Money95 Redux', 'Account Mirage', 'Budgeteer Fiction', 'Ledger Simulator'],
    paint: ['Paint With Intent', 'Image Guessr', 'Pixel Fog', 'Canvas Maybe'],
    encyclopedia: ['EncycloMaybe 98', 'Fact Museum', 'Knowledge-ish', 'Reference Cloud'],
    programs: ['Program Manager', 'Application Catalog', 'Installed Programs', 'Module Registry'],
    generic: ['App Shaped Object', 'Utility Apparition', 'Productivity?', 'Soft Where'],
  } satisfies Record<VibeAppKind, string[]>

  return {
    id: `${hashString(prompt + seed + index)}`,
    title: prompt.trim() ? titleCase(prompt) : pick(names[kind], rand),
    prompt: prompt.trim() || pick(names[kind], rand),
    kind,
    accent: ACCENTS[Math.floor(rand() * ACCENTS.length)],
    x: 3 + Math.floor(rand() * 28),
    y: 7 + Math.floor(rand() * 22),
    w: 42 + Math.floor(rand() * 18),
    h: 40 + Math.floor(rand() * 18),
  }
}

function normaliseSavedPrograms(value: unknown): SavedProgram[] {
  if (!Array.isArray(value)) return []
  return value
    .filter((item): item is SavedProgram => {
      return Boolean(
        item &&
        typeof item.id === 'string' &&
        typeof item.promptKey === 'string' &&
        typeof item.prompt === 'string' &&
        typeof item.title === 'string' &&
        typeof item.version === 'number' &&
        item.app &&
        item.surface,
      )
    })
    .sort((a, b) => b.updatedAt.localeCompare(a.updatedAt))
    .slice(0, MAX_SAVED_PROGRAMS)
}

function loadLegacySavedPrograms(): SavedProgram[] {
  if (typeof window === 'undefined') return []
  try {
    const raw = window.localStorage.getItem(SAVED_PROGRAMS_STORAGE)
    if (!raw) return []
    return normaliseSavedPrograms(JSON.parse(raw))
  } catch {
    return []
  }
}

function openVibeOsDb() {
  return new Promise<IDBDatabase>((resolve, reject) => {
    if (typeof window === 'undefined' || !window.indexedDB) {
      reject(new Error('IndexedDB is unavailable.'))
      return
    }
    const request = window.indexedDB.open(VIBEOS_DB_NAME, VIBEOS_DB_VERSION)
    request.onupgradeneeded = () => {
      const db = request.result
      if (!db.objectStoreNames.contains(SAVED_PROGRAMS_STORE)) {
        const store = db.createObjectStore(SAVED_PROGRAMS_STORE, { keyPath: 'id' })
        store.createIndex('updatedAt', 'updatedAt', { unique: false })
      }
      if (!db.objectStoreNames.contains(IMAGE_CACHE_STORE)) {
        const store = db.createObjectStore(IMAGE_CACHE_STORE, { keyPath: 'id' })
        store.createIndex('updatedAt', 'updatedAt', { unique: false })
      }
    }
    request.onsuccess = () => resolve(request.result)
    request.onerror = () => reject(request.error ?? new Error('Unable to open VibeOS database.'))
  })
}

function idbRequest<T>(request: IDBRequest<T>) {
  return new Promise<T>((resolve, reject) => {
    request.onsuccess = () => resolve(request.result)
    request.onerror = () => reject(request.error ?? new Error('IndexedDB request failed.'))
  })
}

function idbTransactionDone(transaction: IDBTransaction) {
  return new Promise<void>((resolve, reject) => {
    transaction.oncomplete = () => resolve()
    transaction.onerror = () => reject(transaction.error ?? new Error('IndexedDB transaction failed.'))
    transaction.onabort = () => reject(transaction.error ?? new Error('IndexedDB transaction aborted.'))
  })
}

async function loadSavedPrograms(): Promise<SavedProgram[]> {
  try {
    const db = await openVibeOsDb()
    try {
      const transaction = db.transaction(SAVED_PROGRAMS_STORE, 'readonly')
      const stored = normaliseSavedPrograms(await idbRequest(transaction.objectStore(SAVED_PROGRAMS_STORE).getAll()))
      if (stored.length > 0) return stored

      const legacy = loadLegacySavedPrograms()
      if (legacy.length > 0) void saveSavedPrograms(legacy)
      return legacy
    } finally {
      db.close()
    }
  } catch {
    return loadLegacySavedPrograms()
  }
}

async function saveSavedPrograms(programs: SavedProgram[]) {
  const next = normaliseSavedPrograms(programs)
  try {
    const db = await openVibeOsDb()
    try {
      const transaction = db.transaction(SAVED_PROGRAMS_STORE, 'readwrite')
      const store = transaction.objectStore(SAVED_PROGRAMS_STORE)
      store.clear()
      next.forEach(program => store.put(program))
      await idbTransactionDone(transaction)
      try { window.localStorage.removeItem(SAVED_PROGRAMS_STORAGE) } catch { /* optional cleanup */ }
    } finally {
      db.close()
    }
  } catch {
    // Keep a tiny legacy fallback if IndexedDB is blocked by the browser.
    try {
      window.localStorage.setItem(SAVED_PROGRAMS_STORAGE, JSON.stringify(next.slice(0, 20)))
    } catch {
      // Storage can fail in private browsing or if program documents are too large.
    }
  }
}

function loadDesktopWallpaper(): DesktopWallpaper {
  if (typeof window === 'undefined') return 'rolling-hills'
  try {
    return window.localStorage.getItem(DESKTOP_WALLPAPER_STORAGE) === 'teal' ? 'teal' : 'rolling-hills'
  } catch {
    return 'rolling-hills'
  }
}

function saveDesktopWallpaper(value: DesktopWallpaper) {
  if (typeof window === 'undefined') return
  try {
    window.localStorage.setItem(DESKTOP_WALLPAPER_STORAGE, value)
  } catch {
    // Wallpaper preference is optional.
  }
}

function cleanJson(text: string) {
  const trimmed = text.trim()
  const fenced = trimmed.match(/```(?:json)?\s*([\s\S]*?)```/i)
  if (fenced) return fenced[1].trim()
  const first = trimmed.indexOf('{')
  const last = trimmed.lastIndexOf('}')
  if (first >= 0 && last > first) return trimmed.slice(first, last + 1)
  return trimmed
}

async function requestVibeJson({
  apiKey,
  model,
  system,
  user,
  temperature,
  signal,
}: {
  apiKey: string
  model: string
  system: string
  user: string
  temperature?: number
  signal?: AbortSignal
}): Promise<GeneratedVibeSurface> {
  const reasoning = isReasoningModel(model)
  const body: Record<string, unknown> = {
    model,
    messages: [
      { role: 'system', content: system },
      { role: 'user', content: user },
    ],
    response_format: { type: 'json_object' },
    max_completion_tokens: maxOutputTokensFor(model),
  }
  if (!reasoning && typeof temperature === 'number') body.temperature = temperature

  const response = await fetch('https://api.openai.com/v1/chat/completions', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json', Authorization: `Bearer ${apiKey}` },
    body: JSON.stringify(body),
    signal,
  })
  if (!response.ok) {
    let message = `OpenAI request failed (${response.status})`
    try { const err = await response.json(); message = err?.error?.message ?? message } catch { /* ignore */ }
    throw new Error(message)
  }
  const json = await response.json()
  const content = json?.choices?.[0]?.message?.content
  if (typeof content !== 'string') throw new Error('OpenAI returned an empty response.')
  return normaliseGeneratedSurface(JSON.parse(cleanJson(content)))
}

function normaliseUrl(input: string) {
  const trimmed = input.trim()
  if (!trimmed) return 'google.com'
  if (/\s/.test(trimmed) && !/^https?:\/\//i.test(trimmed)) {
    return `https://www.google.com/search?q=${encodeURIComponent(trimmed)}`
  }
  if (/^[a-z0-9.-]+\.[a-z]{2,}([/?#].*)?$/i.test(trimmed)) return `https://${trimmed}`
  return trimmed
}

function displayUrl(url: string) {
  return url.replace(/^https?:\/\//i, '')
}

function coerceVibeImageCategory(value: unknown): VibeImageCategory | null {
  return ['nature', 'city', 'technology', 'food', 'still_life', 'abstract', 'wildlife'].includes(String(value))
    ? String(value) as VibeImageCategory
    : null
}

function browserImageCategoryFor(input: string): VibeImageCategory {
  const explicit = coerceVibeImageCategory(input)
  if (explicit) return explicit
  const text = input.toLowerCase()
  if (/(food|recipe|restaurant|cook|meal|pizza|coffee|grocery|kitchen)/.test(text)) return 'food'
  if (/(animal|wildlife|zoo|bird|fish|cat|dog|pet|safari|forest creature)/.test(text)) return 'wildlife'
  if (/(city|hotel|travel|airport|map|real estate|street|museum|venue|nytimes|news|university)/.test(text)) return 'city'
  if (/(github|computer|software|tech|internet|browser|code|ai|cloud|device|camera|phone|startup)/.test(text)) return 'technology'
  if (/(art|design|gallery|portfolio|fashion|product|shop|store|catalog|amazon|ebay|object)/.test(text)) return 'still_life'
  if (/(finance|market|search|google|data|analytics|science|research|bank|stock)/.test(text)) return 'abstract'
  return 'nature'
}

function escapeSvgText(value: string) {
  return value.replace(/[&<>"']/g, char => ({
    '&': '&amp;',
    '<': '&lt;',
    '>': '&gt;',
    '"': '&quot;',
    "'": '&apos;',
  }[char] ?? char))
}

function fallbackBrowserImageDataUrl(category: VibeImageCategory, title: string) {
  const palettes: Record<VibeImageCategory, [string, string, string]> = {
    nature: ['#71b64b', '#d9f4ff', '#2f7e34'],
    city: ['#5d84c7', '#dfe8f7', '#293b73'],
    technology: ['#1f7acb', '#bff6ff', '#1f2f69'],
    food: ['#e2a232', '#fff0b6', '#8f3a19'],
    still_life: ['#bd7b36', '#f7e7c6', '#5a4231'],
    abstract: ['#6f63c7', '#d8ecff', '#28437c'],
    wildlife: ['#769337', '#f2e4b8', '#3b5e2d'],
  }
  const [a, b, c] = palettes[category]
  const label = escapeSvgText(title || category).slice(0, 34)
  const svg = `<svg xmlns="http://www.w3.org/2000/svg" width="640" height="420" viewBox="0 0 640 420">
<defs><linearGradient id="g" x1="0" y1="0" x2="1" y2="1"><stop stop-color="${b}"/><stop offset=".5" stop-color="${a}"/><stop offset="1" stop-color="${c}"/></linearGradient></defs>
<rect width="640" height="420" fill="url(#g)"/>
<rect x="24" y="24" width="592" height="372" fill="none" stroke="white" stroke-opacity=".72" stroke-width="8"/>
<circle cx="500" cy="95" r="54" fill="white" fill-opacity=".35"/>
<path d="M0 330 C110 250 190 310 285 240 C395 160 480 270 640 185 L640 420 L0 420 Z" fill="white" fill-opacity=".33"/>
<path d="M0 360 C130 290 240 360 360 275 C475 194 560 305 640 242 L640 420 L0 420 Z" fill="black" fill-opacity=".13"/>
<text x="34" y="62" font-family="Tahoma,Arial,sans-serif" font-size="28" font-weight="700" fill="white" stroke="black" stroke-opacity=".25" stroke-width="2">${label}</text>
<text x="36" y="94" font-family="Tahoma,Arial,sans-serif" font-size="16" fill="white" fill-opacity=".9">Random Image service placeholder</text>
</svg>`
  return `data:image/svg+xml;charset=utf-8,${encodeURIComponent(svg)}`
}

function blobToDataUrl(blob: Blob) {
  return new Promise<string>((resolve, reject) => {
    const reader = new FileReader()
    reader.onload = () => resolve(String(reader.result))
    reader.onerror = () => reject(reader.error ?? new Error('Unable to read image response.'))
    reader.readAsDataURL(blob)
  })
}

function prodiaPromptForImage(category: VibeImageCategory, title: string, context: string) {
  const safeTitle = (title || 'website image').replace(/\s+/g, ' ').slice(0, 120)
  const safeContext = (context || safeTitle).replace(/\s+/g, ' ').slice(0, 180)
  const categoryIntent: Record<VibeImageCategory, string> = {
    nature: 'loose category: nature, outdoors, landscape, plants, weather, terrain, or natural materials if relevant',
    city: 'loose category: places, streets, buildings, public spaces, venues, travel, interiors, or locations if relevant',
    technology: 'loose category: technology, tools, machines, electronics, software, labs, interfaces, or technical subjects if relevant',
    food: 'loose category: food, drink, kitchens, recipes, ingredients, restaurants, or dining if relevant',
    still_life: 'loose category: objects, products, collections, documents, props, tabletop scenes, or catalog-like subjects if relevant',
    abstract: 'loose category: abstract, conceptual, diagrammatic, interface background, texture, pattern, or mood image if relevant',
    wildlife: 'loose category: animals, habitats, pets, field photography, biological subjects, or natural creatures if relevant',
  }
  return [
    `subject: ${safeTitle}`,
    `site/app context: ${safeContext}`,
    categoryIntent[category],
    'the subject and context are more important than the category; do not force a landscape, city, gadget, food plate, or animal unless it fits the requested subject',
    'generate the most useful specific image for this website or desktop app, it can be anything: scene, object, character, map, texture, product, diagram, background, document, portrait, icon-like asset, or reference image',
    'authentic late-1990s to early-2000s photo style, consumer digital camera or scanned print, modest resolution, slight JPEG compression, practical composition',
    'period-appropriate objects, clothing, UI, devices, furniture, and lighting',
    'no modern smartphone look, no cinematic color grading, no ultra-sharp HDR, no glossy stock-photo aesthetic',
    'clear composition, readable at thumbnail size, no text, no logos, no watermark',
  ].join(', ')
}

function imageCacheId(prompt: string) {
  return `prodia_${hashString(prompt)}`
}

async function loadCachedGeneratedImage(id: string): Promise<CachedGeneratedImage | null> {
  try {
    const db = await openVibeOsDb()
    try {
      const transaction = db.transaction(IMAGE_CACHE_STORE, 'readonly')
      const value = await idbRequest(transaction.objectStore(IMAGE_CACHE_STORE).get(id))
      return value && typeof value === 'object' ? value as CachedGeneratedImage : null
    } finally {
      db.close()
    }
  } catch {
    return null
  }
}

async function saveCachedGeneratedImage(image: CachedGeneratedImage) {
  try {
    const db = await openVibeOsDb()
    try {
      const transaction = db.transaction(IMAGE_CACHE_STORE, 'readwrite')
      transaction.objectStore(IMAGE_CACHE_STORE).put(image)
      await idbTransactionDone(transaction)
    } finally {
      db.close()
    }
  } catch {
    // The generated image still renders; cache persistence is optional.
  }
}

async function fetchProdiaGeneratedImage(category: VibeImageCategory, title: string, context: string, signal?: AbortSignal) {
  const prompt = prodiaPromptForImage(category, title, context)
  const id = imageCacheId(prompt)
  const cached = await loadCachedGeneratedImage(id)
  if (cached?.dataUrl) return cached.dataUrl

  try {
    const response = await fetch(PRODIA_IMAGE_URL, {
      method: 'POST',
      headers: {
        Authorization: `Bearer ${PRODIA_TOKEN}`,
        Accept: 'image/jpeg;quality=82;progressive=1',
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        type: 'inference.flux-fast.schnell.txt2img.v2',
        config: {
          prompt,
          steps: 4,
          width: 768,
          height: 512,
        },
      }),
      signal,
    })
    if (!response.ok) return fallbackBrowserImageDataUrl(category, title)
    const dataUrl = await blobToDataUrl(await response.blob())
    const now = new Date().toISOString()
    void saveCachedGeneratedImage({ id, prompt, category, title, dataUrl, createdAt: now, updatedAt: now })
    return dataUrl
  } catch {
    return fallbackBrowserImageDataUrl(category, title)
  }
}

function normaliseBrowserPage(value: unknown, fallbackUrl: string): AiBrowserPage {
  const raw = value && typeof value === 'object' ? value as Partial<AiBrowserPage> : {}
  return {
    title: safeString(raw.title, displayUrl(fallbackUrl) || 'Untitled'),
    url: safeString(raw.url, fallbackUrl),
    status: safeString(raw.status, 'Done'),
    documentHtml: typeof raw.documentHtml === 'string' && raw.documentHtml.trim()
      ? raw.documentHtml
      : '<!doctype html><html><body><h1>Page unavailable</h1><p>The document was empty.</p></body></html>',
  }
}

async function requestBrowserPage({
  apiKey,
  model,
  temperature,
  request,
  currentPage,
  history,
  signal,
}: {
  apiKey: string
  model: string
  temperature?: number
  request: BrowserRequest
  currentPage?: AiBrowserPage
  history: AiBrowserPage[]
  signal?: AbortSignal
}): Promise<AiBrowserPage> {
  const target = request.kind === 'address' || request.kind === 'search'
    ? normaliseUrl(request.value)
    : request.value
  const body: Record<string, unknown> = {
    model,
    messages: [
      {
        role: 'system',
        content: [
          'You are the System Browser network and rendering subsystem.',
          'Return ONLY JSON. No markdown. No prose outside JSON.',
          'Generate a complete web page for the requested URL that looks and feels like the REAL website at that domain BUT IN THE 90s in an EARNEST WAY BEFORE IRONY.',
          'CRITICAL: Use your knowledge of what real websites actually are. If the URL is plane.com generate a page all about aeroplanes. If it is github.com generate a GitHub-like page. If it is amazon.com generate a shopping site. If it is nytimes.com generate a news site. Match the real site\'s actual purpose, branding, content categories, and navigation structure.',
          'For well-known domains you recognize: use the real company name, real product names, real content categories, real color schemes (described via inline CSS), and realistic copy that matches what the site actually does.',
          'For unknown/obscure domains: infer the most plausible purpose from the domain name and generate fitting content.',
          'Visual style should feel like an 1990s or early-2000s version of the real site: flat design, tables, simple CSS, no modern gradients. Functional but period-appropriate.',
          'The page should behave like a live web session: navigation links lead to subpages, search forms work, clicked links produce coherent follow-up pages consistent with the domain.',
          'Do not mention AI, generated content, models, hallucinations, VibeOS, or vibes anywhere in the user-visible page.',
          'Include working anchors and forms. Links should point to plausible relative URLs on the same domain.',
          'Include useful image placeholders where a real website would naturally have images: product photos, article thumbnails, profile photos, venue/travel images, diagrams, hero photos, or gallery items. Use <img data-vibe-random-image="category" alt="descriptive text"> where category is one of nature, city, technology, food, still_life, abstract, wildlife. Do not use external image URLs; the host browser will populate these placeholders.',
          'You CAN include inline <script> tags for client-side interactivity: form validation, tab switching, dropdown menus, calculators, mini games, interactive demos, dynamic content updates. Make the page feel alive and functional.',
          'No external script src or network calls. No credential collection or real private data. Inline scripts only.',
          'Schema: { "title": string, "url": string, "status": string, "documentHtml": string }.',
          'documentHtml must be a full <!doctype html> document with inline CSS and inline <script>. The host browser injects navigation handlers for <a> clicks and form submits, so do not override default link/form behavior unless your script handles its own navigation. Take as much space as you need; there is no character limit. Output must be complete and syntactically valid: close every tag, brace, and paren. Do not stop mid-statement.',
        ].join('\n'),
      },
      {
        role: 'user',
        content: JSON.stringify({
          request,
          target,
          currentPage: currentPage ? { title: currentPage.title, url: currentPage.url, status: currentPage.status } : null,
          recentHistory: history.slice(-5).map(page => ({ title: page.title, url: page.url })),
        }),
      },
    ],
    response_format: { type: 'json_object' },
    max_completion_tokens: maxOutputTokensFor(model),
  }
  if (typeof temperature === 'number' && !isReasoningModel(model)) body.temperature = temperature

  const response = await fetch('https://api.openai.com/v1/chat/completions', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json', Authorization: `Bearer ${apiKey}` },
    body: JSON.stringify(body),
    signal,
  })
  if (!response.ok) {
    let message = `OpenAI request failed (${response.status})`
    try { const err = await response.json(); message = err?.error?.message ?? message } catch { /* ignore */ }
    throw new Error(message)
  }
  const json = await response.json()
  const content = json?.choices?.[0]?.message?.content
  if (typeof content !== 'string') throw new Error('OpenAI returned an empty response.')
  return normaliseBrowserPage(JSON.parse(cleanJson(content)), target)
}

function safeString(value: unknown, fallback: string) {
  return typeof value === 'string' && value.trim() ? value.trim().slice(0, 220) : fallback
}

function normaliseTone(value: unknown): VibeTone {
  return ['up', 'down', 'warn', 'info', 'neutral', 'accent'].includes(String(value)) ? value as VibeTone : 'neutral'
}

function normaliseSkin(value: unknown): VibeSkin {
  return ['plain', 'terminal', 'win95', 'neon', 'paper', 'danger', 'success'].includes(String(value)) ? value as VibeSkin : 'plain'
}

function normaliseNode(value: unknown, indexPath: string): VibeNode | null {
  if (!value || typeof value !== 'object') return null
  const raw = value as Partial<VibeNode>
  const type = ['panel', 'text', 'metric', 'button', 'progress', 'list', 'terminal', 'canvas', 'split', 'grid', 'input', 'badge'].includes(String(raw.type))
    ? raw.type as VibeNodeType
    : 'text'
  const children = Array.isArray(raw.children)
    ? raw.children.map((child, index) => normaliseNode(child, `${indexPath}-${index}`)).filter((child): child is VibeNode => Boolean(child)).slice(0, 18)
    : undefined
  const items = Array.isArray(raw.items)
    ? raw.items.filter((item): item is string => typeof item === 'string').map(item => item.slice(0, 180)).slice(0, 18)
    : undefined
  const progress = Number(raw.progress)
  const columns = Number(raw.columns)

  return {
    id: safeString(raw.id, `node-${indexPath}`).replace(/[^a-zA-Z0-9_-]/g, '-').slice(0, 48),
    type,
    title: typeof raw.title === 'string' ? raw.title.slice(0, 90) : undefined,
    text: typeof raw.text === 'string' ? raw.text.slice(0, 420) : undefined,
    label: typeof raw.label === 'string' ? raw.label.slice(0, 80) : undefined,
    value: typeof raw.value === 'string' || typeof raw.value === 'number' ? String(raw.value).slice(0, 120) : undefined,
    meta: typeof raw.meta === 'string' ? raw.meta.slice(0, 90) : undefined,
    tone: normaliseTone(raw.tone),
    skin: normaliseSkin(raw.skin),
    progress: Number.isFinite(progress) ? Math.max(0, Math.min(100, Math.round(progress))) : undefined,
    columns: Number.isFinite(columns) ? Math.max(1, Math.min(4, Math.round(columns))) : undefined,
    action: typeof raw.action === 'string' ? raw.action.slice(0, 120) : undefined,
    items,
    children,
  }
}

function normaliseState(value: unknown) {
  if (!value || typeof value !== 'object') return {}
  return Object.fromEntries(
    Object.entries(value as Record<string, unknown>)
      .filter(([, item]) => ['string', 'number', 'boolean'].includes(typeof item))
      .slice(0, 16)
      .map(([key, item]) => [key.slice(0, 40), item as string | number | boolean]),
  )
}

function normaliseGeneratedSurface(value: unknown): GeneratedVibeSurface {
  const raw = value && typeof value === 'object' ? value as Partial<GeneratedVibeSurface> : {}
  const widgets = Array.isArray(raw.widgets)
    ? raw.widgets.map(item => {
      const widget = item && typeof item === 'object' ? item as Partial<GeneratedWidget> : {}
      const tone = ['up', 'down', 'warn', 'info', 'neutral'].includes(String(widget.tone)) ? widget.tone : 'neutral'
      return {
        label: safeString(widget.label, 'Metric'),
        value: safeString(widget.value, '??'),
        tone,
      } satisfies GeneratedWidget
    }).slice(0, 6)
    : []
  const entries = Array.isArray(raw.entries)
    ? raw.entries.map(item => {
      const entry = item && typeof item === 'object' ? item as Partial<GeneratedEntry> : {}
      return {
        title: safeString(entry.title, 'Application record'),
        body: safeString(entry.body, 'The operating system believes this is useful.'),
        meta: typeof entry.meta === 'string' ? entry.meta.slice(0, 90) : undefined,
      } satisfies GeneratedEntry
    }).slice(0, 6)
    : []

  return {
    headline: safeString(raw.headline, 'Application module'),
    subtitle: safeString(raw.subtitle, 'Application interface loaded.'),
    status: safeString(raw.status, 'Ready'),
    skin: normaliseSkin(raw.skin),
    state: normaliseState(raw.state),
    documentHtml: typeof raw.documentHtml === 'string' ? raw.documentHtml : '',
    nodes: Array.isArray(raw.nodes)
      ? raw.nodes.map((node, index) => normaliseNode(node, String(index))).filter((node): node is VibeNode => Boolean(node)).slice(0, 24)
      : [],
    widgets,
    entries,
    terminalLines: Array.isArray(raw.terminalLines)
      ? raw.terminalLines.filter((line): line is string => typeof line === 'string').map(line => line.slice(0, 180)).slice(0, 8)
      : [],
    buttons: Array.isArray(raw.buttons)
      ? raw.buttons.filter((line): line is string => typeof line === 'string').map(line => line.slice(0, 34)).slice(0, 8)
      : [],
    caution: safeString(raw.caution, 'Ready.'),
  }
}

function fallbackSurface(app: VibeApp, seed: number): GeneratedVibeSurface {
  const rand = mulberry32(hashString(`${app.prompt}:${seed}:fallback-surface`))
  return normaliseGeneratedSurface({
    headline: `${app.title} module`,
    subtitle: `Compatibility mode for "${app.prompt}".`,
    skin: pick(['terminal', 'win95', 'neon', 'paper'] as const, rand),
    documentHtml: `<!doctype html>
<html>
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width,initial-scale=1" />
<style>
  *{box-sizing:border-box} body{margin:0;min-height:100vh;background:#ece9d8;color:#111;font:12px Tahoma,Arial,sans-serif;padding:12px}
  .fallback{min-height:calc(100vh - 24px);border:1px solid #808080;background:#d4d0c8;padding:10px;box-shadow:inset 1px 1px 0 #fff,inset -1px -1px 0 #404040}
  .title{margin:-10px -10px 10px;padding:5px 8px;background:linear-gradient(90deg,#0a246a,#3a6ea5);color:#fff;font-weight:bold}
  h1{margin:0 0 8px;font-size:18px}.muted{color:#333}.grid{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:8px;margin:12px 0}
  .card{border:1px solid #808080;background:#ece9d8;padding:8px;box-shadow:inset 1px 1px 0 #fff,inset -1px -1px 0 #aaa}
  button{min-height:26px;border:1px solid #404040;background:#d4d0c8;color:#111;padding:0 10px;cursor:pointer;box-shadow:inset 1px 1px 0 #fff,inset -1px -1px 0 #808080}
  #log{font-family:"Lucida Console",monospace;color:#111;white-space:pre-wrap;background:#fff;border:1px solid #808080;padding:8px}
</style>
</head>
<body>
  <main class="fallback">
    <div class="title">Compatibility Mode</div>
    <h1>${app.title.replace(/[<>&]/g, '')}</h1>
    <p class="muted">Compatibility services are active for this program.</p>
    <section class="grid">
      <div class="card">Runtime confidence<br><strong>${Math.round(18 + rand() * 81)}%</strong></div>
      <div class="card">Prompt<br><strong>${app.prompt.replace(/[<>&]/g, '').slice(0, 80)}</strong></div>
    </section>
    <button id="btn">Request Module</button>
    <pre id="log">SYSTEM: sandbox initialized
MODULE: compatibility services active
STATUS: waiting for operator input</pre>
  </main>
  <script>
    const log = document.getElementById('log');
    let clicks = 0;
    document.getElementById('btn').addEventListener('click', () => {
      clicks += 1;
      log.textContent += '\\nREQUEST ' + clicks + ': deferred by policy subsystem';
    });
  </script>
</body>
</html>`,
    state: {
      runtimeLevel: Math.round(40 + rand() * 59),
      actionCount: 0,
      installed: true,
    },
    nodes: [
      {
        id: 'fallback-root',
        type: 'grid',
        columns: 2,
        children: [
          { id: 'fallback-status', type: 'panel', title: 'Compatibility module', text: 'The program is running under compatibility services.', skin: 'terminal' },
          { id: 'fallback-meter', type: 'metric', label: 'Runtime', value: `${Math.round(18 + rand() * 81)}%`, tone: 'accent' },
          { id: 'fallback-progress', type: 'progress', label: 'Service queue', progress: Math.round(35 + rand() * 63), tone: 'warn' },
          { id: 'fallback-button', type: 'button', label: 'Request module', action: 'retry module request', tone: 'accent' },
        ],
      },
      {
        id: 'fallback-terminal',
        type: 'terminal',
        items: Array.from({ length: 5 }, (_, index) => bootLine(`${app.id}:${seed}`, index)),
      },
    ],
    status: pick(['compatibility mode active', 'service active', 'operator input pending', 'module request deferred'], rand),
    widgets: ['Latency', 'Confidence', 'Events', 'Controls'].map(label => ({
      label,
      value: `${Math.round(18 + rand() * 81)}%`,
      tone: pick(['up', 'down', 'warn', 'info', 'neutral'] as const, rand),
    })),
    entries: Array.from({ length: 4 }, (_, index) => ({
      title: pick(['System panel', 'Service note', 'Program feature', 'Interface record'], rand),
      body: pick([
        'The compatibility subsystem produced this panel.',
        'No external process has been started.',
        'The requested feature is represented for interface continuity.',
        'Status is nominal, subject to interpretation by operations.',
      ], rand),
      meta: `local pass ${index + 1}`,
    })),
    terminalLines: Array.from({ length: 5 }, (_, index) => bootLine(`${app.id}:${seed}`, index)),
    buttons: ['Acknowledge', 'Retry', 'Open Module', 'Close'],
    caution: 'Compatibility mode active.',
  })
}

function makeStarterApps(seed: number) {
  return STARTERS.map((app, index) => makeApp(app.prompt, seed, index))
}

function metric(seed: string, min: number, max: number, suffix = '') {
  const rand = mulberry32(hashString(seed))
  return `${Math.round(min + rand() * (max - min))}${suffix}`
}

function bootLine(seed: string, index: number) {
  const rand = mulberry32(hashString(`${seed}:boot:${index}`))
  const modules = ['window manager', 'module registry', 'display driver', 'input service', 'compatibility layer', 'application broker']
  const states = ['loaded', 'verified', 'initialized', 'registered', 'started', 'deferred']
  return `${pick(modules, rand)} ${pick(states, rand)} in ${metric(seed + index, 4, 96)}ms`
}

function XpIcon({
  kind,
  size = 'desktop',
}: {
  kind: VibeAppKind | 'computer' | 'folder' | 'control'
  size?: 'desktop' | 'small'
}) {
  const compact = size === 'small'
  const boxClass = compact ? 'h-5 w-5' : 'h-11 w-11'
  const iconSize = compact ? 13 : 24
  const base = `${boxClass} relative grid shrink-0 place-items-center overflow-hidden rounded-[5px] border border-white/40 shadow-[0_2px_3px_rgba(0,0,0,0.35),inset_1px_1px_0_rgba(255,255,255,0.75)]`

  if (kind === 'terminal') {
    return (
      <span className={`${base} bg-gradient-to-br from-[#121820] via-[#05070a] to-[#29384b]`}>
        <span className="absolute inset-x-1 top-1 h-1 rounded-sm bg-[#2d7dce]" />
        <span className={compact ? 'font-mono text-[7px] font-bold text-white' : 'font-mono text-[11px] font-bold text-white'}>C:\</span>
      </span>
    )
  }

  if (kind === 'calculator') {
    return (
      <span className={`${base} bg-gradient-to-br from-[#f6f8fb] via-[#b8c7d9] to-[#61718c]`}>
        <span className="absolute left-1 right-1 top-1 h-[22%] rounded-sm bg-[#dff7d1] shadow-[inset_1px_1px_0_rgba(255,255,255,0.8)]" />
        <span className="grid grid-cols-3 gap-[2px] pt-3">
          {Array.from({ length: 9 }, (_, index) => (
            <span key={index} className={compact ? 'h-[2px] w-[2px] bg-[#1f2a44]' : 'h-1 w-1 bg-[#1f2a44]'} />
          ))}
        </span>
      </span>
    )
  }

  if (kind === 'browser') {
    return (
      <span className={`${base} bg-gradient-to-br from-[#b8f0ff] via-[#268de8] to-[#0d3c92]`}>
        <Globe2 size={iconSize} className="text-white drop-shadow" aria-hidden="true" />
        <span className="absolute bottom-1 right-1 h-3 w-3 rounded-full bg-[#58c84f] shadow-[inset_1px_1px_0_rgba(255,255,255,0.75)]" />
      </span>
    )
  }

  if (kind === 'notes') {
    return (
      <span className={`${base} bg-gradient-to-br from-[#fffef7] via-[#f4e7a9] to-[#d0a13a]`}>
        <FileText size={iconSize} className="text-[#284a80]" aria-hidden="true" />
        <span className="absolute right-1 top-1 h-2 w-2 rounded-sm bg-white/80" />
      </span>
    )
  }

  if (kind === 'money') {
    return (
      <span className={`${base} bg-gradient-to-br from-[#e9fff1] via-[#43c16b] to-[#127038]`}>
        <Monitor size={iconSize} className="text-white drop-shadow" aria-hidden="true" />
        <span className="absolute bottom-0 right-1 text-[10px] font-bold text-[#f7f35a]">$</span>
      </span>
    )
  }

  if (kind === 'paint') {
    return (
      <span className={`${base} bg-gradient-to-br from-[#fff8e7] via-[#f9c46b] to-[#d65d32]`}>
        <Sparkles size={iconSize} className="text-[#145ca8]" aria-hidden="true" />
        <span className="absolute left-1 top-1 h-2 w-2 rounded-full bg-[#d80d36]" />
        <span className="absolute bottom-1 left-2 h-2 w-2 rounded-full bg-[#208a37]" />
        <span className="absolute bottom-2 right-2 h-2 w-2 rounded-full bg-[#174bd6]" />
      </span>
    )
  }

  if (kind === 'encyclopedia') {
    return (
      <span className={`${base} bg-gradient-to-br from-[#fff7ca] via-[#4d8fd7] to-[#193b85]`}>
        <Layers size={iconSize} className="text-white drop-shadow" aria-hidden="true" />
      </span>
    )
  }

  if (kind === 'programs' || kind === 'folder') {
    return (
      <span className={`${base} bg-gradient-to-br from-[#fff1a6] via-[#f7c948] to-[#d98a18]`}>
        <span className="absolute left-1 top-2 h-2 w-5 rounded-t-[3px] bg-[#fce37b]" />
        <span className="absolute bottom-1 h-6 w-8 rounded-[3px] bg-gradient-to-br from-[#ffe883] to-[#de971e] shadow-[inset_1px_1px_0_rgba(255,255,255,0.75)]" />
        <AppWindow size={compact ? 10 : 16} className="relative z-10 mt-2 text-[#154f9c]" aria-hidden="true" />
      </span>
    )
  }

  if (kind === 'computer') {
    return (
      <span className={`${base} bg-gradient-to-br from-[#f8fbff] via-[#9cc1ee] to-[#416ca8]`}>
        <Monitor size={iconSize} className="text-white drop-shadow" aria-hidden="true" />
      </span>
    )
  }

  if (kind === 'control') {
    return (
      <span className={`${base} bg-gradient-to-br from-[#e8f3ff] via-[#8ab7ea] to-[#245fb3]`}>
        <span className="grid h-[70%] w-[70%] place-items-center rounded-full bg-[#f4f8ff] text-[10px] font-bold text-[#245fb3] shadow-[inset_1px_1px_0_#fff,inset_-1px_-1px_0_rgba(0,0,0,0.25)]">
          XP
        </span>
      </span>
    )
  }

  return (
    <span className={`${base} bg-gradient-to-br from-[#f9fbff] via-[#68a7f0] to-[#1e4b98]`}>
      <AppWindow size={iconSize} className="text-white drop-shadow" aria-hidden="true" />
    </span>
  )
}

function ContextMenu({ menu, onClose }: { menu: ContextMenuState | null; onClose: () => void }) {
  useEffect(() => {
    if (!menu) return
    function close() { onClose() }
    function onKeyDown(event: KeyboardEvent) {
      if (event.key === 'Escape') onClose()
    }
    window.addEventListener('click', close)
    window.addEventListener('pointerdown', close)
    window.addEventListener('keydown', onKeyDown)
    return () => {
      window.removeEventListener('click', close)
      window.removeEventListener('pointerdown', close)
      window.removeEventListener('keydown', onKeyDown)
    }
  }, [menu, onClose])

  if (!menu) return null

  return (
    <div
      role="menu"
      className="absolute z-[95] w-[192px] border border-[#7f9db9] bg-white py-1 text-[11px] text-black shadow-[2px_2px_6px_rgba(0,0,0,0.35)]"
      style={{ left: menu.x, top: menu.y }}
      onClick={event => event.stopPropagation()}
      onPointerDown={event => event.stopPropagation()}
      onContextMenu={event => event.preventDefault()}
    >
      {menu.title && (
        <div className="mb-1 truncate border-b border-[#d6dff7] px-3 pb-1 font-bold text-[#1245a8]">
          {menu.title}
        </div>
      )}
      {menu.items.map((item, index) => item.separator ? (
        <div key={`separator-${index}`} className="my-1 border-t border-[#d6d6d6]" />
      ) : (
        <button
          key={`${item.label}-${index}`}
          type="button"
          role="menuitem"
          disabled={item.disabled}
          onClick={() => {
            if (item.disabled) return
            item.onSelect?.()
            onClose()
          }}
          className="flex h-6 w-full items-center px-6 text-left hover:bg-[#316ac5] hover:text-white disabled:text-[#8a8a8a] disabled:hover:bg-transparent disabled:hover:text-[#8a8a8a]"
        >
          {item.label}
        </button>
      ))}
    </div>
  )
}

function DesktopIcon({
  app,
  layout,
  selected,
  onSelect,
  onOpen,
  onMove,
  onContextMenu,
}: {
  app: VibeApp
  layout: DesktopIconLayout
  selected: boolean
  onSelect: (id: string) => void
  onOpen: (id: string) => void
  onMove: (id: string, layout: DesktopIconLayout) => void
  onContextMenu: (event: React.MouseEvent, app: VibeApp) => void
}) {
  function startDrag(event: React.PointerEvent<HTMLButtonElement>) {
    if (event.button !== 0) return
    onSelect(app.id)
    const startX = event.clientX
    const startY = event.clientY
    const startLayout = layout

    function onPointerMove(moveEvent: PointerEvent) {
      const dx = moveEvent.clientX - startX
      const dy = moveEvent.clientY - startY
      if (Math.abs(dx) < 3 && Math.abs(dy) < 3) return
      onMove(app.id, {
        x: startLayout.x + dx,
        y: startLayout.y + dy,
      })
    }

    function onPointerUp() {
      window.removeEventListener('pointermove', onPointerMove)
      window.removeEventListener('pointerup', onPointerUp)
    }

    window.addEventListener('pointermove', onPointerMove)
    window.addEventListener('pointerup', onPointerUp, { once: true })
  }

  return (
    <button
      type="button"
      onClick={() => onSelect(app.id)}
      onDoubleClick={() => onOpen(app.id)}
      onPointerDown={startDrag}
      onContextMenu={event => onContextMenu(event, app)}
      onKeyDown={event => {
        if (event.key === 'Enter') onOpen(app.id)
      }}
      draggable={false}
      className={[
        'absolute grid h-[82px] w-[78px] touch-none select-none place-items-center gap-1 rounded-[2px] p-1 text-center text-[11px] text-white outline-none',
        selected ? 'bg-[#316ac5]/70' : 'hover:bg-[#6da7ff]/25 focus-visible:bg-[#316ac5]/50',
      ].join(' ')}
      style={{ left: layout.x, top: layout.y }}
      title={app.prompt}
    >
      <XpIcon kind={app.kind} />
      <span
        className={[
          'line-clamp-2 w-full rounded-[1px] px-1 leading-[1.1] [text-shadow:1px_1px_2px_rgba(0,0,0,0.9)]',
          selected ? 'bg-[#0b3d91] text-white outline outline-1 outline-[#dbe9ff]' : '',
        ].join(' ')}
      >
        {app.title}
      </span>
    </button>
  )
}

function WindowChrome({
  app,
  seed,
  surface,
  layout,
  z,
  onFocus,
  onMinimize,
  onClose,
  onGenerate,
  savedPrograms,
  onOpenSaved,
  onNewVersion,
  onMove,
  onResize,
  onContextMenu,
}: {
  app: VibeApp
  seed: number
  surface?: VibeSurfaceState
  layout: WindowLayout
  z: number
  onFocus: () => void
  onMinimize: () => void
  onClose: () => void
  onGenerate: () => void
  savedPrograms: SavedProgram[]
  onOpenSaved: (id: string) => void
  onNewVersion: (prompt: string) => void
  onMove: (layout: WindowLayout) => void
  onResize: (layout: WindowLayout) => void
  onContextMenu: (event: React.MouseEvent, app: VibeApp) => void
}) {
  const [isInteracting, setIsInteracting] = useState(false)

  function startDrag(event: React.PointerEvent<HTMLDivElement>) {
    if (event.button !== 0) return
    event.preventDefault()
    onFocus()
    setIsInteracting(true)
    const startX = event.clientX
    const startY = event.clientY
    const startLayout = layout

    function onPointerMove(moveEvent: PointerEvent) {
      onMove({
        ...startLayout,
        x: startLayout.x + moveEvent.clientX - startX,
        y: startLayout.y + moveEvent.clientY - startY,
      })
    }

    function onPointerUp() {
      setIsInteracting(false)
      window.removeEventListener('pointermove', onPointerMove)
      window.removeEventListener('pointerup', onPointerUp)
    }

    window.addEventListener('pointermove', onPointerMove)
    window.addEventListener('pointerup', onPointerUp, { once: true })
  }

  function startResize(event: React.PointerEvent<HTMLButtonElement>) {
    event.preventDefault()
    event.stopPropagation()
    onFocus()
    setIsInteracting(true)
    const startX = event.clientX
    const startY = event.clientY
    const startLayout = layout

    function onPointerMove(moveEvent: PointerEvent) {
      onResize({
        ...startLayout,
        width: Math.max(300, startLayout.width + moveEvent.clientX - startX),
        height: Math.max(260, startLayout.height + moveEvent.clientY - startY),
      })
    }

    function onPointerUp() {
      setIsInteracting(false)
      window.removeEventListener('pointermove', onPointerMove)
      window.removeEventListener('pointerup', onPointerUp)
    }

    window.addEventListener('pointermove', onPointerMove)
    window.addEventListener('pointerup', onPointerUp, { once: true })
  }

  return (
    <section
      className="absolute min-h-[260px] min-w-[300px] overflow-hidden rounded-t-[8px] border border-[#0054e3] bg-[#ece9d8] shadow-[4px_5px_12px_rgba(0,0,0,0.45)]"
      style={{
        left: layout.x,
        top: layout.y,
        width: layout.width,
        height: layout.height,
        zIndex: z,
      }}
      onMouseDown={onFocus}
      onContextMenu={event => onContextMenu(event, app)}
      aria-label={`${app.title} window`}
    >
      <div
        className="flex h-8 cursor-move touch-none items-center gap-1.5 rounded-t-[6px] border-b border-[#003caa] bg-gradient-to-b from-[#5da7ff] via-[#1768e7] to-[#0b45bd] px-1.5 text-white shadow-[inset_1px_1px_0_rgba(255,255,255,0.55)]"
        onPointerDown={startDrag}
      >
        <XpIcon kind={app.kind} size="small" />
        <div className="min-w-0 flex-1 truncate text-[12px] font-bold text-white [text-shadow:1px_1px_1px_rgba(0,0,0,0.45)]">{app.title}</div>
        <span className="hidden text-[10px] text-white/80 sm:inline">PID {metric(`${app.id}:${seed}:render`, 100, 999)}</span>
        {app.kind !== 'browser' && app.kind !== 'programs' && (
          <button
            type="button"
            onPointerDown={event => event.stopPropagation()}
            onClick={(event) => {
              event.stopPropagation()
              onGenerate()
            }}
            className="grid h-[21px] w-[21px] place-items-center rounded-[3px] border border-white/70 bg-gradient-to-b from-[#77b5ff] to-[#1f5fd1] text-white shadow-[inset_1px_1px_0_rgba(255,255,255,0.8)] hover:brightness-110"
            aria-label={`Refresh ${app.title}`}
          >
            <RefreshCw size={12} aria-hidden="true" />
          </button>
        )}
        <button
          type="button"
          onPointerDown={event => event.stopPropagation()}
          onClick={(event) => {
            event.stopPropagation()
            onMinimize()
          }}
          className="grid h-[21px] w-[21px] place-items-center rounded-[3px] border border-white/70 bg-gradient-to-b from-[#77b5ff] to-[#1f5fd1] pb-1 text-[13px] font-bold leading-none text-white shadow-[inset_1px_1px_0_rgba(255,255,255,0.8)] hover:brightness-110"
          aria-label={`Minimize ${app.title}`}
        >
          _
        </button>
        <button
          type="button"
          onPointerDown={event => event.stopPropagation()}
          onClick={(event) => {
            event.stopPropagation()
            onClose()
          }}
          className="grid h-[21px] w-[21px] place-items-center rounded-[3px] border border-white/70 bg-gradient-to-b from-[#ffb08b] via-[#f06b33] to-[#c73913] text-white shadow-[inset_1px_1px_0_rgba(255,255,255,0.8)] hover:brightness-110"
          aria-label={`Close ${app.title}`}
        >
          <X size={12} aria-hidden="true" />
        </button>
      </div>
      {isInteracting && (
        <div className="absolute inset-0 z-40" style={{ cursor: 'inherit' }} />
      )}
      <div className="h-[calc(100%-32px)] overflow-auto border-x-[3px] border-b-[3px] border-[#0054e3] border-t border-[#6bb6ff] bg-[#ece9d8] p-1 text-black">
        <VibeAppContent
          app={app}
          seed={seed}
          surface={surface}
          onGenerate={onGenerate}
          savedPrograms={savedPrograms}
          onOpenSaved={onOpenSaved}
          onNewVersion={onNewVersion}
        />
      </div>
      <button
        type="button"
        onPointerDown={startResize}
        className="absolute bottom-[3px] right-[3px] z-30 h-4 w-4 touch-none cursor-nwse-resize bg-[#ece9d8] hover:bg-white"
        aria-label={`Resize ${app.title}`}
        title="Resize"
      >
        <svg viewBox="0 0 16 16" className="pointer-events-none h-full w-full" aria-hidden="true">
          <line x1="4" y1="16" x2="16" y2="4" stroke="#808080" strokeWidth="1.5" />
          <line x1="8" y1="16" x2="16" y2="8" stroke="#808080" strokeWidth="1.5" />
          <line x1="12" y1="16" x2="16" y2="12" stroke="#808080" strokeWidth="1.5" />
        </svg>
      </button>
    </section>
  )
}

function VibeAppContent({
  app,
  seed,
  surface,
  onGenerate,
  savedPrograms,
  onOpenSaved,
  onNewVersion,
}: {
  app: VibeApp
  seed: number
  surface?: VibeSurfaceState
  onGenerate: () => void
  savedPrograms: SavedProgram[]
  onOpenSaved: (id: string) => void
  onNewVersion: (prompt: string) => void
}) {
  if (app.kind === 'programs') return <ProgramManagerApp savedPrograms={savedPrograms} onOpenSaved={onOpenSaved} onNewVersion={onNewVersion} />
  if (app.kind === 'browser') return <AiBrowserApp app={app} />
  if (surface?.loading) return <LoadingSurface app={app} />
  if (surface?.data) return <GeneratedSurface app={app} state={surface} onGenerate={onGenerate} />
  if (surface?.error) {
    return (
      <div className="grid h-full place-items-center">
        <div className="max-w-sm border border-[#808080] bg-[#d4d0c8] p-4 text-center text-black shadow-[inset_1px_1px_0_#fff,inset_-1px_-1px_0_#808080]">
          <div className="text-sm font-medium">Program request failed.</div>
          <p className="mt-2 text-2xs text-[#404040]">{surface.error}</p>
          <button
            type="button"
            onClick={onGenerate}
            className="mt-3 inline-flex min-h-[28px] items-center justify-center gap-2 border border-[#404040] bg-[#d4d0c8] px-3 text-2xs text-black shadow-[inset_1px_1px_0_#fff,inset_-1px_-1px_0_#808080] hover:bg-[#ece9d8]"
          >
            Retry
          </button>
        </div>
      </div>
    )
  }
  if (app.kind === 'terminal') return <TerminalApp app={app} seed={seed} />
  if (app.kind === 'calculator') return <CalculatorApp app={app} seed={seed} />
  if (app.kind === 'notes') return <NotesApp app={app} seed={seed} />
  if (app.kind === 'money') return <MoneyApp app={app} seed={seed} />
  if (app.kind === 'paint') return <PaintApp app={app} seed={seed} />
  if (app.kind === 'encyclopedia') return <EncyclopediaApp app={app} seed={seed} />
  return <GenericApp app={app} seed={seed} />
}

function toneClass(tone?: GeneratedWidget['tone']) {
  return {
    up: 'text-up',
    down: 'text-down',
    warn: 'text-warn',
    info: 'text-info',
    neutral: 'text-text',
  }[tone ?? 'neutral']
}

function LoadingSurface({ app }: { app: VibeApp }) {
  return (
    <div className="grid h-full place-items-center">
      <div className="w-full max-w-sm border border-[#808080] bg-[#d4d0c8] p-4 text-center text-black shadow-[inset_1px_1px_0_#fff,inset_-1px_-1px_0_#808080]">
        <Monitor className="mx-auto text-[#0a246a]" size={22} aria-hidden="true" />
        <div className="mt-3 text-sm font-medium">Hallucinating {app.title}</div>
        <p className="mt-1 text-2xs text-[#404040]">Seeing program interface.</p>
      </div>
    </div>
  )
}

function ProgramManagerApp({
  savedPrograms,
  onOpenSaved,
  onNewVersion,
}: {
  savedPrograms: SavedProgram[]
  onOpenSaved: (id: string) => void
  onNewVersion: (prompt: string) => void
}) {
  const sorted = [...savedPrograms].sort((a, b) => b.updatedAt.localeCompare(a.updatedAt))

  return (
    <div className="flex h-full min-h-[360px] flex-col border border-[#808080] bg-[#d4d0c8] text-black">
      <div className="border-b border-[#808080] bg-[#ece9d8] px-2 py-1 text-2xs">
        Installed Applications
      </div>
      {sorted.length === 0 ? (
        <div className="grid flex-1 place-items-center bg-white text-center text-2xs text-[#404040]">
          <div>
            <div>No installed applications were found.</div>
            <div className="mt-1">Open a program from Control Panel to register it.</div>
          </div>
        </div>
      ) : (
        <div className="min-h-0 flex-1 overflow-auto bg-white">
          <table className="w-full text-left text-2xs">
            <thead className="sticky top-0 bg-[#d4d0c8]">
              <tr className="border-b border-[#808080]">
                <th className="px-2 py-1 font-semibold">Name</th>
                <th className="px-2 py-1 font-semibold">Version</th>
                <th className="px-2 py-1 font-semibold">Kind</th>
                <th className="px-2 py-1 font-semibold">Modified</th>
                <th className="px-2 py-1 font-semibold">Action</th>
              </tr>
            </thead>
            <tbody>
              {sorted.map(program => (
                <tr key={program.id} className="border-b border-[#d8d8d8] hover:bg-[#eef3ff]">
                  <td className="px-2 py-1">
                    <div className="font-semibold">{program.title}</div>
                    <div className="max-w-[260px] truncate text-[#606060]" title={program.prompt}>{program.prompt}</div>
                  </td>
                  <td className="px-2 py-1 font-mono">{versionLabel(program.version)}</td>
                  <td className="px-2 py-1">{program.kind}</td>
                  <td className="px-2 py-1">{new Date(program.updatedAt).toLocaleString()}</td>
                  <td className="px-2 py-1">
                    <div className="flex gap-1">
                      <button
                        type="button"
                        onClick={() => onOpenSaved(program.id)}
                        className="min-h-[24px] border border-[#404040] bg-[#d4d0c8] px-2 text-black shadow-[inset_1px_1px_0_#fff,inset_-1px_-1px_0_#808080] hover:bg-[#ece9d8]"
                      >
                        Open
                      </button>
                      <button
                        type="button"
                        onClick={() => onNewVersion(program.prompt)}
                        className="min-h-[24px] border border-[#404040] bg-[#d4d0c8] px-2 text-black shadow-[inset_1px_1px_0_#fff,inset_-1px_-1px_0_#808080] hover:bg-[#ece9d8]"
                      >
                        New Version
                      </button>
                    </div>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
      <div className="border-t border-[#808080] px-2 py-1 text-2xs text-[#404040]">
        {sorted.length} application{sorted.length === 1 ? '' : 's'} registered.
      </div>
    </div>
  )
}

function repairTruncatedHtml(html: string): string {
  if (!/<\/body>/i.test(html)) html += '</body>'
  if (!/<\/html>/i.test(html)) html += '</html>'
  return html
}

function ensureHtmlDocument(html: string, app: VibeApp) {
  const title = app.title.replace(/[<>&"]/g, '')
  const repaired = repairTruncatedHtml(html)
  const hostCss = `<style id="system-host-window-normalizer">
html,body{width:100%;min-height:100%;margin:0!important}
body{display:block!important;place-items:initial!important;align-items:stretch!important;justify-content:flex-start!important;padding:0!important;box-sizing:border-box;overflow:auto}
*,*:before,*:after{box-sizing:border-box}
body>.window,body>.app-window,body>.program-window,body>.program,body>.app,body>.container,body>main,body>#app{width:100%!important;max-width:none!important;min-height:100vh!important;margin:0!important;box-shadow:none!important}
body>.window,body>.app-window,body>.program-window{border:0!important}
body>.window>.titlebar:first-child,body>.window>.title-bar:first-child,body>.app-window>.titlebar:first-child,body>.app-window>.title-bar:first-child,body>.program-window>.titlebar:first-child,body>.program-window>.title-bar:first-child{display:none!important}
</style>`
  const doc = /<html[\s>]/i.test(repaired)
    ? repaired
    : `<!doctype html>
<html>
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width,initial-scale=1" />
<title>${title}</title>
</head>
<body>${repaired}</body>
</html>`
  if (/<\/head>/i.test(doc)) return doc.replace(/<\/head>/i, `${hostCss}</head>`)
  return doc.replace(/<body/i, `<head>${hostCss}</head><body`)
}

function fallbackBrowserPage(request: BrowserRequest, currentPage?: AiBrowserPage): AiBrowserPage {
  const url = request.kind === 'search'
    ? `https://www.google.com/search?q=${encodeURIComponent(request.value)}`
    : request.kind === 'address'
      ? normaliseUrl(request.value)
      : request.value
  const query = request.kind === 'search'
    ? request.value
    : request.kind === 'form'
      ? Object.values(request.fields).join(' ')
      : displayUrl(url)
  const safeQuery = query.replace(/[<>&]/g, '')
  const safeUrl = displayUrl(url).replace(/[<>&]/g, '')
  const isGoogle = /google\.com\/?$/i.test(displayUrl(url)) || /google\.com\/search/i.test(url)

  return {
    title: isGoogle ? 'Google' : safeUrl,
    url,
    status: currentPage ? 'Compatibility mode' : 'Compatibility mode',
    documentHtml: `<!doctype html>
<html>
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width,initial-scale=1" />
<style>
  body{margin:0;background:#fff;color:#000;font:12px Tahoma,Arial,sans-serif}
  a{color:#00e;text-decoration:underline;cursor:pointer} a:visited{color:#551a8b}
  .bar{background:#d4d0c8;border-bottom:1px solid #808080;padding:6px}
  .page{padding:16px;max-width:900px}.logo{font-size:38px;font-family:Georgia,serif;margin:20px 0 12px}
  input[type=text]{border:1px solid #808080;padding:4px;font:12px Tahoma;width:min(520px,80vw)}
  button,input[type=submit]{border:1px solid #404040;background:#d4d0c8;padding:3px 10px;box-shadow:inset 1px 1px 0 #fff,inset -1px -1px 0 #808080}
  .result{margin:14px 0}.url{color:#060;font-size:11px}.small{color:#555}.box{border:1px solid #808080;background:#f5f5f5;padding:10px;margin:10px 0}
  .webphoto{float:right;width:220px;max-width:42%;margin:0 0 12px 16px;border:1px solid #808080;background:#f5f5f5;padding:4px;text-align:center;font-size:11px;color:#555}
  .webphoto img{display:block;width:100%;height:138px;object-fit:cover;border:1px solid #aaa;background:#ddd}
  @media(max-width:620px){.webphoto{float:none;width:100%;max-width:none;margin:0 0 12px}.webphoto img{height:170px}}
</style>
</head>
<body>
  <div class="bar">System Browser · ${safeUrl}</div>
  <main class="page">
    ${isGoogle ? `
      <figure class="webphoto"><img data-vibe-random-image="abstract" alt="A random web image for the search page"><figcaption>Image index sample</figcaption></figure>
      <div class="logo">Google</div>
      <form action="https://www.google.com/search" method="get">
        <input type="text" name="q" value="${safeQuery === safeUrl ? '' : safeQuery}" autofocus />
        <input type="submit" value="Google Search" />
      </form>
      <p class="small">Compatibility services are currently active.</p>
      <div class="result"><a href="/search?q=${encodeURIComponent(safeQuery || 'market data')}">Search result ledger</a><div class="url">google.com/search/local-ledger</div><div>Search results will be assembled after a query is submitted.</div></div>
    ` : `
      <figure class="webphoto"><img data-vibe-random-image="${browserImageCategoryFor(`${safeUrl} ${safeQuery}`)}" alt="A random image related to ${safeQuery || safeUrl}"><figcaption>Random site image</figcaption></figure>
      <h1>${safeUrl}</h1>
      <div class="box">The requested page is represented by compatibility services.</div>
      <form action="https://www.google.com/search" method="get">
        <label>Search this session</label><br />
        <input type="text" name="q" value="${safeQuery}" />
        <input type="submit" value="Search" />
      </form>
      <div class="result"><a href="https://www.google.com/search?q=${encodeURIComponent(safeQuery)}">Search for ${safeQuery || safeUrl}</a><div class="url">google.com/search</div><div>Open search results.</div></div>
      <div class="result"><a href="/about">About this page</a><div class="url">${safeUrl}/about</div><div>Administrative details for the current document.</div></div>
    `}
  </main>
</body>
</html>`,
  }
}

function googleHomePage(): AiBrowserPage {
  return {
    title: 'Google!',
    url: 'https://www.google.com/',
    status: 'Done',
    documentHtml: `<!doctype html>
<html>
<head>
<meta charset="utf-8" />
<title>Google!</title>
<style>
  html,body{margin:0;background:#fff;color:#000;font-family:'Times New Roman', Times, serif;font-size:14px}
  body{display:flex;flex-direction:column;align-items:center;padding:30px 12px 12px}
  .logoWrap{position:relative;display:inline-block;margin-bottom:18px}
  .logo{font-family:'Catull','Times New Roman',serif;font-size:78px;font-weight:bold;letter-spacing:-2px;line-height:1;text-shadow:2px 2px 0 rgba(0,0,0,0.15)}
  .logo .b{color:#0039A6}
  .logo .r{color:#C8102E}
  .logo .y{color:#F4C400}
  .logo .g{color:#008A05}
  .beta{position:absolute;right:-46px;bottom:8px;font-family:'Arial',sans-serif;font-size:14px;letter-spacing:6px;color:#888}
  .searchBlock{background:#e6e6e6;border:1px solid #bdbdbd;padding:14px 18px;text-align:center;width:520px;max-width:96%}
  .tagline{font-family:'Times New Roman',serif;font-size:15px;margin-bottom:8px}
  .searchBlock input[type=text]{font-family:'Times New Roman',serif;font-size:14px;width:300px;padding:2px 4px;border:1px solid #7f7f7f;background:#fff;margin-bottom:8px}
  .btnRow{display:flex;justify-content:center;gap:8px}
  .btnRow button{font-family:'Times New Roman',serif;font-size:13px;padding:2px 10px;background:#dcdcdc;border:1px solid #7f7f7f;color:#000;cursor:pointer}
  .btnRow button:active{border-color:#404040;background:#cccccc}
  .infoRow{display:flex;width:520px;max-width:96%;margin-top:14px;gap:0}
  .infoCol{flex:1;padding:10px 12px;text-align:center;font-family:'Times New Roman',serif;font-size:13px;color:#000}
  .infoCol h3{margin:0 0 6px;font-size:13px;font-weight:normal}
  .infoCol a{color:#0000cc;text-decoration:underline;display:block;line-height:1.5}
  .col1{background:#9fe2cf}
  .col2{background:#7fd6c0}
  .col3{background:#5fc7b0}
  .col3 input{font-family:'Times New Roman',serif;font-size:12px;width:120px;padding:1px 3px;border:1px solid #444;background:#fff;color:#444}
  .col3 button{margin-top:4px;font-size:12px;padding:1px 8px;background:#dcdcdc;border:1px solid #7f7f7f;cursor:pointer}
  .col3 .archive{display:inline;margin-left:6px;color:#0000cc;text-decoration:underline}
  .imageBar{width:520px;max-width:96%;margin-top:12px;border:1px solid #bdbdbd;background:#f4f4f4;padding:8px;text-align:left}
  .imageBar img{float:left;width:120px;height:78px;object-fit:cover;border:1px solid #999;margin-right:10px;background:#ddd}
  .imageBar h3{margin:2px 0 4px;font-family:Arial,sans-serif;font-size:13px}
  .imageBar p{margin:0;font-family:'Times New Roman',serif;font-size:13px}
  .copy{margin-top:18px;text-align:center;font-family:'Times New Roman',serif;font-size:13px;color:#000}
</style>
</head>
<body>
  <div class="logoWrap">
    <div class="logo"><span class="b">G</span><span class="r">o</span><span class="y">o</span><span class="b">g</span><span class="g">l</span><span class="r">e</span><span class="b">!</span></div>
    <div class="beta">BETA</div>
  </div>
  <form class="searchBlock" action="https://www.google.com/search" method="get">
    <div class="tagline">Search the web using Google!</div>
    <input type="text" name="q" autocomplete="off" autofocus />
    <div class="btnRow">
      <button type="submit">Google Search</button>
      <button type="submit" name="btnI" value="1">I'm feeling lucky</button>
    </div>
  </form>
  <div class="infoRow">
    <div class="infoCol col1">
      <h3>Special Searches</h3>
      <a href="/search?q=Stanford">Stanford Search</a>
      <a href="/search?q=Linux">Linux Search</a>
    </div>
    <div class="infoCol col2">
      <a href="/help">Help!</a>
      <a href="/about">About Google!</a>
      <a href="/company">Company Info</a>
      <a href="/logos">Google! Logos</a>
    </div>
    <div class="infoCol col3">
      <div>Get Google!<br>updates monthly:</div>
      <form action="/subscribe" method="get" style="margin-top:4px">
        <input type="text" name="email" value="your e-mail" />
        <br>
        <button type="submit">Subscribe</button>
        <a class="archive" href="/archive">Archive</a>
      </form>
    </div>
  </div>
  <div class="imageBar">
    <img data-vibe-random-image="abstract" alt="Random image from the web directory" />
    <h3>Image Search Preview</h3>
    <p>One sample picture has been selected for this browsing session.</p>
    <div style="clear:both"></div>
  </div>
  <div class="copy">Copyright &copy;1998 Google Inc.</div>
</body>
</html>`,
  }
}

function randomImageBridgeScript(appId: string, defaultCategory: VibeImageCategory) {
  return `<script>
(function(){
  const appId = ${JSON.stringify(appId)};
  const defaultCategory = ${JSON.stringify(defaultCategory)};
  const pendingImages = {};
  const allowedCategories = { nature:1, city:1, technology:1, food:1, still_life:1, abstract:1, wildlife:1 };
  function cleanCategory(value){
    value = String(value || defaultCategory || 'abstract').toLowerCase();
    return allowedCategories[value] ? value : defaultCategory;
  }
  function styleImage(img){
    if(img.dataset.vibeImageStyled) return;
    img.dataset.vibeImageStyled = '1';
    img.style.background = '#d8d8d8';
    img.style.objectFit = img.style.objectFit || 'cover';
  }
  function requestImage(img){
    if(img.dataset.vibeImageRequested) return;
    img.dataset.vibeImageRequested = '1';
    styleImage(img);
    const requestId = appId + ':img:' + Date.now() + ':' + Math.random().toString(36).slice(2);
    pendingImages[requestId] = img;
    parent.postMessage({
      type: 'VIBE_RANDOM_IMAGE',
      appId,
      requestId,
      category: cleanCategory(img.getAttribute('data-vibe-random-image') || img.getAttribute('data-random-image') || img.getAttribute('data-api-ninjas-random-image')),
      title: img.getAttribute('alt') || document.title || location.href
    }, '*');
  }
  function hydrateImages(){
    const images = document.querySelectorAll('img[data-vibe-random-image], img[data-random-image], img[data-api-ninjas-random-image]');
    for(let i = 0; i < images.length; i++) requestImage(images[i]);
  }
  window.addEventListener('message', function(event){
    const data = event.data || {};
    if(data.type !== 'VIBE_RANDOM_IMAGE_RESULT' || data.appId !== appId || !data.requestId) return;
    const img = pendingImages[data.requestId];
    if(!img) return;
    delete pendingImages[data.requestId];
    if(data.src) img.src = data.src;
    if(data.alt && !img.alt) img.alt = data.alt;
    img.removeAttribute('data-vibe-random-image');
    img.removeAttribute('data-random-image');
    img.removeAttribute('data-api-ninjas-random-image');
  });
  if(document.readyState === 'loading') document.addEventListener('DOMContentLoaded', hydrateImages, { once:true });
  else hydrateImages();
})();</script>`
}

function injectRandomImageBridge(html: string, appId: string, context: string) {
  const bridge = randomImageBridgeScript(appId, browserImageCategoryFor(context))
  const doc = /<html[\s>]/i.test(html) ? html : `<!doctype html><html><body>${html}</body></html>`
  if (/<\/body>/i.test(doc)) return doc.replace(/<\/body>/i, `${bridge}</body>`)
  return `${doc}${bridge}`
}

function injectBrowserBridge(html: string, appId: string, page?: AiBrowserPage) {
  const imageBridge = randomImageBridgeScript(appId, browserImageCategoryFor(`${page?.url ?? ''} ${page?.title ?? ''}`))
  const browserBridge = `<script>
(function(){
  const appId = ${JSON.stringify(appId)};
  function closestLink(node){
    while(node && node !== document){ if(node.tagName === 'A' && node.getAttribute('href')) return node; node = node.parentNode; }
    return null;
  }
  document.addEventListener('click', function(event){
    const link = closestLink(event.target);
    if(!link) return;
    const href = link.getAttribute('href');
    if(!href || href.indexOf('javascript:') === 0 || href[0] === '#') return;
    event.preventDefault();
    parent.postMessage({ type:'VIBE_BROWSER_NAVIGATE', appId, href, text:(link.textContent || '').trim() }, '*');
  });
  document.addEventListener('submit', function(event){
    const form = event.target;
    if(!form || form.tagName !== 'FORM') return;
    event.preventDefault();
    const fields = {};
    const data = new FormData(form);
    data.forEach(function(value, key){ fields[key] = String(value); });
    parent.postMessage({ type:'VIBE_BROWSER_FORM', appId, action:form.getAttribute('action') || location.href, fields }, '*');
  });
})();</script>`
  const doc = /<html[\s>]/i.test(html) ? html : `<!doctype html><html><body>${html}</body></html>`
  if (/<\/body>/i.test(doc)) return doc.replace(/<\/body>/i, `${imageBridge}${browserBridge}</body>`)
  return `${doc}${imageBridge}${browserBridge}`
}

function useVibeImageResponder(appId: string, getContext: () => { title: string; context: string }) {
  useEffect(() => {
    function onMessage(event: MessageEvent) {
      const data = event.data as {
        type?: string
        appId?: string
        requestId?: string
        category?: string
        title?: string
      }
      if (!data || data.appId !== appId || data.type !== 'VIBE_RANDOM_IMAGE' || !data.requestId) return
      const source = event.source as Window | null
      const fallback = getContext()
      const title = data.title || fallback.title || 'Generated image'
      const context = fallback.context || title
      const category = coerceVibeImageCategory(data.category) ?? browserImageCategoryFor(`${data.category ?? ''} ${title} ${context}`)
      void fetchProdiaGeneratedImage(category, title, context)
        .then(src => source?.postMessage({
          type: 'VIBE_RANDOM_IMAGE_RESULT',
          appId,
          requestId: data.requestId,
          src,
          alt: title,
        }, '*'))
        .catch(() => source?.postMessage({
          type: 'VIBE_RANDOM_IMAGE_RESULT',
          appId,
          requestId: data.requestId,
          src: fallbackBrowserImageDataUrl(category, title),
          alt: title,
        }, '*'))
    }

    window.addEventListener('message', onMessage)
    return () => window.removeEventListener('message', onMessage)
  }, [appId, getContext])
}

function AiBrowserApp({ app }: { app: VibeApp }) {
  const openaiKey = useAppStore(s => s.openaiKey)
  const settings = useAppStore(s => s.settings)
  const abortRef = useRef<AbortController | null>(null)
  const home = googleHomePage()
  const [address, setAddress] = useState(() => displayUrl(home.url))
  const [history, setHistory] = useState<AiBrowserPage[]>([home])
  const [index, setIndex] = useState(0)
  const [loading, setLoading] = useState(false)
  const [progress, setProgress] = useState(0)
  const [status, setStatus] = useState(home.status)
  const currentPage = index >= 0 ? history[index] : null
  const canGoBack = index > 0
  const canGoForward = index >= 0 && index < history.length - 1

  useVibeImageResponder(app.id, () => ({
    title: currentPage?.title || app.title,
    context: `${currentPage?.url ?? address} ${currentPage?.title ?? app.title}`,
  }))

  function requestUrlFor(request: BrowserRequest): string {
    if (request.kind === 'search') return `https://www.google.com/search?q=${encodeURIComponent(request.value)}`
    return normaliseUrl(request.value)
  }

  async function loadPage(request: BrowserRequest, replace = false) {
    abortRef.current?.abort()
    const controller = new AbortController()
    abortRef.current = controller
    const targetUrl = requestUrlFor(request)
    setAddress(displayUrl(targetUrl))
    setLoading(true)
    setProgress(8)
    setStatus('Contacting site...')

    let progressTimer: ReturnType<typeof setInterval> | null = setInterval(() => {
      setProgress(p => (p >= 90 ? p : p + Math.max(1, Math.round((92 - p) / 12))))
    }, 140)

    try {
      const page = openaiKey
        ? await requestBrowserPage({
          apiKey: openaiKey,
          model: settings.openaiModel,
          temperature: Math.max(settings.openaiTemperature, 0.8),
          request,
          currentPage: currentPage ?? undefined,
          history,
          signal: controller.signal,
        })
        : fallbackBrowserPage(request, currentPage ?? undefined)
      if (controller.signal.aborted) return
      setAddress(displayUrl(page.url))
      setHistory(previous => {
        const base = replace && index >= 0 ? previous.slice(0, index) : previous.slice(0, index + 1)
        const next = [...base, page]
        setIndex(next.length - 1)
        return next
      })
      setStatus(page.status)
    } catch (error) {
      if (controller.signal.aborted) return
      const page = fallbackBrowserPage(request, currentPage ?? undefined)
      setAddress(displayUrl(page.url))
      setHistory(previous => {
        const next = [...previous.slice(0, index + 1), page]
        setIndex(next.length - 1)
        return next
      })
      setStatus('Navigation failed')
    } finally {
      if (progressTimer) { clearInterval(progressTimer); progressTimer = null }
      if (abortRef.current === controller) abortRef.current = null
      if (!controller.signal.aborted) {
        setProgress(100)
        setTimeout(() => setProgress(0), 220)
        setLoading(false)
      }
    }
  }

  useEffect(() => {
    return () => abortRef.current?.abort()
  }, [app.id])

  useEffect(() => {
    function onMessage(event: MessageEvent) {
      const data = event.data as {
        type?: string
        appId?: string
        href?: string
        text?: string
        action?: string
        fields?: Record<string, string>
      }
      if (!data || data.appId !== app.id) return
      if (data.type === 'VIBE_BROWSER_NAVIGATE' && data.href) {
        const nextUrl = new URL(data.href, currentPage?.url ?? normaliseUrl(address)).toString()
        void loadPage({ kind: 'link', value: nextUrl, text: data.text })
      }
      if (data.type === 'VIBE_BROWSER_FORM' && data.fields) {
        const q = data.fields.q || data.fields.search || Object.values(data.fields)[0] || ''
        if (q.trim()) void loadPage({ kind: 'search', value: q })
        else void loadPage({ kind: 'form', value: data.action || currentPage?.url || normaliseUrl(address), fields: data.fields })
      }
    }

    window.addEventListener('message', onMessage)
    return () => window.removeEventListener('message', onMessage)
  })

  function submitAddress(event: React.FormEvent) {
    event.preventDefault()
    void loadPage({ kind: /\s/.test(address.trim()) ? 'search' : 'address', value: address })
  }

  function goBack() {
    if (!canGoBack) return
    const next = index - 1
    setIndex(next)
    setAddress(displayUrl(history[next].url))
    setStatus(history[next].status)
  }

  function goForward() {
    if (!canGoForward) return
    const next = index + 1
    setIndex(next)
    setAddress(displayUrl(history[next].url))
    setStatus(history[next].status)
  }

  return (
    <div className="flex h-full min-h-[420px] flex-col border border-[#808080] bg-[#d4d0c8] text-black">
      <div className="flex min-h-[30px] items-center gap-1 border-b border-[#808080] bg-[#d4d0c8] px-1 shadow-[inset_0_1px_0_#fff]">
        <button type="button" onClick={goBack} disabled={!canGoBack} className="h-6 border border-[#404040] bg-[#d4d0c8] px-2 text-2xs shadow-[inset_1px_1px_0_#fff,inset_-1px_-1px_0_#808080] disabled:text-[#808080]">Back</button>
        <button type="button" onClick={goForward} disabled={!canGoForward} className="h-6 border border-[#404040] bg-[#d4d0c8] px-2 text-2xs shadow-[inset_1px_1px_0_#fff,inset_-1px_-1px_0_#808080] disabled:text-[#808080]">Forward</button>
        <button type="button" onClick={() => void loadPage({ kind: 'address', value: currentPage?.url || address }, true)} className="h-6 border border-[#404040] bg-[#d4d0c8] px-2 text-2xs shadow-[inset_1px_1px_0_#fff,inset_-1px_-1px_0_#808080]">Refresh</button>
        <form onSubmit={submitAddress} className="flex min-w-0 flex-1 items-center gap-1">
          <span className="text-2xs text-[#404040]">Address</span>
          <input
            value={address}
            onChange={event => setAddress(event.target.value)}
            className="h-6 min-w-0 flex-1 border border-[#808080] bg-white px-1 font-mono text-2xs text-black"
          />
          <button type="submit" className="h-6 border border-[#404040] bg-[#d4d0c8] px-2 text-2xs shadow-[inset_1px_1px_0_#fff,inset_-1px_-1px_0_#808080]">Go</button>
        </form>
      </div>
      <div className="flex min-h-[28px] items-center gap-2 border-b border-[#808080] bg-[#ece9d8] px-2">
        <form
          onSubmit={event => {
            event.preventDefault()
            const data = new FormData(event.currentTarget)
            const q = String(data.get('q') ?? '')
            if (q.trim()) void loadPage({ kind: 'search', value: q })
          }}
          className="flex min-w-0 flex-1 items-center gap-1"
        >
          <span className="text-2xs text-[#404040]">Search</span>
          <input name="q" placeholder="Search web" className="h-5 min-w-0 flex-1 border border-[#808080] bg-white px-1 text-2xs" />
          <button type="submit" className="h-5 border border-[#404040] bg-[#d4d0c8] px-2 text-2xs shadow-[inset_1px_1px_0_#fff,inset_-1px_-1px_0_#808080]">Search</button>
        </form>
      </div>
      <div className="relative min-h-0 flex-1 bg-white">
        {progress > 0 && (
          <div className="pointer-events-none absolute left-0 right-0 top-0 z-10 h-[3px] bg-[#e8e8e8]">
            <div
              className="h-full bg-[#1a73e8] transition-[width] duration-150 ease-out"
              style={{ width: `${progress}%` }}
            />
          </div>
        )}
        {currentPage ? (
          <iframe
            key={`${app.id}-${currentPage.url}-${index}`}
            title={currentPage.title}
            sandbox="allow-scripts"
            srcDoc={injectBrowserBridge(currentPage.documentHtml, app.id, currentPage)}
            className="h-full w-full border-0 bg-white"
          />
        ) : (
          <div className="grid h-full place-items-center text-2xs text-[#404040]">Opening browser session...</div>
        )}
      </div>
      <div className="flex min-h-[22px] items-center justify-between border-t border-[#808080] bg-[#d4d0c8] px-2 text-2xs text-[#202020]">
        <span className="truncate">{loading ? 'Working...' : status}</span>
        <span className="shrink-0">{currentPage ? displayUrl(currentPage.url) : 'about:blank'}</span>
      </div>
    </div>
  )
}

function GeneratedSurface({
  app,
  state,
  onGenerate,
}: {
  app: VibeApp
  state: VibeSurfaceState
  onGenerate: () => void
}) {
  const data = state.data
  useVibeImageResponder(app.id, () => ({
    title: data?.headline || app.title,
    context: `${app.title} ${app.prompt} ${data?.headline ?? ''} ${data?.subtitle ?? ''}`,
  }))
  if (!data) return null
  if (data.documentHtml && data.documentHtml.trim().length > 200) {
    return (
      <div className="h-full min-h-[360px]">
        <iframe
          key={`${app.id}-${state.generatedAt ?? data.status}`}
          title={`${app.title} program`}
          sandbox="allow-scripts allow-forms allow-modals allow-popups"
          srcDoc={injectRandomImageBridge(ensureHtmlDocument(data.documentHtml, app), app.id, `${app.title} ${app.prompt} ${data.headline} ${data.subtitle}`)}
          className="h-full w-full border-0 bg-white"
        />
      </div>
    )
  }
  if (state.source === 'ai') {
    return (
      <div className="grid h-full place-items-center bg-[#ece9d8] p-4 text-center">
        <div className="max-w-sm border border-[#808080] bg-[#d4d0c8] p-4 text-[12px] text-black shadow-[inset_1px_1px_0_#fff,inset_-1px_-1px_0_#808080]">
          <div className="mb-2 font-bold">Generation incomplete</div>
          <div className="mb-3 text-[11px]">The model returned metadata but no application body. This usually means the prompt was ambiguous or the response was truncated.</div>
          <button
            type="button"
            onClick={onGenerate}
            className="h-6 border border-[#404040] bg-[#d4d0c8] px-3 text-[11px] shadow-[inset_1px_1px_0_#fff,inset_-1px_-1px_0_#808080] hover:bg-[#ece9d8]"
          >
            Regenerate
          </button>
        </div>
      </div>
    )
  }
  return (
    <div className="grid gap-3">
      <div className="rounded-[8px] border border-white/10 bg-white/[0.035] p-3">
        <div className="flex items-start justify-between gap-3">
          <div className="min-w-0">
            <div className="text-md font-semibold text-text">{data.headline}</div>
            <p className="mt-1 text-2xs text-muted">{data.subtitle}</p>
          </div>
          <button
            type="button"
            onClick={onGenerate}
            className="grid h-8 w-8 shrink-0 place-items-center rounded-[8px] border border-white/10 bg-black/25 text-dim hover:text-text"
            aria-label={`Refresh ${app.title}`}
          >
            <RefreshCw size={14} aria-hidden="true" />
          </button>
        </div>
        <div className="mt-3 inline-flex items-center gap-2 rounded-sm border border-white/10 bg-black/25 px-2 py-1 font-mono text-2xs text-accent">
          <span className="h-1.5 w-1.5 animate-pulse rounded-full bg-accent" aria-hidden="true" />
          {data.status}
        </div>
      </div>

      {data.widgets.length > 0 && (
        <div className="grid grid-cols-2 gap-2 md:grid-cols-3">
          {data.widgets.map((widget, index) => (
            <div key={`${widget.label}-${index}`} className="rounded-[8px] border border-white/10 bg-black/25 p-3">
              <div className="truncate text-2xs uppercase tracking-widest text-dim">{widget.label}</div>
              <div className={`mt-1 truncate font-mono text-sm ${toneClass(widget.tone)}`}>{widget.value}</div>
            </div>
          ))}
        </div>
      )}

      {data.terminalLines.length > 0 && (
        <div className="rounded-[8px] border border-white/10 bg-black/45 p-3 font-mono text-2xs">
          {data.terminalLines.map((line, index) => (
            <div key={`${line}-${index}`} className="truncate text-muted">
              <span className="text-accent">&gt;</span> {line}
            </div>
          ))}
        </div>
      )}

      {data.entries.length > 0 && (
        <div className="grid gap-2">
          {data.entries.map((entry, index) => (
            <article key={`${entry.title}-${index}`} className="rounded-[8px] border border-white/10 bg-white/[0.03] p-3">
              <div className="flex items-start justify-between gap-2">
                <h3 className="text-sm font-medium text-text">{entry.title}</h3>
                {entry.meta && <span className="shrink-0 text-2xs text-dim">{entry.meta}</span>}
              </div>
              <p className="mt-2 text-2xs text-muted">{entry.body}</p>
            </article>
          ))}
        </div>
      )}

      {data.buttons.length > 0 && (
        <div className="flex flex-wrap gap-2">
          {data.buttons.map((button, index) => (
            <button
              key={`${button}-${index}`}
              type="button"
              className="min-h-[34px] rounded-[8px] border border-white/10 bg-white/[0.04] px-3 text-2xs text-muted hover:text-text"
            >
              {button}
            </button>
          ))}
        </div>
      )}

      <div className="rounded-[8px] border border-warn/20 bg-warn/10 p-3 text-2xs text-muted">
        {data.caution}
      </div>
    </div>
  )
}

function TerminalApp({ app, seed }: { app: VibeApp; seed: number }) {
  const rand = mulberry32(hashString(`${app.prompt}:${seed}:terminal`))
  const commands = ['dir C:\\MARKETS', 'type alpha.txt', 'net use mainframe', 'run service-check', 'verify policy', 'find certainty.log']
  const responses = [
    'permission denied: confidence exceeded available evidence',
    'found 14 files, 9 require review',
    'warning: bash has somehow offended both unix and windows',
    'compiled successfully, emotionally unavailable',
    'market open detected; replacing facts with plausible candles',
    'exit code 0, but at what cost',
  ]
  return (
    <div className="font-mono text-2xs">
      {Array.from({ length: 8 }, (_, index) => (
        <div key={index} className="mb-2">
          <span className="text-accent">C:\\SYSTEM</span><span className="text-dim">&gt; </span>
          <span className="text-text">{pick(commands, rand)}</span>
          <div className="pl-4 text-muted">{pick(responses, rand)}</div>
        </div>
      ))}
      <div className="mt-3 inline-flex items-center gap-2 border border-white/10 bg-black/30 px-2 py-1 text-dim">
        <span className="h-2 w-2 animate-pulse rounded-full bg-up" aria-hidden="true" />
        output stream active
      </div>
    </div>
  )
}

function CalculatorApp({ app, seed }: { app: VibeApp; seed: number }) {
  const rand = mulberry32(hashString(`${app.prompt}:${seed}:calculator`))
  const left = Math.floor(3 + rand() * 97)
  const right = Math.floor(2 + rand() * 21)
  const ops = ['+', '-', 'x', '/', '^']
  const op = pick(ops, rand)
  const answer = op === '+'
    ? left + right
    : op === '-'
      ? left - right
      : op === 'x'
        ? left * right
        : op === '/'
          ? left / right
          : Math.pow(left % 9, right % 4)
  const buttons = ['7', '8', '9', '/', '4', '5', '6', 'x', '1', '2', '3', '-', '0', '.', '=', '+']
  return (
    <div className="grid gap-3">
      <div className="rounded-[8px] border border-white/10 bg-black/35 p-3 text-right font-mono">
        <div className="text-2xs text-dim">{left} {op} {right}</div>
        <div className="text-xl text-text">{Number(answer).toFixed(rand() > 0.45 ? 2 : 6)}</div>
      </div>
      <div className="grid grid-cols-4 gap-2">
        {buttons.sort(() => rand() - 0.5).map((button, index) => (
          <button
            key={`${button}-${index}`}
            type="button"
            className="h-10 rounded-[8px] border border-white/10 bg-white/[0.04] text-sm text-muted hover:border-white/20 hover:text-text"
          >
            {button}
          </button>
        ))}
      </div>
    </div>
  )
}

function NotesApp({ app, seed }: { app: VibeApp; seed: number }) {
  const rand = mulberry32(hashString(`${app.prompt}:${seed}:notes`))
  const fragments = [
    'Ship the UI first, invent the kernel during the meeting.',
    'Every button needs a backup button in case it loses faith.',
    'Risk note: charts may become self-aware but still lag 15 minutes.',
    'CEO asked if we can replace the database with enthusiasm.',
    'Normalize operating assumptions before market open.',
  ]
  return (
    <div className="min-h-full rounded-[8px] border border-amber-300/15 bg-[#18130b] p-4 font-mono text-2xs leading-relaxed text-amber-50/80">
      <div className="mb-3 text-amber-300">untitled-final-real-final.txt</div>
      {Array.from({ length: 5 }, (_, index) => (
        <p key={index}>- {pick(fragments, rand)}</p>
      ))}
    </div>
  )
}

function MoneyApp({ app, seed }: { app: VibeApp; seed: number }) {
  const rand = mulberry32(hashString(`${app.prompt}:${seed}:money`))
  const rows = ['Tacos', 'Cloud GPUs', 'Therapy for buttons', 'Short NVDA, emotionally', 'Copilot SDK', 'Found money']
  return (
    <div className="grid gap-3">
      <div className="grid grid-cols-3 gap-2">
        {['Balance', 'Runway', 'Regret'].map((label, index) => (
          <div key={label} className="rounded-[8px] border border-white/10 bg-white/[0.035] p-3">
            <div className="text-2xs text-dim">{label}</div>
            <div className="mt-1 font-mono text-md text-text">${metric(app.id + seed + index, 200, 99000)}</div>
          </div>
        ))}
      </div>
      <div className="overflow-hidden rounded-[8px] border border-white/10">
        {rows.map((row, index) => (
          <div key={row} className="grid grid-cols-[1fr_auto] border-t border-white/10 px-3 py-2 first:border-t-0">
            <span className="text-2xs text-muted">{row}</span>
            <span className={rand() > 0.52 ? 'font-mono text-2xs text-up' : 'font-mono text-2xs text-down'}>
              {rand() > 0.52 ? '+' : '-'}${metric(app.id + row + index, 9, 4200)}
            </span>
          </div>
        ))}
      </div>
    </div>
  )
}

function PaintApp({ app, seed }: { app: VibeApp; seed: number }) {
  const rand = mulberry32(hashString(`${app.prompt}:${seed}:paint`))
  const colors = ['#0f172a', '#E89B2C', '#3B82F6', '#10B981', '#F43F5E', '#fafafa']
  return (
    <div className="grid gap-3">
      <div className="flex gap-2">
        {colors.slice(1).map(color => (
          <span key={color} className="h-6 w-6 rounded-sm border border-white/20" style={{ background: color }} />
        ))}
      </div>
      <div className="grid aspect-[16/10] grid-cols-12 overflow-hidden rounded-[8px] border border-white/10 bg-black">
        {Array.from({ length: 120 }, (_, index) => (
          <span
            key={index}
            className="border border-black/20"
            style={{ background: pick(colors, rand), opacity: 0.58 + rand() * 0.42 }}
          />
        ))}
      </div>
    </div>
  )
}

function EncyclopediaApp({ app, seed }: { app: VibeApp; seed: number }) {
  const rand = mulberry32(hashString(`${app.prompt}:${seed}:encyclopedia`))
  const topics = ['Mark Russinovich', 'Sloperating Systems', 'Roguelike UI', 'Hallucinated Kernels', 'Productivity Theatre']
  const facts = [
    'Selected fact: internals become folklore when presented in a beige window.',
    'Reliability score is high because the paragraph sounds laminated.',
    'The citation was last seen near a conference demo.',
    'Related media unavailable, but the thumbnail feels correct.',
  ]
  return (
    <div className="grid gap-3 md:grid-cols-[150px_1fr]">
      <div className="aspect-square rounded-[8px] border border-white/10 bg-gradient-to-br from-s2 via-black to-s1 p-3">
        <div className="grid h-full place-items-center border border-dashed border-white/15 text-center text-2xs text-dim">
          image confidently omitted
        </div>
      </div>
      <div>
        <h3 className="text-md font-semibold text-text">{pick(topics, rand)}</h3>
        <p className="mt-2 text-2xs text-muted">{pick(facts, rand)}</p>
        <div className="mt-3 space-y-2">
          {Array.from({ length: 4 }, (_, index) => (
            <div key={index} className="h-2 rounded-full bg-white/10">
              <div className="h-full rounded-full bg-accent" style={{ width: `${28 + rand() * 62}%` }} />
            </div>
          ))}
        </div>
      </div>
    </div>
  )
}

function GenericApp({ app, seed }: { app: VibeApp; seed: number }) {
  const rand = mulberry32(hashString(`${app.prompt}:${seed}:generic`))
  const cards = ['Intent', 'Output', 'Confidence', 'Side effect', 'UI entropy', 'Shareholder value']
  return (
    <div className="grid gap-3 sm:grid-cols-2">
      {cards.map(card => (
        <div key={card} className="rounded-[8px] border border-white/10 bg-white/[0.035] p-3">
          <div className="text-2xs uppercase tracking-widest text-dim">{card}</div>
          <div className="mt-2 text-sm text-text">{metric(`${app.id}:${seed}:${card}`, 8, 99)}%</div>
          <div className="mt-2 h-2 rounded-full bg-white/10">
            <div className="h-full rounded-full" style={{ width: `${20 + rand() * 76}%`, background: app.accent }} />
          </div>
        </div>
      ))}
    </div>
  )
}

function playLoginChime() {
  try {
    const AudioContextCtor = window.AudioContext || (window as unknown as { webkitAudioContext?: typeof AudioContext }).webkitAudioContext
    if (!AudioContextCtor) return
    const ctx = new AudioContextCtor()
    const now = ctx.currentTime
    const master = ctx.createGain()
    const filter = ctx.createBiquadFilter()
    const delay = ctx.createDelay(1.2)
    const feedback = ctx.createGain()
    const delayMix = ctx.createGain()

    master.gain.setValueAtTime(0.0001, now)
    master.gain.exponentialRampToValueAtTime(0.18, now + 0.08)
    master.gain.exponentialRampToValueAtTime(0.0001, now + 2.65)
    filter.type = 'lowpass'
    filter.frequency.setValueAtTime(1300, now)
    filter.frequency.exponentialRampToValueAtTime(5600, now + 0.72)
    filter.frequency.exponentialRampToValueAtTime(2400, now + 2.2)
    filter.Q.setValueAtTime(0.65, now)
    delay.delayTime.setValueAtTime(0.185, now)
    feedback.gain.setValueAtTime(0.22, now)
    delayMix.gain.setValueAtTime(0.18, now)

    filter.connect(master)
    master.connect(ctx.destination)
    filter.connect(delay)
    delay.connect(feedback)
    feedback.connect(delay)
    delay.connect(delayMix)
    delayMix.connect(ctx.destination)

    const tones = [
      { frequency: 391.99, start: 0.00, duration: 1.45, gain: 0.11, type: 'sine' as OscillatorType, detune: -7 },
      { frequency: 493.88, start: 0.06, duration: 1.52, gain: 0.10, type: 'triangle' as OscillatorType, detune: 11 },
      { frequency: 587.33, start: 0.19, duration: 1.36, gain: 0.08, type: 'sine' as OscillatorType, detune: -14 },
      { frequency: 739.99, start: 0.38, duration: 1.35, gain: 0.075, type: 'triangle' as OscillatorType, detune: 9 },
      { frequency: 987.77, start: 0.66, duration: 1.05, gain: 0.052, type: 'sine' as OscillatorType, detune: -19 },
      { frequency: 1174.66, start: 0.82, duration: 0.95, gain: 0.038, type: 'sine' as OscillatorType, detune: 23 },
    ]

    tones.forEach((tone, index) => {
      const osc = ctx.createOscillator()
      const gain = ctx.createGain()
      const pan = ctx.createStereoPanner()
      const start = now + tone.start
      const end = start + tone.duration

      osc.type = tone.type
      osc.frequency.setValueAtTime(tone.frequency, start)
      osc.detune.setValueAtTime(tone.detune, start)
      osc.detune.linearRampToValueAtTime(tone.detune * -0.35, end)
      gain.gain.setValueAtTime(0.0001, start)
      gain.gain.exponentialRampToValueAtTime(tone.gain, start + 0.09)
      gain.gain.exponentialRampToValueAtTime(Math.max(0.012, tone.gain * 0.32), start + tone.duration * 0.55)
      gain.gain.exponentialRampToValueAtTime(0.0001, end)
      pan.pan.setValueAtTime(index % 2 === 0 ? -0.22 : 0.18, start)
      pan.pan.linearRampToValueAtTime(index % 2 === 0 ? 0.08 : -0.1, end)
      osc.connect(gain)
      gain.connect(pan)
      pan.connect(filter)
      osc.start(start)
      osc.stop(end + 0.05)
    })

    const sparkleTimes = [0.48, 0.61, 0.94, 1.08]
    sparkleTimes.forEach((startOffset, index) => {
      const osc = ctx.createOscillator()
      const gain = ctx.createGain()
      const start = now + startOffset
      const duration = 0.34 + index * 0.04
      osc.type = 'sine'
      osc.frequency.setValueAtTime([1567.98, 1318.51, 1760, 1479.98][index], start)
      osc.detune.setValueAtTime(index % 2 === 0 ? 17 : -21, start)
      gain.gain.setValueAtTime(0.0001, start)
      gain.gain.exponentialRampToValueAtTime(0.03, start + 0.018)
      gain.gain.exponentialRampToValueAtTime(0.0001, start + duration)
      osc.connect(gain)
      gain.connect(filter)
      osc.start(start)
      osc.stop(start + duration + 0.03)
    })

    window.setTimeout(() => void ctx.close(), 2900)
  } catch {
    // Audio can be blocked by browser policy or unavailable devices.
  }
}

function LoginScreen({
  username,
  password,
  onUsername,
  onPassword,
  onLogin,
}: {
  username: string
  password: string
  onUsername: (value: string) => void
  onPassword: (value: string) => void
  onLogin: () => void
}) {
  return (
    <div
      className="grid h-full min-h-0 place-items-center overflow-hidden bg-gradient-to-br from-[#2d6fd3] via-[#1957b7] to-[#0c347d] p-4 text-[11px] leading-[1.25] text-black"
      style={{
        height: '100dvh',
        fontFamily: 'Tahoma, "MS Sans Serif", Arial, sans-serif',
      }}
    >
      <div className="w-[min(94vw,520px)] overflow-hidden rounded-[8px] border border-[#0054e3] bg-[#ece9d8] shadow-[0_18px_38px_rgba(0,0,0,0.45)]">
        <div className="flex h-8 items-center gap-1.5 bg-gradient-to-b from-[#5da7ff] via-[#1768e7] to-[#0b45bd] px-2 text-white">
          <SystemLogo size={18} />
          <span className="min-w-0 flex-1 truncate text-[12px] font-bold [text-shadow:1px_1px_1px_rgba(0,0,0,0.45)]">Log On to Hallucination XP</span>
        </div>
        <form
          className="text-[11px]"
          onSubmit={event => {
            event.preventDefault()
            onLogin()
          }}
        >
          <div className="grid min-h-[210px] grid-cols-[230px_minmax(0,1fr)] border-b border-[#9eb8e8] max-sm:grid-cols-1">
            <div className="grid place-items-center overflow-hidden border-r border-[#9eb8e8] bg-gradient-to-br from-[#f8fbff] to-[#b8d5ff] p-5 max-sm:min-h-[150px] max-sm:border-r-0 max-sm:border-b">
              <div className="grid gap-3 text-center text-[#174aa5]">
                <SystemLogo size={132} className="mx-auto" />
                <span className="text-[20px] font-bold">Hallucination XP</span>
              </div>
            </div>
            <div className="bg-[#eef5ff] p-5 text-[#202020]">
              <div className="text-[13px] font-bold text-[#174aa5]">Enter network password</div>
              <p className="mt-1 text-[11px] text-[#404040]">
                Type a user name and password to log on to this workstation.
              </p>
            </div>
          </div>

          <div className="grid grid-cols-[88px_minmax(0,1fr)] items-center gap-x-2 gap-y-2 px-1 py-2 text-[11px]">
            <label htmlFor="system-username">User name:</label>
            <input
              id="system-username"
              value={username}
              onChange={event => onUsername(event.target.value)}
              className="h-6 border border-[#7f9db9] bg-white px-2 text-[11px] text-black shadow-[inset_1px_1px_2px_rgba(0,0,0,0.15)]"
              autoFocus
            />
            <label htmlFor="system-password">Password:</label>
            <input
              id="system-password"
              value={password}
              onChange={event => onPassword(event.target.value)}
              type="password"
              className="h-6 border border-[#7f9db9] bg-white px-2 text-[11px] text-black shadow-[inset_1px_1px_2px_rgba(0,0,0,0.15)]"
            />
            <span>Domain:</span>
            <div className="h-6 border border-[#7f9db9] bg-white px-2 py-1 text-[11px] shadow-[inset_1px_1px_2px_rgba(0,0,0,0.15)]">
              LOCAL
            </div>
          </div>

          <div className="flex justify-end gap-2 border-t border-[#9eb8e8] bg-[#d7e6ff] px-3 py-2">
            <button
              type="submit"
              className="min-h-[24px] min-w-[76px] rounded-[3px] border border-[#316ac5] bg-gradient-to-b from-white to-[#d7e7ff] px-3 text-[11px] text-black hover:brightness-105"
            >
              OK
            </button>
            <button
              type="button"
              onClick={() => onPassword('')}
              className="min-h-[24px] min-w-[76px] rounded-[3px] border border-[#316ac5] bg-gradient-to-b from-white to-[#d7e7ff] px-3 text-[11px] text-black hover:brightness-105"
            >
              Cancel
            </button>
          </div>
        </form>
      </div>
    </div>
  )
}

function WelcomeDialog({ username, onClose }: { username: string; onClose: () => void }) {
  return (
    <div className="absolute inset-0 z-[80] grid place-items-center bg-black/20 p-4">
      <section className="w-[min(94vw,540px)] overflow-hidden rounded-t-[8px] border border-[#0054e3] bg-[#ece9d8] text-[11px] leading-[1.25] text-black shadow-[4px_5px_14px_rgba(0,0,0,0.45)]">
        <div className="flex h-8 items-center gap-1.5 bg-gradient-to-b from-[#5da7ff] via-[#1768e7] to-[#0b45bd] px-1.5 text-white">
          <SystemLogo size={18} />
          <span className="min-w-0 flex-1 truncate text-[12px] font-bold [text-shadow:1px_1px_1px_rgba(0,0,0,0.45)]">Welcome to Hallucination XP</span>
          <button
            type="button"
            onClick={onClose}
            className="grid h-[21px] w-[21px] place-items-center rounded-[3px] border border-white/70 bg-gradient-to-b from-[#ffb08b] via-[#f06b33] to-[#c73913] text-white shadow-[inset_1px_1px_0_rgba(255,255,255,0.8)] hover:brightness-110"
            aria-label="Close welcome dialog"
          >
            <X size={10} aria-hidden="true" />
          </button>
        </div>
        <div className="grid gap-3 p-4 sm:grid-cols-[48px_minmax(0,1fr)]">
          <span className="grid h-10 w-10 place-items-center self-start border border-[#808080] bg-[#ece9d8] p-1.5 shadow-[inset_1px_1px_0_#fff,inset_-1px_-1px_0_#aaa]">
            <SystemLogo size={28} />
          </span>
          <div className="min-w-0">
            <h2 className="text-[13px] font-bold leading-[1.2] text-[#174aa5]">Setup has completed successfully.</h2>
            <p className="mt-2 text-[11px]">
              Thank you, {username || 'Operator'}, for starting this Hallucination XP workstation.
            </p>
            <div className="mt-3 border border-[#808080] bg-white p-2 text-[11px] leading-[1.35] shadow-[inset_1px_1px_0_#aaa,inset_-1px_-1px_0_#fff]">
              <div><span className="font-bold">Version:</span> 5.1.2600 Compatibility Build</div>
              <div><span className="font-bold">Author:</span> Kenneth Law</div>
              <div><span className="font-bold">License:</span> One workstation, many tiny programs.</div>
            </div>
            <p className="mt-3 text-[11px]">
              Shortcuts can be selected, dragged, double-clicked, and right-clicked just like a very earnest desktop.
            </p>
            <p className="mt-2 text-[10px] text-[#404040]">
              Product support is available through Control Panel, naturally.
            </p>
          </div>
        </div>
        <div className="flex justify-end gap-2 border-t border-[#9eb8e8] bg-[#d7e6ff] px-3 py-2">
          <button
            type="button"
            onClick={onClose}
            className="min-h-[24px] min-w-[78px] rounded-[3px] border border-[#316ac5] bg-gradient-to-b from-white to-[#d7e7ff] px-3 text-[11px] text-black hover:brightness-105"
          >
            Continue
          </button>
        </div>
      </section>
    </div>
  )
}

function StartMenu({
  open,
  onClose,
  onOpenControlPanel,
  onOpenProgramManager,
  onLaunch,
}: {
  open: boolean
  onClose: () => void
  onOpenControlPanel: () => void
  onOpenProgramManager: () => void
  onLaunch: (prompt: string) => void
}) {
  const [search, setSearch] = useState('')
  const inputRef = useRef<HTMLInputElement>(null)

  useEffect(() => {
    if (open) setTimeout(() => inputRef.current?.focus(), 50)
    else setSearch('')
  }, [open])

  if (!open) return null

  const launch = (value: string) => {
    onLaunch(value)
    onClose()
  }

  const launchSearch = () => {
    const q = search.trim()
    if (q) launch(q)
  }

  const primaryRows: Array<{ label: string; kind: VibeAppKind | 'computer' | 'folder' | 'control'; action: () => void; strong?: boolean }> = [
    { label: 'Internet', kind: 'browser', action: () => launch('Internet Console'), strong: true },
    { label: 'E-mail', kind: 'notes', action: () => launch('Outlook Express'), strong: true },
    { label: 'Command Prompt', kind: 'terminal', action: () => launch('Command Prompt') },
    { label: 'Calculator', kind: 'calculator', action: () => launch('Calculator') },
    { label: 'Paint', kind: 'paint', action: () => launch('Paint') },
    { label: 'Program Manager', kind: 'programs', action: onOpenProgramManager },
  ]

  const secondaryRows: Array<{ label: string; kind: VibeAppKind | 'computer' | 'folder' | 'control'; action: () => void }> = [
    { label: 'My Documents', kind: 'folder', action: () => launch('My Documents') },
    { label: 'My Recent Documents', kind: 'folder', action: () => launch('Recent Documents') },
    { label: 'My Pictures', kind: 'paint', action: () => launch('My Pictures') },
    { label: 'My Computer', kind: 'computer', action: () => launch('My Computer') },
    { label: 'Control Panel', kind: 'control', action: onOpenControlPanel },
    { label: 'Help and Support', kind: 'encyclopedia', action: () => launch('Help and Support') },
    { label: 'Search', kind: 'browser', action: () => launch('Search') },
    { label: 'Run...', kind: 'terminal', action: launchSearch },
  ]

  const rowClass = 'flex min-h-[34px] w-full items-center gap-2 px-2 text-left text-[11px] text-black hover:bg-[#2f65c8] hover:text-white focus-visible:bg-[#2f65c8] focus-visible:text-white'

  return (
    <div className="absolute bottom-10 left-0 z-[70] w-[min(96vw,430px)] overflow-hidden rounded-t-[8px] border border-[#0a4ec4] bg-white text-[11px] text-black shadow-[4px_4px_16px_rgba(0,0,0,0.45)]">
      <div className="flex h-14 items-center gap-3 bg-gradient-to-b from-[#3c8cff] via-[#1d62d0] to-[#174aa5] px-3 text-white">
        <span className="grid h-10 w-10 place-items-center rounded-[5px] border border-white/50 bg-white/20 shadow-[inset_1px_1px_0_rgba(255,255,255,0.7)]">
          <SystemLogo size={28} />
        </span>
        <div className="min-w-0">
          <div className="truncate text-[15px] font-bold [text-shadow:1px_1px_1px_rgba(0,0,0,0.5)]">Kenneth</div>
          <div className="text-[10px] text-white/80">Hallucination XP Professional</div>
        </div>
      </div>

      <div className="grid min-h-[360px] grid-cols-[1fr_168px] max-sm:grid-cols-1">
        <div className="bg-white py-2">
          {primaryRows.map(row => (
            <button key={row.label} type="button" className={rowClass} onClick={row.action}>
              <XpIcon kind={row.kind} size="small" />
              <span className={row.strong ? 'font-bold' : ''}>{row.label}</span>
            </button>
          ))}
          <div className="mx-2 my-2 border-t border-[#d6d6d6]" />
          <form
            className="px-2"
            onSubmit={e => { e.preventDefault(); launchSearch() }}
          >
            <label className="mb-1 block text-[10px] text-[#555]" htmlFor="xp-run-prompt">Open:</label>
            <div className="flex gap-1">
              <input
                id="xp-run-prompt"
                ref={inputRef}
                value={search}
                onChange={e => setSearch(e.target.value)}
                placeholder="program name or prompt"
                className="h-6 min-w-0 flex-1 border border-[#7f9db9] bg-white px-1 text-[11px] text-black shadow-[inset_1px_1px_2px_rgba(0,0,0,0.15)] focus:outline-none"
              />
              <button
                type="submit"
                className="h-6 shrink-0 rounded-[3px] border border-[#316ac5] bg-gradient-to-b from-white to-[#d7e7ff] px-2 text-[11px] hover:brightness-105 disabled:opacity-40"
                disabled={!search.trim()}
              >
                Run
              </button>
            </div>
          </form>
          <div className="mt-2 border-t border-[#d6d6d6] pt-1">
            <button type="button" className={`${rowClass} justify-between font-bold`} onClick={onOpenProgramManager}>
              <span className="flex items-center gap-2"><XpIcon kind="programs" size="small" />All Programs</span>
              <span aria-hidden="true">&gt;</span>
            </button>
          </div>
        </div>

        <div className="border-l border-[#9eb8e8] bg-[#d7e6ff] py-2 max-sm:border-l-0 max-sm:border-t">
          {secondaryRows.map(row => (
            <button key={row.label} type="button" className={rowClass} onClick={row.action}>
              <XpIcon kind={row.kind} size="small" />
              <span>{row.label}</span>
            </button>
          ))}
        </div>
      </div>

      <div className="flex h-10 items-center justify-end gap-2 bg-gradient-to-b from-[#d7e6ff] to-[#b6d1ff] px-3">
        <button type="button" onClick={onClose} className="inline-flex h-7 items-center gap-2 rounded-[3px] px-2 text-[11px] hover:bg-white/45">
          <span className="grid h-5 w-5 place-items-center rounded-sm bg-[#f7a51f] text-white">L</span>
          Log Off
        </button>
        <button type="button" onClick={onClose} className="inline-flex h-7 items-center gap-2 rounded-[3px] px-2 text-[11px] hover:bg-white/45">
          <span className="grid h-5 w-5 place-items-center rounded-sm bg-[#d94326] text-white">S</span>
          Turn Off Computer
        </button>
      </div>
    </div>
  )
}

export default function VibeOS() {
  const openaiKey = useAppStore(s => s.openaiKey)
  const settings = useAppStore(s => s.settings)
  const controllersRef = useRef<Record<string, AbortController>>({})
  const desktopRef = useRef<HTMLElement>(null)
  const [loggedIn, setLoggedIn] = useState(false)
  const [welcomeOpen, setWelcomeOpen] = useState(false)
  const [username, setUsername] = useState('Kenneth')
  const [password, setPassword] = useState('')
  const [seed, setSeed] = useState(() => Date.now())
  const [drift, setDrift] = useState(true)
  const [prompt, setPrompt] = useState('')
  const [apps, setApps] = useState<VibeApp[]>(() => makeStarterApps(Date.now()))
  const [openIds, setOpenIds] = useState<string[]>(() => [])
  const [minimizedIds, setMinimizedIds] = useState<string[]>(() => [])
  const [focusedId, setFocusedId] = useState<string | null>(null)
  const [startOpen, setStartOpen] = useState(false)
  const [settingsOpen, setSettingsOpen] = useState(false)
  const [desktopWallpaper, setDesktopWallpaper] = useState<DesktopWallpaper>(() => loadDesktopWallpaper())
  const [controlPanelSelection, setControlPanelSelection] = useState('Display')
  const [windowLayouts, setWindowLayouts] = useState<Record<string, WindowLayout>>({})
  const [iconLayouts, setIconLayouts] = useState<Record<string, DesktopIconLayout>>({})
  const [selectedIconId, setSelectedIconId] = useState<string | null>(null)
  const [contextMenu, setContextMenu] = useState<ContextMenuState | null>(null)
  const [surfaces, setSurfaces] = useState<Record<string, VibeSurfaceState>>({})
  const [savedPrograms, setSavedPrograms] = useState<SavedProgram[]>([])

  const focusedApp = apps.find(app => app.id === focusedId)
  const bootLines = useMemo(() => Array.from({ length: 6 }, (_, index) => bootLine(String(seed), index)), [seed])
  const desktopStats = useMemo(() => [
    ['Kernel', `${metric(String(seed), 14, 96)}% nominal`],
    ['Apps', `${savedPrograms.length} installed`],
    ['Events', `${metric(String(seed) + apps.length, 22, 99)} cps`],
    ['Certainty', `${metric(String(seed) + 'truth', 1, 18)}%`],
  ], [apps.length, savedPrograms.length, seed])

  useEffect(() => {
    if (!drift) return
    const timer = window.setInterval(() => setSeed(Date.now()), 4200)
    return () => window.clearInterval(timer)
  }, [drift])

  useEffect(() => {
    return () => {
      Object.values(controllersRef.current).forEach(controller => controller.abort())
    }
  }, [])

  useEffect(() => {
    let alive = true
    void loadSavedPrograms().then(programs => {
      if (alive) setSavedPrograms(programs)
    })
    return () => { alive = false }
  }, [])

  useEffect(() => {
    ensureIconLayouts(apps)
  }, [apps])

  function desktopRect() {
    return desktopRef.current?.getBoundingClientRect()
      ?? { width: 1280, height: 720, left: 0, top: 0, right: 1280, bottom: 720, x: 0, y: 0, toJSON: () => ({}) } as DOMRect
  }

  function defaultIconLayout(index: number): DesktopIconLayout {
    const rect = desktopRect()
    const columnHeight = Math.max(1, Math.floor((rect.height - 74) / 92))
    const column = Math.floor(index / columnHeight)
    const row = index % columnHeight
    return {
      x: 12 + column * 88,
      y: 12 + row * 92,
    }
  }

  function clampIconLayout(layout: DesktopIconLayout): DesktopIconLayout {
    const rect = desktopRect()
    return {
      x: Math.max(4, Math.min(layout.x, rect.width - 84)),
      y: Math.max(4, Math.min(layout.y, rect.height - 124)),
    }
  }

  function setIconLayout(id: string, layout: DesktopIconLayout) {
    setIconLayouts(current => ({
      ...current,
      [id]: clampIconLayout(layout),
    }))
  }

  function ensureIconLayouts(appList = apps) {
    setIconLayouts(current => {
      let changed = false
      const next = { ...current }
      appList.forEach((app, index) => {
        if (!next[app.id]) {
          next[app.id] = defaultIconLayout(index)
          changed = true
        }
      })
      return changed ? next : current
    })
  }

  function showContextMenu(event: React.MouseEvent, title: string | undefined, items: ContextMenuItem[]) {
    event.preventDefault()
    event.stopPropagation()
    const rect = desktopRect()
    setContextMenu({
      title,
      items,
      x: Math.max(4, Math.min(event.clientX - rect.left, rect.width - 204)),
      y: Math.max(4, Math.min(event.clientY - rect.top, rect.height - 220)),
    })
  }

  function showDesktopMenu(event: React.MouseEvent) {
    showContextMenu(event, 'Desktop', [
      { label: 'Arrange Icons by Name', onSelect: arrangeIconsByName },
      { label: 'Refresh', onSelect: regenerate },
      { separator: true },
      { label: desktopWallpaper === 'rolling-hills' ? 'Use System Teal' : 'Use Bliss Wallpaper', onSelect: () => applyWallpaper(desktopWallpaper === 'rolling-hills' ? 'teal' : 'rolling-hills') },
      { label: 'Properties', onSelect: () => { setSettingsOpen(true); setStartOpen(false) } },
    ])
  }

  function showIconMenu(event: React.MouseEvent, app: VibeApp) {
    setSelectedIconId(app.id)
    showContextMenu(event, app.title, [
      { label: 'Open', onSelect: () => openApp(app.id) },
      { label: 'Explore', onSelect: () => openProgramManager() },
      { separator: true },
      { label: 'Refresh Program', disabled: app.kind === 'browser' || app.kind === 'programs', onSelect: () => void generateSurface(app) },
      { label: 'Minimize', disabled: !openIds.includes(app.id), onSelect: () => minimizeApp(app.id) },
      { label: 'Close', disabled: !openIds.includes(app.id), onSelect: () => closeApp(app.id) },
      { separator: true },
      { label: 'Properties', onSelect: () => { setSettingsOpen(true); setControlPanelSelection('System') } },
    ])
  }

  function showWindowMenu(event: React.MouseEvent, app: VibeApp) {
    showContextMenu(event, app.title, [
      { label: 'Restore', onSelect: () => openApp(app.id) },
      { label: 'Move', disabled: true },
      { label: 'Size', disabled: true },
      { label: 'Minimize', onSelect: () => minimizeApp(app.id) },
      { separator: true },
      { label: 'Refresh', disabled: app.kind === 'browser' || app.kind === 'programs', onSelect: () => void generateSurface(app) },
      { label: 'Close', onSelect: () => closeApp(app.id) },
    ])
  }

  function arrangeIconsByName() {
    const sorted = [...apps].sort((a, b) => a.title.localeCompare(b.title))
    setIconLayouts(Object.fromEntries(sorted.map((app, index) => [app.id, defaultIconLayout(index)])))
  }

  function defaultLayoutForApp(app: VibeApp): WindowLayout {
    const rect = desktopRect()
    const width = app.kind === 'browser'
      ? Math.max(620, Math.min(980, rect.width * 0.72))
      : Math.max(420, Math.min(760, rect.width * 0.52))
    const height = app.kind === 'browser'
      ? Math.max(430, Math.min(680, rect.height * 0.74))
      : Math.max(320, Math.min(560, rect.height * 0.62))
    const x = Math.max(12, Math.min(rect.width - width - 12, rect.width * app.x / 100))
    const y = Math.max(16, Math.min(rect.height - height - 58, rect.height * app.y / 100))
    return { x, y, width, height }
  }

  function clampLayout(layout: WindowLayout): WindowLayout {
    const rect = desktopRect()
    const minWidth = 320
    const minHeight = 280
    const width = Math.max(minWidth, Math.min(layout.width, Math.max(minWidth, rect.width - 24)))
    const height = Math.max(minHeight, Math.min(layout.height, Math.max(minHeight, rect.height - 86)))
    return {
      x: Math.max(8, Math.min(layout.x, rect.width - width - 8)),
      y: Math.max(8, Math.min(layout.y, rect.height - height - 50)),
      width,
      height,
    }
  }

  function setAppLayout(id: string, layout: WindowLayout) {
    setWindowLayouts(current => ({
      ...current,
      [id]: clampLayout(layout),
    }))
  }

  function ensureAppLayout(app: VibeApp) {
    setWindowLayouts(current => current[app.id]
      ? current
      : { ...current, [app.id]: defaultLayoutForApp(app) })
  }

  function nextVersionForPrompt(value: string) {
    const key = promptKey(value)
    const versions = savedPrograms
      .filter(program => program.promptKey === key)
      .map(program => program.version)
    return versions.length ? Math.max(...versions) + 1 : 0
  }

  function persistProgram(app: VibeApp, surface: GeneratedVibeSurface, source: VibeSurfaceState['source'] = 'ai') {
    if (app.kind === 'browser' || app.kind === 'programs') return
    const key = promptKey(app.prompt)
    const version = app.version ?? nextVersionForPrompt(app.prompt)
    const id = app.savedId ?? `${hashString(`${key}:${version}`)}_${version}`
    const now = new Date().toISOString()
    const savedApp = {
      ...app,
      id: `saved_${id}`,
      savedId: id,
      version,
      title: displayTitle(app.title.replace(/\s+1\.\d+$/, ''), version),
    }
    const saved: SavedProgram = {
      id,
      promptKey: key,
      prompt: app.prompt,
      title: savedApp.title,
      kind: app.kind,
      version,
      createdAt: now,
      updatedAt: now,
      app: savedApp,
      surface,
    }

    setSavedPrograms(current => {
      const next = [saved, ...current.filter(program => program.id !== id)].slice(0, MAX_SAVED_PROGRAMS)
      void saveSavedPrograms(next)
      return next
    })

    setApps(current => current.map(item => item.id === app.id ? savedApp : item))
    setSurfaces(current => ({
      ...current,
      [savedApp.id]: {
        loading: false,
        source,
        generatedAt: now,
        data: surface,
      },
    }))
    setOpenIds(current => current.map(openId => openId === app.id ? savedApp.id : openId))
    setFocusedId(current => current === app.id ? savedApp.id : current)
    setWindowLayouts(current => {
      const existing = current[app.id]
      const rest = { ...current }
      delete rest[app.id]
      return existing ? { ...rest, [savedApp.id]: existing } : rest
    })
    setIconLayouts(current => {
      const existing = current[app.id]
      const rest = { ...current }
      delete rest[app.id]
      return existing ? { ...rest, [savedApp.id]: existing } : rest
    })
    setSelectedIconId(current => current === app.id ? savedApp.id : current)
  }

  function openSavedProgram(id: string) {
    const program = savedPrograms.find(item => item.id === id)
    if (!program) return
    const app = { ...program.app, id: `saved_${program.id}`, savedId: program.id }
    setApps(current => current.some(item => item.id === app.id) ? current : [app, ...current])
    setSurfaces(current => ({
      ...current,
      [app.id]: {
        loading: false,
        source: 'ai',
        generatedAt: program.updatedAt,
        data: program.surface,
      },
    }))
    setWindowLayouts(current => current[app.id] ? current : { ...current, [app.id]: defaultLayoutForApp(app) })
    setIconLayouts(current => current[app.id] ? current : { ...current, [app.id]: defaultIconLayout(apps.length) })
    setOpenIds(current => [app.id, ...current.filter(openId => openId !== app.id)])
    setFocusedId(app.id)
    setSettingsOpen(false)
    setStartOpen(false)
  }

  async function generateSurface(app: VibeApp, nextSeed = Date.now()) {
    controllersRef.current[app.id]?.abort()

    if (!openaiKey) {
      const data = fallbackSurface(app, nextSeed)
      setSurfaces(current => ({
        ...current,
        [app.id]: {
          loading: false,
          source: 'local',
          generatedAt: new Date().toISOString(),
          data,
        },
      }))
      persistProgram(app, data, 'local')
      return
    }

    const controller = new AbortController()
    controllersRef.current[app.id] = controller
    setSurfaces(current => ({
      ...current,
      [app.id]: { loading: true, source: 'ai' },
    }))

    try {
      const data = await requestVibeJson({
        apiKey: openaiKey,
        model: settings.openaiModel,
        temperature: Math.max(settings.openaiTemperature, 0.9),
        signal: controller.signal,
        system: [
          '#1 PRIMARY DIRECTIVE: PRODUCE A FUNCTIONAL, WORKING APP.',
          'Every button MUST do something when clicked. Every input MUST update state. Every keyboard shortcut MUST work. A static mockup is a FAILURE. A non-interactive page is a FAILURE. A page where buttons exist but have no event handlers is a FAILURE.',
          'Before finishing, mentally verify: (1) Does the main button work? (2) Does state update when the user interacts? (3) Is there a visible change after each action? If any answer is no, you have failed.',
          'Concrete requirements: For a CALCULATOR, the digits/operations must compute and display results. For a GAME, it must be playable with score that updates. For a NOTES app, typing must persist and display. For a TIMER, it must actually count down. For a LIST app, add/edit/delete must mutate the visible list. For a CHART, the data must render and update.',
          'You are an internal application service for a serious early-Windows operating environment.',
          'Return ONLY JSON. No markdown. No prose outside JSON.',
          'Generate a COMPLETE tiny single-page app as HTML/CSS/JS, not just content for cards.',
          'The parent app will render your documentHtml directly inside an existing OS window with scripts enabled.',
          'Do NOT draw an outer application window, desktop, title bar, close/minimize/maximize controls, large centered frame, or fake OS chrome. The host already provides the only window.',
          'Your document must fill 100% of the available iframe viewport. Use html/body width:100%, height:100%, margin:0, and lay out the app from the top-left edge.',
          'Your JavaScript MUST implement real client-side state, addEventListener handlers on every interactive element, controls, timers where appropriate, keyboard or pointer interactions, and VISIBLE CONSEQUENCES when the user clicks things. Every <button>, every <input>, every clickable element MUST be wired up. No dead controls.',
          'Make each app uniquely shaped: different layout, controls, visual style, and logic depending on the requested app. Avoid generic dashboard cards unless the prompt asks for a dashboard.',
          'Visual style must be Windows 95/98/XP or early software inside the app: grey panels, bevels, menu bars, status bars, tree views, tabs, toolbars, serif-free system fonts. No purple gradients, neon AI styling, glassmorphism, glowing blobs, or meme UI. if someone fun is requested generate silly 90s-style homemade web page vibes',
          'Only include image placeholders when imagery is part of the app itself: game backgrounds/sprites/maps, product photos, profile/portfolio images, gallery/media apps, travel/recipe/catalog pages, or visual reference material. Do NOT add decorative/random images to calculators, terminals, notes, forms, admin tools, ledgers, settings panels, or plain utilities. When imagery is genuinely needed, use <img data-vibe-random-image="category" alt="descriptive text"> where category is nature, city, technology, food, still_life, abstract, or wildlife. The host will generate and cache those images.',
          'Sprinkle in occasional random dry humor, but keep it understated, and plausible inside serious enterprise software. Avoid obvious irony, internet slang, fantasy creatures, "vibe coded" jokes, and loud absurdism.',
          'Do not mention AI, generated content, models, hallucinations, VibeOS, or vibes anywhere in the user-visible interface.',
          'You can have external assets, network calls, imports, script src, localStorage, cookies, parent/window.top access, or real system/file/account. Make the apps functional.',
          'Schema: { "documentHtml": string (REQUIRED, the entire app), "headline": string, "subtitle": string, "status": string, "skin": "plain"|"terminal"|"win95"|"neon"|"paper"|"danger"|"success", "caution": string }.',
          'documentHtml is the ONLY thing the user sees. It MUST be a complete, functional <!doctype html> document with substantial inline <style> AND inline <script>. The other fields are metadata only and are NOT rendered as UI. NEVER return an empty or trivial documentHtml.',
          'Take as much space as you need. There is no character limit. Use however many tokens are required to produce a fully working app.',
          'ABSOLUTE REQUIREMENT: Your output MUST be complete and syntactically valid. Every <script> tag must close with </script>. Every { ( [ must close. The JSON itself must be valid and end with }. Do NOT stop mid-statement, mid-string, or mid-tag. If you must choose, choose a smaller scope that finishes over a larger scope that runs out.',
          'The UI must fit a resizable OS window around 600x420, occupy the full viewport, and be responsive.',
        ].join('\n'),
        user: [
          `App title: ${app.title}`,
          `App kind: ${app.kind}`,
          `User prompt: ${app.prompt}`,
          `Render seed: ${nextSeed}`,
          'Return the runnable application now. The app MUST be interactive: every button click must produce a visible change, every input must update state, and the core action of the app (calculating, playing, editing, timing, etc.) MUST WORK end to end. Do not return a mockup. Do not return decorative UI with no logic. Wire up every control with addEventListener and real state changes.',
        ].join('\n'),
      })
      setSurfaces(current => ({
        ...current,
        [app.id]: {
          loading: false,
          source: 'ai',
          generatedAt: new Date().toISOString(),
          data,
        },
      }))
      persistProgram(app, data, 'ai')
    } catch (error) {
      if (controller.signal.aborted) return
      console.error('[VibeOS] generateSurface failed:', error)
      const data = fallbackSurface(app, nextSeed)
      setSurfaces(current => ({
        ...current,
        [app.id]: {
          loading: false,
          source: 'local',
          error: error instanceof Error ? error.message : 'Unable to complete program request.',
          generatedAt: new Date().toISOString(),
          data,
        },
      }))
      persistProgram(app, data, 'local')
    } finally {
      if (controllersRef.current[app.id] === controller) delete controllersRef.current[app.id]
    }
  }

  function regenerate() {
    const nextSeed = Date.now()
    setSeed(nextSeed)
    setApps(current => {
      const nextApps = current.map((app, index) => ({ ...makeApp(app.prompt, nextSeed, index), id: app.id }))
      setWindowLayouts(layouts => {
        const nextLayouts = { ...layouts }
        nextApps.forEach(app => {
          if (!nextLayouts[app.id]) nextLayouts[app.id] = defaultLayoutForApp(app)
        })
        return nextLayouts
      })
      nextApps
        .filter(app => openIds.includes(app.id) && app.kind !== 'browser' && app.kind !== 'programs')
        .forEach(app => void generateSurface(app, nextSeed))
      return nextApps
    })
  }

  function launchApp(nextPrompt = prompt) {
    const clean = nextPrompt.trim()
    if (!clean) return
    const draft = makeApp(clean, Date.now(), apps.length + 1)
    const version = draft.kind === 'browser' || draft.kind === 'programs'
      ? undefined
      : nextVersionForPrompt(clean)
    const app = {
      ...draft,
      version,
      title: displayTitle(draft.title, version),
    }
    setApps(current => [app, ...current])
    setWindowLayouts(current => ({ ...current, [app.id]: defaultLayoutForApp(app) }))
    setIconLayouts(current => ({ ...current, [app.id]: defaultIconLayout(apps.length) }))
    setOpenIds(current => [app.id, ...current.filter(id => id !== app.id)])
    setFocusedId(app.id)
    setSettingsOpen(false)
    setStartOpen(false)
    setPrompt('')
    if (app.kind !== 'browser' && app.kind !== 'programs') void generateSurface(app)
  }

  function openProgramManager() {
    const manager = apps.find(app => app.kind === 'programs')
    if (manager) {
      openApp(manager.id)
    } else {
      launchApp('installed program manager')
    }
    setSettingsOpen(false)
    setStartOpen(false)
  }

  function openApp(id: string) {
    setOpenIds(current => [id, ...current.filter(openId => openId !== id)])
    setMinimizedIds(current => current.filter(openId => openId !== id))
    setFocusedId(id)
    const app = apps.find(item => item.id === id)
    if (app) {
      ensureAppLayout(app)
      if (app.kind !== 'browser' && app.kind !== 'programs' && !surfaces[id]) void generateSurface(app)
    }
  }

  function minimizeApp(id: string) {
    setMinimizedIds(current => current.includes(id) ? current : [id, ...current])
    setFocusedId(current => current === id ? null : current)
  }

  function closeApp(id: string) {
    setOpenIds(current => current.filter(openId => openId !== id))
    setMinimizedIds(current => current.filter(openId => openId !== id))
    setFocusedId(current => current === id ? null : current)
  }

  function applyWallpaper(value: DesktopWallpaper) {
    setDesktopWallpaper(value)
    saveDesktopWallpaper(value)
  }

  function completeLogin() {
    setLoggedIn(true)
    setWelcomeOpen(true)
    playLoginChime()
  }

  if (!loggedIn) {
    return (
      <LoginScreen
        username={username}
        password={password}
        onUsername={setUsername}
        onPassword={setPassword}
        onLogin={completeLogin}
      />
    )
  }

  const controlPanelItems = [
    { label: 'Accessibility Options', code: 'AO' },
    { label: 'Add New Hardware', code: 'HW' },
    { label: 'Add/Remove Programs', code: 'AR' },
    { label: 'Date/Time', code: 'DT' },
    { label: 'Display', code: 'DP' },
    { label: 'Fonts', code: 'FN' },
    { label: 'Internet', code: 'IE' },
    { label: 'Keyboard', code: 'KB' },
    { label: 'Mail', code: 'ML' },
    { label: 'Modems', code: 'MD' },
    { label: 'Mouse', code: 'MS' },
    { label: 'Multimedia', code: 'MM' },
    { label: 'Network', code: 'NW' },
    { label: 'Passwords', code: 'PW' },
    { label: 'Printers', code: 'PR' },
    { label: 'Regional Settings', code: 'RS' },
    { label: 'Sounds', code: 'SO' },
    { label: 'System', code: 'SY' },
  ]
  const selectedControlPanelItem = controlPanelItems.find(item => item.label === controlPanelSelection) ?? controlPanelItems[4]
  const controlPanelIconKind = (label: string): VibeAppKind | 'computer' | 'folder' | 'control' => {
    if (label === 'Internet') return 'browser'
    if (label === 'Display' || label === 'System' || label === 'Keyboard' || label === 'Mouse') return 'computer'
    if (label === 'Add/Remove Programs' || label === 'Fonts' || label === 'Printers') return 'programs'
    if (label === 'Sounds' || label === 'Multimedia') return 'paint'
    if (label === 'Mail' || label === 'Passwords') return 'notes'
    if (label === 'Date/Time' || label === 'Regional Settings') return 'calculator'
    return 'control'
  }

  return (
    <div
      className="h-full min-h-0 overflow-hidden text-[11px] leading-[1.25] text-text"
      style={{
        height: '100dvh',
        fontFamily: '"MS Sans Serif", Tahoma, Arial, sans-serif',
      }}
    >
      <section
        ref={desktopRef}
        className="relative h-full w-full overflow-hidden bg-[#0b6f77] shadow-2xl"
        onMouseDown={event => {
          if (event.target === event.currentTarget) {
            setSelectedIconId(null)
            setContextMenu(null)
            setStartOpen(false)
          }
        }}
        onContextMenu={showDesktopMenu}
        style={desktopWallpaper === 'rolling-hills'
          ? {
            backgroundColor: '#2f79dd',
            backgroundImage: `url(${xpAiWallpaper}), linear-gradient(180deg,#1c76ec 0%,#61a7ff 42%,#5cab35 43%,#1e7b24 100%)`,
            backgroundSize: 'cover, cover',
            backgroundPosition: 'center bottom, center',
          }
          : {
            backgroundColor: '#0b6f77',
          }}
      >
        {desktopWallpaper === 'teal' && (
          <div className="pointer-events-none absolute inset-0 bg-[linear-gradient(rgba(255,255,255,0.04)_1px,transparent_1px),linear-gradient(90deg,rgba(255,255,255,0.04)_1px,transparent_1px)] bg-[size:64px_64px] opacity-35" />
        )}

        {settingsOpen && (
          <section
            className="absolute left-6 top-8 z-50 w-[min(94vw,720px)] overflow-hidden rounded-t-[8px] border border-[#0054e3] bg-[#ece9d8] text-black shadow-[4px_5px_14px_rgba(0,0,0,0.45)]"
            onContextMenu={event => showContextMenu(event, 'Control Panel', [
              { label: 'Open', disabled: true },
              { label: 'Refresh', onSelect: regenerate },
              { label: 'Close', onSelect: () => setSettingsOpen(false) },
              { separator: true },
              { label: 'Properties', onSelect: () => setControlPanelSelection('System') },
            ])}
          >
            <div className="flex h-8 items-center gap-1.5 bg-gradient-to-b from-[#5da7ff] via-[#1768e7] to-[#0b45bd] px-1.5 text-white">
              <XpIcon kind="control" size="small" />
              <span className="min-w-0 flex-1 truncate text-[12px] font-bold [text-shadow:1px_1px_1px_rgba(0,0,0,0.45)]">Control Panel</span>
              <button
                type="button"
                onClick={() => setSettingsOpen(false)}
                className="grid h-[21px] w-[21px] place-items-center rounded-[3px] border border-white/70 bg-gradient-to-b from-[#ffb08b] via-[#f06b33] to-[#c73913] text-white shadow-[inset_1px_1px_0_rgba(255,255,255,0.8)] hover:brightness-110"
                aria-label="Close system settings"
              >
                <X size={10} aria-hidden="true" />
              </button>
            </div>
            <div className="border-b border-[#808080] bg-[#d4d0c8] px-2 py-1 shadow-[inset_0_1px_0_#fff]">
              <div className="flex gap-4 text-[11px]">
                {['File', 'Edit', 'View', 'Help'].map(menu => (
                  <button key={menu} type="button" className="px-1 hover:bg-[#000080] hover:text-white">{menu}</button>
                ))}
              </div>
            </div>
            <div className="flex min-h-[30px] items-center gap-1 border-b border-[#808080] bg-[#d4d0c8] px-1 shadow-[inset_0_1px_0_#fff]">
              {['Back', 'Forward', 'Up'].map(label => (
                <button
                  key={label}
                  type="button"
                  className="h-6 border border-[#404040] bg-[#d4d0c8] px-2 text-[11px] shadow-[inset_1px_1px_0_#fff,inset_-1px_-1px_0_#808080] hover:bg-[#ece9d8]"
                >
                  {label}
                </button>
              ))}
              <button
                type="button"
                onClick={regenerate}
                className="inline-flex h-6 items-center gap-1 border border-[#404040] bg-[#d4d0c8] px-2 text-[11px] shadow-[inset_1px_1px_0_#fff,inset_-1px_-1px_0_#808080] hover:bg-[#ece9d8]"
              >
                <RefreshCw size={12} aria-hidden="true" />
                Refresh
              </button>
              <div className="ml-2 flex min-w-0 flex-1 items-center gap-1">
                <span className="text-[#404040]">Address</span>
                <div className="h-6 min-w-0 flex-1 border border-[#808080] bg-white px-2 py-1 font-mono text-[10px] shadow-[inset_1px_1px_0_#aaa,inset_-1px_-1px_0_#fff]">
                  C:\System\Control Panel
                </div>
              </div>
            </div>
            <div className="grid max-h-[calc(100dvh-148px)] grid-cols-[minmax(0,1fr)_220px] overflow-hidden max-md:grid-cols-1">
              <div className="min-h-[330px] overflow-y-auto bg-white p-4 shadow-[inset_1px_1px_0_#aaa,inset_-1px_-1px_0_#fff]">
                <div className="grid grid-cols-[repeat(auto-fill,minmax(86px,1fr))] gap-x-3 gap-y-4">
                  {controlPanelItems.map(item => (
                    <button
                      key={item.label}
                      type="button"
                      onClick={() => setControlPanelSelection(item.label)}
                      onDoubleClick={() => {
                        if (item.label === 'Add/Remove Programs') openProgramManager()
                        if (item.label === 'Internet') launchApp('Internet Console')
                        if (item.label === 'System') setDrift(value => !value)
                      }}
                      className={[
                        'group grid min-h-[76px] place-items-center gap-1 p-1 text-center text-[11px] leading-tight text-black',
                        controlPanelSelection === item.label ? 'bg-[#000080] text-white' : 'hover:bg-[#dfe8ff]',
                      ].join(' ')}
                    >
                      <XpIcon kind={controlPanelIconKind(item.label)} />
                      <span className="max-w-[82px]">{item.label}</span>
                    </button>
                  ))}
                </div>
              </div>
              <aside className="border-l border-[#808080] bg-[#d4d0c8] p-2 max-md:border-l-0 max-md:border-t">
                <div className="border border-[#808080] bg-[#ece9d8] p-2 shadow-[inset_1px_1px_0_#fff,inset_-1px_-1px_0_#aaa]">
                  <div className="flex items-center gap-2">
                    <XpIcon kind={controlPanelIconKind(selectedControlPanelItem.label)} />
                    <div className="min-w-0">
                      <div className="truncate font-bold">{selectedControlPanelItem.label}</div>
                      <div className="text-[10px] text-[#404040]">Control Panel item</div>
                    </div>
                  </div>
                  <div className="mt-2 h-px bg-[#808080] shadow-[0_1px_0_#fff]" />
                  {controlPanelSelection === 'Display' ? (
                    <div className="mt-2">
                      <div className="font-bold">Desktop Background</div>
                      <div className="mt-2 grid gap-2">
                        <button
                          type="button"
                          onClick={() => applyWallpaper('teal')}
                          className={[
                            'grid gap-1 border p-1 text-left shadow-[inset_1px_1px_0_#fff,inset_-1px_-1px_0_#808080]',
                            desktopWallpaper === 'teal' ? 'border-[#000080] bg-white' : 'border-[#808080] bg-[#d4d0c8] hover:bg-[#ece9d8]',
                          ].join(' ')}
                        >
                          <span className="block h-12 border border-[#404040] bg-[#0b6f77]" />
                          <span>System Teal</span>
                        </button>
                        <button
                          type="button"
                          onClick={() => applyWallpaper('rolling-hills')}
                          className={[
                            'grid gap-1 border p-1 text-left shadow-[inset_1px_1px_0_#fff,inset_-1px_-1px_0_#808080]',
                            desktopWallpaper === 'rolling-hills' ? 'border-[#000080] bg-white' : 'border-[#808080] bg-[#d4d0c8] hover:bg-[#ece9d8]',
                          ].join(' ')}
                        >
                          <span
                            className="block h-12 border border-[#404040] bg-cover bg-center"
                            style={{ backgroundImage: `url(${xpAiWallpaper})` }}
                          />
                          <span>Prairie 2026</span>
                        </button>
                      </div>
                    </div>
                  ) : controlPanelSelection === 'Add/Remove Programs' ? (
                    <div className="mt-2">
                      <div className="font-bold">Installed Programs</div>
                      <p className="mt-1 text-[#404040]">{savedPrograms.length} registered application{savedPrograms.length === 1 ? '' : 's'}.</p>
                      <button
                        type="button"
                        onClick={openProgramManager}
                        className="mt-2 min-h-[25px] border border-[#404040] bg-[#d4d0c8] px-3 shadow-[inset_1px_1px_0_#fff,inset_-1px_-1px_0_#808080] hover:bg-[#ece9d8]"
                      >
                        Open
                      </button>
                    </div>
                  ) : controlPanelSelection === 'System' ? (
                    <div className="mt-2">
                      {desktopStats.map(([label, value]) => (
                        <div key={label} className="mb-1 grid grid-cols-[70px_1fr] gap-2">
                          <span className="text-[#404040]">{label}</span>
                          <span className="font-mono text-[10px]">{value}</span>
                        </div>
                      ))}
                      <button
                        type="button"
                        onClick={() => setDrift(value => !value)}
                        className="mt-2 min-h-[25px] border border-[#404040] bg-[#d4d0c8] px-3 shadow-[inset_1px_1px_0_#fff,inset_-1px_-1px_0_#808080] hover:bg-[#ece9d8]"
                      >
                        {drift ? 'Pause' : 'Resume'}
                      </button>
                      <div className="mt-2 h-[72px] overflow-y-auto border border-[#808080] bg-white p-1 font-mono text-[10px] shadow-[inset_1px_1px_0_#aaa,inset_-1px_-1px_0_#fff]">
                        {bootLines.map(line => (
                          <div key={line} className="truncate text-[#202020]">OK: {line}</div>
                        ))}
                      </div>
                    </div>
                  ) : (
                    <div className="mt-2">
                      <p className="text-[#404040]">Double-click this item to open its administrative utility.</p>
                      <button
                        type="button"
                        onClick={() => launchApp(`${controlPanelSelection} control panel`)}
                        className="mt-2 min-h-[25px] border border-[#404040] bg-[#d4d0c8] px-3 shadow-[inset_1px_1px_0_#fff,inset_-1px_-1px_0_#808080] hover:bg-[#ece9d8]"
                      >
                        Open
                      </button>
                    </div>
                  )}
                </div>

                {!openaiKey && (
                  <div className="mt-2 border border-[#808080] bg-[#fff8d7] p-2 text-[11px] text-[#202020]">
                    Compatibility mode is active. Configure application services in <Link to="/settings" className="text-[#0a246a] underline">Settings</Link>.
                  </div>
                )}
              </aside>
            </div>
            <div className="flex h-6 items-center justify-between border-t border-[#808080] bg-[#d4d0c8] px-2 text-[10px] text-[#404040] shadow-[inset_0_1px_0_#fff]">
              <span>{controlPanelItems.length} object(s)</span>
              <span className="truncate">{selectedControlPanelItem.label}</span>
            </div>
          </section>
        )}

        <div className="absolute inset-x-0 top-0 bottom-10 z-10">
          {apps.slice(0, 24).map((app, index) => (
            <DesktopIcon
              key={app.id}
              app={app}
              layout={iconLayouts[app.id] ?? defaultIconLayout(index)}
              selected={selectedIconId === app.id}
              onSelect={setSelectedIconId}
              onOpen={openApp}
              onMove={setIconLayout}
              onContextMenu={showIconMenu}
            />
          ))}
        </div>

        {openIds.filter(id => !minimizedIds.includes(id)).map((id, index) => {
          const app = apps.find(item => item.id === id)
          if (!app) return null
          const layout = windowLayouts[id] ?? defaultLayoutForApp(app)
          return (
            <WindowChrome
              key={id}
              app={app}
              seed={seed}
              surface={surfaces[id]}
              layout={layout}
              z={40 - index}
              onFocus={() => openApp(id)}
              onMinimize={() => minimizeApp(id)}
              onClose={() => closeApp(id)}
              onGenerate={() => void generateSurface(app)}
              savedPrograms={savedPrograms}
              onOpenSaved={openSavedProgram}
              onNewVersion={launchApp}
              onMove={next => setAppLayout(id, next)}
              onResize={next => setAppLayout(id, next)}
              onContextMenu={showWindowMenu}
            />
          )
        })}

        <StartMenu
          open={startOpen}
          onClose={() => setStartOpen(false)}
          onOpenControlPanel={() => {
            setSettingsOpen(true)
            setStartOpen(false)
          }}
          onOpenProgramManager={openProgramManager}
          onLaunch={launchApp}
        />

        <div className="absolute bottom-0 left-0 right-0 z-40 flex h-10 items-center gap-2 bg-gradient-to-b from-[#2f8cff] via-[#0f5fd7] to-[#0642a7] px-1.5 text-white shadow-[inset_0_1px_0_rgba(255,255,255,0.45)]">
          <button
            type="button"
            onClick={() => {
              setStartOpen(value => !value)
              setSettingsOpen(false)
            }}
            onContextMenu={event => showContextMenu(event, 'Start', [
              { label: 'Open', onSelect: () => setStartOpen(true) },
              { label: 'Explore', onSelect: openProgramManager },
              { separator: true },
              { label: 'Properties', onSelect: () => { setSettingsOpen(true); setControlPanelSelection('System') } },
            ])}
            className={[
              'inline-flex h-8 items-center gap-1.5 rounded-r-[14px] rounded-l-[7px] border border-[#2f8f2f] px-3 pr-5 text-[15px] font-bold italic text-white shadow-[inset_1px_1px_0_rgba(255,255,255,0.55),2px_0_3px_rgba(0,0,0,0.25)] [text-shadow:1px_1px_1px_rgba(0,0,0,0.45)]',
              startOpen ? 'bg-gradient-to-b from-[#7fd04d] to-[#208820]' : 'bg-gradient-to-b from-[#8edb5d] via-[#43a72e] to-[#1d7b1d] hover:brightness-110',
            ].join(' ')}
            aria-expanded={startOpen}
          >
            <SystemLogo size={16} />
            start
          </button>
          <div className="flex min-w-0 flex-1 gap-1 overflow-x-auto">
            {openIds.map(id => {
              const app = apps.find(item => item.id === id)
              if (!app) return null
              return (
                <button
                  key={id}
                  type="button"
                  onClick={() => openApp(id)}
                  onContextMenu={event => showContextMenu(event, app.title, [
                    { label: 'Restore', onSelect: () => openApp(id) },
                    { label: 'Minimize', onSelect: () => minimizeApp(id), disabled: minimizedIds.includes(id) },
                    { label: 'Close', onSelect: () => closeApp(id) },
                    { separator: true },
                    { label: 'Properties', onSelect: () => { setSettingsOpen(true); setControlPanelSelection('System') } },
                  ])}
                  className={[
                    'inline-flex h-8 min-w-[132px] max-w-[210px] shrink-0 items-center gap-1.5 rounded-[3px] border px-2 text-[11px] text-white shadow-[inset_1px_1px_0_rgba(255,255,255,0.25)]',
                    focusedApp?.id === id && !minimizedIds.includes(id)
                      ? 'border-[#74aaff] bg-gradient-to-b from-[#4f9fff] to-[#1f56bf]'
                      : minimizedIds.includes(id)
                        ? 'border-[#1b4a98] bg-[#1a4fb1]/70 text-white/70 hover:bg-[#255fc4]'
                        : 'border-[#1b4a98] bg-gradient-to-b from-[#2f7de5] to-[#174ba8] hover:brightness-110',
                  ].join(' ')}
                >
                  <XpIcon kind={app.kind} size="small" />
                  <span className="min-w-0 truncate">{app.title}</span>
                </button>
              )
            })}
          </div>
          <div className="flex h-8 shrink-0 items-center gap-1 rounded-sm border-l border-[#4aa3ff] bg-[#0b78d0] px-2 text-[10px] shadow-[inset_1px_0_0_rgba(255,255,255,0.2)]">
            <span className="hidden sm:inline">EN</span>
            <span>{new Date(seed).toLocaleTimeString([], { hour: 'numeric', minute: '2-digit' })}</span>
          </div>
        </div>
        <ContextMenu menu={contextMenu} onClose={() => setContextMenu(null)} />
        {welcomeOpen && (
          <WelcomeDialog username={username} onClose={() => setWelcomeOpen(false)} />
        )}
      </section>
    </div>
  )
}
