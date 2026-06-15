import { useEffect, useRef, useState } from 'react'
import { Link } from 'react-router-dom'
import {
  AlertTriangle,
  BarChart3,
  Brain,
  Camera,
  CheckCircle2,
  ClipboardList,
  Loader2,
  Mic,
  Play,
  RotateCcw,
  Settings,
  Square,
  Timer,
  Video,
} from 'lucide-react'
import { isReasoningModel, useAppStore } from '../store/useAppStore'

type InterviewFormat = 'hirevue' | 'behavioral' | 'technical' | 'case' | 'motivational' | 'mixed'
type PracticePhase = 'setup' | 'ready' | 'practice' | 'analyzing' | 'report'
type PracticeMode = 'thinking' | 'speaking' | 'complete'
type RoleSeniority = 'internship' | 'graduate' | 'experienced' | 'general'

interface RoleDraft {
  company: string
  roleTitle: string
  roleDetails: string
  format: InterviewFormat
  questionCount: number
  thinkingTimeSec: number
  speakingTimeSec: number
  focusAreas: string
}

interface InterviewQuestion {
  id: number
  question: string
  thinkingTimeSec: number
  speakingTimeSec: number
  competency: string
  evaluationFocus: string[]
}

interface InterviewPlan {
  roleTitle: string
  company: string
  format: string
  generatedAt: string
  questions: InterviewQuestion[]
}

interface TranscriptMetrics {
  wordCount: number
  fillerCount: number
  hedgeCount: number
  numbersUsed: number
  wordsPerMinute: number
  starSignals: number
  roleKeywordHits: number
}

interface VideoMetrics {
  frameCount: number
  avgBrightness: number
  brightnessConsistency: number
  motionStability: number
  resolution: string
}

interface InterviewReport {
  overallScore: number
  scores: Record<string, number>
  transcriptMetrics: TranscriptMetrics
  videoMetrics: VideoMetrics
  strengths: string[]
  criticalFeedback: string[]
  improvementPlan: string[]
  report: string
}

interface SpeechRecognitionLike {
  continuous: boolean
  interimResults: boolean
  lang: string
  onresult: ((event: SpeechRecognitionEventLike) => void) | null
  onerror: (() => void) | null
  onend: (() => void) | null
  start: () => void
  stop: () => void
}

interface SpeechRecognitionEventLike {
  resultIndex: number
  results: ArrayLike<{
    isFinal: boolean
    0: { transcript: string }
  }>
}

type SpeechRecognitionCtor = new () => SpeechRecognitionLike

declare global {
  interface Window {
    SpeechRecognition?: SpeechRecognitionCtor
    webkitSpeechRecognition?: SpeechRecognitionCtor
  }
}

const INPUT_CLASS = 'min-h-[44px] w-full rounded-[8px] border border-border bg-bg px-3 py-2 text-sm text-text outline-none placeholder:text-dim focus-visible:border-accent'
const PANEL_CLASS = 'border border-border bg-s1/85'
const BUTTON_CLASS = 'inline-flex min-h-[40px] items-center justify-center gap-2 rounded-[8px] border border-border bg-s2 px-4 py-2 text-2xs font-medium text-muted transition-colors hover:border-border-strong hover:text-text disabled:cursor-not-allowed disabled:opacity-45'

const DEFAULT_DRAFT: RoleDraft = {
  company: '',
  roleTitle: '',
  roleDetails: '',
  format: 'hirevue',
  questionCount: 5,
  thinkingTimeSec: 30,
  speakingTimeSec: 120,
  focusAreas: 'leadership, stakeholder management, ambiguity, communication, measurable impact',
}

const FORMAT_LABELS: Record<InterviewFormat, string> = {
  hirevue: 'HireVue Classic',
  behavioral: 'Behavioral',
  technical: 'Technical',
  case: 'Case',
  motivational: 'Motivational',
  mixed: 'Mixed',
}

const QUESTION_BANK: Record<InterviewFormat, string[]> = {
  hirevue: [
    'Tell me about yourself and why this role is the right next step.',
    'Describe a time you had to solve a difficult problem with limited information.',
    'Tell me about a time you received critical feedback and changed your approach.',
    'Give an example of a time you influenced someone who did not report to you.',
    'Why do you want to work at this company in this role?',
  ],
  behavioral: [
    'Tell me about a time you took ownership of a project that was falling behind.',
    'Describe a conflict with a teammate and how you handled it.',
    'Tell me about a time you had to learn something quickly to deliver results.',
    'Give an example of a decision you made using imperfect data.',
    'Describe your most measurable professional achievement.',
  ],
  technical: [
    'Walk me through a technical decision you made and the trade-offs you considered.',
    'Explain a complex project you built as if I am a cross-functional stakeholder.',
    'Tell me about a time you diagnosed a bug or system issue under pressure.',
    'Describe how you would validate that your solution is working.',
    'What technical area would you need to deepen for this role?',
  ],
  case: [
    'How would you prioritize the first 30 days if hired into this role?',
    'A key metric is declining. Walk through how you would diagnose the cause.',
    'How would you decide between two competing initiatives with limited resources?',
    'Design a practical plan to improve a process used by this team.',
    'How would you communicate a risky recommendation to senior stakeholders?',
  ],
  motivational: [
    'What interests you most about this role and company?',
    'Which part of the role would stretch you the most?',
    'What type of work environment helps you do your best work?',
    'What are your long-term career goals?',
    'Why should the team choose you over another qualified candidate?',
  ],
  mixed: [
    'Tell me about yourself and connect your experience to this role.',
    'Describe a time you delivered measurable impact under constraints.',
    'Walk me through how you would approach a realistic challenge in this job.',
    'Tell me about a mistake you made and what changed afterward.',
    'What would make you successful in the first six months here?',
  ],
}

const COMPETENCIES = [
  'role motivation',
  'structured communication',
  'ownership',
  'problem solving',
  'collaboration',
  'adaptability',
  'customer judgment',
  'measurable impact',
]

function clamp(value: number, min = 0, max = 100) {
  return Math.min(max, Math.max(min, Math.round(value)))
}

function scoreLabel(score: number) {
  if (score >= 85) return 'Strong'
  if (score >= 70) return 'Ready'
  if (score >= 55) return 'Developing'
  return 'Needs work'
}

function secondsLabel(total: number) {
  const min = Math.floor(total / 60)
  const sec = total % 60
  return `${String(min).padStart(2, '0')}:${String(sec).padStart(2, '0')}`
}

function seededIndex(seed: string, offset: number, length: number) {
  let hash = 0
  for (let i = 0; i < seed.length; i++) hash = (hash * 31 + seed.charCodeAt(i) + offset) >>> 0
  return hash % length
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

function roleKeywords(draft: RoleDraft) {
  const source = `${draft.roleTitle} ${draft.company} ${draft.roleDetails} ${draft.focusAreas}`.toLowerCase()
  const stop = new Set(['the', 'and', 'for', 'with', 'that', 'this', 'from', 'role', 'team', 'will', 'you', 'are', 'have'])
  return Array.from(new Set((source.match(/[a-z]{4,}/g) ?? []).filter(word => !stop.has(word)))).slice(0, 24)
}

function fallbackPlan(draft: RoleDraft): InterviewPlan {
  const format = draft.format === 'mixed' ? 'mixed' : draft.format
  const bank = QUESTION_BANK[format]
  const seed = `${draft.company}|${draft.roleTitle}|${draft.roleDetails}|${draft.focusAreas}|${draft.format}`
  const keywords = roleKeywords(draft)
  const questions: InterviewQuestion[] = Array.from({ length: draft.questionCount }, (_, index) => {
    const base = bank[seededIndex(seed, index + 7, bank.length)]
    const roleHook = keywords[index % Math.max(1, keywords.length)] ?? 'impact'
    return {
      id: index + 1,
      question: index === 0
        ? base
        : `${base} In your answer, connect it to ${roleHook} and the outcomes this role needs.`,
      thinkingTimeSec: draft.thinkingTimeSec,
      speakingTimeSec: draft.speakingTimeSec,
      competency: COMPETENCIES[seededIndex(seed, index + 19, COMPETENCIES.length)],
      evaluationFocus: [
        'specific situation and stakes',
        'clear action taken by you',
        'quantified result or learning',
      ],
    }
  })

  return {
    roleTitle: draft.roleTitle || 'Target role',
    company: draft.company || 'Target company',
    format: FORMAT_LABELS[draft.format],
    generatedAt: new Date().toISOString(),
    questions,
  }
}

function normalisePlan(value: unknown, draft: RoleDraft): InterviewPlan {
  const raw = value && typeof value === 'object' ? value as Partial<InterviewPlan> : {}
  const fallback = fallbackPlan(draft)
  const rawQuestions = Array.isArray(raw.questions) ? raw.questions : []
  const questions = rawQuestions.slice(0, draft.questionCount).map((q, index) => {
    const item = q && typeof q === 'object' ? q as Partial<InterviewQuestion> : {}
    return {
      id: index + 1,
      question: typeof item.question === 'string' && item.question.trim()
        ? item.question.trim()
        : fallback.questions[index]?.question ?? fallback.questions[0].question,
      thinkingTimeSec: Number.isFinite(Number(item.thinkingTimeSec))
        ? clamp(Number(item.thinkingTimeSec), 10, 180)
        : draft.thinkingTimeSec,
      speakingTimeSec: Number.isFinite(Number(item.speakingTimeSec))
        ? clamp(Number(item.speakingTimeSec), 30, 300)
        : draft.speakingTimeSec,
      competency: typeof item.competency === 'string' && item.competency.trim()
        ? item.competency.trim()
        : fallback.questions[index]?.competency ?? 'structured communication',
      evaluationFocus: Array.isArray(item.evaluationFocus)
        ? item.evaluationFocus.filter((v): v is string => typeof v === 'string').slice(0, 4)
        : fallback.questions[index]?.evaluationFocus ?? [],
    }
  })

  return {
    roleTitle: typeof raw.roleTitle === 'string' && raw.roleTitle.trim() ? raw.roleTitle.trim() : fallback.roleTitle,
    company: typeof raw.company === 'string' && raw.company.trim() ? raw.company.trim() : fallback.company,
    format: typeof raw.format === 'string' && raw.format.trim() ? raw.format.trim() : fallback.format,
    generatedAt: typeof raw.generatedAt === 'string' ? raw.generatedAt : new Date().toISOString(),
    questions: questions.length ? questions : fallback.questions,
  }
}

async function requestJson<T>({
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
}): Promise<T> {
  const body: Record<string, unknown> = {
    model,
    messages: [
      { role: 'system', content: system },
      { role: 'user', content: user },
    ],
    response_format: { type: 'json_object' },
  }
  if (typeof temperature === 'number' && !isReasoningModel(model)) body.temperature = temperature

  const response = await fetch('https://api.openai.com/v1/chat/completions', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      Authorization: `Bearer ${apiKey}`,
    },
    body: JSON.stringify(body),
    signal,
  })

  if (!response.ok) {
    let message = `OpenAI request failed (${response.status})`
    try {
      const error = await response.json()
      message = error?.error?.message ?? message
    } catch {
      // ignore parse errors
    }
    throw new Error(message)
  }

  const json = await response.json()
  const content = json?.choices?.[0]?.message?.content
  if (typeof content !== 'string') throw new Error('OpenAI returned an empty response.')
  return JSON.parse(cleanJson(content)) as T
}

function transcriptMetrics(transcript: string, durationSec: number, draft: RoleDraft): TranscriptMetrics {
  const words = transcript.toLowerCase().match(/[a-z0-9']+/g) ?? []
  const fillerWords = new Set(['um', 'uh', 'like', 'actually', 'basically', 'literally', 'sort', 'kinda'])
  const hedges = new Set(['maybe', 'probably', 'perhaps', 'guess', 'think', 'just'])
  const keywordSet = new Set(roleKeywords(draft))
  const starTerms = ['situation', 'task', 'action', 'result', 'first', 'then', 'because', 'therefore', 'outcome', 'measured']
  return {
    wordCount: words.length,
    fillerCount: words.filter(word => fillerWords.has(word)).length,
    hedgeCount: words.filter(word => hedges.has(word)).length,
    numbersUsed: (transcript.match(/\b\d+([.,]\d+)?%?\b/g) ?? []).length,
    wordsPerMinute: durationSec > 0 ? Math.round(words.length / (durationSec / 60)) : 0,
    starSignals: starTerms.filter(term => transcript.toLowerCase().includes(term)).length,
    roleKeywordHits: words.filter(word => keywordSet.has(word)).length,
  }
}

function detectRoleSeniority(draft: RoleDraft): RoleSeniority {
  const source = `${draft.roleTitle} ${draft.roleDetails}`.toLowerCase()
  if (/\b(intern|internship|summer analyst|summer associate|vacationer|placement|co-?op|work experience)\b/.test(source)) {
    return 'internship'
  }
  if (/\b(graduate|new grad|grad program|graduate program|entry[- ]level|junior|analyst program|rotational)\b/.test(source)) {
    return 'graduate'
  }
  if (/\b(senior|lead|principal|manager|director|head of|vp|vice president|experienced|staff)\b/.test(source)) {
    return 'experienced'
  }
  return 'general'
}

function scoringCalibration(draft: RoleDraft, question: InterviewQuestion, durationSec: number) {
  const seniority = detectRoleSeniority(draft)
  const minutes = Math.max(1, Math.round(durationSec / 15) / 4)
  const shared = {
    seniority,
    answerLength: `${minutes} minutes`,
    questionSpeakingLimitSec: question.speakingTimeSec,
    universalRule: 'Score the answer against the role level and the time limit. Reward clear prioritisation and do not expect an exhaustive textbook answer in a short HireVue response.',
  }

  if (seniority === 'internship') {
    return {
      ...shared,
      evaluatorMode: 'supportive internship assessor',
      expectedStandard: [
        'Evaluate potential, structured thinking, teachability, and directionally correct reasoning.',
        'For a 90-120 second answer, a strong response can cover a practical sequence, 2-3 core concepts, and one clear communication/risk point.',
        'Do not require specialist desk-level details, live market levels, flows, swaps jargon, or every possible product linkage unless the question explicitly asks for them.',
        'Treat missing advanced nuance as coaching feedback, not as a major score penalty when the core logic is sound.',
      ],
      scoreGuide: {
        '80-89': 'strong internship answer; clear framework, mostly correct logic, good communication, minor gaps',
        '70-79': 'solid internship answer with useful substance and fixable gaps',
        '60-69': 'developing but plausible; some structure or content issues, still shows potential',
        '<60': 'mostly off-question, seriously unclear, or technically misleading',
      },
      feedbackStyle: 'Be honest but not harsh. Keep criticalFeedback to the highest-leverage 3-4 items and avoid phrasing that implies experienced-professional expectations.',
    }
  }

  if (seniority === 'graduate') {
    return {
      ...shared,
      evaluatorMode: 'balanced graduate-role assessor',
      expectedStandard: [
        'Expect a clearer framework and more specific examples than an internship answer.',
        'For a short answer, prioritise the most important concepts rather than exhaustive coverage.',
        'Missing depth should reduce the score only when it weakens the main answer or communication.',
      ],
      scoreGuide: {
        '80-89': 'strong graduate answer; structured, specific, and role-aware',
        '70-79': 'ready or close; clear core answer with some missing specificity',
        '60-69': 'developing; understandable but too generic or uneven',
        '<60': 'substantial gaps, unclear communication, or weak role fit',
      },
      feedbackStyle: 'Be direct and practical, with criticism framed as the next improvement step.',
    }
  }

  if (seniority === 'experienced') {
    return {
      ...shared,
      evaluatorMode: 'experienced-hire assessor',
      expectedStandard: [
        'Expect sharper judgment, prioritisation, domain fluency, and stakeholder-ready communication.',
        'For a short answer, still reward concise tradeoffs rather than exhaustive detail.',
      ],
      scoreGuide: {
        '80-89': 'strong professional answer; specific, credible, and decision-ready',
        '70-79': 'competent with notable gaps',
        '60-69': 'below expected experienced-hire standard',
        '<60': 'major gaps for the level',
      },
      feedbackStyle: 'Be candid and specific, but avoid piling on low-value criticisms.',
    }
  }

  return {
    ...shared,
    evaluatorMode: 'balanced role-level assessor',
    expectedStandard: [
      'Infer the likely level from the role context and avoid applying senior professional standards by default.',
      'For a short answer, reward strong prioritisation over exhaustive detail.',
    ],
    scoreGuide: {
      '80-89': 'strong answer for the apparent level',
      '70-79': 'solid answer with fixable gaps',
      '60-69': 'developing answer',
      '<60': 'major clarity, relevance, or correctness issues',
    },
    feedbackStyle: 'Be useful, specific, and proportionate.',
  }
}

function buildFallbackReport({
  draft,
  question,
  transcript,
  durationSec,
  videoMetrics,
}: {
  draft: RoleDraft
  question: InterviewQuestion
  transcript: string
  durationSec: number
  videoMetrics: VideoMetrics
}): InterviewReport {
  const metrics = transcriptMetrics(transcript, durationSec, draft)
  const fillerRate = metrics.wordCount ? metrics.fillerCount / metrics.wordCount : 0
  const hedgeRate = metrics.wordCount ? metrics.hedgeCount / metrics.wordCount : 0
  const paceScore = metrics.wordsPerMinute
    ? 100 - Math.min(45, Math.abs(metrics.wordsPerMinute - 145) * 0.8)
    : 35
  const structure = clamp(38 + metrics.starSignals * 11 + (/\b(result|outcome|impact|learned)\b/i.test(transcript) ? 16 : 0))
  const evidence = clamp(42 + metrics.numbersUsed * 12 + (/\b(increased|reduced|improved|delivered|launched|saved)\b/i.test(transcript) ? 18 : 0))
  const roleFit = clamp(45 + metrics.roleKeywordHits * 6 + (transcript.toLowerCase().includes(draft.company.toLowerCase()) ? 8 : 0))
  const clarity = clamp(paceScore - fillerRate * 260 - hedgeRate * 130)
  const concision = clamp(100 - Math.abs(durationSec - question.speakingTimeSec * 0.82) * 0.42)
  const confidence = clamp(76 - fillerRate * 320 - hedgeRate * 190 + (metrics.wordCount > 80 ? 8 : 0))
  const delivery = clamp((clarity + concision + paceScore) / 3)
  const visualPresence = clamp(
    35 + videoMetrics.avgBrightness * 0.35 + videoMetrics.brightnessConsistency * 0.22 + videoMetrics.motionStability * 0.28,
  )
  const scores = { structure, evidence, roleFit, clarity, concision, confidence, delivery, visualPresence }
  const overallScore = clamp(Object.values(scores).reduce((sum, value) => sum + value, 0) / Object.values(scores).length)

  const criticalFeedback = [
    structure < 72
      ? 'Your answer needs a sharper STAR arc. State the situation in one sentence, name your personal action, then finish with a measurable result.'
      : 'The structure is workable, but tighten the opening so the evaluator understands the stakes faster.',
    evidence < 72
      ? 'The answer is under-evidenced. Add numbers, before-and-after comparison, scale, or a specific stakeholder outcome.'
      : 'Evidence is present. Make the metric more central rather than leaving it as a side detail.',
    clarity < 72
      ? 'Delivery needs cleaner pacing. Reduce filler words and aim for a controlled 120-160 words per minute.'
      : 'Delivery is clear enough; the next lift is making the first 15 seconds more decisive.',
    visualPresence < 68
      ? 'Video quality or stability may weaken perceived confidence. Improve lighting, camera height, and stillness before a real attempt.'
      : 'Video presence is serviceable. Keep the same setup and focus on answer content.',
  ]

  return {
    overallScore,
    scores,
    transcriptMetrics: metrics,
    videoMetrics,
    strengths: [
      metrics.wordCount > 90 ? 'Enough answer volume to evaluate substance.' : 'Concise answer length that can be expanded with evidence.',
      metrics.roleKeywordHits > 2 ? 'Some alignment to the target role language.' : 'Clear opportunity to add role-specific language.',
      videoMetrics.frameCount > 0 ? 'Camera stream captured measurable visual signals.' : 'Transcript-only scoring remained available.',
    ],
    criticalFeedback,
    improvementPlan: [
      'Write a 4-line STAR outline before recording: context, action, metric, learning.',
      'Add one quantified result and one decision trade-off to every answer.',
      'Record one more take with the same question and target 120-160 words per minute.',
      'Review the first 20 seconds and remove any preamble that does not answer the question.',
    ],
    report: `For "${question.question}", the answer scored ${overallScore}/100. The repeatable rubric weighted structure, evidence, role fit, clarity, concision, confidence, delivery, and visual presence equally. The biggest improvement lever is ${criticalFeedback[0].toLowerCase()}`,
  }
}

function normaliseReport(value: unknown, fallback: InterviewReport): InterviewReport {
  const raw = value && typeof value === 'object' ? value as Partial<InterviewReport> : {}
  const rawScores = raw.scores && typeof raw.scores === 'object' ? raw.scores : {}
  const scores = Object.fromEntries(
    Object.keys(fallback.scores).map(key => {
      const rawScore = Number((rawScores as Record<string, unknown>)[key])
      return [key, Number.isFinite(rawScore) ? clamp(rawScore) : 0]
    }),
  )
  for (const [key, value] of Object.entries(rawScores as Record<string, unknown>)) {
    if (key in scores) continue
    const rawScore = Number(value)
    if (Number.isFinite(rawScore)) scores[key] = clamp(rawScore)
  }
  const scoreValues = Object.values(scores)
  const fallbackOverall = scoreValues.length
    ? clamp(scoreValues.reduce((sum, value) => sum + value, 0) / scoreValues.length)
    : 0
  const rawOverall = Number(raw.overallScore)
  const cleanList = (items: unknown, emptyMessage: string) => {
    const values = Array.isArray(items) ? items.filter((v): v is string => typeof v === 'string' && Boolean(v.trim())).slice(0, 6) : []
    return values.length ? values : [emptyMessage]
  }

  return {
    overallScore: Number.isFinite(rawOverall) ? clamp(rawOverall) : fallbackOverall,
    scores,
    transcriptMetrics: fallback.transcriptMetrics,
    videoMetrics: fallback.videoMetrics,
    strengths: cleanList(raw.strengths, 'AI returned no strengths for this report. Retry generation for a complete assessment.').slice(0, 5),
    criticalFeedback: cleanList(raw.criticalFeedback, 'AI returned no critical feedback for this report. Retry generation for a complete assessment.'),
    improvementPlan: cleanList(raw.improvementPlan, 'Retry the AI report after confirming transcript capture and the OpenAI key.'),
    report: typeof raw.report === 'string' && raw.report.trim()
      ? raw.report.trim()
      : 'AI returned structured scores without a written summary. Retry generation for a complete narrative report.',
  }
}

function missingTranscriptNotice(canUseSpeechRecognition: boolean) {
  return canUseSpeechRecognition
    ? 'No final speech transcript was captured for this attempt. The AI report will evaluate observable delivery signals and mark content-based scoring as limited.'
    : 'This browser did not provide live speech recognition. The AI report will evaluate timing and video delivery metrics, but content-based scoring is limited.'
}

function emptyVideoMetrics(): VideoMetrics {
  return {
    frameCount: 0,
    avgBrightness: 0,
    brightnessConsistency: 0,
    motionStability: 0,
    resolution: 'unavailable',
  }
}

function summariseFrames(frames: Array<{ brightness: number; delta: number }>, resolution: string): VideoMetrics {
  if (!frames.length) return { ...emptyVideoMetrics(), resolution }
  const avgBrightness = frames.reduce((sum, frame) => sum + frame.brightness, 0) / frames.length
  const brightnessVariance = frames.reduce((sum, frame) => sum + Math.abs(frame.brightness - avgBrightness), 0) / frames.length
  const avgDelta = frames.reduce((sum, frame) => sum + frame.delta, 0) / frames.length
  return {
    frameCount: frames.length,
    avgBrightness: clamp(avgBrightness),
    brightnessConsistency: clamp(100 - brightnessVariance * 1.8),
    motionStability: clamp(100 - avgDelta * 1.6),
    resolution,
  }
}

function ScorePill({ label, score }: { label: string; score: number }) {
  const color = score >= 80 ? 'text-up' : score >= 65 ? 'text-accent' : score >= 50 ? 'text-warn' : 'text-down'
  return (
    <div className="border border-border bg-bg/55 p-3">
      <div className="mb-2 flex items-center justify-between gap-3">
        <span className="text-2xs uppercase tracking-widest text-dim">{label}</span>
        <span className={`text-sm tabnum font-semibold ${color}`}>{score}</span>
      </div>
      <div className="h-1.5 bg-s2" aria-hidden="true">
        <div className="h-full bg-accent" style={{ width: `${score}%` }} />
      </div>
    </div>
  )
}

export default function HireVuePrep() {
  const [draft, setDraft] = useState<RoleDraft>(DEFAULT_DRAFT)
  const [plan, setPlan] = useState<InterviewPlan | null>(null)
  const [phase, setPhase] = useState<PracticePhase>('setup')
  const [mode, setMode] = useState<PracticeMode>('thinking')
  const [currentQuestionIndex, setCurrentQuestionIndex] = useState(0)
  const [timeLeft, setTimeLeft] = useState(DEFAULT_DRAFT.thinkingTimeSec)
  const [transcript, setTranscript] = useState('')
  const [interimTranscript, setInterimTranscript] = useState('')
  const [recordingUrl, setRecordingUrl] = useState('')
  const [report, setReport] = useState<InterviewReport | null>(null)
  const [statusMessage, setStatusMessage] = useState('')
  const [errorMessage, setErrorMessage] = useState('')

  const videoRef = useRef<HTMLVideoElement>(null)
  const streamRef = useRef<MediaStream | null>(null)
  const recorderRef = useRef<MediaRecorder | null>(null)
  const chunksRef = useRef<Blob[]>([])
  const recognitionRef = useRef<SpeechRecognitionLike | null>(null)
  const startedAtRef = useRef<number>(0)
  const framesRef = useRef<Array<{ brightness: number; delta: number }>>([])
  const lastFrameRef = useRef<number[] | null>(null)
  const resolutionRef = useRef('unavailable')
  const abortRef = useRef<AbortController | null>(null)

  const settings = useAppStore(s => s.settings)
  const openaiKey = useAppStore(s => s.openaiKey)
  const activeQuestion = plan?.questions[currentQuestionIndex] ?? null
  const canUseSpeechRecognition = typeof window !== 'undefined' && Boolean(window.SpeechRecognition || window.webkitSpeechRecognition)

  function updateDraft(patch: Partial<RoleDraft>) {
    setDraft(prev => ({ ...prev, ...patch }))
  }

  function attachCameraPreview() {
    const video = videoRef.current
    const stream = streamRef.current
    if (!video || !stream) return
    if (video.srcObject !== stream) video.srcObject = stream
    void video.play().catch(() => {
      setStatusMessage('Camera is recording, but the browser blocked preview autoplay. Click the preview if it stays paused.')
    })
  }

  async function generatePlan() {
    setErrorMessage('')
    setStatusMessage(openaiKey ? 'Generating structured interview JSON...' : 'Generating local practice JSON...')
    setReport(null)
    setRecordingUrl('')
    setTranscript('')
    setInterimTranscript('')
    abortRef.current?.abort()
    const controller = new AbortController()
    abortRef.current = controller

    try {
      let nextPlan: InterviewPlan
      if (openaiKey) {
        const response = await requestJson<InterviewPlan>({
          apiKey: openaiKey,
          model: settings.openaiModel,
          temperature: settings.openaiTemperature,
          signal: controller.signal,
          system: [
            'You generate realistic HireVue-style interview practice plans.',
            'Return only valid JSON. No markdown.',
            'Schema: { "roleTitle": string, "company": string, "format": string, "generatedAt": ISO string, "questions": [{ "id": number, "question": string, "thinkingTimeSec": number, "speakingTimeSec": number, "competency": string, "evaluationFocus": string[] }] }.',
            'Questions must be specific to the role details and suitable for recorded one-way interviews.',
          ].join(' '),
          user: JSON.stringify({
            roleTitle: draft.roleTitle,
            company: draft.company,
            roleDetails: draft.roleDetails,
            format: FORMAT_LABELS[draft.format],
            questionCount: draft.questionCount,
            thinkingTimeSec: draft.thinkingTimeSec,
            speakingTimeSec: draft.speakingTimeSec,
            focusAreas: draft.focusAreas,
          }),
        })
        nextPlan = normalisePlan(response, draft)
      } else {
        nextPlan = fallbackPlan(draft)
      }
      setPlan(nextPlan)
      setCurrentQuestionIndex(0)
      setTimeLeft(nextPlan.questions[0]?.thinkingTimeSec ?? draft.thinkingTimeSec)
      setMode('thinking')
      setPhase('ready')
      setStatusMessage(openaiKey ? 'AI JSON ready.' : 'Local JSON ready. Add an OpenAI key in Settings for AI reports.')
    } catch (error) {
      if (controller.signal.aborted) return
      setErrorMessage(error instanceof Error ? error.message : 'Unable to generate interview JSON.')
      const nextPlan = fallbackPlan(draft)
      setPlan(nextPlan)
      setCurrentQuestionIndex(0)
      setTimeLeft(nextPlan.questions[0].thinkingTimeSec)
      setMode('thinking')
      setPhase('ready')
      setStatusMessage('OpenAI failed, so a local question plan was generated.')
    }
  }

  async function startPractice() {
    if (!activeQuestion) return
    setErrorMessage('')
    setStatusMessage('Requesting camera and microphone...')
    setTranscript('')
    setInterimTranscript('')
    setReport(null)
    if (recordingUrl) URL.revokeObjectURL(recordingUrl)
    setRecordingUrl('')
    framesRef.current = []
    lastFrameRef.current = null
    chunksRef.current = []

    try {
      const stream = await navigator.mediaDevices.getUserMedia({ video: true, audio: true })
      streamRef.current = stream
      const videoTrack = stream.getVideoTracks()[0]
      const settings = videoTrack?.getSettings()
      resolutionRef.current = settings?.width && settings?.height ? `${settings.width}x${settings.height}` : 'available'

      const recorder = new MediaRecorder(stream)
      recorderRef.current = recorder
      recorder.ondataavailable = event => {
        if (event.data.size > 0) chunksRef.current.push(event.data)
      }
      recorder.onstop = () => {
        const blob = new Blob(chunksRef.current, { type: recorder.mimeType || 'video/webm' })
        setRecordingUrl(URL.createObjectURL(blob))
      }
      recorder.start(1000)

      startSpeechRecognition()
      startedAtRef.current = Date.now()
      setMode('thinking')
      setTimeLeft(activeQuestion.thinkingTimeSec)
      setPhase('practice')
      window.requestAnimationFrame(attachCameraPreview)
      setStatusMessage(canUseSpeechRecognition ? 'Recording started. Transcript capture is active.' : 'Recording started. Transcript capture is unavailable in this browser.')
    } catch (error) {
      setErrorMessage(error instanceof Error ? error.message : 'Camera or microphone permission failed.')
      setStatusMessage('')
    }
  }

  function startSpeechRecognition() {
    const Recognition = window.SpeechRecognition || window.webkitSpeechRecognition
    if (!Recognition) return
    const recognition = new Recognition()
    recognition.continuous = true
    recognition.interimResults = true
    recognition.lang = 'en-US'
    recognition.onresult = event => {
      let finalText = ''
      let interimText = ''
      for (let i = event.resultIndex; i < event.results.length; i++) {
        const part = event.results[i][0]?.transcript ?? ''
        if (event.results[i].isFinal) finalText += `${part} `
        else interimText += `${part} `
      }
      if (finalText) setTranscript(prev => `${prev} ${finalText}`.trim())
      setInterimTranscript(interimText.trim())
    }
    recognition.onerror = () => setStatusMessage('Speech transcript paused. Recording continues.')
    recognitionRef.current = recognition
    try {
      recognition.start()
    } catch {
      // Some browsers throw if recognition starts too quickly after permission.
    }
  }

  function sampleVideoFrame() {
    const video = videoRef.current
    if (!video || video.videoWidth === 0 || video.videoHeight === 0) return
    const canvas = document.createElement('canvas')
    canvas.width = 96
    canvas.height = 54
    const ctx = canvas.getContext('2d')
    if (!ctx) return
    ctx.drawImage(video, 0, 0, canvas.width, canvas.height)
    const data = ctx.getImageData(0, 0, canvas.width, canvas.height).data
    const luminance: number[] = []
    let sum = 0
    for (let i = 0; i < data.length; i += 16) {
      const y = data[i] * 0.2126 + data[i + 1] * 0.7152 + data[i + 2] * 0.0722
      luminance.push(y)
      sum += y
    }
    const brightness = (sum / luminance.length / 255) * 100
    const previous = lastFrameRef.current
    const delta = previous
      ? luminance.reduce((acc, value, index) => acc + Math.abs(value - (previous[index] ?? value)), 0) / luminance.length / 255 * 100
      : 0
    framesRef.current.push({ brightness, delta })
    lastFrameRef.current = luminance
  }

  async function stopPractice(autoAnalyze = true) {
    recognitionRef.current?.stop()
    recognitionRef.current = null
    if (recorderRef.current?.state === 'recording') recorderRef.current.stop()
    streamRef.current?.getTracks().forEach(track => track.stop())
    streamRef.current = null
    if (videoRef.current) videoRef.current.srcObject = null
    setMode('complete')
    if (autoAnalyze) {
      setPhase('analyzing')
      setStatusMessage(openaiKey ? 'Generating AI report from transcript and delivery metrics...' : 'OpenAI key required for an AI report.')
      await analyzeAttempt()
    }
  }

  async function analyzeAttempt() {
    if (!activeQuestion) return
    const durationSec = Math.max(1, Math.round((Date.now() - startedAtRef.current) / 1000))
    const videoMetrics = summariseFrames(framesRef.current, resolutionRef.current)
    const finalTranscript = `${transcript} ${interimTranscript}`.trim()
    const fallback = buildFallbackReport({
      draft,
      question: activeQuestion,
      transcript: finalTranscript,
      durationSec,
      videoMetrics,
    })

    if (!openaiKey) {
      setReport(null)
      setPhase('ready')
      setStatusMessage('Recording saved. Add an OpenAI key in Settings to generate the report.')
      setErrorMessage('AI report generation requires an OpenAI key. No deterministic report was generated.')
      return
    }

    try {
      const calibration = scoringCalibration(draft, activeQuestion, durationSec)
      const response = await requestJson<InterviewReport>({
        apiKey: openaiKey,
        model: settings.openaiModel,
        temperature: 0.1,
        system: [
          'You are a calibrated HireVue interview evaluator.',
          'Return only valid JSON using this schema: { "overallScore": number, "scores": object, "strengths": string[], "criticalFeedback": string[], "improvementPlan": string[], "report": string }.',
          'Scores must be evidence-based and anchored only to the supplied role context, question, transcript, transcript metrics, timing, and video metrics.',
          'Use the supplied seniorityCalibration as the scoring standard. Do not apply graduate or experienced-hire standards to internship roles.',
          'For short timed answers, reward good prioritisation. Do not penalize a candidate for failing to mention every possible nuance when the core answer is directionally sound.',
          'If the transcript is unavailable, say that content scoring is limited, avoid inventing answer content, and grade structure, evidence, and role fit conservatively.',
          'Be realistic, fair, and useful. Do not inflate generic or missing answers, but do not be harsher than the role level warrants.',
        ].join(' '),
        user: JSON.stringify({
          role: draft,
          question: activeQuestion,
          transcript: finalTranscript || missingTranscriptNotice(canUseSpeechRecognition),
          transcriptAvailable: Boolean(finalTranscript),
          durationSec,
          transcriptMetrics: fallback.transcriptMetrics,
          videoMetrics,
          seniorityCalibration: calibration,
          scoringArchitecture: {
            dimensions: Object.keys(fallback.scores),
            scale: '0-100',
            rule: 'Score only observable transcript content, timing, and deterministic video metrics. Do not use the deterministic fallback scores as the final report.',
          },
        }),
      })
      const nextReport = normaliseReport(response, fallback)
      setReport(nextReport)
      setPhase('report')
      setStatusMessage(finalTranscript ? 'AI report ready.' : 'AI report ready with transcript limitations.')
    } catch (error) {
      setReport(null)
      setPhase('ready')
      setStatusMessage('AI report generation failed. No deterministic report was generated.')
      setErrorMessage(error instanceof Error ? error.message : 'Unable to run OpenAI analysis.')
    }
  }

  function resetAttempt() {
    void stopPractice(false)
    setPhase(plan ? 'ready' : 'setup')
    setMode('thinking')
    setTimeLeft(activeQuestion?.thinkingTimeSec ?? draft.thinkingTimeSec)
    setTranscript('')
    setInterimTranscript('')
    setReport(null)
    setErrorMessage('')
    setStatusMessage('')
  }

  useEffect(() => {
    if (phase !== 'practice') return
    attachCameraPreview()
    const attachTimer = window.setTimeout(attachCameraPreview, 0)
    const interval = window.setInterval(sampleVideoFrame, 2500)
    return () => {
      window.clearTimeout(attachTimer)
      window.clearInterval(interval)
    }
  }, [phase])

  useEffect(() => {
    if (phase !== 'practice' || mode === 'complete') return
    const timer = window.setInterval(() => {
      setTimeLeft(prev => {
        if (prev > 1) return prev - 1
        if (mode === 'thinking' && activeQuestion) {
          setMode('speaking')
          return activeQuestion.speakingTimeSec
        }
        void stopPractice(true)
        return 0
      })
    }, 1000)
    return () => window.clearInterval(timer)
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [phase, mode, activeQuestion?.id])

  useEffect(() => () => {
    abortRef.current?.abort()
    recognitionRef.current?.stop()
    streamRef.current?.getTracks().forEach(track => track.stop())
    if (recordingUrl) URL.revokeObjectURL(recordingUrl)
  }, [recordingUrl])

  return (
    <div className="flex h-full flex-col">
      <div className="flex shrink-0 items-center gap-3 border-b border-border bg-s1 px-4 py-2">
        <Video size={16} className="text-accent" aria-hidden="true" />
        <span className="text-sm font-medium text-text">HireVue Prep</span>
        <span className="text-2xs text-muted">structured prompts, timed recording, transcript and video scoring</span>
        <div className="ml-auto flex items-center gap-2 text-2xs text-dim">
          <span className={openaiKey ? 'text-up' : 'text-warn'}>{openaiKey ? 'OpenAI ready' : 'Local mode'}</span>
        </div>
      </div>

      <div className="grid flex-1 grid-cols-1 overflow-auto lg:grid-cols-[360px_minmax(0,1fr)]">
        <aside className="border-b border-border bg-bg/55 p-4 lg:border-b-0 lg:border-r">
          <div className="mb-4 flex items-start gap-3">
            <ClipboardList size={18} className="mt-1 text-accent" aria-hidden="true" />
            <div>
              <h1 className="text-lg font-semibold text-text">Interview setup</h1>
              <p className="mt-1 text-2xs leading-relaxed text-muted">
                Add the role context, generate question JSON, then launch a timed recorded answer.
              </p>
            </div>
          </div>

          <div className="space-y-3">
            <label className="block">
              <span className="mb-1 block text-2xs text-muted">Company</span>
              <input className={INPUT_CLASS} value={draft.company} onChange={event => updateDraft({ company: event.target.value })} placeholder="TradeSmart Analytics" />
            </label>
            <label className="block">
              <span className="mb-1 block text-2xs text-muted">Role title</span>
              <input className={INPUT_CLASS} value={draft.roleTitle} onChange={event => updateDraft({ roleTitle: event.target.value })} placeholder="Summer Quantitative Analyst (26/27)" />
            </label>
            <label className="block">
              <span className="mb-1 block text-2xs text-muted">Role details</span>
              <textarea
                className={`${INPUT_CLASS} min-h-[132px] resize-y`}
                value={draft.roleDetails}
                onChange={event => updateDraft({ roleDetails: event.target.value })}
                placeholder="Paste job description, company values, interview notes, key responsibilities, and anything the recruiter mentioned."
              />
            </label>
            <label className="block">
              <span className="mb-1 block text-2xs text-muted">Format</span>
              <select className={INPUT_CLASS} value={draft.format} onChange={event => updateDraft({ format: event.target.value as InterviewFormat })}>
                {Object.entries(FORMAT_LABELS).map(([value, label]) => (
                  <option key={value} value={value}>{label}</option>
                ))}
              </select>
            </label>

            <div className="grid grid-cols-3 gap-2">
              <label className="block">
                <span className="mb-1 block text-2xs text-muted">Questions</span>
                <input className={INPUT_CLASS} type="number" min={1} max={10} value={draft.questionCount} onChange={event => updateDraft({ questionCount: clamp(Number(event.target.value), 1, 10) })} />
              </label>
              <label className="block">
                <span className="mb-1 block text-2xs text-muted">Think</span>
                <input className={INPUT_CLASS} type="number" min={10} max={180} value={draft.thinkingTimeSec} onChange={event => updateDraft({ thinkingTimeSec: clamp(Number(event.target.value), 10, 180) })} />
              </label>
              <label className="block">
                <span className="mb-1 block text-2xs text-muted">Speak</span>
                <input className={INPUT_CLASS} type="number" min={30} max={300} value={draft.speakingTimeSec} onChange={event => updateDraft({ speakingTimeSec: clamp(Number(event.target.value), 30, 300) })} />
              </label>
            </div>

            <label className="block">
              <span className="mb-1 block text-2xs text-muted">Scoring focus</span>
              <input className={INPUT_CLASS} value={draft.focusAreas} onChange={event => updateDraft({ focusAreas: event.target.value })} />
            </label>

            <button type="button" className={`${BUTTON_CLASS} w-full border-accent bg-accent text-bg hover:text-bg`} onClick={() => void generatePlan()}>
              <Brain size={15} aria-hidden="true" />
              Generate JSON
            </button>

            {!openaiKey && (
              <div className="border border-warn/40 bg-warn/10 p-3">
                <div className="flex gap-2">
                  <Settings size={15} className="mt-0.5 shrink-0 text-warn" aria-hidden="true" />
                  <p className="text-2xs leading-relaxed text-muted">
                    Local mode can prepare questions, but reports require OpenAI. Add an OpenAI key in <Link to="/settings" className="text-accent hover:text-text">Settings</Link> for real AI feedback.
                  </p>
                </div>
              </div>
            )}
          </div>
        </aside>

        <main className="min-w-0 overflow-auto p-4">
          {(statusMessage || errorMessage) && (
            <div className="mb-4 grid gap-2">
              {statusMessage && (
                <div className="flex items-center gap-2 border border-border bg-s1 px-3 py-2 text-2xs text-muted">
                  <CheckCircle2 size={14} className="text-up" aria-hidden="true" />
                  {statusMessage}
                </div>
              )}
              {errorMessage && (
                <div className="flex items-center gap-2 border border-down/40 bg-down/10 px-3 py-2 text-2xs text-down">
                  <AlertTriangle size={14} aria-hidden="true" />
                  {errorMessage}
                </div>
              )}
            </div>
          )}

          <div className="grid gap-4 xl:grid-cols-[minmax(0,1fr)_420px]">
            <section className={PANEL_CLASS}>
              <div className="flex items-center gap-2 border-b border-border px-4 py-3">
                <Camera size={16} className="text-accent" aria-hidden="true" />
                <h2 className="text-sm font-semibold text-text">Practice room</h2>
                {activeQuestion && (
                  <span className="ml-auto text-2xs tabnum text-dim">
                    Q{currentQuestionIndex + 1}/{plan?.questions.length ?? 0}
                  </span>
                )}
              </div>

              <div className="grid gap-4 p-4 lg:grid-cols-[minmax(0,1fr)_260px]">
                <div>
                  <div className="aspect-video overflow-hidden border border-border bg-black">
                    {phase === 'practice' ? (
                      <video ref={videoRef} autoPlay muted playsInline className="h-full w-full scale-x-[-1] object-cover" />
                    ) : recordingUrl ? (
                      <video src={recordingUrl} controls className="h-full w-full object-cover" />
                    ) : (
                      <div className="flex h-full items-center justify-center text-center text-2xs text-dim">
                        Camera preview appears when practice starts.
                      </div>
                    )}
                  </div>

                  <div className="mt-3 min-h-[120px] border border-border bg-bg/60 p-3">
                    <div className="mb-2 flex items-center gap-2">
                      <Mic size={14} className="text-accent" aria-hidden="true" />
                      <span className="text-2xs font-medium uppercase tracking-widest text-dim">Transcript</span>
                    </div>
                    <p className="text-sm leading-relaxed text-muted">
                      {transcript || interimTranscript ? (
                        <>
                          {transcript} <span className="text-dim">{interimTranscript}</span>
                        </>
                      ) : (
                        canUseSpeechRecognition
                          ? 'Transcript will appear while you answer.'
                          : 'This browser does not expose live speech recognition. You can still record; AI reporting will use timing and video metrics only.'
                      )}
                    </p>
                  </div>
                </div>

                <div className="space-y-3">
                  <div className="border border-border bg-bg/55 p-4 text-center">
                    <div className="flex items-center justify-center gap-2 text-2xs uppercase tracking-widest text-dim">
                      <Timer size={14} aria-hidden="true" />
                      {mode}
                    </div>
                    <div className="mt-2 text-4xl tabnum font-semibold text-text">{secondsLabel(timeLeft)}</div>
                  </div>

                  {activeQuestion ? (
                    <div className="border border-border bg-bg/55 p-4">
                      <p className="text-2xs uppercase tracking-widest text-dim">{activeQuestion.competency}</p>
                      <p className="mt-2 text-sm leading-relaxed text-text">{activeQuestion.question}</p>
                      <div className="mt-3 flex flex-wrap gap-2">
                        {activeQuestion.evaluationFocus.map(item => (
                          <span key={item} className="border border-border px-2 py-1 text-2xs text-muted">{item}</span>
                        ))}
                      </div>
                    </div>
                  ) : (
                    <div className="border border-border bg-bg/55 p-4 text-sm text-muted">
                      Generate the interview JSON to unlock the practice room.
                    </div>
                  )}

                  <div className="grid grid-cols-2 gap-2">
                    <button type="button" className={BUTTON_CLASS} disabled={!activeQuestion || phase === 'practice' || phase === 'analyzing'} onClick={() => void startPractice()}>
                      <Play size={15} aria-hidden="true" />
                      Launch
                    </button>
                    <button type="button" className={BUTTON_CLASS} disabled={phase !== 'practice'} onClick={() => void stopPractice(true)}>
                      <Square size={15} aria-hidden="true" />
                      Stop
                    </button>
                    <button type="button" className={BUTTON_CLASS} disabled={phase !== 'practice' || mode !== 'thinking'} onClick={() => {
                      if (activeQuestion) {
                        setMode('speaking')
                        setTimeLeft(activeQuestion.speakingTimeSec)
                      }
                    }}>
                      <Mic size={15} aria-hidden="true" />
                      Answer
                    </button>
                    <button type="button" className={BUTTON_CLASS} onClick={resetAttempt}>
                      <RotateCcw size={15} aria-hidden="true" />
                      Reset
                    </button>
                  </div>

                  {phase === 'analyzing' && (
                    <div className="flex items-center gap-2 border border-accent/40 bg-accent/10 p-3 text-2xs text-accent">
                      <Loader2 size={14} className="animate-spin" aria-hidden="true" />
                      Building report...
                    </div>
                  )}
                </div>
              </div>
            </section>

            <section className={PANEL_CLASS}>


            </section>
          </div>

          {report && (
            <section className={`${PANEL_CLASS} mt-4`}>
              <div className="flex flex-wrap items-center gap-3 border-b border-border px-4 py-3">
                <BarChart3 size={16} className="text-accent" aria-hidden="true" />
                <h2 className="text-sm font-semibold text-text">Full report</h2>
                <span className="ml-auto text-2xs text-muted">Overall</span>
                <span className="text-lg tabnum font-semibold text-accent">{report.overallScore}</span>
                <span className="text-2xs text-dim">{scoreLabel(report.overallScore)}</span>
              </div>

              <div className="grid gap-4 p-4 xl:grid-cols-[minmax(0,1fr)_360px]">
                <div className="space-y-4">
                  <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-4">
                    {Object.entries(report.scores).map(([label, score]) => (
                      <ScorePill key={label} label={label} score={score} />
                    ))}
                  </div>

                  <div className="border border-border bg-bg/55 p-4">
                    <h3 className="text-sm font-semibold text-text">Evaluator summary</h3>
                    <p className="mt-2 text-sm leading-relaxed text-muted">{report.report}</p>
                  </div>

                  <div className="grid gap-4 lg:grid-cols-2">
                    <div className="border border-border bg-bg/55 p-4">
                      <h3 className="text-sm font-semibold text-text">Critical feedback</h3>
                      <ul className="mt-3 space-y-2 text-sm leading-relaxed text-muted">
                        {report.criticalFeedback.map(item => <li key={item}>- {item}</li>)}
                      </ul>
                    </div>
                    <div className="border border-border bg-bg/55 p-4">
                      <h3 className="text-sm font-semibold text-text">Steps to improve</h3>
                      <ol className="mt-3 space-y-2 text-sm leading-relaxed text-muted">
                        {report.improvementPlan.map((item, index) => <li key={item}>{index + 1}. {item}</li>)}
                      </ol>
                    </div>
                  </div>
                </div>

                <div className="space-y-4">
                  <div className="border border-border bg-bg/55 p-4">
                    <h3 className="text-sm font-semibold text-text">Transcript metrics</h3>
                    <dl className="mt-3 grid grid-cols-2 gap-2 text-2xs">
                      {Object.entries(report.transcriptMetrics).map(([key, value]) => (
                        <div key={key} className="border border-border bg-s1 p-2">
                          <dt className="uppercase tracking-widest text-dim">{key}</dt>
                          <dd className="mt-1 tabnum text-sm text-text">{value}</dd>
                        </div>
                      ))}
                    </dl>
                  </div>

                  <div className="border border-border bg-bg/55 p-4">
                    <h3 className="text-sm font-semibold text-text">Video metrics</h3>
                    <dl className="mt-3 grid grid-cols-2 gap-2 text-2xs">
                      {Object.entries(report.videoMetrics).map(([key, value]) => (
                        <div key={key} className="border border-border bg-s1 p-2">
                          <dt className="uppercase tracking-widest text-dim">{key}</dt>
                          <dd className="mt-1 tabnum text-sm text-text">{value}</dd>
                        </div>
                      ))}
                    </dl>
                  </div>

                  <div className="border border-border bg-bg/55 p-4">
                    <h3 className="text-sm font-semibold text-text">Strengths</h3>
                    <ul className="mt-3 space-y-2 text-sm leading-relaxed text-muted">
                      {report.strengths.map(item => <li key={item}>- {item}</li>)}
                    </ul>
                  </div>
                </div>
              </div>
            </section>
          )}
        </main>
      </div>
    </div>
  )
}
