import { useEffect, useState } from 'react'
import type { ReactNode } from 'react'
import { useLocation, useNavigate } from 'react-router-dom'
import { AlertTriangle, X } from 'lucide-react'
import { hasSystemSettingsCookie } from '../../store/useAppStore'
import TopBar from './TopBar'
import StatusBar from './StatusBar'
import CommandPalette from './CommandPalette'
import Onboarding from '../onboarding/Onboarding'
import heroImage from '../../assets/landing-hero.jpg'

const SHORTCUTS: Record<string, string> = {
  '1': '/', '2': '/market', '3': '/system',
  '4': '/technical', '5': '/daily-lineup', '6': '/portfolio',
  '7': '/docs', '8': '/settings',
}

export default function Shell({ children }: { children: ReactNode }) {
  const navigate = useNavigate()
  const location = useLocation()
  const [cmdOpen, setCmdOpen] = useState(false)
  const [legalNoticeOpen, setLegalNoticeOpen] = useState(true)
  const [onboardingOpen, setOnboardingOpen] = useState(() => !hasSystemSettingsCookie())
  const showDashboardBackground = location.pathname !== '/'

  useEffect(() => {
    function onKey(e: KeyboardEvent) {
      if (onboardingOpen) {
        setCmdOpen(false)
        return
      }
      if (e.altKey && SHORTCUTS[e.key]) {
        e.preventDefault()
        navigate(SHORTCUTS[e.key])
      }
      if ((e.ctrlKey || e.metaKey) && e.key === 'k') {
        e.preventDefault()
        setCmdOpen(o => !o)
      }
      if (e.key === 'Escape') setCmdOpen(false)
    }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [navigate, onboardingOpen])

  const isHome = !showDashboardBackground

  return (
    <div className="relative flex flex-col overflow-hidden bg-bg" style={{ height: '100vh' }}>
      {/* Always mounted so it never remounts/resizes on route change */}
      <img
        src={heroImage}
        alt=""
        aria-hidden="true"
        className={[
          'pointer-events-none absolute inset-0 h-full w-full object-cover',
          isHome ? 'theme-hero-image' : 'theme-hero-image-blurred opacity-10',
        ].join(' ')}
      />
      <div className={[
        'pointer-events-none absolute inset-0',
        isHome ? 'landing-hero-overlay' : 'dashboard-hero-overlay',
      ].join(' ')} />
      <TopBar onCommandPalette={() => setCmdOpen(true)} />
      <main
        className="relative z-10 flex-1 overflow-auto"
        style={{ marginTop: 'var(--topbar-height)', marginBottom: 'var(--statusbar-height)' }}
      >
        {children}
      </main>
      <StatusBar />
      {legalNoticeOpen && !onboardingOpen && (
        <div
          className="fixed inset-0 z-[70] flex items-center justify-center bg-black/65 px-4 backdrop-blur-sm"
          role="presentation"
        >
          <section
            role="dialog"
            aria-modal="true"
            aria-labelledby="legal-disclaimer-title"
            aria-describedby="legal-disclaimer-description"
            className="w-full max-w-md border border-border-strong bg-s1 shadow-2xl"
          >
            <div className="flex items-start gap-3 border-b border-border px-4 py-3">
              <span className="mt-0.5 flex h-8 w-8 shrink-0 items-center justify-center border border-warn/50 bg-warn/10 text-warn">
                <AlertTriangle size={18} aria-hidden="true" />
              </span>
              <div className="min-w-0">
                <p id="legal-disclaimer-title" className="text-sm font-medium text-text">
                  Legal disclaimer
                </p>
                <p id="legal-disclaimer-description" className="mt-1 text-2xs leading-5 text-muted">
                  TradeSmart is an experimental research tool. Content, simulations, signals, and portfolio outputs are for informational purposes only and are not financial advice.
                </p>
              </div>
              <button
                type="button"
                onClick={() => setLegalNoticeOpen(false)}
                className="ml-auto flex h-8 w-8 shrink-0 items-center justify-center text-dim hover:text-text"
                aria-label="Dismiss legal disclaimer"
              >
                <X size={16} aria-hidden="true" />
              </button>
            </div>
            <div className="flex justify-end px-4 py-3">
              <button
                type="button"
                autoFocus
                onClick={() => setLegalNoticeOpen(false)}
                className="border border-accent bg-accent px-3 py-1.5 text-2xs font-medium text-bg hover:opacity-90"
              >
                I understand
              </button>
            </div>
          </section>
        </div>
      )}
      {onboardingOpen && (
        <Onboarding onComplete={() => setOnboardingOpen(false)} />
      )}
      <CommandPalette open={cmdOpen} onClose={() => setCmdOpen(false)} />
    </div>
  )
}
