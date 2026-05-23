import { useEffect, useState } from 'react'
import type { ReactNode } from 'react'
import { useLocation, useNavigate } from 'react-router-dom'
import { hasSystemSettingsCookie } from '../../store/useAppStore'
import TopBar from './TopBar'
import StatusBar from './StatusBar'
import CommandPalette from './CommandPalette'
import Onboarding, { AppIntroOverlay } from '../onboarding/Onboarding'
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
  const [onboardingOpen, setOnboardingOpen] = useState(() => !hasSystemSettingsCookie())
  const [appIntroOpen, setAppIntroOpen] = useState(() => hasSystemSettingsCookie())
  const showDashboardBackground = location.pathname !== '/'

  useEffect(() => {
    function onKey(e: KeyboardEvent) {
      if (onboardingOpen || appIntroOpen) {
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
  }, [appIntroOpen, navigate, onboardingOpen])

  const isHome = !showDashboardBackground

  return (
    <div className="relative flex flex-col overflow-hidden bg-bg" style={{ height: '100dvh' }}>
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
      {onboardingOpen && (
        <Onboarding onComplete={() => setOnboardingOpen(false)} />
      )}
      {appIntroOpen && !onboardingOpen && (
        <AppIntroOverlay onComplete={() => setAppIntroOpen(false)} />
      )}
      <CommandPalette open={cmdOpen} onClose={() => setCmdOpen(false)} />
    </div>
  )
}
