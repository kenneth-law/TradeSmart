import { useEffect, useState } from 'react'
import type { ReactNode } from 'react'
import { useNavigate } from 'react-router-dom'
import TopBar from './TopBar'
import StatusBar from './StatusBar'
import CommandPalette from './CommandPalette'

const SHORTCUTS: Record<string, string> = {
  '1': '/', '2': '/backtest', '3': '/integrated',
  '4': '/market', '5': '/portfolio', '6': '/docs',
}

export default function Shell({ children }: { children: ReactNode }) {
  const navigate = useNavigate()
  const [cmdOpen, setCmdOpen] = useState(false)

  useEffect(() => {
    function onKey(e: KeyboardEvent) {
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
  }, [navigate])

  return (
    <div className="flex flex-col bg-bg" style={{ height: '100vh' }}>
      <TopBar onCommandPalette={() => setCmdOpen(true)} />
      <main
        className="flex-1 overflow-auto"
        style={{ marginTop: 36, marginBottom: 22 }}
      >
        {children}
      </main>
      <StatusBar />
      <CommandPalette open={cmdOpen} onClose={() => setCmdOpen(false)} />
    </div>
  )
}
