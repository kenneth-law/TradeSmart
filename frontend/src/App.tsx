import { Navigate, Routes, Route } from 'react-router-dom'
import Shell from './components/layout/Shell'
import LandingPage from './pages/LandingPage'
import Technical from './pages/Technical'
import Results from './pages/Results'
import StockDetail from './pages/StockDetail'
import System from './pages/Backtest'
import Portfolio from './pages/Portfolio'
import Research from './pages/Research'
import DailyLineup from './pages/DailyLineup'
import Documentation from './pages/Documentation'
import Settings from './pages/Settings'

export default function App() {
  return (
    <Shell>
      <Routes>
        <Route path="/"            element={<LandingPage />} />
        <Route path="/technical"   element={<Technical />} />
        <Route path="/results"     element={<Results />} />
        <Route path="/stock/:ticker" element={<StockDetail />} />
        <Route path="/system"      element={<System />} />
        <Route path="/backtest"    element={<System />} />
        <Route path="/integrated"  element={<Navigate to="/system" replace />} />
        <Route path="/market"      element={<Research />} />
        <Route path="/research"    element={<Navigate to="/market" replace />} />
        <Route path="/daily-lineup" element={<DailyLineup />} />
        <Route path="/portfolio"   element={<Portfolio />} />
        <Route path="/docs"        element={<Documentation />} />
        <Route path="/docs/:type"  element={<Documentation />} />
        <Route path="/settings"    element={<Settings />} />
      </Routes>
    </Shell>
  )
}
