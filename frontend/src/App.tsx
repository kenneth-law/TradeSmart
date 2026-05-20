import { Routes, Route } from 'react-router-dom'
import Shell from './components/layout/Shell'
import LandingPage from './pages/LandingPage'
import Technical from './pages/Technical'
import Results from './pages/Results'
import StockDetail from './pages/StockDetail'
import System from './pages/Backtest'
import MarketOverview from './pages/MarketOverview'
import Portfolio from './pages/Portfolio'
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
        <Route path="/integrated"  element={<System />} />
        <Route path="/market"      element={<MarketOverview />} />
        <Route path="/portfolio"   element={<Portfolio />} />
        <Route path="/docs"        element={<Documentation />} />
        <Route path="/docs/:type"  element={<Documentation />} />
        <Route path="/settings"    element={<Settings />} />
      </Routes>
    </Shell>
  )
}
