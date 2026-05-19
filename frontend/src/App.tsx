import { Routes, Route } from 'react-router-dom'
import Shell from './components/layout/Shell'
import Dashboard from './pages/Dashboard'
import Results from './pages/Results'
import StockDetail from './pages/StockDetail'
import Backtest from './pages/Backtest'
import IntegratedSystem from './pages/IntegratedSystem'
import MarketOverview from './pages/MarketOverview'
import Portfolio from './pages/Portfolio'
import Documentation from './pages/Documentation'

export default function App() {
  return (
    <Shell>
      <Routes>
        <Route path="/"            element={<Dashboard />} />
        <Route path="/results"     element={<Results />} />
        <Route path="/stock/:ticker" element={<StockDetail />} />
        <Route path="/backtest"    element={<Backtest />} />
        <Route path="/integrated"  element={<IntegratedSystem />} />
        <Route path="/market"      element={<MarketOverview />} />
        <Route path="/portfolio"   element={<Portfolio />} />
        <Route path="/docs"        element={<Documentation />} />
        <Route path="/docs/:type"  element={<Documentation />} />
      </Routes>
    </Shell>
  )
}
