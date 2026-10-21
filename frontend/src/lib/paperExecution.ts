export type ExecutionAlgo = 'smart' | 'pov' | 'twap' | 'vwap' | 'arrival' | 'open' | 'close' | 'iceberg'
export type StockOrderType = 'market' | 'limit' | 'stop' | 'stop-limit'
export type TimeInForce = 'DAY' | 'IOC' | 'FOK'

export interface ExecutionParameters {
  algo: ExecutionAlgo
  orderType: StockOrderType
  timeInForce: TimeInForce
  limitPrice?: number
  stopPrice?: number
  durationSeconds: number
  slices: number
  participationRate: number
  displayQuantity: number
  latencyMinMs: number
  latencyMaxMs: number
  spreadBps: number
  baseSlippageBps: number
  impactCoefficient: number
  volatilityPct: number
  availableVolume: number
  randomize: boolean
  urgency: 1 | 2 | 3 | 4 | 5
  completion: 'must-complete' | 'price-sensitive' | 'liquidity-only'
  maxParticipationRate: number
  maxSlippageBps: number
  minChildQuantity: number
  maxChildQuantity: number
  useLitVenues: boolean
  useDarkPools: boolean
  useMidpoint: boolean
  includeAuction: boolean
  darkExposureSeconds: number
  minDarkFill: number
  darkFillProbabilityPct: number
  stealthPct: number
  volatilityPausePct: number
  startDelaySeconds: number
}

export interface ExecutionSlice {
  quantity: number
  delayMs: number
  expectedPrice: number
  slippageBps: number
  participationPct: number
  venue: 'LIT' | 'DARK' | 'MIDPOINT' | 'AUCTION' | 'SMART'
  instruction: 'POST' | 'JOIN' | 'MID-PEG' | 'CROSS' | 'SWEEP' | 'SEEK'
}

export interface ExecutionPlan {
  slices: ExecutionSlice[]
  expectedAveragePrice: number
  expectedSlippageBps: number
  expectedDurationMs: number
}

export const DEFAULT_EXECUTION_PARAMETERS: ExecutionParameters = {
  algo: 'smart',
  orderType: 'market',
  timeInForce: 'DAY',
  durationSeconds: 8,
  slices: 5,
  participationRate: 10,
  displayQuantity: 100,
  latencyMinMs: 80,
  latencyMaxMs: 280,
  spreadBps: 4,
  baseSlippageBps: 1,
  impactCoefficient: 12,
  volatilityPct: 1.5,
  availableVolume: 100_000,
  randomize: true,
  urgency: 3,
  completion: 'must-complete',
  maxParticipationRate: 15,
  maxSlippageBps: 35,
  minChildQuantity: 1,
  maxChildQuantity: 50_000,
  useLitVenues: true,
  useDarkPools: true,
  useMidpoint: true,
  includeAuction: false,
  darkExposureSeconds: 10,
  minDarkFill: 100,
  darkFillProbabilityPct: 35,
  stealthPct: 50,
  volatilityPausePct: 5,
  startDelaySeconds: 0,
}

function clamp(value: number, min: number, max: number) {
  return Math.min(max, Math.max(min, value))
}

function splitInteger(total: number, weights: number[]) {
  const normalized = weights.map(weight => Math.max(0.0001, weight))
  const weightTotal = normalized.reduce((sum, weight) => sum + weight, 0)
  const raw = normalized.map(weight => total * weight / weightTotal)
  const parts = raw.map(Math.floor)
  let remainder = total - parts.reduce((sum, part) => sum + part, 0)
  raw
    .map((value, index) => ({ index, fraction: value - Math.floor(value) }))
    .sort((a, b) => b.fraction - a.fraction)
    .forEach(({ index }) => {
      if (remainder > 0) {
        parts[index] += 1
        remainder -= 1
      }
    })
  return parts.filter(part => part > 0)
}

function sliceWeights(algo: ExecutionAlgo, count: number) {
  if (algo === 'vwap') {
    // A compact U-shaped intraday volume curve.
    return Array.from({ length: count }, (_, index) => {
      const x = count === 1 ? 0 : index / (count - 1)
      return 0.65 + 1.7 * Math.pow(Math.abs(x - 0.5) * 2, 1.7)
    })
  }
  if (algo === 'smart') {
    return Array.from({ length: count }, (_, index) => Math.pow(0.82, index))
  }
  if (algo === 'arrival' || algo === 'open') {
    return Array.from({ length: count }, (_, index) => Math.pow(0.68, index))
  }
  if (algo === 'close') {
    return Array.from({ length: count }, (_, index) => Math.pow(0.68, count - index - 1))
  }
  return Array.from({ length: count }, () => 1)
}

export function orderIsTriggered(
  side: 'buy' | 'sell',
  marketPrice: number,
  parameters: ExecutionParameters,
) {
  const limit = Number(parameters.limitPrice)
  const stop = Number(parameters.stopPrice)
  const limitReached = !Number.isFinite(limit) || (side === 'buy' ? marketPrice <= limit : marketPrice >= limit)
  const stopReached = !Number.isFinite(stop) || (side === 'buy' ? marketPrice >= stop : marketPrice <= stop)
  if (parameters.orderType === 'limit') return limitReached
  if (parameters.orderType === 'stop') return stopReached
  if (parameters.orderType === 'stop-limit') return stopReached && limitReached
  return true
}

export function buildExecutionPlan(
  side: 'buy' | 'sell',
  quantity: number,
  marketPrice: number,
  raw: ExecutionParameters,
): ExecutionPlan {
  const qty = Math.max(1, Math.floor(quantity))
  const volume = Math.max(qty, raw.availableVolume)
  const durationMs = Math.max(0, raw.durationSeconds * 1000)
  let count = raw.algo === 'smart' ? Math.min(3, qty) : clamp(Math.floor(raw.slices), 1, Math.min(50, qty))

  if (raw.algo === 'iceberg') count = clamp(Math.ceil(qty / Math.max(1, raw.displayQuantity)), 1, 50)
  if (raw.algo === 'pov') {
    const capacityPerSlice = Math.max(1, Math.floor((volume / Math.max(1, count)) * raw.participationRate / 100))
    count = clamp(Math.ceil(qty / capacityPerSlice), 1, Math.min(50, qty))
  }

  count = Math.max(count, Math.ceil(qty / Math.max(1, raw.maxChildQuantity)))
  const participationCapacity = Math.max(1, Math.floor(volume * raw.maxParticipationRate / 100))
  count = Math.max(count, Math.ceil(qty / participationCapacity))
  count = clamp(count, 1, Math.min(50, Math.max(1, Math.floor(qty / Math.max(1, raw.minChildQuantity)))))

  const quantities = splitInteger(qty, sliceWeights(raw.algo, count)).flatMap(part => {
    const chunks = Math.ceil(part / Math.max(1, raw.maxChildQuantity))
    return splitInteger(part, Array.from({ length: chunks }, () => 1))
  })
  const minLatency = Math.max(0, raw.latencyMinMs)
  const maxLatency = Math.max(minLatency, raw.latencyMaxMs)
  let weightedPrice = 0
  let weightedBps = 0

  const slices = quantities.map((sliceQty, index) => {
    const position = quantities.length === 1 ? 0 : index / (quantities.length - 1)
    const jitter = raw.randomize ? 0.8 + Math.random() * 0.4 : 1
    const latency = minLatency + (maxLatency - minLatency) * (raw.randomize ? Math.random() : 0.5)
    const delayMs = raw.startDelaySeconds * 1000 + latency + durationMs * position * jitter
    const participationPct = sliceQty / volume * 100
    const deadlineRamp = raw.completion === 'must-complete' ? position * (5 - raw.urgency) : 0
    const effectiveUrgency = clamp(raw.urgency + deadlineRamp, 1, 5)
    const urgencyMultiplier = 0.55 + effectiveUrgency * 0.18
    const impactBps = raw.impactCoefficient * Math.sqrt(Math.max(0, participationPct) / 10) * urgencyMultiplier
    const volatilityBps = raw.volatilityPct * 100 * Math.sqrt(Math.max(delayMs, 1) / 23_400_000)
    const noiseBps = raw.randomize ? (Math.random() - 0.35) * volatilityBps : volatilityBps * 0.15
    const passiveSpread = effectiveUrgency <= 2 ? -raw.spreadBps * 0.2 : effectiveUrgency >= 4 ? raw.spreadBps / 2 : raw.spreadBps / 4
    const slippageBps = clamp(Math.max(-raw.spreadBps / 3, passiveSpread + raw.baseSlippageBps + impactBps + noiseBps), -raw.maxSlippageBps, raw.maxSlippageBps)
    const direction = side === 'buy' ? 1 : -1
    let expectedPrice = marketPrice * (1 + direction * slippageBps / 10_000)
    if (Number.isFinite(Number(raw.limitPrice)) && (raw.orderType === 'limit' || raw.orderType === 'stop-limit')) {
      expectedPrice = side === 'buy' ? Math.min(expectedPrice, Number(raw.limitPrice)) : Math.max(expectedPrice, Number(raw.limitPrice))
    }
    const darkEligible = raw.useDarkPools && sliceQty >= raw.minDarkFill && delayMs <= raw.darkExposureSeconds * 1000
    const darkDraw = raw.randomize ? Math.random() : 0.45
    const venue: ExecutionSlice['venue'] = raw.includeAuction && (raw.algo === 'open' || raw.algo === 'close') && index === quantities.length - 1
      ? 'AUCTION'
      : darkEligible && darkDraw < clamp(raw.darkFillProbabilityPct / 100 + raw.stealthPct / 500, 0, 0.95)
        ? (raw.useMidpoint ? 'MIDPOINT' : 'DARK')
        : raw.useLitVenues ? 'LIT' : 'SMART'
    const instruction: ExecutionSlice['instruction'] = venue === 'AUCTION' ? 'SEEK'
      : venue === 'MIDPOINT' ? 'MID-PEG'
      : venue === 'DARK' ? 'SEEK'
      : effectiveUrgency >= 5 ? 'SWEEP'
      : effectiveUrgency >= 4 ? 'CROSS'
      : effectiveUrgency <= 1 ? 'POST'
      : 'JOIN'
    weightedPrice += expectedPrice * sliceQty
    weightedBps += slippageBps * sliceQty
    return { quantity: sliceQty, delayMs: Math.round(delayMs), expectedPrice, slippageBps, participationPct, venue, instruction }
  })

  return {
    slices,
    expectedAveragePrice: weightedPrice / qty,
    expectedSlippageBps: weightedBps / qty,
    expectedDurationMs: Math.max(...slices.map(slice => slice.delayMs)),
  }
}
