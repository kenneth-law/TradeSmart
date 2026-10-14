interface ProdiaImageConfig {
  prompt: string
  width: number
  height: number
  steps?: number
  seed?: number
}

export async function generateProdiaImage(
  apiKey: string,
  config: ProdiaImageConfig,
  accept = 'image/jpeg',
  signal?: AbortSignal,
): Promise<Blob> {
  const response = await fetch('/api/vibeos/images', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      api_key: apiKey.trim().replace(/^Bearer\s+/i, ''),
      accept,
      job: { type: 'inference.flux-fast.schnell.txt2img.v2', config },
    }),
    signal,
  })
  if (!response.ok) {
    const payload = await response.json().catch(() => null) as { error?: string } | null
    const message = typeof payload?.error === 'string'
      ? payload.error
      : `Image generation failed (HTTP ${response.status}).`
    throw new Error(message)
  }
  const blob = await response.blob()
  if (!blob.type.startsWith('image/')) throw new Error('Image generation returned an invalid image.')
  return blob
}
