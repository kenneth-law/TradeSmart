import { useEffect, useRef, useState } from 'react'
import type { SSEMessage } from '../types'

type SSEStatus = 'idle' | 'connecting' | 'streaming' | 'complete' | 'error'

export function useSSE(url: string | null) {
  const [messages, setMessages] = useState<SSEMessage[]>([])
  const [progress, setProgress] = useState(0)
  const [status, setStatus] = useState<SSEStatus>('idle')
  const [lastMessage, setLastMessage] = useState<SSEMessage | null>(null)
  const esRef = useRef<EventSource | null>(null)

  useEffect(() => {
    if (!url) {
      setStatus('idle')
      setMessages([])
      setProgress(0)
      return
    }

    setStatus('connecting')
    setMessages([])
    setProgress(0)

    const es = new EventSource(url)
    esRef.current = es

    es.onopen = () => setStatus('streaming')

    es.onmessage = (e) => {
      try {
        const data: SSEMessage = JSON.parse(e.data)
        setLastMessage(data)

        if (data.progress !== undefined) setProgress(data.progress)

        if (data.message || data.error) {
          setMessages((prev) => [...prev, data])
        }

        if (data.status === 'complete') {
          setStatus('complete')
          setProgress(100)
          es.close()
        }

        if (data.error && !data.status) {
          setStatus('error')
          es.close()
        }
      } catch {
        // malformed frame, skip
      }
    }

    es.onerror = () => {
      setStatus('error')
      es.close()
    }

    return () => {
      es.close()
      esRef.current = null
    }
  }, [url])

  const reset = () => {
    esRef.current?.close()
    setMessages([])
    setProgress(0)
    setStatus('idle')
    setLastMessage(null)
  }

  return { messages, progress, status, lastMessage, reset }
}
