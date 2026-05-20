import { Link } from 'react-router-dom'
import { ArrowRight } from 'lucide-react'
import heroImage from '../assets/landing-hero.png'

const ENTRY_POINTS = [
  { label: 'Technical', to: '/technical' },
  { label: 'System', to: '/system' },
  { label: 'Market', to: '/market' },
  { label: 'Portfolio', to: '/portfolio' },
  { label: 'Docs', to: '/docs' },
]

export default function LandingPage() {
  return (
    <div className="relative min-h-full overflow-hidden bg-black text-white">
      <img
        src={heroImage}
        alt=""
        aria-hidden="true"
        className="absolute inset-0 h-full w-full object-cover"
      />
      <div className="absolute inset-0 bg-[radial-gradient(circle_at_50%_34%,rgba(255,255,255,0.14),transparent_24%),linear-gradient(180deg,rgba(0,0,0,0.22)_0%,rgba(0,0,0,0.72)_54%,rgba(0,0,0,0.96)_100%),linear-gradient(90deg,rgba(0,0,0,0.82)_0%,rgba(0,0,0,0.38)_48%,rgba(0,0,0,0.70)_100%)]" />
      <div className="absolute inset-x-0 bottom-0 h-72 bg-gradient-to-t from-black via-black/82 to-transparent" />

      <section className="relative z-10 flex min-h-[calc(100vh-58px)] flex-col items-center px-6 py-8 sm:px-10 lg:px-16">
        <div className="mt-[12vh] w-full max-w-6xl text-center">
          <h1 className="mx-auto max-w-5xl text-5xl font-semibold text-white sm:text-6xl lg:text-7xl">
            Market research in full focus.
          </h1>
          <p className="mx-auto mt-6 max-w-3xl text-base leading-7 text-white/68 sm:text-md">
            Screen equities, test strategy ideas, and review portfolio context from one private research terminal.
          </p>
        </div>

        <div className="mt-auto mb-[9vh] w-full max-w-5xl">
          <p className="mb-4 text-center text-2xs font-medium uppercase tracking-[0.28em] text-accent">
            Modules
          </p>
          <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-3">
            {ENTRY_POINTS.map(item => (
              <Link
                key={item.to}
                to={item.to}
                className="group relative min-h-24 overflow-hidden rounded-[8px] border border-white/12 bg-black/42 px-8 py-5 text-left backdrop-blur-md transition hover:border-accent/70 hover:bg-white/8"
              >
                <div className="absolute inset-0 opacity-45 transition group-hover:opacity-70">
                  <div className="absolute inset-0 bg-[linear-gradient(120deg,rgba(232,155,44,0.16),transparent_34%),radial-gradient(circle_at_85%_20%,rgba(255,255,255,0.12),transparent_24%)]" />
                  <div className="absolute bottom-0 left-0 right-0 h-px bg-white/14" />
                </div>
                <div className="relative flex h-full items-center justify-between gap-8">
                  <span className="text-lg font-medium text-white sm:text-xl">{item.label}</span>
                  <span className="flex shrink-0 items-center justify-center text-white/72 transition group-hover:translate-x-1 group-hover:text-accent">
                    <ArrowRight size={22} strokeWidth={1.8} aria-hidden="true" />
                  </span>
                </div>
              </Link>
            ))}
          </div>
        </div>

        <footer className="absolute bottom-4 left-6 text-2xs text-white/42 sm:left-10 lg:left-16">
          Copyright Kenneth Law 2026
        </footer>
      </section>
    </div>
  )
}
