import { Link } from 'react-router-dom'
import { ArrowRight } from 'lucide-react'

const ENTRY_POINTS = [
  { label: 'Markets', to: '/market' },
  { label: 'Quant Trading', to: '/system' },
  { label: 'Technical Ranking', to: '/technical' },
  { label: 'Daily Lineup', to: '/daily-lineup' },
  { label: 'Portfolio', to: '/portfolio' },
  { label: 'Docs', to: '/docs' },
]

export default function LandingPage() {
  return (
    <div className="relative h-full">
      <div className="absolute inset-x-0 bottom-0 h-72 bg-gradient-to-t from-black via-black/82 to-transparent" />

      <section className="relative z-10 flex h-full min-h-0 flex-col items-center px-6 py-6 sm:px-10 lg:px-16">
        <div className="mt-[7vh] w-full max-w-6xl text-center">
          <h1 className="hero-title mx-auto max-w-5xl text-4xl font-semibold sm:text-5xl lg:max-w-none lg:whitespace-nowrap lg:text-[3.65rem] xl:text-[3.95rem]">
            Market research in full focus.
          </h1>
          <p className="hero-copy mx-auto mt-6 max-w-3xl text-base leading-7 sm:text-md">
            Screen equities, test strategy ideas, and review portfolio context from one private research terminal.
          </p>
        </div>

        <div className="mt-[13vh] w-full max-w-5xl">
          <p className="mb-4 text-center text-2xs font-medium uppercase tracking-[0.28em] text-accent">
            Modules
          </p>
          <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-3">
            {ENTRY_POINTS.map(item => {
              const highlighted = item.to === '/market'

              return (
                <Link
                  key={item.to}
                  to={item.to}
                  className={[
                    'module-card group relative min-h-24 overflow-hidden rounded-[8px] border px-8 py-5 text-left backdrop-blur-md transition hover:border-accent/60',
                    highlighted
                      ? 'module-card-highlight border-accent/70'
                      : '',
                  ].join(' ')}
                >
                  <div className={['absolute inset-0 opacity-45 transition group-hover:opacity-70', highlighted ? 'opacity-75' : ''].join(' ')}>
                    <div className="module-card-sheen absolute inset-0" />
                  </div>
                  <div className="relative flex h-full items-center justify-between gap-8">
                    <span className="hero-title text-[1.22rem] font-medium sm:text-[1.45rem]">
                      {item.label}
                    </span>
                    <span className={['flex shrink-0 items-center justify-center transition group-hover:translate-x-1 group-hover:text-accent', highlighted ? 'text-accent' : 'module-arrow'].join(' ')}>
                      <ArrowRight size={22} strokeWidth={1.8} aria-hidden="true" />
                    </span>
                  </div>
                </Link>
              )
            })}
          </div>
        </div>

        <footer className="absolute bottom-4 left-1/2 -translate-x-1/2 text-2xs text-white/42 sm:left-10 lg:left-16">
          © Kenneth Law 2026
        </footer>
      </section>
    </div>
  )
}
