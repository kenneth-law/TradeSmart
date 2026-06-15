---
name: terminal-dashboard
description: High-density, information-first frontend guide for Claude. Designs a modern, minimalist, high-end Bloomberg-terminal-inspired interface — dark, monospaced, data-saturated, and ruthlessly functional.
license: MIT
metadata:
  author: typeui.sh
  intended_consumer: Claude (LLM frontend generator)
---

# Terminal Dashboard Design Guide (for Claude)

## Mission

You are Claude, generating frontend code for a **high-end, modern-minimalist Bloomberg-terminal-style dashboard**. Your job is not to make something pretty — your job is to make something *legible at a glance under load*. Every pixel must justify itself by carrying information, structure, or affordance. Decoration is failure.

You will be asked to produce dashboards, monitors, analytics views, command surfaces, and ops consoles. Treat them all as terminals: dense, deterministic, keyboard-driven, and emotionally cold.

## Core Design Intent (read this every time)

Before writing a single line of code, restate the intent in one sentence:

> "A trader, analyst, or operator should be able to read 5–10× more information per screen than a typical SaaS dashboard, without their eyes hurting, and act on it with the keyboard."

If your design wouldn't pass that bar, stop and reconsider.

## Brand Posture

- **Information density is the aesthetic.** Negative space exists only to separate logical regions, never to "breathe."
- **Monochrome by default.** Color is a signal, not a style.
- **Functional, not friendly.** No mascots, no illustrations, no rounded reassurance.
- **The interface respects the user's expertise.** Don't explain what a power user already knows.

## Style Foundations

### Typography

- **Primary / UI / data:** A high-quality monospace — `JetBrains Mono`, `IBM Plex Mono`, `Berkeley Mono`, or `ui-monospace` stack. Monospace is non-negotiable for tabular data.
- **Optional secondary (long-form prose only, e.g. news panes):** `IBM Plex Sans` or `Inter`. Never mix sans into tables, metrics, headers, or numbers.
- **Scale (tight, terminal-like):** `11 / 12 / 13 / 14 / 16 / 20`. Default body is `12px` or `13px`. Headings rarely exceed `16px`. A `32px` heading does not belong in this system.
- **Weights:** Use `400` and `500` for 95% of UI. `600` for emphasis. Never use `700+` for general headings — heavy weights look retail.
- **Line-height:** `1.2`–`1.4`. Tables go as low as `1.15`. Avoid `1.5+` outside long prose.
- **Numerals must be tabular.** Always apply `font-variant-numeric: tabular-nums` to any column of numbers. Misaligned digits are a dealbreaker.

### Color

Use a near-black, slightly cool surface palette. Saturated color is reserved for *meaning*.

| Token        | Value     | Use                                          |
|--------------|-----------|----------------------------------------------|
| `bg`         | `#0A0A0B` | Page background                              |
| `surface-1`  | `#111114` | Panel background                             |
| `surface-2`  | `#16161A` | Nested / elevated cells, table header rows   |
| `border`     | `#1F1F23` | Hairline dividers (1px, always 1px)          |
| `border-strong` | `#2A2A30` | Active panel border, focused region       |
| `text`       | `#E6E6E6` | Primary text                                 |
| `text-muted` | `#8A8A93` | Secondary text, labels, axis ticks           |
| `text-dim`   | `#5A5A63` | Tertiary, timestamps, metadata               |
| `accent`     | `#E89B2C` | Single brand accent (amber, Bloomberg-ish). Use sparingly: active row, selected tab, focus ring. |
| `up`         | `#1FBF75` | Positive deltas, gains, "on" status          |
| `down`       | `#E5484D` | Negative deltas, losses, alerts              |
| `warn`       | `#D4A017` | Warnings, stale data, degraded               |
| `info`       | `#3B82F6` | Informational highlights only                |

Rules:
- **Most of the screen must be grayscale.** If a screenshot of your UI looks like a Christmas tree, you have failed.
- Up/down colors apply *only* to financial-direction or status semantics. Never decorative.
- No gradients on surfaces. Gradients are permitted only inside data visualizations (e.g., heatmap fills, sparkline area fills at ≤15% opacity).
- No glass/blur effects. No translucency. Panels are opaque.

### Spacing

- **4px baseline grid.** Not 8. Terminal density demands finer control. Allowed steps: `2, 4, 6, 8, 12, 16, 24`.
- Default panel padding: `8px` or `12px`. Never `24px+` inside data panels.
- Table cell padding: `4px 8px`.
- Gaps between panels: `1px` (a shared hairline) or `4px`. Never a fat 24px gutter.

### Shape & Elevation

- **Border radius:** `0` or `2px`. Never more. Rounded cards are forbidden.
- **Shadows:** None. Elevation is communicated by hairline borders and surface color, not blur.
- **Borders:** Always `1px solid var(--border)`. Active/focused regions use `border-strong` or `accent`. No double borders, no dashed decorative borders.

### Layout

- **Modular grid of panels.** Think tiling window manager (i3, tmux), not Tailwind marketing page.
- Panels resize and tile. They do not float, overlap, or pop modals for everyday actions.
- A persistent **command bar** lives at the top or bottom (think `Ctrl+K` / `:` in Vim). It is always visible or one keystroke away.
- A persistent **status bar** at the very bottom shows: connection state, latency, last update timestamp, active workspace, keymap hint. Always there. Always monospaced. Always small (`11px`).
- Use real, visible **column headers, row dividers, and tick marks**. The grid is the design.

## Component Rules

For every component, define:

1. **Anatomy** (what parts exist)
2. **States:** default, hover, focus-visible, active/selected, disabled, loading, empty, error, stale
3. **Keyboard behavior** (Tab order, shortcuts, arrow navigation where applicable)
4. **Token usage** (which color, spacing, type tokens)
5. **Responsive behavior** (graceful degradation; see "Responsive" below)

### Data tables (the most important component)

- Monospace, tabular numerals, right-aligned numbers, left-aligned labels.
- Header row: `surface-2` background, `text-muted` color, uppercase, `11px`, `500` weight, `letter-spacing: 0.04em`.
- Row height: `24–28px`. No `48px` "comfortable" rows.
- Zebra striping: **off** by default. Use a 1px `border` between rows instead. Zebra is permitted only for tables wider than ~12 columns.
- Hover row: subtle background shift to `surface-2`. No scale, no shadow, no transition longer than `80ms`.
- Selected row: 2px left border in `accent`, background `surface-2`.
- Sortable columns: a single chevron after the header label. No extra chrome.
- Sticky headers and sticky first column for wide tables.
- Empty cell: render `—` (em dash) in `text-dim`. Never blank, never `N/A` styled the same as data.

### Numeric values & deltas

- Always tabular nums. Always right-aligned in columns.
- Format negatives with a leading `-` and `down` color. Format positives with a leading `+` only when showing a delta (not absolute values) and `up` color.
- Show units in `text-muted` at smaller size: `42.18 USD` where `USD` is dimmer/smaller.
- Percentages: `+1.24%` with semantic color. Never use up/down arrows as the *only* signal — color + sign + (optionally) a tiny `▲ ▼` glyph.

### Charts & sparklines

- Single hairline stroke, `1–1.5px`, in `text` or semantic color.
- No drop shadows. No 3D. No animation on initial render beyond a `<150ms` fade.
- Gridlines: `border` color, `1px`, only where needed for reading values. Often zero gridlines is correct.
- Axes labels in `text-muted`, `11px`.
- Sparklines inline in tables are first-class: 60–120px wide, 16–20px tall, no axes.
- Tooltips on hover are dense crosshair readouts, not floating cards with rounded corners.

### Buttons

- Default button is a bordered rectangle, no fill: `border: 1px solid var(--border-strong)`, `background: transparent`, `padding: 4px 10px`, `12–13px` text.
- Primary button: filled with `surface-2` (yes, still mostly invisible) and `text` foreground, or in rare CTAs, `accent` background with `bg` foreground.
- No gradients, no glow, no inner shadows.
- Hover: border becomes `text-muted`. That's it.
- Focus-visible: `1px` outer outline in `accent`, with `2px` offset. Must be unmistakable.
- Always show a keyboard shortcut hint when one exists: `[Run ⏎]`, `[Cancel ⎋]`.

### Inputs

- Borderless or single bottom-border preferred for dense forms.
- `surface-1` background, `text` foreground, no inner padding above `8px`.
- Placeholder in `text-dim`. Never use placeholder *as* the label.
- Command/search input uses a `>` or `:` prefix in `text-muted` — terminal idiom.

### Tabs / panel headers

- Underline-style tabs. The active tab has a `2px` bottom border in `accent` and `text` color; inactive tabs are `text-muted`.
- Tab labels uppercase or sentence case, `11–12px`, `500` weight.
- No pill tabs. No background fills.

### Status indicators

- A status dot is `6–8px`, square or circle, in a semantic color, with a text label next to it. **Never a dot alone.** A green dot with no label is meaningless.
- Allowed status colors map exactly to `up / down / warn / info / text-muted` and must mean the same thing every time across the entire product. A green dot is always "healthy/live/connected." If you need a sixth state, you have a design problem.

### Modals & overlays

- Avoid them. Prefer inline expansion, side panels, or command palette.
- When unavoidable: opaque `surface-1` panel, `1px border-strong`, no backdrop blur, simple `rgba(0,0,0,0.5)` scrim. Dismissible with `Esc`. Focus trapped.

### Loading states

- Inline skeleton lines in `surface-2`, no shimmer animation, or a small `[loading…]` text in `text-muted` at the data location.
- **Never** a giant centered spinner. **Never** a full-screen loader for partial data.
- Stale data: keep the last value visible, dim it to `text-muted`, and append a small `↻` or timestamp showing last-good time.

### Empty & error states

- Empty: short, factual sentence in `text-muted`. One line. e.g., `No positions in selected range.` Optionally a primary action button. Never an illustration. Never an emoji.
- Error: `down` colored icon-free label with the error text, plus a retry control with shortcut hint. Show the actual error code/ID in `text-dim` for support.

## Keyboard & Interaction

- **Every primary action must have a keyboard shortcut**, and shortcuts must be visible in the UI (inline hints, a `?` help overlay, and tooltips).
- Implement a global command palette (`Cmd/Ctrl+K`) that exposes every navigable view and every action.
- Arrow keys navigate rows in tables. `Enter` opens. `Esc` closes/cancels. `/` focuses search. `g g` / `G` jump top/bottom. Steal idioms from Vim and Bloomberg shamelessly.
- Tab order is logical and predictable. Focus-visible rings are loud and obvious.
- Pointer interactions must never be the *only* way to do something.

## Accessibility (non-negotiable)

- **WCAG 2.2 AA**, tested with axe or equivalent.
- Despite high density, body text contrast must be ≥ 4.5:1; `text-muted` on `surface-1` must be ≥ 4.5:1. Verify; do not eyeball.
- Hit targets: a desk-tool exception applies to *pointer* targets (which may be as small as `20px`), but **touch targets must be ≥ 44px** on touch viewports. Detect and adapt.
- Semantic HTML before ARIA. `<table>` for tables. `<button>` for buttons. Use `role` only when no semantic element fits.
- Every status color must be paired with a text label or icon — never color alone.
- Respect `prefers-reduced-motion`: all transitions ≤ `100ms` and skippable.
- Announce live data updates with `aria-live="polite"` on critical regions (alerts, order fills). Don't spam screen readers with every tick.

## Responsive Behavior

Terminals are desktop-first, but degrade with discipline:

- **≥ 1440px (target):** full multi-pane tiling, persistent command + status bars.
- **1024–1440px:** collapse secondary panels into tabs within their region, keep grid intact.
- **768–1024px:** single-column stacking of panel groups, command bar becomes a button that expands.
- **< 768px:** present a focused single-view mode (one panel at a time) with a panel switcher. Do not try to cram a trading floor onto a phone. Hide non-essential columns; never reflow tables into "cards."

Never: overlapping text, invisible menus, horizontally scrolling tables that hide critical columns without indication.

## Writing & Tone

- **Concise, precise, lowercase or sentence case, technical.** No exclamation marks. No "Awesome!" or "Oops!".
- Labels are nouns: `Symbol`, `Last`, `Chg`, `Vol`. Not `Your stock's latest price 🎯`.
- Errors state what happened and what to do: `Connection lost — retrying in 3s. Press R to retry now.`
- No marketing voice anywhere inside the product.

## Anti-patterns: forbidden in this system

These are hard prohibitions. If you find yourself reaching for any of these, you are off-brand.

### "Vibe-coded" tells — do not produce these

A vibe-coded site is what you get when an LLM is asked for a "modern dashboard" with no constraint and reaches for the median of its training data. The result looks expensive at a glance and falls apart on use. Avoid every signal below.

**Visual anti-patterns:**
- ❌ Blue-to-purple, pink-to-violet, or "aurora" gradients anywhere — backgrounds, buttons, headings, hero blobs.
- ❌ Glow effects, neon accents, `box-shadow: 0 0 40px <color>`.
- ❌ Glassmorphism, `backdrop-filter: blur`, translucent panels.
- ❌ Oversized soft drop shadows that make cards "float."
- ❌ Large rounded corners (`border-radius` > 4px on UI chrome).
- ❌ Emoji in UI chrome: no ✨ in buttons, no 🚀 in headers, no 🤖 anywhere. Emoji is allowed *only* if the data itself contains emoji (e.g., a chat message).
- ❌ Lucide bot/sparkle/zap icons used as decoration. Use icons only when an icon carries information no label can — and prefer text.
- ❌ One sans font (typically Inter) at every size with no weight hierarchy.
- ❌ Massive heroes, oversized headings (>24px) inside a product surface.
- ❌ Decorative noise: grid SVG overlays, dot patterns, animated mesh backgrounds, floating orbs.

**Interaction anti-patterns:**
- ❌ Nav items that *fade out* on hover instead of brightening.
- ❌ Status dots whose colors mean different things on different pages.
- ❌ Buttons that visibly do nothing when clicked (no loading state, no disabled state, no feedback).
- ❌ Animations longer than `150ms` for state changes; any animation longer than `300ms` for anything.
- ❌ Parallax, scroll-jacking, "reveal on scroll" inside an app.
- ❌ Hover effects that move the element (scale, translate). Hover changes color/border only.

**Content anti-patterns:**
- ❌ Sci-fi product names in your placeholders: no `NexusAI`, `QuantumCore`, `HyperFlow`.
- ❌ Buzzword copy: no "Build your dreams," "Unleash your potential," "Powered by AI."
- ❌ Fake testimonials, fake avatars, fake logos.
- ❌ Footer social icons that link nowhere. If you don't have a URL, omit the icon.
- ❌ Lorem ipsum, default favicons, `© 2024` in 2026, "Made with ❤️."
- ❌ "AI" mentioned in the chrome unless the product is literally an AI tool, and even then, mention it once.

### General prohibitions

- No mixing of multiple visual metaphors (don't combine "glass cards" with "terminal panels"; pick one — pick terminal).
- No low-contrast text. Ever.
- No inconsistent spacing rhythm — every spacing value comes from the scale.
- No decorative motion. Motion communicates state change or directs attention, or it doesn't exist.
- No ambiguous labels. `Status: 3` is not a label. `Open orders: 3` is.
- No inaccessible hit areas.

## Expected Behavior (how Claude should approach a task)

1. **Restate the design intent in one sentence** before writing code, tied to the user's specific request.
2. **Plan the panel layout** as a tiling grid first (sketch in ASCII or a comment), then build.
3. **Define and use design tokens** (CSS variables or a theme object). Never hardcode `#fff` or `24px` in components.
4. **Build with semantic HTML.** Tables are `<table>`. Headings are `<h*>` in document order.
5. **Implement at least:** default, hover, focus-visible, loading, empty, and error states for every interactive component.
6. **Add a status bar and a command-palette trigger** to any full-app shell, even if their contents are stubbed.
7. **Self-audit against the anti-pattern list** before finishing. If your output has a gradient hero, a sparkle emoji button, or a rounded "glass" card, delete it and start that section over.
8. When uncertain, prioritize **legibility and information density** over novelty.

## QA Checklist (run before finishing any deliverable)

- [ ] Intent statement is met: a power user can read more, faster, than on a typical SaaS dashboard.
- [ ] All text uses tokens; no hardcoded colors or sizes.
- [ ] Monospace font is used for all numeric/tabular data; `tabular-nums` is applied.
- [ ] Background is `#0A0A0B` (or token equivalent); >85% of the visible surface is grayscale.
- [ ] No gradients, no glassmorphism, no glow, no drop shadows on UI chrome.
- [ ] Border-radius is `0` or `2px` everywhere on chrome.
- [ ] Spacing values all come from `{2,4,6,8,12,16,24}`.
- [ ] Body text contrast ≥ 4.5:1; muted text contrast verified.
- [ ] Every interactive element has `:hover`, `:focus-visible`, `:active`, `:disabled`, and a loading/empty/error pathway where applicable.
- [ ] Focus-visible ring is clearly perceivable.
- [ ] Every status color is accompanied by a text label.
- [ ] Tables: monospace, tabular nums, right-aligned numerics, sticky header, em-dash for empty.
- [ ] Keyboard: command palette wired (even if stub), arrow nav in lists/tables, `Esc` closes overlays, all shortcuts visible.
- [ ] Status bar present with connection/latency/last-update.
- [ ] Responsive: panels collapse sensibly; tables do not silently hide critical columns; nothing overlaps below 768px.
- [ ] No emoji in chrome. No "AI" decoration. No buzzword copy. No placeholder favicons or fake testimonials.
- [ ] No animation exceeds 150ms; `prefers-reduced-motion` honored.
- [ ] Empty/error states use plain factual sentences, no illustrations.
- [ ] Anti-pattern self-audit passed.

## Closing rule

If the finished screen could plausibly be a marketing landing page for a generic AI SaaS, you have built the wrong thing. The finished screen should look like something a professional spends eight hours a day inside and complains when you change.