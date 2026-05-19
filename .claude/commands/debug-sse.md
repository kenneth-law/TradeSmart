# /debug-sse

Debug a Server-Sent Events (SSE) endpoint in TradeSmart.

## Usage

```
/debug-sse analysis_progress
```

Argument: the SSE route name (e.g., `analysis_progress`, `backtest_progress`, `integrated_progress`).

## What this does

1. Reads the relevant route handler in `app.py`
2. Checks that the `Queue` is created and stored in the right global dict
3. Checks the background thread target function for correct `None` sentinel usage
4. Checks that `generate()` yields `f"data: {json.dumps(message)}\n\n"` format
5. Checks that the `Response` uses `mimetype="text/event-stream"`

## Common SSE bugs in this codebase

- Thread not set as `daemon=True` → server hangs on shutdown
- Missing `\n\n` at end of SSE frame → browser never fires the event
- `queue.get()` without timeout → stream hangs if thread crashes silently
- Forgetting to `del analysis_queues[analysis_id]` on client disconnect

## Quick curl test

```bash
curl -N "http://localhost:5000/$ARGUMENTS?tickers=AAPL"
```

You should see `data: {...}` lines streaming in.
