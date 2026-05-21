#!/usr/bin/env bash
set -e

ROOT="$(cd "$(dirname "$0")" && pwd)"
FRONTEND="$ROOT/frontend"
VENV="$ROOT/.venv"
PYTHON_BASE="/Users/kenlaw/.local/share/uv/python/cpython-3.12.11-macos-aarch64-none/bin/python"
PYTHON="$VENV/bin/python3"
BACKEND_PORT=5001
FRONTEND_PORT=5173

# ── Local environment ────────────────────────────────────────────────────────
if [ -f "$ROOT/.env" ]; then
  echo "Loading local environment from .env"
  set -a
  # shellcheck disable=SC1091
  source "$ROOT/.env"
  set +a
fi

# ── Virtualenv setup ─────────────────────────────────────────────────────────
if [ ! -f "$VENV/bin/activate" ]; then
  echo "Creating project virtualenv at .venv …"
  "$PYTHON_BASE" -m venv "$VENV"
fi

if ! "$PYTHON" -c "import flask, pandas, plotly, markdown, websocket" &>/dev/null; then
  echo "Installing Python dependencies …"
  "$VENV/bin/pip" install -q \
    "flask==2.3.3" "pandas>=2.0" "numpy<2" "plotly" \
    "scikit-learn" "yfinance" "curl_cffi" "beautifulsoup4" \
    "markdown" "matplotlib" "joblib" "tqdm" "openai" "Werkzeug" \
    "websocket-client" 2>&1 | \
    grep -E "^(ERROR|Successfully installed)" || true
fi

# ── Quick import check ────────────────────────────────────────────────────────
if ! "$PYTHON" -c "import flask, pandas, plotly" &>/dev/null; then
  echo "ERROR: Flask/pandas/plotly not importable from .venv. Try:"
  echo "  .venv/bin/pip install -r requirements.txt"
  exit 1
fi

# ── Frontend deps ────────────────────────────────────────────────────────────
if [ ! -d "$FRONTEND/node_modules" ]; then
  echo "Installing frontend npm packages …"
  cd "$FRONTEND" && npm install && cd "$ROOT"
fi

# ── Port check ───────────────────────────────────────────────────────────────
if lsof -i ":$BACKEND_PORT" -n -P &>/dev/null; then
  echo "ERROR: Port $BACKEND_PORT is already in use."
  lsof -i ":$BACKEND_PORT" -n -P | head -5
  exit 1
fi

# ── Launch Flask ──────────────────────────────────────────────────────────────
echo "Starting Flask  → http://localhost:$BACKEND_PORT"
cd "$ROOT"
PORT=$BACKEND_PORT "$PYTHON" app.py &>/tmp/tradesmart-flask.log &
FLASK_PID=$!

# ── Wait for Flask ────────────────────────────────────────────────────────────
echo -n "Waiting for Flask"
for i in $(seq 1 40); do
  if curl -sf "http://localhost:$BACKEND_PORT/healthcheck" &>/dev/null; then
    echo " ready"
    break
  fi
  if ! kill -0 $FLASK_PID 2>/dev/null; then
    echo ""
    echo "ERROR: Flask crashed. Last log:"
    tail -20 /tmp/tradesmart-flask.log
    exit 1
  fi
  printf "."
  sleep 0.5
done

# ── Launch Vite ───────────────────────────────────────────────────────────────
echo "Starting Vite   → http://localhost:$FRONTEND_PORT"
cd "$FRONTEND"
npm run dev &
VITE_PID=$!

# ── Summary ───────────────────────────────────────────────────────────────────
echo ""
echo "  Backend  →  http://localhost:$BACKEND_PORT   (pid $FLASK_PID)"
echo "  Frontend →  http://localhost:$FRONTEND_PORT   (pid $VITE_PID)"
echo ""
echo "  Open:  http://localhost:$FRONTEND_PORT"
echo "  Logs:  tail -f /tmp/tradesmart-flask.log"
echo ""
echo "  Ctrl+C to stop both servers"

trap 'printf "\nStopping…\n"; kill $FLASK_PID $VITE_PID 2>/dev/null; wait 2>/dev/null; echo "Done."' INT TERM

wait
