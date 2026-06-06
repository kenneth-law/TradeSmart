#!/usr/bin/env bash
set -e

ROOT="$(cd "$(dirname "$0")" && pwd)"
FRONTEND="$ROOT/frontend"
VENV="$ROOT/.venv"
PYTHON_BASE="/Users/kenlaw/.local/share/uv/python/cpython-3.12.11-macos-aarch64-none/bin/python"
PYTHON="$VENV/bin/python3"
BACKEND_PORT=5001
FRONTEND_PORT=5173
LAUNCHED_PIDS=()

port_in_use() {
  lsof -i ":$1" -n -P &>/dev/null
}

port_owner() {
  lsof -i ":$1" -n -P | head -5
}

port_pid() {
  lsof -nP -tiTCP:"$1" -sTCP:LISTEN | head -1
}

backend_ready() {
  curl -sf "http://localhost:$BACKEND_PORT/healthcheck" &>/dev/null
}

frontend_ready() {
  curl -sf "http://localhost:$FRONTEND_PORT" &>/dev/null
}

cleanup() {
  printf "\nStopping launched services...\n"
  if [ ${#LAUNCHED_PIDS[@]} -gt 0 ]; then
    kill "${LAUNCHED_PIDS[@]}" 2>/dev/null || true
    wait "${LAUNCHED_PIDS[@]}" 2>/dev/null || true
  fi
  echo "Done."
}

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

# ── Launch Flask ──────────────────────────────────────────────────────────────
if port_in_use "$BACKEND_PORT"; then
  if backend_ready; then
    FLASK_PID="$(port_pid "$BACKEND_PORT")"
    FLASK_STATUS="already running"
    echo "Flask already running → http://localhost:$BACKEND_PORT (pid $FLASK_PID)"
  else
    echo "ERROR: Port $BACKEND_PORT is already in use, but Flask is not healthy."
    port_owner "$BACKEND_PORT"
    exit 1
  fi
else
  echo "Starting Flask  → http://localhost:$BACKEND_PORT"
  cd "$ROOT"
  PORT=$BACKEND_PORT "$PYTHON" app.py &>/tmp/tradesmart-flask.log &
  FLASK_PID=$!
  FLASK_STATUS="started"
  LAUNCHED_PIDS+=("$FLASK_PID")
fi

# ── Wait for Flask ────────────────────────────────────────────────────────────
if [ "$FLASK_STATUS" = "started" ]; then
  echo -n "Waiting for Flask"
  for i in $(seq 1 40); do
    if backend_ready; then
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
fi

# ── Launch Vite ───────────────────────────────────────────────────────────────
if port_in_use "$FRONTEND_PORT"; then
  if frontend_ready; then
    VITE_PID="$(port_pid "$FRONTEND_PORT")"
    VITE_STATUS="already running"
    echo "Vite already running  → http://localhost:$FRONTEND_PORT (pid $VITE_PID)"
  else
    echo "ERROR: Port $FRONTEND_PORT is already in use, but Vite is not responding."
    port_owner "$FRONTEND_PORT"
    exit 1
  fi
else
  echo "Starting Vite   → http://localhost:$FRONTEND_PORT"
  cd "$FRONTEND"
  npm run dev &
  VITE_PID=$!
  VITE_STATUS="started"
  LAUNCHED_PIDS+=("$VITE_PID")
fi

# ── Summary ───────────────────────────────────────────────────────────────────
echo ""
echo "  Backend  →  http://localhost:$BACKEND_PORT   (pid $FLASK_PID, $FLASK_STATUS)"
echo "  Frontend →  http://localhost:$FRONTEND_PORT   (pid $VITE_PID, $VITE_STATUS)"
echo ""
echo "  Open:  http://localhost:$FRONTEND_PORT"
echo "  Logs:  tail -f /tmp/tradesmart-flask.log"
echo ""
if [ ${#LAUNCHED_PIDS[@]} -gt 0 ]; then
  echo "  Ctrl+C to stop services launched by this script"
else
  echo "  Both services were already running; nothing new to stop."
fi

trap cleanup INT TERM

if [ ${#LAUNCHED_PIDS[@]} -gt 0 ]; then
  wait "${LAUNCHED_PIDS[@]}"
fi
