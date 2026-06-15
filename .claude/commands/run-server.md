# /run-server

Start the TradeSmart Flask development server.

## Usage

```
/run-server
```

## What this does

Starts `app.py` on `http://localhost:5000` in debug mode. The server auto-reloads on file changes.

## Steps

```bash
python app.py
```

Or, if you need to specify a port:

```bash
flask --app app run --debug --port 5000
```

Check that `yfinance_cookie_patch.py` is applied (it is imported automatically via `data_retrieval.py`).
The `yf_session` is created once at startup — restart the server to refresh it.
