# /add-module

Scaffold a new module in the `modules/` directory following the project's conventions.

## Usage

```
/add-module risk_management
```

## What this does

Creates `modules/<name>.py` with standard boilerplate: imports, a module-level logger via `utils.log_message`, and placeholder public functions. Also reminds you to import the module in `app.py` and `main.py`.

## Checklist for a new module

1. Create `modules/$ARGUMENTS.py` with:
   - Docstring describing the module's purpose
   - Import `from modules.utils import log_message`
   - At least one public function with a clear signature
2. Import and wire it into `app.py` (add route if needed) and/or `main.py`
3. Add any new pip packages to `requirements.txt`
4. Update `CLAUDE.md` architecture table

## Template

```python
"""
$ARGUMENTS module — <one-line description>.
"""

from modules.utils import log_message


def example_function(ticker: str) -> dict:
    log_message(f"Processing {ticker}")
    # TODO: implement
    return {}
```
