# Plan: New Feature — <Feature Name>

## Goal

One sentence: what this feature does and why it matters.

## Affected Files

| File | Change |
|---|---|
| `modules/xxx.py` | Add new logic |
| `app.py` | Add route(s) |
| `templates/xxx.html` | Add UI |

## Steps

- [ ] 1. Implement core logic in `modules/`
- [ ] 2. Add Flask route in `app.py`
- [ ] 3. Wire SSE progress if long-running (Queue + Thread + `/xxx_progress` endpoint)
- [ ] 4. Add Jinja2 template in `templates/`
- [ ] 5. Test with `python app.py` and curl/browser
- [ ] 6. Update `CLAUDE.md` if architecture changes

## SSE Checklist (if applicable)

- [ ] Global queue dict created: `xxx_queues = {}`
- [ ] Queue created per request with unique ID
- [ ] Background thread set as `daemon=True`
- [ ] Thread target sends `None` sentinel when done
- [ ] `generate()` yields `f"data: {json.dumps(msg)}\n\n"`
- [ ] `Response(..., mimetype="text/event-stream")` returned
- [ ] Client disconnect cleans up queue dict entry

## Open Questions

- 
