"""Same-origin bridge for VibeOS image generation using the user's Prodia key."""

import re

import requests
from flask import Blueprint, Response, jsonify, request


prodia_images = Blueprint('prodia_images', __name__)
PRODIA_JOB_URL = 'https://inference.prodia.com/v2/job'
# Keep model selection on the server so a provider migration does not depend
# on users refreshing cached frontend bundles.
# Schema: https://docs.prodia.com/job-types/inference-flux-2-klein-4b-txt2img-v1/
PRODIA_JOB_TYPE = 'inference.flux-2.klein.4b.txt2img.v1'
LEGACY_JOB_TYPE = 'inference.flux-fast.schnell.txt2img.v2'


@prodia_images.post('/api/vibeos/images')
def generate_image():
    payload = request.get_json(silent=True)
    if not isinstance(payload, dict):
        return jsonify(error='Image request must be a JSON object.'), 400

    # The browser sends the saved key in the same-origin body. Construct the
    # upstream header here instead of relying on cross-origin browser auth.
    key = payload.get('api_key')
    key = re.sub(r'^Bearer\s+', '', key.strip(), flags=re.IGNORECASE) if isinstance(key, str) else ''
    if not key or key.lower() == 'bearer':
        return jsonify(error='Save a Prodia v2 API key in Settings before generating images.'), 401
    if re.search(r'\s', key):
        return jsonify(error='The Prodia API key contains whitespace. Copy the v2 token again from the Prodia dashboard.'), 400

    job = payload.get('job')
    config = job.get('config') if isinstance(job, dict) else None
    if not isinstance(config, dict) or job.get('type') not in (None, PRODIA_JOB_TYPE, LEGACY_JOB_TYPE):
        return jsonify(error='Unsupported Prodia image job.'), 400
    if set(config) - {'prompt', 'width', 'height', 'steps', 'seed'}:
        return jsonify(error='Unsupported image configuration.'), 400
    prompt = config.get('prompt')
    if not isinstance(prompt, str) or not 1 <= len(prompt) <= 32768:
        return jsonify(error='Image prompt must contain 1–32768 characters.'), 400
    for field in ('width', 'height', 'steps', 'seed'):
        value = config.get(field)
        if field in config and (isinstance(value, bool) or not isinstance(value, int)):
            return jsonify(error=f'{field} must be an integer.'), 400
    for field in ('width', 'height'):
        if field in config and not 512 <= config[field] <= 2048:
            return jsonify(error=f'{field} must be between 512 and 2048.'), 400
    if 'steps' in config and not 1 <= config['steps'] <= 4:
        return jsonify(error='steps must be between 1 and 4.'), 400
    if 'seed' in config and not 0 <= config['seed'] <= 2**64 - 1:
        return jsonify(error='seed must be an unsigned 64-bit integer.'), 400
    accept = payload.get('accept', 'image/jpeg')
    if accept not in ('image/jpeg', 'image/jpeg;quality=82;progressive=1',
                      'image/jpeg;quality=84;progressive=1', 'image/jpeg;quality=90'):
        return jsonify(error='Unsupported image output format.'), 400

    try:
        upstream = requests.post(
            PRODIA_JOB_URL,
            headers={'Authorization': f'Bearer {key}', 'Accept': accept},
            json={'type': PRODIA_JOB_TYPE, 'config': config},
            timeout=(10, 120),
            allow_redirects=False,
        )
    except requests.Timeout:
        return jsonify(error='Prodia image generation timed out. Please try again.'), 504
    except requests.RequestException:
        return jsonify(error='Could not reach Prodia. Please try again.'), 502

    content_type = upstream.headers.get('Content-Type', '')
    if upstream.status_code == 200 and content_type.lower().startswith('image/'):
        return Response(upstream.content, content_type=content_type,
                        headers={'Cache-Control': 'no-store'})

    if 300 <= upstream.status_code < 400:
        return jsonify(error='Prodia returned an unexpected redirect.'), 502
    if upstream.status_code == 200:
        return jsonify(error='Prodia returned an unexpected response instead of an image.'), 502
    try:
        detail = upstream.json().get('error')
    except (ValueError, AttributeError):
        detail = None
    if not isinstance(detail, str):
        detail = (upstream.text.strip()[:500]
                  if content_type.lower().startswith('text/plain') else '')
        detail = detail or f'Prodia image generation failed (HTTP {upstream.status_code}).'
    # Never reflect credentials if an upstream error happens to contain them.
    detail = detail.replace(key, '[redacted]')[:500]
    response = jsonify(error=detail)
    response.status_code = upstream.status_code
    response.headers['Cache-Control'] = 'no-store'
    if upstream.headers.get('Retry-After'):
        response.headers['Retry-After'] = upstream.headers['Retry-After']
    return response
