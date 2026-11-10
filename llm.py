"""Language-model providers for the lab guide: Anthropic, OpenAI, Google Gemini, OpenRouter, Ollama, or any
OpenAI-compatible server (LM Studio, vLLM…).

Two ways to configure:
- Lab settings (the owner): chosen in the guide's Settings tab, saved to ~/.config/malecns-lab/llm.json
  (mode 600, outside the repo). Editable only from the Mac itself (http://127.0.0.1) or with MALECNS_ADMIN_TOKEN,
  because the lab is public through the tunnel. Keys are never sent back to a browser, only their last 4 characters.
- A visitor's own key: kept in that visitor's browser and sent with each question; used for that request, never
  stored on the server. Only hosted providers are allowed this way (no local or custom URLs from the internet).

Environment variables still work and fill any gap: ANTHROPIC_API_KEY, OPENAI_API_KEY, GEMINI_API_KEY,
OPENROUTER_API_KEY, and the older ~/.config/malecns-lab/anthropic-key file.
"""
import json
import os
import urllib.error
import urllib.request
from pathlib import Path
from urllib.parse import quote, urlparse

CONFIG_DIR = Path(os.environ.get('MALECNS_CONFIG_DIR', Path.home() / '.config' / 'malecns-lab'))
SETTINGS = CONFIG_DIR / 'llm.json'
OLD_KEY = CONFIG_DIR / 'anthropic-key'

PROVIDERS = {
    'anthropic': dict(label='Anthropic (Claude)', kind='anthropic', base='https://api.anthropic.com', env='ANTHROPIC_API_KEY',
                      model='claude-sonnet-5-5', key=True, visitor=True, keys_url='https://console.anthropic.com/settings/keys'),
    'openai': dict(label='OpenAI', kind='openai', base='https://api.openai.com/v1', env='OPENAI_API_KEY',
                   model='', key=True, visitor=True, keys_url='https://platform.openai.com/api-keys'),
    'gemini': dict(label='Google Gemini', kind='gemini', base='https://generativelanguage.googleapis.com/v1beta', env='GEMINI_API_KEY',
                   model='', key=True, visitor=True, keys_url='https://aistudio.google.com/apikey'),
    'openrouter': dict(label='OpenRouter (many models)', kind='openai', base='https://openrouter.ai/api/v1', env='OPENROUTER_API_KEY',
                       model='', key=True, visitor=True, keys_url='https://openrouter.ai/keys'),
    'ollama': dict(label='Ollama (local, free)', kind='openai', base='http://127.0.0.1:11434/v1', env=None,
                   model='llama3.1', key=False, visitor=False, keys_url=''),
    'custom': dict(label='OpenAI-compatible server', kind='openai', base='', env=None,
                   model='', key=False, visitor=False, keys_url=''),
}
DEFAULT = dict(provider='', models={}, keys={}, base_urls={})


class LLMError(Exception):
    pass


# --- settings -----------------------------------------------------------------------------------
def load():
    s = json.loads(json.dumps(DEFAULT))
    try:
        if SETTINGS.exists():
            raw = json.loads(SETTINGS.read_text())
            if isinstance(raw, dict):
                for k in DEFAULT:
                    if isinstance(raw.get(k), type(DEFAULT[k])): s[k] = raw[k]
    except (OSError, ValueError):
        pass
    return s


def save(s):
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    tmp = SETTINGS.with_suffix('.tmp')
    fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, 'w') as f: json.dump(s, f, indent=1)
    os.replace(tmp, SETTINGS)
    try: os.chmod(SETTINGS, 0o600)
    except OSError: pass


def stored_key(s, provider):
    k = (s['keys'].get(provider) or '').strip()
    if k: return k
    env = PROVIDERS[provider]['env']
    if env and os.environ.get(env, '').strip(): return os.environ[env].strip()
    if provider == 'anthropic' and OLD_KEY.exists():
        try: return OLD_KEY.read_text().strip()
        except OSError: return ''
    return ''


def config_for(s, provider):
    p = PROVIDERS[provider]
    return dict(provider=provider, kind=p['kind'], label=p['label'],
                base=(s['base_urls'].get(provider) or p['base']).rstrip('/'),
                model=(s['models'].get(provider) or p['model']).strip(),
                key=stored_key(s, provider) if p['key'] else '')


def usable(cfg):
    return bool(cfg and cfg['base'] and cfg['model'] and (cfg['key'] or not PROVIDERS[cfg['provider']]['key']))


def active():
    """The lab's configured model, or None. With nothing chosen, an Anthropic key alone (env or old file) still works."""
    s = load()
    provider = s['provider']
    if not provider and stored_key(s, 'anthropic'): provider = 'anthropic'
    if provider not in PROVIDERS: return None
    cfg = config_for(s, provider)
    return cfg if usable(cfg) else None


def visitor(override):
    """A visitor's own provider, model and key for one request. Hosted providers only."""
    if not isinstance(override, dict) or not override.get('key'): return None
    provider = str(override.get('provider', ''))
    if provider not in PROVIDERS or not PROVIDERS[provider]['visitor']: raise LLMError('That provider cannot be used with a browser key')
    p = PROVIDERS[provider]
    cfg = dict(provider=provider, kind=p['kind'], label=p['label'], base=p['base'],
               model=str(override.get('model') or p['model']).strip()[:120], key=str(override['key']).strip()[:400])
    if not cfg['model']: raise LLMError(f"Choose a {p['label']} model")
    return cfg


def mask(k):
    return ('…' + k[-4:]) if k and len(k) > 8 else ('set' if k else '')


def view(local):
    """Settings for the browser. Keys are only ever masked; non-local viewers see just what is active."""
    s, cfg = load(), active()
    out = dict(active=dict(provider=cfg['provider'], label=cfg['label'], model=cfg['model']) if cfg else None, editable=local,
               providers=[dict(id=k, label=v['label'], needs_key=v['key'], visitor=v['visitor'], default_model=v['model'],
                               keys_url=v['keys_url'], base=v['base']) for k, v in PROVIDERS.items()])
    if local:
        out['settings'] = dict(provider=s['provider'], models=s['models'], base_urls=s['base_urls'],
                               keys={k: mask(stored_key(s, k)) for k, v in PROVIDERS.items() if v['key']},
                               key_source={k: ('lab settings' if s['keys'].get(k) else 'environment' if stored_key(s, k) else '')
                                           for k, v in PROVIDERS.items() if v['key']})
    return out


def check_base(url):
    u = urlparse(url)
    if u.scheme not in ('http', 'https') or not u.netloc: raise ValueError('Base URL must start with http:// or https://')
    return url.rstrip('/')


def update(changes):
    """Apply a change from the Settings tab (only called for local or admin requests)."""
    if not isinstance(changes, dict): raise ValueError('Expected settings')
    s = load()
    if 'provider' in changes:
        p = str(changes['provider'] or '')
        if p and p not in PROVIDERS: raise ValueError('Unknown provider')
        s['provider'] = p
    for k, v in (changes.get('models') or {}).items():
        if k in PROVIDERS: s['models'][k] = str(v or '').strip()[:120]
    for k, v in (changes.get('base_urls') or {}).items():
        if k in PROVIDERS: s['base_urls'][k] = check_base(str(v).strip()) if str(v or '').strip() else ''
    for k, v in (changes.get('keys') or {}).items():
        if k in PROVIDERS and PROVIDERS[k]['key']:
            v = str(v or '').strip()
            if v: s['keys'][k] = v[:400]
            else: s['keys'].pop(k, None)          # empty clears the stored key
    save(s)
    return s


# --- calling ------------------------------------------------------------------------------------
def _post(url, body, headers, timeout):
    req = urllib.request.Request(url, data=json.dumps(body).encode(), method='POST', headers={'content-type': 'application/json', **headers})
    return _open(req, timeout)


def _open(req, timeout):
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r: return json.loads(r.read())
    except urllib.error.HTTPError as e:
        try: detail = json.loads(e.read()); detail = (detail.get('error') or {}).get('message') if isinstance(detail.get('error'), dict) else detail.get('error') or detail
        except Exception: detail = e.reason
        raise LLMError(f'{e.code}: {str(detail)[:200]}')
    except (urllib.error.URLError, TimeoutError, OSError) as e:
        raise LLMError(f'unreachable ({getattr(e, "reason", e)})')
    except ValueError:
        raise LLMError('unreadable reply')


def chat(cfg, system, messages, max_tokens=900, timeout=60):
    """messages: [{role: user|assistant, content: str}], alternating and starting with user."""
    kind = cfg['kind']
    if kind == 'anthropic':
        out = _post(cfg['base'] + '/v1/messages', dict(model=cfg['model'], max_tokens=max_tokens, system=system, messages=messages),
                    {'x-api-key': cfg['key'], 'anthropic-version': '2023-06-01'}, timeout)
        return ''.join(b.get('text', '') for b in out.get('content', []) if b.get('type') == 'text').strip()
    if kind == 'gemini':
        body = dict(systemInstruction=dict(parts=[dict(text=system)]),
                    contents=[dict(role='model' if m['role'] == 'assistant' else 'user', parts=[dict(text=m['content'])]) for m in messages],
                    generationConfig=dict(maxOutputTokens=max_tokens))
        out = _post(f"{cfg['base']}/models/{quote(cfg['model'], safe='')}:generateContent", body, {'x-goog-api-key': cfg['key']}, timeout)
        cands = out.get('candidates') or []
        return ''.join(p.get('text', '') for p in ((cands[0].get('content') or {}).get('parts') or [])).strip() if cands else ''
    headers = {'Authorization': f"Bearer {cfg['key']}"} if cfg['key'] else {}
    if cfg['provider'] == 'openrouter': headers.update({'X-Title': 'MaleCNS Lab'})
    body = dict(model=cfg['model'], messages=[dict(role='system', content=system)] + messages)
    body['max_completion_tokens' if cfg['provider'] == 'openai' else 'max_tokens'] = max_tokens
    out = _post(cfg['base'] + '/chat/completions', body, headers, timeout)
    choices = out.get('choices') or []
    return (((choices[0].get('message') or {}).get('content')) or '').strip() if choices else ''


def models(cfg, timeout=15):
    """Model names the provider offers to this key, for the Settings picker."""
    kind = cfg['kind']
    if kind == 'anthropic':
        req = urllib.request.Request(cfg['base'] + '/v1/models?limit=100', headers={'x-api-key': cfg['key'], 'anthropic-version': '2023-06-01'})
        return [m['id'] for m in _open(req, timeout).get('data', [])]
    if kind == 'gemini':
        req = urllib.request.Request(cfg['base'] + '/models?pageSize=200', headers={'x-goog-api-key': cfg['key']})
        return [m['name'].split('/', 1)[-1] for m in _open(req, timeout).get('models', []) if 'generateContent' in m.get('supportedGenerationMethods', [])]
    req = urllib.request.Request(cfg['base'] + '/models', headers={'Authorization': f"Bearer {cfg['key']}"} if cfg['key'] else {})
    return sorted(m['id'] for m in _open(req, timeout).get('data', []))
