"""The lab guide: tours, explanations, questions and model proposals for every page.

Works in two modes. Without a language model it answers from its own knowledge (helper_kb.py) and the saved
benchmarks: page purpose, measured against assumed, findings, glossary, ranked proposals. With a model
(Anthropic, OpenAI, Gemini, OpenRouter, Ollama or any OpenAI-compatible server; see llm.py) it hands the same
material to that model, which answers free-form and drafts new proposals.

Because the lab may be public through a tunnel, the lab's own model is rate-limited per visitor and per day
(not for requests from the Mac itself). Visitors can bring their own key, used only for their request.
"""
import json
import math
import os
import re
import threading
import time
from collections import Counter, defaultdict, deque
from pathlib import Path

import helper_kb as kb
import llm

BASE = Path(__file__).resolve().parent
PER_IP_HOUR = int(os.environ.get('MALECNS_HELPER_PER_HOUR', '40'))
PER_DAY = int(os.environ.get('MALECNS_HELPER_PER_DAY', '300'))

STOP = set('a an the of to in on for and or is are was be it this that what which how why does do can with by from as at i me my we our you your about there their these those into than then so not no if when where who whom any some more most much many very just also only its it\'s show tell explain mean means page here'.split())


def words(text):
    return [w for w in re.findall(r"[a-z0-9][a-z0-9\-']*", str(text).lower()) if w not in STOP and len(w) > 1]


def llm_status():
    cfg = llm.active()
    return dict(provider=cfg['provider'], label=cfg['label'], model=cfg['model']) if cfg else None


def page_entry(path):
    path = (path or '/').rstrip('/') or '/'
    return path, kb.PAGES.get(path)


# --- knowledge as searchable documents -------------------------------------------------------
def documents():
    docs = []
    for term, text in kb.GLOSSARY.items():
        docs.append(dict(kind='term', title=term, text=text, link=None, boost=term.lower()))
    for path, p in kb.PAGES.items():
        docs.append(dict(kind='page', title=p['title'], link=path,
                         text=f"{p['purpose']} Measured: {p['measured']} Assumed: {p['assumed']}", boost=p['title'].lower()))
        for f in p['findings']:
            docs.append(dict(kind='finding', title=p['title'], link=path, text=f, boost=''))
        for sel, t, text in p['steps']:
            docs.append(dict(kind='control', title=f"{p['title']}: {t}", link=path, text=text, boost=t.lower()))
    for title, path, text in kb.OPEN_PROBLEMS:
        docs.append(dict(kind='open problem', title=title, link=path, text=text, boost=''))
    for pr in kb.PROPOSALS:
        docs.append(dict(kind='proposal', title=pr['title'], link=pr['pages'][0], text=pr['idea'], extra=pr['tags'], boost=''))
    return docs


DOCS = documents()
_DF = Counter(w for d in DOCS for w in set(words(d['title'] + ' ' + d['text'] + ' ' + d.get('extra', ''))))


def search(query, page=None, k=6):
    q = words(query)
    if not q: return []
    n = len(DOCS)
    scored = []
    for d in DOCS:
        bag = Counter(words(d['title'] + ' ' + d['text'] + ' ' + d.get('extra', ''))); size = sum(bag.values()) or 1
        s = 0.0
        for w in q:
            tf = bag.get(w, 0)
            if not tf:   # match plural or partial forms ("kenyon" in "kenyon-cell")
                tf = sum(c for t, c in bag.items() if len(w) > 3 and (t.startswith(w) or w.startswith(t) and len(t) > 3)) * 0.6
            if tf: s += (tf * 2.2 / (tf + 1.2 * (0.25 + 0.75 * size / 30))) * math.log(1 + n / (1 + _DF.get(w, 0)))
        if d['boost'] and d['boost'] in query.lower(): s += 6 + len(d['boost']) / 4   # exact term named
        if page and d['link'] == page: s *= 1.25
        if s > 0: scored.append((s, d))
    scored.sort(key=lambda t: -t[0])
    return [d for _, d in scored[:k]]


def terms_in(text):
    """Glossary terms that occur in a piece of text, longest first, each once."""
    low = ' ' + str(text).lower() + ' '
    found = []
    for term in sorted(kb.GLOSSARY, key=len, reverse=True):
        t = term.lower()
        if re.search(r'(?<![a-z0-9])' + re.escape(t) + r's?(?![a-z0-9])', low) and not any(t in f.lower() for f in found):
            found.append(term)
    return sorted(found, key=lambda t: low.find(t.lower()))   # in reading order


# --- live numbers from the saved benchmarks ---------------------------------------------------
def flatten(obj, prefix='', out=None, limit=160):
    out = [] if out is None else out
    if len(out) >= limit: return out
    if isinstance(obj, dict):
        for k, v in obj.items(): flatten(v, f'{prefix}.{k}' if prefix else str(k), out, limit)
    elif isinstance(obj, list):
        if obj and all(isinstance(v, (int, float)) for v in obj):
            vals = obj[:12]; out.append(f"{prefix} = [{', '.join(f'{v:.3g}' if isinstance(v, float) else str(v) for v in vals)}{', …' if len(obj) > 12 else ''}]")
        else:
            for i, v in enumerate(obj[:8]): flatten(v, f'{prefix}[{i}]', out, limit)
    elif isinstance(obj, float): out.append(f'{prefix} = {obj:.4g}')
    elif isinstance(obj, (int, bool)) or (isinstance(obj, str) and len(obj) < 80): out.append(f'{prefix} = {obj}')
    return out


def data_digest(path, chars=7000):
    _, p = page_entry(path)
    if not p or not p.get('data'): return ''
    f = BASE / p['data']
    if not f.exists(): return f"(no saved benchmark yet: {p['data']})"
    try: obj = json.loads(f.read_text())
    except (OSError, ValueError): return ''
    text = '\n'.join(flatten(obj))
    return text[:chars]


# --- page guide (tour + explain) ---------------------------------------------------------------
def guide(path):
    path, p = page_entry(path)
    if not p:
        return dict(path=path, known=False, title='This page', purpose='The guide has no notes for this page yet. Ask a question and it will search what it knows.',
                    steps=[], findings=[], measured='', assumed='', terms=[], suggestions=suggestions(path), llm=llm_status())
    text = ' '.join([p['purpose'], p['measured'], p['assumed']] + p['findings'] + [s[2] for s in p['steps']])
    return dict(path=path, known=True, title=p['title'], purpose=p['purpose'], measured=p['measured'], assumed=p['assumed'],
                findings=p['findings'], steps=[dict(selector=s, title=t, text=x) for s, t, x in p['steps']],
                terms=[dict(term=t, text=kb.GLOSSARY[t]) for t in terms_in(text)][:20],
                problems=[dict(title=t, text=x) for t, l, x in kb.OPEN_PROBLEMS if l == path],
                suggestions=suggestions(path), llm=llm_status())


def suggestions(path):
    _, p = page_entry(path)
    base = ['What is measured and what is assumed here?', 'What did the experiments find?', 'What should I try first?']
    extra = {'/optogenetics': ['Why doesn\'t the fly walk?', 'What is MDN?', 'What does the shuffled control show?'],
             '/scaffold': ['Why does the reference frame help so much?', 'Why is there no gain on digits?', 'What is prediction-assisted pooling?'],
             '/objects': ['What does the lamina do?', 'How is this different from Monty?'],
             '/memory': ['What is a compartment?', 'How does extinction work?'],
             '/navigation': ['What is path integration?', 'What is a ring attractor?'],
             '/expansion': ['Why does backprop forget?', 'What is a sparse expansion?']}
    return (extra.get(path, []) + base)[:5] if p else ['What is a connectome?', 'Which page should I start with?', 'What are the open problems?']


def glossary():
    return [dict(term=t, text=x) for t, x in sorted(kb.GLOSSARY.items(), key=lambda kv: kv[0].lower())]


# --- answering ---------------------------------------------------------------------------------
def local_answer(question, path, selection=None):
    path, p = page_entry(path)
    ql = question.lower()
    parts, sources = [], []
    if selection:
        found = terms_in(selection)
        if found:
            parts.append('Terms in what you pointed at:\n' + '\n'.join(f'- **{t}**: {kb.GLOSSARY[t]}' for t in found[:6]))
        elif p:
            parts.append(f"That is part of **{p['title']}**. {p['purpose']}")
    if p and re.search(r'measured|assumed|assumption|trust|real', ql):
        parts.append(f"On **{p['title']}**:\n\n**Measured:** {p['measured']}\n\n**Assumed:** {p['assumed']}")
    if p and re.search(r'find|result|show|learn|outcome|work', ql) and p['findings']:
        found = p['findings']
        if selection:   # only the findings about what was pointed at
            sw = set(words(selection))
            found = [f for f in found if len(sw & set(words(f))) >= 3] or found[:1]
        parts.append('What the experiments found:\n' + '\n'.join(f'- {f}' for f in found))
    if re.search(r'open problem|missing|next|what now|improve|todo|gap', ql):
        parts.append('Open problems:\n' + '\n'.join(f'- **{t}** ([page]({l})): {x}' for t, l, x in kb.OPEN_PROBLEMS))
    if p and re.search(r'start|first|try|how do i|use this', ql):
        parts.append('A good first pass:\n' + '\n'.join(f'{i + 1}. **{t}**: {x}' for i, (_, t, x) in enumerate(p['steps'][:6])) + '\n\nOr press **Tour** to have each part highlighted.')
    if re.search(r'start|which page|where', ql) and not p:
        parts.append('Start with [Motion lab](/) for wiring → movement, [Virtual optogenetics](/optogenetics) for the real-fly tests, [Memory](/memory) for learning, and [Scaffold learner](/scaffold) for learning from little data.')
    if not parts:
        hits = search(question + ' ' + (selection or ''), path)
        terms = [d for d in hits if d['kind'] == 'term']
        if terms and re.search(r'^(what|who)\b|mean|define|is an? ', ql):
            d = terms[0]; parts.append(f"**{d['title']}**: {d['text']}")
            hits = [h for h in hits if h is not d]
        rest = hits[:4]
        if rest:
            parts.append('\n'.join(f"- **{d['title']}**" + (f" ([{kb.PAGES.get(d['link'], {}).get('title', d['link'])}]({d['link']}))" if d['link'] else '') + f": {d['text']}" for d in rest))
            sources = [d['link'] for d in rest if d['link']]
    if not parts:
        parts.append('I could not find that in the lab\'s notes. Try a term from the page, or press **Explain** to see the glossary.')
    return '\n\n'.join(parts), sorted(set(sources))


def system_prompt(path, selection=None):
    path, p = page_entry(path)
    pages = '\n'.join(f"- {v['title']} ({k}): {v['purpose']}" for k, v in kb.PAGES.items())
    gl = '\n'.join(f'- {t}: {x}' for t, x in kb.GLOSSARY.items())
    probs = '\n'.join(f'- {t} ({l}): {x}' for t, l, x in kb.OPEN_PROBLEMS)
    here = ''
    if p:
        here = (f"\n\nTHE PAGE THE USER IS ON: {p['title']} ({path})\nPurpose: {p['purpose']}\nMeasured: {p['measured']}\nAssumed: {p['assumed']}\n"
                f"Findings: {' | '.join(p['findings'])}\nControls on the page: " + '; '.join(f'{t}: {x}' for _, t, x in p['steps']))
        digest = data_digest(path)
        if digest: here += f"\n\nLIVE NUMBERS FROM THE PAGE'S SAVED BENCHMARK ({p['data']}), flattened:\n{digest}"
    if selection: here += f'\n\nTHE USER POINTED AT THIS PART OF THE PAGE:\n"""{selection[:1500]}"""'
    return f"""You are the guide inside MaleCNS Lab, a research web app that simulates parts of the male fruit-fly (Drosophila) central nervous system from its connectome (MaleCNS v1.0), and uses its design to build learners that need little data (sparse distributed representations, Monty-style reference frames, HTM prediction).

Your jobs: walk users through pages, explain what things mean in plain language, interpret results honestly, and propose new models or experiments.

Rules:
- Always keep measured (from the connectome) separate from assumed (model choices). Never claim the model reproduces a real fly unless a result with controls shows it. Shuffled and sham controls matter: if they do the same, say the effect is not wiring-specific.
- Quote numbers only from the material below; if a number is not there, say you don't have it.
- Be concise: short paragraphs or bullets, plain words, define jargon the first time. Link pages as markdown links like [Memory](/memory).
- When proposing a model, give: idea, what measured data it uses, hypothesis, test, controls, baseline, success criterion, and files it would touch in this repo. Prefer proposals that could show the measured wiring matters.

PAGES:
{pages}

OPEN PROBLEMS:
{probs}

GLOSSARY:
{gl}{here}"""


class Limiter:
    def __init__(self, per_hour=PER_IP_HOUR, per_day=PER_DAY):
        self.per_hour, self.per_day = per_hour, per_day
        self.ip = defaultdict(deque); self.day = deque(); self.lock = threading.Lock()

    def allow(self, who, now=None):
        now = time.time() if now is None else now
        with self.lock:
            q = self.ip[who]
            while q and now - q[0] > 3600: q.popleft()
            while self.day and now - self.day[0] > 86400: self.day.popleft()
            if len(q) >= self.per_hour or len(self.day) >= self.per_day: return False
            q.append(now); self.day.append(now); return True


LIMIT = Limiter()


VISITOR_LIMIT = Limiter(per_hour=int(os.environ.get('MALECNS_HELPER_VISITOR_PER_HOUR', '120')), per_day=10 ** 9)


def pick_model(override=None, who='local', local=False):
    """(config, None) when a model may answer, or (None, reason) when the guide must use its notes.
    A visitor's own key wins for that request; otherwise the lab's model, rate-limited unless asked from the Mac itself."""
    try:
        cfg = llm.visitor(override)
    except llm.LLMError as e:
        return None, str(e)
    if cfg:
        return (cfg, None) if VISITOR_LIMIT.allow(who) else (None, 'Too many questions from here for now')
    cfg = llm.active()
    if not cfg: return None, None
    if local or LIMIT.allow(who): return cfg, None
    return None, 'Question limit reached for now'


def tag(cfg):
    return dict(mode='llm', provider=cfg['provider'], model=cfg['model'], label=cfg['label'])


def clean_history(history):
    msgs = []
    for m in (history or [])[-8:]:
        if isinstance(m, dict) and m.get('role') in ('user', 'assistant') and isinstance(m.get('text'), str) and m['text'].strip():
            msgs.append(dict(role=m['role'], content=m['text'][:2000]))
    while msgs and msgs[0]['role'] != 'user': msgs.pop(0)
    return msgs


def ask(question, path='/', history=None, selection=None, who='local', override=None, local=False):
    question = str(question or '').strip()[:1500]
    if not question and not selection: raise ValueError('Ask a question')
    if not question: question = 'Explain this part of the page in plain language: what it shows and what it means.'
    selection = str(selection)[:1500] if selection else None
    cfg, note = pick_model(override, who, local)
    if note: note += '; answered from the lab notes.'
    if cfg:
        msgs = clean_history(history) + [dict(role='user', content=question)]
        # Consecutive same-role messages are not allowed: merge them.
        merged = []
        for m in msgs:
            if merged and merged[-1]['role'] == m['role']: merged[-1]['content'] += '\n\n' + m['content']
            else: merged.append(m)
        try:
            text = llm.chat(cfg, system_prompt(path, selection), merged)
            if text: return dict(answer=text, sources=[], **tag(cfg))
            note = f"{cfg['label']} returned nothing; answered from the lab notes."
        except llm.LLMError as e:
            note = f"{cfg['label']}: {e}; answered from the lab notes."
    text, sources = local_answer(question, path, selection)
    return dict(answer=text, mode='local', sources=sources, note=note)


# --- proposals ---------------------------------------------------------------------------------
def rank_proposals(goal='', path='/', k=4):
    g = set(words(goal))
    scored = []
    for pr in kb.PROPOSALS:
        bag = set(words(' '.join([pr['title'], pr['tags'], pr['idea'], pr['hypothesis']])))
        s = len(g & bag) * 3 + sum(1 for w in g for t in bag if len(w) > 3 and w != t and (t.startswith(w) or w.startswith(t))) * 1.5
        if path in pr['pages']: s += 2 + (2 if pr['pages'][0] == path else 0)
        scored.append((s, pr))
    scored.sort(key=lambda t: -t[0])
    return [dict(pr, score=round(s, 1)) for s, pr in scored[:k]]


FIELDS = ['title', 'idea', 'uses', 'hypothesis', 'test', 'controls', 'success', 'files', 'effort']


def propose(goal='', path='/', who='local', override=None, local=False):
    goal = str(goal or '').strip()[:800]
    picks = rank_proposals(goal, path)
    cfg, note = pick_model(override, who, local) if goal else (None, None)
    if note: note += '; showing the catalogue.'
    if cfg:
        ask_ = (f'Goal: {goal}\n\nPropose 3 new models or experiments for this lab that serve the goal. Build on what exists; '
                'you may refine one of the catalogue proposals below or invent new ones. Reply with only a JSON array of objects with keys '
                + ', '.join(FIELDS) + ' (effort is S, M or L). Catalogue:\n'
                + '\n'.join(f"- {p['title']}: {p['idea']}" for p in kb.PROPOSALS))
        try:
            text = llm.chat(cfg, system_prompt(path), [dict(role='user', content=ask_)], max_tokens=1800, timeout=90)
            m = re.search(r'\[.*\]', text, re.S)
            items = json.loads(m.group(0)) if m else []
            items = [dict({f: str(it.get(f, '')) for f in FIELDS}, id=f'llm-{i}', pages=[path], source='llm')
                     for i, it in enumerate(items) if isinstance(it, dict) and it.get('title')][:3]
            if items: return dict(proposals=items + [dict(p, source='catalogue') for p in picks[:2]], **tag(cfg))
            note = f"{cfg['label']}'s reply could not be read; showing the catalogue."
        except (llm.LLMError, ValueError) as e:
            note = f"{cfg['label']}: {e}; showing the catalogue."
    return dict(proposals=[dict(p, source='catalogue') for p in picks], mode='local', note=note)


def build_request(proposal):
    """Text to paste into a build session: what to make and how to judge it."""
    p = {f: str(proposal.get(f, '')) for f in FIELDS}
    return (f"MaleCNS Lab: build \"{p['title']}\".\n\nIdea: {p['idea']}\nUses (measured): {p['uses']}\nHypothesis: {p['hypothesis']}\n"
            f"Test: {p['test']}\nControls: {p['controls']}\nSuccess: {p['success']}\nFiles: {p['files']}\n\n"
            'Keep measured and assumed separate on the page, run three or more seeds with the controls, add tests, and report honestly if it fails.')


# --- settings tab: try a provider, list its models --------------------------------------------------
def candidate(o, local, who):
    """A provider config to test or list models for: a visitor's own key, or (local only) the lab's settings with edits."""
    provider = str(o.get('provider', ''))
    if provider not in llm.PROVIDERS: raise ValueError('Unknown provider')
    if o.get('own') or not local:
        cfg = llm.visitor(dict(provider=provider, key=o.get('key'), model=o.get('model') or 'x'))
        if not cfg: raise ValueError('Enter a key first')
        if not VISITOR_LIMIT.allow(who): raise ValueError('Too many requests from here for now')
        return cfg
    s = llm.load()
    cfg = llm.config_for(s, provider)
    if o.get('key'): cfg['key'] = str(o['key']).strip()
    if o.get('model'): cfg['model'] = str(o['model']).strip()
    if o.get('base_url'): cfg['base'] = llm.check_base(str(o['base_url']).strip())
    if llm.PROVIDERS[provider]['key'] and not cfg['key']: raise ValueError('Enter a key first')
    if not cfg['base']: raise ValueError('Enter the server URL')
    return cfg


def list_models(o, local=False, who='local'):
    cfg = candidate(o, local, who)
    try: return dict(models=llm.models(cfg)[:300])
    except (llm.LLMError, KeyError, TypeError) as e: raise ValueError(f"{cfg['label']}: {e}")


def test(o, local=False, who='local'):
    cfg = candidate(o, local, who)
    if not cfg['model'] or cfg['model'] == 'x': raise ValueError('Choose a model first')
    try: text = llm.chat(cfg, 'You are a connection test.', [dict(role='user', content='Reply with the single word: ready')], max_tokens=20, timeout=30)
    except llm.LLMError as e: raise ValueError(f"{cfg['label']}: {e}")
    return dict(ok=True, reply=text[:80], label=cfg['label'], model=cfg['model'])
