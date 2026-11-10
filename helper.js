/* Lab guide: a floating window on every page.
   Tour: steps through the page and rings each part. Explain: what the page measures and assumes, what it found,
   a glossary, terms marked on the page, and "point at something". Ask: questions, answered from the lab's notes
   and saved benchmarks, or by Claude when the server has a key. Propose: new models ranked for a goal.
   Loaded by ink-skin.js, so every page gets it without changes to the page. Press ? to open. */
(function () {
  'use strict';
  if (window.__labGuide) return; window.__labGuide = true;
  const PATH = location.pathname.replace(/\/+$/, '') || '/';
  const esc = s => String(s ?? '').replace(/[&<>"]/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c]));
  const store = {
    get(k, d) { try { const v = localStorage.getItem('hx-' + k); return v == null ? d : JSON.parse(v); } catch (_) { return d; } },
    set(k, v) { try { localStorage.setItem('hx-' + k, JSON.stringify(v)); } catch (_) {} },
  };
  const IC = {
    guide: '<circle cx="12" cy="12" r="9"/><path d="M9.5 9.3a2.6 2.6 0 0 1 5 .9c0 1.7-2.5 2.3-2.5 4"/><circle cx="12" cy="17.2" r=".6"/>',
  };
  const svg = n => `<svg class="hx-ic" viewBox="0 0 24 24" aria-hidden="true">${IC[n]}</svg>`;

  // Small, safe markdown: escape first, then bold, code, links (relative or http), lists, paragraphs.
  function md(src) {
    const inline = t => esc(t)
      .replace(/`([^`]+)`/g, '<code>$1</code>')
      .replace(/\*\*([^*]+)\*\*/g, '<b>$1</b>')
      .replace(/\[([^\]]+)\]\(((?:\/|https?:\/\/)[^)\s]*)\)/g, (m, a, h) => `<a href="${h}"${h[0] === '/' ? '' : ' target="_blank" rel="noopener"'}>${a}</a>`);
    const out = []; let list = null;
    for (const raw of String(src).split('\n')) {
      const line = raw.trimEnd(), ul = line.match(/^\s*[-*•]\s+(.*)/), ol = line.match(/^\s*\d+[.)]\s+(.*)/), h = line.match(/^#{1,4}\s+(.*)/);
      if (ul || ol) {
        const tag = ul ? 'ul' : 'ol';
        if (list !== tag) { if (list) out.push(`</${list}>`); out.push(`<${tag}>`); list = tag; }
        out.push(`<li>${inline((ul || ol)[1])}</li>`); continue;
      }
      if (list) { out.push(`</${list}>`); list = null; }
      if (h) out.push(`<p><b>${inline(h[1])}</b></p>`);
      else if (line.trim()) out.push(`<p>${inline(line)}</p>`);
    }
    if (list) out.push(`</${list}>`);
    return out.join('');
  }

  async function api(path, body) {
    const r = await fetch(path, body ? { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) } : {});
    const j = await r.json().catch(() => ({}));
    if (!r.ok) throw new Error(j.error || r.statusText);
    return j;
  }

  let G = null, GLOSS = null, win, body, foot, ring, tip, tab = store.get('tab', 'tour');
  let tourAt = -1, tourOn = false, pointing = false, termsOn = false, busy = false;
  const history = store.get('history', []);   // shared across pages: [{role, text, path, sel?, mode?}]

  function build() {
    if (document.querySelector('.hx-launch')) return;
    const launch = document.createElement('button');
    launch.className = 'hx-launch'; launch.type = 'button';
    launch.innerHTML = `${svg('guide')}<span>Guide</span><kbd>?</kbd>`;
    launch.title = 'Lab guide: tour, explanations, questions, proposals';
    launch.onclick = () => toggle();
    win = document.createElement('div');
    win.className = 'hx-win'; win.hidden = true; win.setAttribute('role', 'dialog'); win.setAttribute('aria-label', 'Lab guide');
    win.innerHTML = `<div class="hx-top"><b>Lab guide<small class="hx-sub">Loading…</small></b><span class="hx-badge" title="">notes</span>
        <button class="hx-x" data-a="min" title="Close (Esc)">×</button></div>
      <div class="hx-tabs" role="tablist">${['tour', 'explain', 'ask', 'propose'].map(t =>
        `<button role="tab" data-t="${t}">${t[0].toUpperCase() + t.slice(1)}</button>`).join('')}</div>
      <div class="hx-body"></div>
      <div class="hx-foot" hidden><textarea placeholder="Ask about this page, a term, a result…" aria-label="Question"></textarea><button class="hx-btn pri" data-a="send">Ask</button></div>`;
    ring = document.createElement('div'); ring.className = 'hx-ring'; ring.hidden = true;
    tip = document.createElement('div'); tip.className = 'hx-tip'; tip.hidden = true;
    document.body.append(launch, win, ring, tip);
    body = win.querySelector('.hx-body'); foot = win.querySelector('.hx-foot');

    const pos = store.get('pos', null);
    if (pos && innerWidth > 640) Object.assign(win.style, { left: Math.min(pos.x, innerWidth - 340) + 'px', top: Math.max(8, Math.min(pos.y, innerHeight - 200)) + 'px', right: 'auto', bottom: 'auto' });
    drag(win.querySelector('.hx-top'));
    win.querySelectorAll('.hx-tabs button').forEach(b => b.onclick = () => show(b.dataset.t));
    win.querySelector('[data-a=min]').onclick = () => toggle(false);
    const ta = foot.querySelector('textarea');
    ta.addEventListener('keydown', e => { if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); send(ta.value); } });
    ta.addEventListener('input', () => { ta.style.height = '40px'; ta.style.height = Math.min(120, ta.scrollHeight) + 'px'; });
    foot.querySelector('[data-a=send]').onclick = () => send(ta.value);

    addEventListener('keydown', e => {
      const typing = /^(INPUT|TEXTAREA|SELECT)$/.test(document.activeElement?.tagName) || document.activeElement?.isContentEditable;
      if (e.key === '?' && !typing && !e.metaKey && !e.ctrlKey) { e.preventDefault(); toggle(); }
      else if (e.key === 'Escape') { if (pointing) stopPoint(); else if (tourOn) endTour(); else if (!win.hidden && !typing) toggle(false); }
      else if (tourOn && !typing && (e.key === 'ArrowRight' || e.key === 'ArrowLeft')) { e.preventDefault(); go(tourAt + (e.key === 'ArrowRight' ? 1 : -1)); }
    });
    addEventListener('scroll', place, { passive: true }); addEventListener('resize', place);
    if (store.get('open', false)) toggle(true);
    // First visit to the lab: point out the guide once.
    if (!store.get('seen', false)) { store.set('seen', true); setTimeout(() => toggle(true), 900); }
  }

  function drag(handle) {
    let sx, sy, ox, oy;
    handle.addEventListener('pointerdown', e => {
      if (e.target.closest('button') || innerWidth <= 640) return;
      const r = win.getBoundingClientRect(); sx = e.clientX; sy = e.clientY; ox = r.left; oy = r.top;
      handle.setPointerCapture(e.pointerId);
      const move = ev => { Object.assign(win.style, { left: Math.max(0, Math.min(innerWidth - 120, ox + ev.clientX - sx)) + 'px', top: Math.max(0, Math.min(innerHeight - 60, oy + ev.clientY - sy)) + 'px', right: 'auto', bottom: 'auto' }); };
      const up = () => { handle.removeEventListener('pointermove', move); handle.removeEventListener('pointerup', up); const r2 = win.getBoundingClientRect(); store.set('pos', { x: r2.left, y: r2.top }); };
      handle.addEventListener('pointermove', move); handle.addEventListener('pointerup', up);
    });
  }

  async function load() {
    if (G) return G;
    try { G = await api('/api/helper/guide?path=' + encodeURIComponent(PATH)); }
    catch (e) { G = { title: 'This page', known: false, steps: [], findings: [], terms: [], suggestions: [], llm: false, error: e.message }; }
    win.querySelector('.hx-sub').textContent = G.title;
    const badge = win.querySelector('.hx-badge');
    badge.textContent = G.llm ? 'Claude' : 'lab notes'; badge.classList.toggle('on', !!G.llm);
    badge.title = G.llm ? 'Questions and proposals are answered by Claude, grounded in the lab\'s notes and saved results.'
      : 'Answers come from the lab\'s own notes and saved results. Put an Anthropic API key in ~/.config/malecns-lab/anthropic-key for free-form answers.';
    return G;
  }

  function toggle(open) {
    open = open ?? win.hidden;
    win.hidden = !open; store.set('open', open);
    if (open) { load().then(() => show(tab)); }
    else { endTour(); stopPoint(); }
  }

  function show(t) {
    tab = t; store.set('tab', t);
    win.querySelectorAll('.hx-tabs button').forEach(b => b.setAttribute('aria-selected', b.dataset.t === t));
    foot.hidden = t !== 'ask';
    if (t !== 'tour' && tourOn) endTour();
    ({ tour: renderTour, explain: renderExplain, ask: renderAsk, propose: renderPropose })[t]();
  }

  // ---------- tour ----------
  const target = s => {
    if (!s) return null;
    const el = document.querySelector(s); if (!el) return null;
    const r = el.getBoundingClientRect(); return (r.width || r.height) ? el : null;
  };
  function renderTour() {
    if (!G) return;
    const steps = G.steps || [];
    if (!steps.length) { body.innerHTML = `<p>${esc(G.purpose || 'No tour for this page yet.')}</p><p class="hx-dim">Try <b>Ask</b> or <b>Explain</b>.</p>`; return; }
    body.innerHTML = `<p>${esc(G.purpose)}</p>
      <div class="hx-nav"><button class="hx-btn pri" data-a="start">${tourOn ? 'Restart tour' : 'Start the tour'}</button><span class="hx-dim" style="font-size:12.5px">← → to move, Esc to stop</span></div>
      <ol class="hx-steps">${steps.map((s, i) => `<li data-i="${i}" class="${i === tourAt && tourOn ? 'on' : ''}${s.selector && !target(s.selector) ? ' miss' : ''}"><i>${i + 1}</i><div><b>${esc(s.title)}</b><span>${esc(s.text)}</span></div></li>`).join('')}</ol>
      ${tourOn ? `<div class="hx-nav"><button class="hx-btn" data-a="back"${tourAt <= 0 ? ' disabled' : ''}>Back</button><span class="sp"></span><span class="hx-dim">${tourAt + 1} of ${steps.length}</span><button class="hx-btn" data-a="end">End</button><button class="hx-btn pri" data-a="next">${tourAt >= steps.length - 1 ? 'Finish' : 'Next'}</button></div>` : ''}`;
    body.querySelector('[data-a=start]').onclick = () => go(0);
    body.querySelectorAll('.hx-steps li').forEach(li => li.onclick = () => go(+li.dataset.i));
    const q = a => body.querySelector(`[data-a=${a}]`);
    if (q('back')) q('back').onclick = () => go(tourAt - 1);
    if (q('next')) q('next').onclick = () => go(tourAt + 1);
    if (q('end')) q('end').onclick = endTour;
    body.querySelector('.hx-steps li.on')?.scrollIntoView({ block: 'nearest' });
  }
  function go(i) {
    const steps = G.steps || [];
    if (i < 0) return; if (i >= steps.length) return endTour();
    tourOn = true; tourAt = i;
    const el = target(steps[i].selector);
    ring.classList.remove('point');
    if (el) { el.scrollIntoView({ behavior: 'smooth', block: 'center' }); ring.hidden = false; ring._el = el; setTimeout(place, 350); place(); }
    else { ring.hidden = true; ring._el = null; if (!steps[i].selector) scrollTo({ top: 0, behavior: 'smooth' }); }
    renderTour();
  }
  function endTour() { tourOn = false; tourAt = -1; ring.hidden = true; ring._el = null; if (tab === 'tour' && G) renderTour(); }
  function place() {
    const el = ring._el; if (!el || ring.hidden) return;
    const r = el.getBoundingClientRect(), p = 6;
    Object.assign(ring.style, { left: r.left - p + 'px', top: r.top - p + 'px', width: r.width + 2 * p + 'px', height: r.height + 2 * p + 'px' });
  }

  // ---------- explain ----------
  async function renderExplain() {
    if (!G) return;
    body.innerHTML = `${G.known ? `<h3>What this page does</h3><p>${esc(G.purpose)}</p>
      <div class="hx-ma"><div class="m"><b>Measured.</b> ${esc(G.measured)}</div><div class="a"><b>Assumed.</b> ${esc(G.assumed)}</div></div>
      ${G.findings.length ? `<h3>What it found</h3><ul>${G.findings.map(f => `<li>${esc(f)}</li>`).join('')}</ul>` : ''}
      ${(G.problems || []).length ? `<h3>Open problems</h3><ul>${G.problems.map(p => `<li><b>${esc(p.title)}.</b> ${esc(p.text)}</li>`).join('')}</ul>` : ''}` : `<p>${esc(G.purpose)}</p>`}
      <h3>On the page</h3>
      <div class="hx-chips"><button class="hx-chip" data-a="terms">${termsOn ? 'Unmark terms' : 'Mark terms on the page'}</button><button class="hx-chip" data-a="point">Point at something to explain</button></div>
      ${G.terms.length ? `<h3>Terms used here</h3><dl class="hx-gl">${G.terms.map(t => `<dt>${esc(t.term)}</dt><dd>${esc(t.text)}</dd>`).join('')}</dl>` : ''}
      <h3>Glossary</h3><input class="hx-in" type="search" placeholder="Search terms" aria-label="Search terms"><dl class="hx-gl hx-all"></dl>`;
    body.querySelector('[data-a=terms]').onclick = () => { markTerms(!termsOn); renderExplain(); };
    body.querySelector('[data-a=point]').onclick = startPoint;
    if (!GLOSS) { try { GLOSS = await api('/api/helper/glossary'); } catch (_) { GLOSS = []; } }
    const input = body.querySelector('.hx-in'), list = body.querySelector('.hx-all');
    if (!input) return;
    const draw = () => { const q = input.value.trim().toLowerCase();
      const hits = GLOSS.filter(g => !q || (g.term + ' ' + g.text).toLowerCase().includes(q)).slice(0, q ? 40 : 12);
      list.innerHTML = hits.map(g => `<dt>${esc(g.term)}</dt><dd>${esc(g.text)}</dd>`).join('') + (!q && GLOSS.length > 12 ? `<dd class="hx-dim">${GLOSS.length - 12} more: search above.</dd>` : ''); };
    input.oninput = draw; draw();
    if (explainTerm) { input.value = explainTerm; draw(); input.scrollIntoView({ block: 'start' }); explainTerm = null; }
  }
  let explainTerm = null;

  const SKIP = /^(SCRIPT|STYLE|TEXTAREA|INPUT|SELECT|OPTION|BUTTON|CANVAS|SVG|CODE|PRE|KBD|A)$/;
  async function markTerms(on) {
    termsOn = on;
    if (!on) {
      document.querySelectorAll('.hx-term').forEach(s => { const p = s.parentNode; p.replaceChild(document.createTextNode(s.textContent), s); p.normalize(); });
      return;
    }
    if (!GLOSS) { try { GLOSS = await api('/api/helper/glossary'); } catch (_) { GLOSS = []; } }
    const terms = GLOSS.map(g => g.term).filter(t => t.length > 2).sort((a, b) => b.length - a.length);
    const def = Object.fromEntries(GLOSS.map(g => [g.term.toLowerCase(), g]));
    const re = new RegExp('(?<![\\w-])(' + terms.map(t => t.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')).join('|') + ')s?(?![\\w-])', 'gi');
    const root = document.querySelector('main') || document.body, used = {};
    const walker = document.createTreeWalker(root, NodeFilter.SHOW_TEXT, { acceptNode(n) {
      for (let p = n.parentElement; p && p !== root.parentElement; p = p.parentElement) {
        if (SKIP.test(p.tagName.toUpperCase()) || p.classList.contains('hx-term') || p.closest('.hx-win,.ink-header,.ink-rail,.ink-toc,label')) return NodeFilter.FILTER_REJECT;
      }
      return n.nodeValue.trim().length > 2 ? NodeFilter.FILTER_ACCEPT : NodeFilter.FILTER_REJECT; } });
    const nodes = []; while (walker.nextNode()) nodes.push(walker.currentNode);
    for (const n of nodes) {
      const text = n.nodeValue; re.lastIndex = 0; let m, last = 0, frag = null;
      while ((m = re.exec(text))) {
        const key = m[1].toLowerCase(); if (!def[key] || (used[key] = (used[key] || 0) + 1) > 2) continue;
        frag = frag || document.createDocumentFragment();
        frag.append(text.slice(last, m.index));
        const s = document.createElement('span'); s.className = 'hx-term'; s.dataset.term = def[key].term; s.textContent = m[0];
        frag.append(s); last = m.index + m[0].length;
      }
      if (frag) { frag.append(text.slice(last)); n.parentNode.replaceChild(frag, n); }
    }
  }
  document.addEventListener('mouseover', e => {
    const t = e.target.closest?.('.hx-term'); if (!t || !GLOSS) return;
    const g = GLOSS.find(x => x.term === t.dataset.term); if (!g) return;
    tip.innerHTML = `<b>${esc(g.term)}</b>${esc(g.text)}`; tip.hidden = false;
    const r = t.getBoundingClientRect();
    tip.style.left = Math.min(innerWidth - 330, Math.max(8, r.left)) + 'px';
    tip.style.top = (r.bottom + 8 + 120 > innerHeight ? r.top - tip.offsetHeight - 8 : r.bottom + 8) + 'px';
  });
  document.addEventListener('mouseout', e => { if (e.target.closest?.('.hx-term')) tip.hidden = true; });
  document.addEventListener('click', e => {
    const t = e.target.closest?.('.hx-term'); if (!t || pointing) return;
    explainTerm = t.dataset.term; if (win.hidden) toggle(true); show('explain');
  });

  // ---------- point at something ----------
  const CONTAINER = '.wb-card,.card,.panel,section,.notice,table,figure,fieldset,.bench,li,tr';
  function context(el) {
    const box = el.closest(CONTAINER) || el;
    const head = (box.querySelector('h1,h2,h3') || el.closest('section,.wb-card,.card')?.querySelector('h2,h3'))?.textContent.trim();
    let text = (el.tagName === 'CANVAS' ? '' : el.innerText || '').trim();
    if (el.tagName === 'CANVAS' || text.length < 20) text = (box.innerText || '').trim();
    if (el.tagName === 'CANVAS') text = `[a chart${el.id ? ' (#' + el.id + ')' : ''}] ` + text;
    if (el.matches('input,select')) { const lab = el.closest('label')?.innerText || document.querySelector(`label[for="${el.id}"]`)?.innerText || ''; text = `Control "${lab.trim()}" (${el.id || el.name || el.tagName.toLowerCase()}), value ${el.value}. ` + text; }
    return ((head && !text.startsWith(head) ? head + ': ' : '') + text).replace(/\s+\n/g, '\n').slice(0, 1400);
  }
  let pointEl = null;
  function startPoint() {
    if (tourOn) endTour();
    pointing = true; document.body.classList.add('hx-pointing'); ring.classList.add('point'); win.style.opacity = '.35';
    body.insertAdjacentHTML('afterbegin', '<p class="hx-dim hx-pointnote"><b>Click anything on the page</b> (a chart, a number, a control). Esc to cancel.</p>');
  }
  function stopPoint() {
    pointing = false; document.body.classList.remove('hx-pointing'); ring.classList.remove('point'); ring.hidden = true; ring._el = null; win.style.opacity = '';
    body?.querySelector('.hx-pointnote')?.remove();
  }
  document.addEventListener('mousemove', e => {
    if (!pointing) return;
    const el = document.elementFromPoint(e.clientX, e.clientY);
    if (!el || el.closest('.hx-win,.hx-launch')) { ring.hidden = true; return; }
    pointEl = el.closest('canvas,table,input,select,.notice,h2,p,li,tr') || el;
    ring._el = pointEl; ring.hidden = false; place();
  }, true);
  document.addEventListener('click', e => {
    if (!pointing || e.target.closest('.hx-win,.hx-launch')) return;
    e.preventDefault(); e.stopPropagation();
    const sel = context(pointEl || e.target); stopPoint();
    show('ask'); send('', sel);
  }, true);

  // ---------- ask ----------
  function renderAsk() {
    const mine = history.slice(-30);
    body.innerHTML = (mine.length ? '' : `<p>Ask what anything on this page means, how to use it, what the results say, or what to build next.</p>`)
      + mine.map(m => m.role === 'user'
        ? `<div class="hx-msg u"><div>${m.sel ? `<div class="sel">${esc(m.sel)}</div>` : ''}${esc(m.text || 'Explain this')}</div></div>`
        : `<div class="hx-msg a"><div>${md(m.text)}</div><small>${m.mode === 'claude' ? 'Claude' : 'Lab notes'}${m.note ? ' · ' + esc(m.note) : ''}${m.path && m.path !== PATH ? ' · asked on ' + esc(m.path) : ''}</small></div>`).join('')
      + (busy ? '<div class="hx-msg a"><div class="hx-dim hx-dots">Thinking</div></div>' : '')
      + `<div class="hx-chips">${(G?.suggestions || []).map(s => `<button class="hx-chip">${esc(s)}</button>`).join('')}${mine.length ? '<button class="hx-chip" data-a="clear">Clear</button>' : ''}</div>`;
    body.querySelectorAll('.hx-chip').forEach(c => c.onclick = () => c.dataset.a === 'clear' ? (history.length = 0, store.set('history', history), renderAsk()) : send(c.textContent));
    body.scrollTop = body.scrollHeight;
    if (!pointing) foot.querySelector('textarea').focus({ preventScroll: true });
  }
  async function send(text, sel) {
    text = String(text || '').trim();
    if (busy || (!text && !sel)) return;
    if (tab !== 'ask') show('ask');
    const ta = foot.querySelector('textarea'); ta.value = ''; ta.style.height = '40px';
    const prior = history.slice(-8).map(m => ({ role: m.role, text: m.text }));
    history.push({ role: 'user', text, sel, path: PATH }); busy = true; renderAsk();
    try {
      const r = await api('/api/helper/ask', { question: text, path: PATH, history: prior, selection: sel || null });
      history.push({ role: 'assistant', text: r.answer, mode: r.mode, note: r.note, path: PATH });
    } catch (e) { history.push({ role: 'assistant', text: 'Could not reach the lab server: ' + e.message, mode: 'local', path: PATH }); }
    busy = false; while (history.length > 40) history.shift(); store.set('history', history); renderAsk();
  }

  // ---------- propose ----------
  const GOALS = ['Make the fly walk', 'Learn from less data', 'Show the measured wiring matters', 'Go beyond sensorimotor: concepts and reasoning', 'More realistic physiology'];
  let proposals = null, goalText = store.get('goal', '');
  function renderPropose() {
    body.innerHTML = `<p>Describe what you want the lab to do next. Proposals build on what the lab already has, each with a hypothesis, a test and controls.</p>
      <input class="hx-in" placeholder="Goal, e.g. recognise objects from a few looks" value="${esc(goalText)}" aria-label="Goal">
      <div class="hx-chips">${GOALS.map(g => `<button class="hx-chip">${esc(g)}</button>`).join('')}</div>
      <div class="hx-nav"><button class="hx-btn pri" data-a="go">Propose models</button><span class="hx-dim" style="font-size:12.5px">${G?.llm ? 'Claude drafts new ones' : 'From the lab\'s catalogue, ranked for your goal and this page'}</span></div>
      <div class="hx-props" style="margin-top:12px">${busy ? '<p class="hx-dim hx-dots">Drafting</p>' : ''}</div>`;
    const input = body.querySelector('.hx-in');
    body.querySelectorAll('.hx-chip').forEach(c => c.onclick = () => { input.value = c.textContent; run(); });
    input.addEventListener('keydown', e => { if (e.key === 'Enter') run(); });
    body.querySelector('[data-a=go]').onclick = run;
    async function run() {
      if (busy) return; goalText = input.value.trim(); store.set('goal', goalText); busy = true; renderPropose();
      try { proposals = await api('/api/helper/propose', { goal: goalText, path: PATH }); }
      catch (e) { proposals = { proposals: [], note: 'Could not reach the lab server: ' + e.message }; }
      busy = false; renderPropose();
    }
    if (proposals && !busy) drawCards(body.querySelector('.hx-props'));
  }
  const LABELS = { idea: 'Idea', uses: 'Uses (measured)', hypothesis: 'Hypothesis', test: 'Test', controls: 'Controls', success: 'Success looks like', files: 'Files' };
  function drawCards(box) {
    const ps = proposals.proposals || [];
    box.innerHTML = (proposals.note ? `<p class="hx-dim">${esc(proposals.note)}</p>` : '') + (ps.length ? '' : '<p class="hx-dim">No proposals.</p>')
      + ps.map((p, i) => `<div class="hx-card${i === 0 ? ' open' : ''}" data-i="${i}"><h4>${esc(p.title)}<em>${p.source === 'claude' ? 'new · ' : ''}effort ${esc(p.effort || '?')}</em></h4>
        <p class="hx-dim" style="margin:0">${esc(p.hypothesis)}</p>
        <dl>${Object.entries(LABELS).map(([k, l]) => p[k] ? `<dt>${l}</dt><dd>${esc(p[k])}</dd>` : '').join('')}</dl>
        <div class="row"><button class="hx-btn" data-a="more">${i === 0 ? 'Less' : 'Details'}</button><button class="hx-btn" data-a="ask">Discuss</button><button class="hx-btn" data-a="copy">Copy build request</button>
        ${(p.pages || []).filter(x => x !== PATH).slice(0, 2).map(x => `<a class="hx-btn" href="${esc(x)}">Open ${esc(x)}</a>`).join('')}</div></div>`).join('');
    box.querySelectorAll('.hx-card').forEach(card => {
      const p = ps[+card.dataset.i];
      card.querySelector('[data-a=more]').onclick = e => { card.classList.toggle('open'); e.target.textContent = card.classList.contains('open') ? 'Less' : 'Details'; };
      card.querySelector('[data-a=ask]').onclick = () => send(`Tell me more about the proposal "${p.title}": what would it show, what are the risks, and how would you build it here?`);
      card.querySelector('[data-a=copy]').onclick = async e => {
        const b = e.target;
        try { const r = await api('/api/helper/build-request', { proposal: p }); await navigator.clipboard.writeText(r.text); b.textContent = 'Copied'; }
        catch (_) { b.textContent = 'Copy failed'; }
        setTimeout(() => { b.textContent = 'Copy build request'; }, 1600);
      };
    });
  }

  window.labGuide = { open: () => toggle(true), ask: q => { toggle(true); send(q); }, tour: () => { toggle(true); load().then(() => { show('tour'); go(0); }); } };
  if (document.body) build(); else document.addEventListener('DOMContentLoaded', build);
})();
