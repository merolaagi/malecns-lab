/* Ink skin for MaleCNS Lab: shell (header, section tabs, rail, search, "On this page") and canvas colours.
   Loaded synchronously in <head>, before any page script, so the canvas colour map is in place before the
   pages draw. The pages' own code and markup are unchanged; their old header is hidden by ink-skin.css. */
(function () {
  if (typeof window === 'undefined' || typeof document === 'undefined') return;

  // --- canvas colours: the pages draw a dark theme with hard-coded hexes; map them to the light Ink palette ---
  const HEX = {
    '#0d130f': '#ffffff', '#121914': '#ffffff', '#101411': '#ffffff', '#0b110d': '#ffffff', '#132018': '#ffffff',
    '#171d19': '#f3f3f1', '#18221b': '#f3f3f1', '#132015': '#f3f3f1', '#1b2a20': '#f3f3f1', '#202b24': '#f0f0ee',
    '#2b372e': '#e6e6e3', '#1e2a21': '#ececea', '#253228': '#e6e6e3', '#243229': '#e6e6e3', '#3a473d': '#d5d5d1', '#364b2b': '#d9e2f6',
    '#e7eee5': '#262626', '#d6e2d8': '#262626', '#ffffff': '#262626', '#fff': '#262626',
    '#99aa9c': '#6e6e6e', '#aab7ad': '#6e6e6e', '#a9b6ad': '#6e6e6e', '#6c806f': '#8a8a8a', '#6b7c6e': '#8a8a8a',
    '#5b6a5e': '#9a9a9a', '#55635a': '#9a9a9a',
    '#c2e899': '#2f5fd0', '#a9d68a': '#2f5fd0', '#8fc47c': '#2552b5', '#46602f': '#9fb6ea', '#15200e': '#ffffff',
    '#6fc7bc': '#2aa0a0', '#8ab4f8': '#4aa7ee', '#f28b82': '#d0524a', '#e8907a': '#d0524a', '#f0a07a': '#d0524a', '#6d2f2b': '#f0c4c0',
    '#e9be75': '#e9a23b', '#f2c46f': '#e9a23b', '#6d5a2c': '#f0d9a8', '#d4a5f5': '#8150e6', '#a9abed': '#8150e6', '#6fb07a': '#3f9a70',
  };
  const RGB = {
    '231,238,229': '0,0,0', '194,232,153': '47,95,208', '153,170,156': '110,110,110', '242,139,130': '208,82,74',
    '111,199,188': '42,160,160', '233,190,117': '233,162,59', '138,180,248': '74,167,238', '0,0,0': '0,0,0',
  };
  const map = v => {
    if (typeof v !== 'string') return v;
    const s = v.trim().toLowerCase();
    // Pages where green means "correct" or "positive" (data-ink-green="status") keep it green, as a status colour.
    if ((s === '#c2e899' || s === '#a9d68a') && document.documentElement.dataset.inkGreen === 'status') return '#1f8a4c';
    if (HEX[s]) return HEX[s];
    const m = s.match(/^rgba?\(\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*(,.*)?\)$/);
    if (m) {
      const key = `${m[1]},${m[2]},${m[3]}`;
      if (key === '0,0,0' && m[4]) return `rgba(255,255,255${m[4]})`;         // dark scrims become light ones
      if (key === '194,232,153' && document.documentElement.dataset.inkGreen === 'status') return `rgba(31,138,76${m[4] || ',1'})`;
      if (RGB[key]) return `rgba(${RGB[key]}${m[4] || ',1'})`;
    }
    return v;
  };
  const patch = (proto, prop) => {
    const d = proto && Object.getOwnPropertyDescriptor(proto, prop);
    if (!d || !d.set) return;
    Object.defineProperty(proto, prop, { configurable: true, get: d.get, set(v) { d.set.call(this, map(v)); } });
  };
  if (window.CanvasRenderingContext2D) {
    for (const p of ['fillStyle', 'strokeStyle', 'shadowColor']) patch(CanvasRenderingContext2D.prototype, p);
  }
  // Sans-serif in figures too: the pages ask for Georgia, serif or monospace in a few labels.
  const fd = window.CanvasRenderingContext2D && Object.getOwnPropertyDescriptor(CanvasRenderingContext2D.prototype, 'font');
  if (fd && fd.set) {
    const SANS = 'Inter, -apple-system, "Segoe UI", Helvetica, Arial, sans-serif';
    Object.defineProperty(CanvasRenderingContext2D.prototype, 'font', { configurable: true, get: fd.get, set(v) {
      fd.set.call(this, typeof v === 'string' ? v.replace(/(Georgia\s*,\s*)?\bserif\b|Georgia|\bmonospace\b|\bsans-serif\b/g, SANS) : v);
    } });
  }
  if (window.CanvasGradient) {
    const add = CanvasGradient.prototype.addColorStop;
    CanvasGradient.prototype.addColorStop = function (o, c) { return add.call(this, o, map(c)); };
  }

  // --- pages, grouped into sections (each becomes a header tab) ---
  const SECTIONS = [
    ['Motion & body', [
      ['/', 'Motion lab', 'walk'], ['/arena', 'Arena', 'arena'], ['/arena3d', '3D arena', 'cube'],
      ['/body', 'Body studio', 'fly'], ['/optogenetics', 'Virtual optogenetics', 'bolt']]],
    ['Cells & circuits', [
      ['/neuron', 'Neuron workbench', 'neuron'], ['/synapse', 'Synapses', 'synapse'], ['/explore', 'Inside the circuit', 'graph'],
      ['/atlas', 'Circuit atlas', 'network'], ['/regions', 'Brain regions', 'brain'], ['/3d', '3D viewer', 'cube']]],
    ['Learning & memory', [
      ['/learning', 'Odour learning', 'lab'], ['/memory', 'Memory', 'layers'], ['/objects', 'Object memory', 'cup'],
      ['/numerosity', 'Numerosity', 'hash'], ['/math', 'Math classroom', 'sigma']]],
    ['Vision & navigation', [
      ['/vision', 'Vision network', 'eye'], ['/navigation', 'Navigation', 'compass']]],
    ['Theory', [
      ['/expansion', 'Sparse expansion', 'fan'], ['/theory', 'Theory', 'idea'], ['/spectrum', 'Spectra', 'wave'],
      ['/inhibition', 'Inhibition', 'minus'], ['/scaffold', 'Scaffold learner', 'layers'], ['/backbone', 'Connectome backbone', 'network']]],
  ];
  const ICONS = {
    walk: '<circle cx="13" cy="4.5" r="1.8"/><path d="m9 21 2.5-6.5L14 17v4M8 11l3-3.5 3 1 2.5 3M11 7.5 11.5 14.5"/>',
    arena: '<circle cx="12" cy="12" r="9"/><path d="M8 15c1.5-4 4.5-6 8-6"/><circle cx="16" cy="9" r="1.4"/>',
    cube: '<path d="m12 3 8 4.5v9L12 21l-8-4.5v-9z"/><path d="m4 7.5 8 4.5 8-4.5M12 12v9"/>',
    fly: '<ellipse cx="12" cy="13" rx="2.6" ry="5.5"/><circle cx="12" cy="6" r="1.8"/><path d="M9.6 10.5 4 8.5M14.4 10.5 20 8.5M9.6 13 5 15M14.4 13 19 15M10 16.5 7 20M14 16.5 17 20"/>',
    bolt: '<path d="M13 2 5 13.5h6L10 22l8-11.5h-6z"/>',
    neuron: '<circle cx="12" cy="12" r="3"/><path d="M12 9V3M9.5 13.5 4 17M14.5 13.5 20 17M9 11 4.5 8M15 11l4.5-3M12 15v6"/>',
    synapse: '<path d="M4 7c3 0 5 2 5 5s-2 5-5 5"/><path d="M20 7c-3 0-5 2-5 5s2 5 5 5"/><path d="M11.5 10.5h1M11.5 13.5h1"/>',
    graph: '<circle cx="6" cy="6" r="2.5"/><circle cx="18" cy="6" r="2.5"/><circle cx="12" cy="18" r="2.5"/><path d="M8.3 7.2 10.8 16M15.7 7.2 13.2 16M8.5 6h7"/>',
    network: '<rect x="9" y="2.5" width="6" height="5" rx="1"/><rect x="2.5" y="16.5" width="6" height="5" rx="1"/><rect x="15.5" y="16.5" width="6" height="5" rx="1"/><path d="M12 7.5V12M5.5 16.5V12h13v4.5"/>',
    brain: '<path d="M9 4a3 3 0 0 0-3 3 3 3 0 0 0-2 5 3 3 0 0 0 2 5 3 3 0 0 0 6 1V5a2 2 0 0 0-3-1z"/><path d="M15 4a3 3 0 0 1 3 3 3 3 0 0 1 2 5 3 3 0 0 1-2 5 3 3 0 0 1-6 1"/>',
    lab: '<path d="M9 3h6M10 3v6L4.5 18.5A1.7 1.7 0 0 0 6 21h12a1.7 1.7 0 0 0 1.5-2.5L14 9V3"/><path d="M7 15h10"/>',
    layers: '<path d="m12 3 9 5-9 5-9-5z"/><path d="m3 13 9 5 9-5"/>',
    cup: '<path d="M5 6h11l-1.3 13a2 2 0 0 1-2 1.8H8.3a2 2 0 0 1-2-1.8z"/><path d="M16 9h1.5a2.5 2.5 0 0 1 0 5H15.5"/>',
    hash: '<path d="M5 9h15M4 15h15M10 3 8 21M16 3l-2 18"/>',
    sigma: '<path d="M18 5H6l6 7-6 7h12"/>',
    eye: '<path d="M2.5 12S6 5.5 12 5.5 21.5 12 21.5 12 18 18.5 12 18.5 2.5 12 2.5 12z"/><circle cx="12" cy="12" r="3"/>',
    compass: '<circle cx="12" cy="12" r="9"/><path d="m15.5 8.5-2 5-5 2 2-5z"/>',
    fan: '<circle cx="4" cy="12" r="1.6"/><path d="M5.5 12 19 4M5.5 12 19 9.3M5.5 12H19M5.5 12 19 14.7M5.5 12 19 20"/>',
    idea: '<path d="M9 18h6M10 21h4"/><path d="M12 3a6 6 0 0 0-3.6 10.8c.7.6 1.1 1.3 1.1 2.2h5c0-.9.4-1.6 1.1-2.2A6 6 0 0 0 12 3z"/>',
    wave: '<path d="M3 12c2-6 4-6 6 0s4 6 6 0 4-6 6 0"/>',
    minus: '<circle cx="12" cy="12" r="9"/><path d="M8 12h8"/>',
    search: '<circle cx="11" cy="11" r="7"/><path d="m20 20-3.5-3.5"/>',
    dot: '<circle cx="12" cy="12" r="3"/>',
  };
  const icon = (n, cls = 'ink-i') => `<svg class="${cls}" viewBox="0 0 24 24" aria-hidden="true">${ICONS[n] || ICONS.dot}</svg>`;
  const esc = s => String(s ?? '').replace(/[&<>"]/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c]));
  const store = { get(k) { try { return localStorage.getItem(k); } catch (_) { return null; } }, set(k, v) { try { localStorage.setItem(k, v); } catch (_) {} } };

  const here = location.pathname.replace(/\/+$/, '') || '/';
  let current = SECTIONS.findIndex(([, pages]) => pages.some(([p]) => p === here));
  if (current < 0) current = 0;
  store.set('ink-last-' + current, here);

  function build() {
    if (document.querySelector('.ink-header')) return;
    const header = document.createElement('header');
    header.className = 'ink-header';
    header.innerHTML = `<a class="ink-brand" href="/"><b>MaleCNS Lab</b><small>fruit-fly connectome, circuit to behaviour</small></a>
      <div class="ink-search">${icon('search', '')}<input type="search" placeholder="Search pages" aria-label="Search pages"><kbd>${/Mac|iP/.test(navigator.platform) ? '⌘' : 'Ctrl'} K</kbd><div class="ink-results"></div></div>
      <div class="ink-tabs" role="tablist">${SECTIONS.map(([name], i) => `<button role="tab" aria-selected="${i === current}" data-i="${i}">${esc(name)}</button>`).join('')}</div>`;
    const rail = document.createElement('div');
    rail.className = 'ink-rail';
    const [name, pages] = SECTIONS[current];
    rail.innerHTML = `<div class="grp">${esc(name)}</div>` + pages.map(([p, label, ic]) =>
      `<a href="${p}"${p === here ? ' class="on" aria-current="page"' : ''}>${icon(ic)}<span>${esc(label)}</span></a>`).join('');
    document.body.prepend(header, rail);

    header.querySelectorAll('.ink-tabs button').forEach(b => b.addEventListener('click', () => {
      const i = +b.dataset.i; location.href = store.get('ink-last-' + i) || SECTIONS[i][1][0][0];
    }));

    // search: every page, plus this page's headings (read when searching, since the page may still be loading)
    const input = header.querySelector('.ink-search input'), out = header.querySelector('.ink-results');
    const pageEntries = SECTIONS.flatMap(([sec, ps]) => ps.map(([p, label]) => ({ title: label, sub: sec, go: () => { location.href = p; } })));
    const headingEntries = () => headings().map(h => ({ title: h.textContent.trim(), sub: 'On this page', go: () => h.scrollIntoView({ behavior: 'smooth' }) }));
    let hits = [], sel = 0;
    const render = () => {
      const q = input.value.trim().toLowerCase();
      hits = q ? pageEntries.concat(headingEntries()).filter(e => (e.title + ' ' + e.sub).toLowerCase().includes(q)).slice(0, 10) : [];
      sel = 0;
      out.innerHTML = hits.map((e, i) => `<div data-i="${i}"${i ? '' : ' class="on"'}>${esc(e.title)}<small>${esc(e.sub)}</small></div>`).join('');
    };
    input.addEventListener('input', render);
    input.addEventListener('keydown', e => {
      if (e.key === 'ArrowDown' || e.key === 'ArrowUp') {
        e.preventDefault(); if (!hits.length) return;
        sel = (sel + (e.key === 'ArrowDown' ? 1 : hits.length - 1)) % hits.length;
        out.querySelectorAll('div').forEach((d, i) => d.classList.toggle('on', i === sel));
      } else if (e.key === 'Enter' && hits[sel]) { hits[sel].go(); input.value = ''; render(); input.blur(); }
      else if (e.key === 'Escape') { input.value = ''; render(); input.blur(); }
    });
    out.addEventListener('mousedown', e => { const d = e.target.closest('[data-i]'); if (d) { e.preventDefault(); hits[+d.dataset.i].go(); input.value = ''; render(); } });
    input.addEventListener('blur', () => setTimeout(() => { out.innerHTML = ''; }, 150));
    addEventListener('keydown', e => {
      if ((e.metaKey || e.ctrlKey) && e.key.toLowerCase() === 'k') { e.preventDefault(); input.focus(); input.select(); }
    });
  }

  function headings() {
    return [...document.querySelectorAll('main h2, body > section h2')]
      .filter(h => h.textContent.trim() && !/^loading/i.test(h.textContent.trim()) && !h.closest('.ink-header'));
  }

  // "On this page": the page's own h2 headings, once the page is parsed
  function buildToc() {
    if (document.querySelector('.ink-toc')) return;
    const heads = headings();
    if (heads.length >= 3) {
      heads.forEach((h, i) => { if (!h.id) h.id = 'sec-' + i + '-' + h.textContent.trim().toLowerCase().replace(/[^a-z0-9]+/g, '-').slice(0, 40); });
      const toc = document.createElement('div');
      toc.className = 'ink-toc';
      toc.innerHTML = '<b>On this page</b>' + heads.map(h => `<a href="#${h.id}">${esc(h.textContent.trim())}</a>`).join('');
      document.body.append(toc);
      document.body.classList.add('ink-has-toc');
      const links = [...toc.querySelectorAll('a')];
      const spy = () => {
        let k = 0; heads.forEach((h, i) => { if (h.getBoundingClientRect().top < 160) k = i; });
        links.forEach((a, i) => a.classList.toggle('on', i === k));
      };
      addEventListener('scroll', spy, { passive: true }); spy();
    }
  }

  // Page scripts sit at the end of <body>; build as soon as <body> exists so the shell is there before they run.
  if (document.body) build();
  else {
    const mo = new MutationObserver(() => { if (document.body) { mo.disconnect(); build(); } });
    mo.observe(document.documentElement, { childList: true });
  }
  document.addEventListener('DOMContentLoaded', () => { build(); buildToc(); });

  // The lab guide (tour, explain, ask, propose) on every page.
  if (!document.querySelector('link[href="/helper.css"]')) {
    const css = document.createElement('link'); css.rel = 'stylesheet'; css.href = '/helper.css';
    const js = document.createElement('script'); js.src = '/helper.js'; js.defer = true;
    document.head.append(css, js);
  }
})();
