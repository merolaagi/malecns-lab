/* Single-neuron engine for the workbench. Pure, deterministic, no DOM.
   Assumed biophysics, stated plainly:
   - Two compartments: a passive dendrite that receives all synapses, coupled to an active
     spike initiation zone (SIZ). Insect neurons are unipolar, so the soma is left off the path.
   - SIZ uses squid Hodgkin-Huxley kinetics (1952 parameters) as a placeholder. Drosophila
     channels (para Na+, Shaker/Shab K+) have different kinetics.
   - Each input fires at `rate` Hz with random timing. Each spike adds gmax * weight of
     conductance to its channel, decaying with a 5 ms time constant.
   - weight = log(1 + synapses) / log(1 + strongest input), the same count transform as the
     network models. Excitatory reversal 0 mV, inhibitory -80 mV.
   Units: mV, ms, mS/cm², µF/cm² = 1. */
(function (root) {
  const E_EXC = 0, E_INH = -80, TAU_SYN = 5, G_LEAK_D = 0.3, G_COUPLE = 0.5;

  function rng(seed) { let s = seed >>> 0; return () => { s = (s * 1664525 + 1013904223) >>> 0; return s / 4294967296; }; }

  function weights(synapses) {
    const max = Math.max(1, ...synapses);
    return synapses.map(s => Math.log1p(s) / Math.log1p(max));
  }

  function rates(V) {
    const am = Math.abs(V + 40) < 1e-6 ? 1 : 0.1 * (V + 40) / (1 - Math.exp(-(V + 40) / 10));
    const bm = 4 * Math.exp(-(V + 65) / 18);
    const ah = 0.07 * Math.exp(-(V + 65) / 20);
    const bh = 1 / (1 + Math.exp(-(V + 35) / 10));
    const an = Math.abs(V + 55) < 1e-6 ? 0.1 : 0.01 * (V + 55) / (1 - Math.exp(-(V + 55) / 10));
    const bn = 0.125 * Math.exp(-(V + 65) / 80);
    return { am, bm, ah, bh, an, bn };
  }

  // channels: [{weight, sign (+1/-1/0), active}]
  function simulate({ channels, gmax = 0.065, rate = 80, T = 200, onset = 20, seed = 7, dt = 0.01, every = 10 }) {
    const r = rng(seed), C = channels.length;
    const times = channels.map(ch => {
      if (!ch.active || !ch.sign) return [];
      const k = Math.round(rate / 1000 * (T - onset)), t = [];
      for (let j = 0; j < k; j++) t.push(onset + r() * (T - onset));
      return t.sort((a, b) => a - b);
    });
    const ix = new Array(C).fill(0), g = new Float64Array(C), dec = Math.exp(-dt / TAU_SYN);
    let Vd = -65, V = -65, m = 0.0529, h = 0.5961, n = 0.3177, up = false, peak = -65;
    const trace = { t: [], vd: [], v: [], m: [], h: [], n: [], ina: [], ik: [] }, spikes = [], events = [];
    const steps = Math.round(T / dt);
    for (let s = 0; s < steps; s++) {
      const t = s * dt;
      let Iexc = 0, Iinh = 0;
      for (let j = 0; j < C; j++) {
        while (ix[j] < times[j].length && times[j][ix[j]] <= t) { g[j] += gmax * channels[j].weight; ix[j]++; events.push([t, j]); }
        g[j] *= dec;
        if (channels[j].sign > 0) Iexc += g[j] * (E_EXC - Vd); else if (channels[j].sign < 0) Iinh += g[j] * (E_INH - Vd);
      }
      Vd += dt * (-G_LEAK_D * (Vd + 65) + Iexc + Iinh - G_COUPLE * (Vd - V));
      const k = rates(V);
      m += dt * (k.am * (1 - m) - k.bm * m); h += dt * (k.ah * (1 - h) - k.bh * h); n += dt * (k.an * (1 - n) - k.bn * n);
      const INa = 120 * m * m * m * h * (V - 50), IK = 36 * n * n * n * n * (V + 77);
      V += dt * (-INa - IK - 0.3 * (V + 54.4) - G_COUPLE * (V - Vd));
      if (V > 0 && !up) { spikes.push(t); up = true; }
      if (V < -30) up = false;
      if (Vd > peak) peak = Vd;
      if (s % every === 0) {
        trace.t.push(t); trace.vd.push(Vd); trace.v.push(V); trace.m.push(m); trace.h.push(h); trace.n.push(n);
        trace.ina.push(INa); trace.ik.push(IK);
      }
    }
    return { trace, spikes, events, peakVd: peak };
  }

  // Fraction of random input patterns (each input on with probability p) that produce >= 1 spike.
  function fireProbability(channels, { p = 0.2, trials = 60, gmax = 0.065, rate = 80, seed = 99 } = {}) {
    const r = rng(seed); let fired = 0;
    for (let k = 0; k < trials; k++) {
      const pattern = channels.map(ch => ({ ...ch, active: r() < p }));
      if (simulate({ channels: pattern, gmax, rate, seed: k + 1 }).spikes.length) fired++;
    }
    return fired / trials;
  }

  const api = { simulate, weights, fireProbability, rng, E_EXC, E_INH };
  if (typeof module !== 'undefined' && module.exports) module.exports = api; else root.NeuronCore = api;
})(typeof window !== 'undefined' ? window : globalThis);
