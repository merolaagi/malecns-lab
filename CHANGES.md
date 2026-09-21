# Changes

Newest first. `tools/sync.sh` uses the top entry as the commit message.

## Guard sync against backup zips

- `tools/sync.sh` and `tools/autosync.sh` now only pick up `malecns-lab.zip` and browser-renamed `malecns-lab (N).zip`, not any file starting with `malecns-lab`. A backup named `malecns-lab_v1.zip` had matched and replaced the project.
- Sync refuses zips without `CHANGES.md` and `tools/sync.sh`, and zips that would delete more than three tracked files, unless `MALECNS_FORCE=1` is set.

## Add automatic GitHub sync on download

- `tools/autosync.sh install` starts a macOS launchd agent that watches Downloads. Each new `malecns-lab*.zip` is checked, applied, tested, committed with its CHANGES.md entry and pushed, with a notification for the result. `watch` runs the same job in a Terminal tab; `status` and `uninstall` manage it.
- `tools/sync.sh` now refuses to apply a zip whose newest CHANGES entry is older than the committed one, and archives older leftover downloads after a successful sync so they can never be applied later.

## Add neuron workbench and numerosity experiment

- New `/neuron` workbench: opens any cell from either circuit with its measured inputs and simulates a two-compartment Hodgkin–Huxley membrane (dendrite, spike initiation zone, soma off-path) with voltage, gating and current traces. Kenyon cells can be driven by the numerosity codes. Linked from every atlas cell panel.
- New `/numerosity` experiment (`numerosity.py`): choose the larger number on the measured PN→KC→MBON circuit, comparing an HTM-style scalar SDR code with a one-hot code, shuffled wiring and no plasticity, across interpolation, middle-block and extrapolation held-out designs. Benchmark saved in `data/numerosity-benchmark.json`.
- Finding: overlapping codes interpolate to unseen numbers (100% vs 48% one-hot), fail on wide gaps, and reverse on extrapolation (10%). Shuffled wiring does as well as measured wiring.
- `/api/neuron` and `/api/numerosity` endpoints; `neuron_inputs.py`, `neuron-core.js`; tests in `test_numerosity.py`, `test_neuron.py`, `neuron-core.test.js`.

## Add circuit atlas and GitHub sync tooling

- New `/atlas` page: circuit, layer and anatomy views of all 807 cells and 21,161 measured edges, with per-cell, per-group and per-region panels that flag known modeling problems.
- `atlas-core.js` with node tests, and `test_atlas.py` route tests.
- `tools/setup_repo.sh` creates the GitHub repo; `tools/sync.sh` applies each downloaded iteration, runs tests, commits and pushes.
- `.gitignore` now excludes raw MaleCNS tables and `.DS_Store`.
- `CHANGES.md` added.

## Baseline: locomotion model and odor-learning lab

- Sparse LIF model on a MaleCNS locomotion subset driving an engineered six-leg readout.
- Trial-based odor-learning rate model on measured PN, KC, MBON and PAM cells.
