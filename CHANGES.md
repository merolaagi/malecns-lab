# Changes

Newest first. `tools/sync.sh` uses the top entry as the commit message.

## Merge vision, explorer and math modules; add trainable compound-eye layer

- Merged the separate project copy into the repo: the trainable visual graph (`/vision`, 209 measured right-eye cells across 19 columns), the neuron explorer with skeleton and membrane models (`/explore`), and the math classroom and SDR arithmetic experiments (`/math`), with their data, saved weights and tests. Its `build_vision.py` and `test_vision.py` are renamed `build_vision_patch.py` and `test_vision_patch.py`, since the repo's vision bridge already uses those names.
- New trainable eye in front of the measured graph: shared photoreceptor adaptation and gain, then a learned column projection that is either retinotopic (own column plus six hex neighbours) or unrestricted. It starts as an exact identity; eye gradients are checked against finite differences.
- Vision benchmark regenerated for six architectures and three seeds. The four existing architectures reproduce the previous benchmark exactly. The retinotopic eye improves hazard calibration (Brier 0.057 vs 0.064), the unrestricted eye improves motion (13.7° vs 16.5° direction error); the learned adaptation stays near zero.
- The saved-weights reproduction test now allows 1e-4 instead of 1e-6, since float32 results differ by about 4e-6 across machines and would otherwise fail on the Mac.
- One navigation bar across all eight pages.

## Protect generated data during sync; restore eye responses

- A sync deleted `data/eye-responses.json` because the user pushed it after that iteration's zip was built. `tools/sync.sh` now never deletes files under `data/`; it lists the ones it keeps. Other files removed by an iteration are still deleted, and the safety limit on deletions still applies.
- `data/eye-responses.json` (pretrained flyvis run, five models) restored from history.
- README records the eye result: six of eight direction checks pass; the horizontal detectors match anatomy, the vertical pair is less clean.

## Fix simulations behind the HTTPS tunnel

- The server's same-origin check only accepted `http://<host>`, so behind the Cloudflare tunnel (`https://malecns.fueldeskpro.com`) every simulation request was refused with "Origin not allowed" and the motion lab had nothing to animate. It now accepts `http://` and `https://` for the request's own host and still refuses other sites.
- Route tests cover the local origin, the HTTPS proxy origin, and cross-site requests.

## Keep consensus transmitters; flag per-synapse disagreements

- Real data showed per-synapse transmitter predictions fail systematically for Kenyon cells (4,060 of 4,064 predicted dopamine) and motor neurons (mostly histamine or acetylcholine), while agreeing with the consensus for all other locomotion cells.
- `build_quality.py` now also records each cell's per-body consensus and predicted transmitter and whether its synapses agree. The consensus stays the model's transmitter.
- The workbench takes learning-circuit input signs from the consensus. Using per-synapse calls would have zeroed Kenyon cell input to MBONs.
- Atlas flags now describe disagreements instead of suggesting that unclear motor neurons were resolved.
- README records the first coverage and transmitter results.

## Add data quality layer: coverage and per-synapse transmitters

- `build_quality.py` streams the official body-statistics and per-synapse transmitter tables for the lab's cells only and writes `data/quality.json`: input and output coverage of each cell by the lab's subset, and per-synapse transmitter shares and sign probabilities.
- Atlas cell panels and the neuron workbench show coverage and transmitter evidence, flagging low coverage, mixed evidence and "unclear" cells whose synapses mostly agree. `/api/quality` serves the file.
- Workbench PN inputs take their sign from per-synapse predictions when available, replacing the all-excitatory assumption.
- `test_quality.py` checks coverage arithmetic, transmitter aggregation, filtering to lab cells and summary counts on synthetic tables.

## Make autosync install reliable

- `tools/autosync.sh install` validates the agent file, waits until launchd has fully unloaded the previous agent, and retries loading up to three times. Reinstalling immediately after an unload previously failed with "Bootstrap failed: 5: Input/output error".

## Restart the lab server after every sync

- `tools/lab.sh start|stop|restart|status|open|logs` runs the lab server in the background, logging to `.server.log` and tracking its PID in `.server.pid` (both git-ignored). It only ever stops the server it started.
- `tools/sync.sh` restarts the server after each successful sync, so every iteration goes live with its commit. Set `MALECNS_NO_RESTART=1` to skip.
- Autosync's launchd agent now keeps the restarted server alive after the sync job exits (`AbandonProcessGroup`), and its notification includes the lab URL. Rerun `tools/autosync.sh install` once to pick this up.

## Add eye-response runner (vision milestone 2)

- `vision_eye.py` runs pretrained flyvis eye models on gratings in eight directions, a looming disc and flashes, and saves compact responses of the 34 output cell types to `data/eye-responses.json`, with motion tuning per T4/T5 subtype and anatomical-consistency checks against T4a. Downloads pretrained models into git-ignored `flyvis-data/` on first run.
- Optional `requirements-vision.txt` (flyvis 1.2.0, PyTorch) keeps the core lab install unchanged.
- `test_vision_eye.py` checks stimulus geometry and drift direction, tuning math, and an optional end-to-end smoke test with an untrained network.
- `data/vision-bridge.json` (built from MaleCNS on the user's machine) now ships with the project, and the README summarizes the bridge result.

## Add vision-to-steering bridge extraction

- `build_vision.py` extracts, from the full MaleCNS raw tables, the anatomical routes from the 34 flyvis output cell types to the steering descending neurons DNa01, DNa02 and DNg13, at the (cell type, side) level, with up to three hops. Influence is the product of input fractions and ranks routes; it does not predict activity. Output goes to `data/vision-bridge.json`.
- `test_vision.py` checks the path arithmetic, denominators that include untyped inputs, intermediate ranking and provenance on synthetic tables.
- README: "Vision to walking" plan with four milestones and known caveats.

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
