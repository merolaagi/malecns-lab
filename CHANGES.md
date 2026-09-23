# Changes

Newest first. `tools/sync.sh` uses the top entry as the commit message.

## Draw the arena agents as flies

- Arena agents are drawn to scale (2.5 mm body) with abdomen, thorax, head, antennae, six legs animated by each agent's own leg oscillators, and wings that fold back except when a male extends one while singing, reusing the motion lab's fly drawing.
- The arena trace now carries each agent's leg phases and strides; the view fits itself to the trajectories, and faint haloes show the odour field.

## Drive each leg joint from its own motor neurons

- `bodyplan.py` maps motor neurons to leg joints by the muscle their type names identify (coxa protractors/retractors, trochanter levator/depressors, tibia extensor/flexors, tarsus levator/depressors) and turns firing-rate differences into joint angles. All six legs can drive coxa, trochanter and tibia; tarsus levators exist only for the front legs; 59 motor neurons have no muscle name or leg and stay unused.
- `/arena` flies are drawn to scale (2.5 mm body) and posed by those joint angles, with planted feet marked, an extended wing while singing, and a joint panel per agent. New body mode "from the planted feet" derives translation and turning from foot motion instead of pooled stride averages.
- Noted on the page: compound eyes cannot move, and neck and antennal motor neurons are not in the subset, so head, eyes and antennae are static.
- Correction: the 59 motor neurons leaving by accessory nerves are leg-muscle motor neurons, so pooling them into leg stride is reasonable. The atlas flag said they were not leg motor neurons; it now names the muscle and reserves doubt for the 8 abdominal-nerve cells.
- `test_bodyplan.py` and new arena tests for joint drive and the kinematic body mode.

## Add three-fly arena; fix inverted steering in target mode

- New `/arena` page and `arena.py`: two agents labelled male and one labelled female, each running the same measured locomotion circuit, coupled by engineered odour, vision and courtship-song channels. Four scenarios (rivalry, courtship, food, threat) and sensory controls (no vision, no odour, no song, isolated, shuffled connectome, silenced VNC), with an animated replay, pairwise distance plot and saved two-seed benchmark.
- Stated plainly in the module, the page and the README: MaleCNS is one male fly with no courtship circuitry in the subset, so all three agents share one connectome and "female" is a label on engineered emissions and responses.
- Findings: odour is the channel that finds the female and the food (closest approach 0.2 mm with odour, 5.6 mm without; 3 of 3 reach food with odour, none without); vision changes little; rivalry yields 5.7 s with both males near her and about 5 s of song each. The faster food arrivals with vision are an artifact of the engineered drive rule and are labelled as such.
- Bug fix: the motion lab's target mode steered away from its target because the bearing-to-bias sign was inverted relative to the body readout. `model.steer()` now holds the correct sign and is shared by the motion lab and the arena; target mode reaches within 1.5 mm over 20 s.
- `test_arena.py`: steering sign, target-mode regression, reproducibility, scenario behaviour, sensory controls, silenced VNC and validation.

## Add theory page: four predictions tested against simulation

- `theory.py` and `/theory`: states the sparse-expansion model (fixed expansion, top-k code, associative readout as a kernel sum) and tests four predictions with thresholds fixed in advance.
- P1 refuted as first stated and replaced: forgetting under the error-driven rule does not track code overlap (r = 0.27); normalising class weights at test time cuts it by up to 0.29, so recency bias dominates. New `DeltaNormalised` readout isolates the two mechanisms.
- P2 supported: two-shape classes peak at an interior sparsity (0.01 dense, 0.02 measured).
- P3 supported: dense random codes follow the exact bivariate-normal overlap curve (RMSE 0.028); measured wiring deviates (RMSE 0.098) because its weights are non-negative with uneven in-degrees.
- P4 supported for f ≥ 0.05 and broken at f = 0.01, with the measured wrong-score skew (+0.71, +0.25, −0.01) explaining where and why.
- `test_theory.py`: analytic endpoints and monotonicity, the normalised readout, strict-JSON results and regression checks on each conclusion.

## Add sparse-expansion continual-learning experiment

- New `/expansion` page and `expansion.py`: class-incremental learning on bundled 8×8 digits, comparing the measured PN→KC expansion with 5% active cells and a reward-gated associative readout against shuffled, random sparse and dense random wiring, no sparsity, no expansion, an error-driven readout and a backpropagation MLP, plus an all-at-once upper bound. Two task variants (single-digit and two-shape classes) and a few-shot setting.
- Benchmark (five seeds, three conditions) in `data/expansion-benchmark.json`. Findings: the associative rule prevents catastrophic forgetting (about 5% vs 84–100%) even on raw pixels; sparse expansion helps only for classes with several shapes (about +6 points); measured wiring performs like shuffled and random wiring.
- `data/digits.npz` (UCI handwritten digits, 45 KB) bundled so the experiment runs offline.
- `test_expansion.py`; route tests for the new page and the 3D assets.

## Fix region and outline meshes in the 3D view

- Neuron meshes worked, but region shapes and one outline piece failed with "Invalid path": the proxy's character allow-list rejected fragment names listed in real mesh manifests. Directories stay strictly checked; file names may now hold any printable characters except slashes and `..`, and are URL-encoded when fetched and when cached.
- Path errors now include the refused path, and the 3D page shows each item's failure or fallback reason under its row instead of only in a status line that later messages overwrote.

## Simplify large meshes on the server

- The probe confirmed every viewer layer uses the legacy precomputed mesh format and that bucket region labels match the lab's names; the only failure was DNa02's full-resolution mesh, a single file over 25 MB.
- `meshes.py` and `/api/mesh/<layer>/<id>`: fetch all fragments once (raw files up to 400 MB, not cached), merge, simplify by vertex clustering to per-layer triangle targets, and cache only the simplified mesh. One build per mesh even under concurrent requests.
- The 3D page now loads all meshes, neurons, regions and outlines, through this endpoint and shows the simplification in each item's tooltip.
- `test_meshes.py`: round-trip encoding, simplification to target with shape preserved and no broken triangles, fragment assembly and caching, input checks.

## Add 3D view and Neuroglancer links

- New `/3d` page: official neuron meshes (skeleton fallback) and brain-region shapes from the public MaleCNS bucket, with brain and VNC outlines, presets for the steering descending neurons and their input regions, and URL-encoded state. three.js r128 is bundled locally.
- `gcs.py` and `/api/gcs/...`: whitelisted, validated, size-capped, cached read-only proxy to `gs://flyem-male-cns`, handling gzip-encoded objects.
- "View in 3D" and "Neuroglancer ↗" links on atlas cells, workbench cells and region panels; `ng.js` builds official-viewer links and decodes legacy precomputed meshes.
- `tools/probe_ng.py` records the formats of the layers the viewer uses in `data/ng-probe.json`.
- Tests: `test_gcs.py` (path rules, gzip, caching, errors), `ng.test.js` (mesh decoding, unit conversion, region labels, link round-trip), and a route test that the proxy refuses other paths.

## neuPrint-style region matrix with hover details

- `/regions` hover box for every cell: "A → B", flow, neuron count, what share of A-driven output lands in B, what share of B's output is driven from A, and the reverse direction. The hovered row and column are highlighted.
- Regions ordered by neuPrint's region hierarchy by default, with dividers between top-level divisions (central brain, optic lobes, ventral nerve cord); name and activity orders and a linear/log colour switch are available. Every region is labelled, and the matrix uses the full page width.
- `build_regions.py` now also counts, for each region pair, the traced neurons with input in A and output in B, the second number neuPrint shows. Rebuild to add it; the page works without it.

## Fix silent sync exit; let local caches coexist with syncs

- `tools/sync.sh` silently stopped whenever an iteration removed no files: an `[ -n "$f" ] && ...` test in the deletion loop returned failure on the empty list, and `set -e` ended the script. Since the data-protection change, every such sync, including autosync's, stopped before applying anything. Fixed, with the same pattern removed elsewhere.
- Sync now only refuses when tracked files have local edits. Untracked files, such as skeletons the neuron explorer caches, no longer block it unless the incoming zip would overwrite them.
- Explorer-downloaded skeletons in `data/skeletons/` are git-ignored except the bundled one.

## Add brain-region connectivity from neuPrint

- `build_regions.py` reads every traced neuron's region profile from neuPrint (keyset-paginated custom query, token from `NEUPRINT_APPLICATION_CREDENTIALS`) and writes `data/regions.json`: per-region input and output totals, a neuron-routed region-to-region flow matrix over primary regions, and input/output region profiles for every lab cell.
- New `/regions` page: interactive flow matrix with a panel for each region showing what it drives, what drives it, and which lab cells work there. Atlas cell panels and the neuron workbench show each cell's top input and output regions.
- `test_regions.py` checks pagination, flow arithmetic, exclusion of overlapping super-regions and lab profiles against a fake neuPrint.
- Optional `requirements-regions.txt` (neuprint-python). "Brain regions" added to every page's navigation.

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
