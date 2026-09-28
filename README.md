# MaleCNS Lab — circuit to motion

A runnable first research prototype: **807 anatomically identified neurons, 21,161 directed measured connections, and 329,230 anatomical synapses**, extracted from MaleCNS v1.0. A sparse leaky integrate-and-fire (LIF) model drives a simplified six-leg body readout in a local experiment UI.

This is a **real anatomical subset with assumed dynamics**. It is not a whole-brain emulation, recovered natural gait, or demonstration of an advance over published work. It implements the infrastructure for testing that research direction.

## Run locally

Python 3.11 or later is recommended. From this directory:

```sh
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/python server.py
```

Open the URL printed by the server. It listens only on `127.0.0.1`; the operating system chooses an available port on first startup. The port is saved in `.runtime.json`, checked by binding on restart, and replaced with a free port if occupied. No other listener is stopped.

In the original workspace, packages are already installed. Start with:

```sh
../../work/venv/bin/python server.py
```

The UI starts an 8-second experiment. Adjust the descending input, left/right bias, synaptic gain or sensory feedback. Run intact, VNC-silenced, left-DN-silenced, no-feedback, no-stimulus and shuffled controls. Replay, scrub, compare sessions and export the complete run as JSON.

Exports include the exact parameters, anatomical source hashes, neuron IDs in firing-rate order, trajectory and sampled raster. The raster is deliberately capped and subsampled for export size; `neuron_hz` and total spike counts use all steps. The visual fly and plot replay saved results; dragging playback does not rerun the model.

## Repository and iterations

The project is versioned with git and published with `tools/setup_repo.sh` (default `merolaagi/malecns-lab`, private; set `MALECNS_VISIBILITY=public` to change). Each new iteration arrives as a `malecns-lab*.zip` in `~/Downloads`. `tools/sync.sh` applies the newest one to `~/Sites/malecns-lab`, removes files the iteration dropped, leaves untracked local files such as `raw-data/` and `.venv` alone, runs the full test suite, commits with the top entry of `CHANGES.md` as the message, pushes, and moves the zip to `~/Downloads/malecns-lab-applied/`. It refuses to run over uncommitted local edits and commits nothing if tests fail. It also skips a zip older than the committed version and archives leftover older downloads. Files under `data/` are never deleted by a sync, because results generated on this machine (bridge, eye responses, quality) can be pushed after an iteration zip was built. After each successful sync the lab server is restarted in the background (`tools/lab.sh start|stop|restart|status|open|logs`; set `MALECNS_NO_RESTART=1` to skip). `tools/autosync.sh install` runs all of this automatically whenever a new zip lands in Downloads (macOS launchd agent, log in `~/Library/Logs/malecns-lab-autosync.log`); `tools/autosync.sh watch` does the same in a Terminal tab if macOS blocks background access to Downloads. Paths can be overridden with `MALECNS_PROJECT` and `MALECNS_DOWNLOADS`.

## What is measured

Source: [MaleCNS v1.0 downloads](https://male-cns.janelia.org/download/). Original tables: neuron annotations, connection counts, and neurotransmitter predictions. Each source URL and SHA-256 is in `data/circuit.json`. The anatomical data is [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). Attribution: FlyEM / HHMI Janelia, University of Cambridge, MRC Laboratory of Molecular Biology, and Google Research; see the [MaleCNS project and linked publication](https://male-cns.janelia.org/).

The subset is selected by anatomy, before evaluating simulated behavior:

1. Six Traced DNa01, DNa02 and DNg13 descending neurons.
2. Up to 80 strongest VNC intrinsic/leg motor targets per seed, requiring at least five synapses for selection.
3. Leg motor targets of these selected cells with at least five synapses.
4. Up to 20 strongly connected proprioceptive sensory cells per leg and side, annotated as chordotonal organ or campaniform sensilla, in ProLN/MesoLN/MetaLN.
5. All positive measured connections among the selected cells, including counts below the selection threshold.

There are 312 selected motor cells and 91 selected sensory cells. Truncating the circuit removes external inputs and feedback loops; this is a major scientific limitation.

The small extracted dataset is bundled, so the application does not need internet or a neuPrint account. To reproduce it from official raw tables (~1.1 GB):

```sh
.venv/bin/python fetch_data.py /path/to/raw-data
# If tables are already downloaded as annotations.feather, weights.feather,
# neurotransmitters.feather:
.venv/bin/python build_dataset.py /path/to/raw-data
```

## What is assumed

**Neural dynamics:** 1 ms time step, 20 ms membrane constant, threshold 1, reset 0, two-step refractory counter, 10 ms synaptic trace, 50 ms firing-rate filter. The voltage update is `v += dt/tau * (-v + external + gain*W@syn + noise)`. All cells use the same dynamics; this is not cell-type-specific physiology. Synapses increment their source trace by one on a spike; trace decay is exponential. Noise is independent Gaussian current with standard deviation 0.015 per step.

**Weights:** `log(1 + synapse_count)`, multiplied by an assumed transmitter sign and divided by total absolute incoming weight within the subset. Acetylcholine is positive; GABA and glutamate are negative inside the CNS. Other/unresolved transmitters contribute zero. These rules are explicit approximations: transmitter predictions do not fully determine receptor-specific effects. Counts are not measured physiological conductances. Normalization is computed before silencing, so ablations do not renormalize remaining inputs.

**Stimulation:** selected descending cells receive current after 0.5 seconds. Pulse mode stops at 1.5 seconds. Target mode computes target bearing from exact world coordinates and injects a hand-designed signal into descending neurons. It does not model visual perception, smell, feeding, or food recognition. Right-side bias specifies stimulus side, not a guaranteed turning direction.

**Body:** motor firing is pooled separately for each leg, using annotated side and leg subclass. Six engineered phase oscillators run at up to 5 Hz. Motor rate/50 sets oscillator speed and a bounded stride value. Mean left/right stride controls forward speed and rotation in an unbounded 2D plane. The fly drawing is a replay of those engineered cycles. There are no muscles, force balance, collision physics, or wing aerodynamics.

**Feedback:** annotated sensory cells receive a hand-designed stance-times-stride current. The stance signal comes from the engineered oscillator; it is not measured load or joint angle. This closes a computational feedback loop without claiming a realistic proprioceptive model.

**Default selection:** gain 10 was chosen after exploratory inspection at 7, 10, 14, 18 and 20 to produce visible, mostly unsaturated motor activity. It was not fitted to recordings or chosen to recover a biological effect. Feedback gain 2 was selected after inspecting 0, 0.6, 1.5 and 2 because lower values did not produce an observable feedback effect at the default synaptic gain. This is an engineering choice, not a physiological estimate. Higher gains can create excessive firing and saturate the body readout; the comparison table exposes clipping.

## Tests and experiments

```sh
.venv/bin/python -m unittest -v test_model.py
.venv/bin/python benchmark.py
```

The tests check data integrity, deterministic replay, silence enforcement, zero-gain motor isolation, negative controls, shuffled controls, transmitter handling, finite exports and input validation. They validate software behavior only.

`data/benchmark.json` contains five seeds for each of six conditions at the default settings, plus gain sensitivity at 7, 10 and 14. These are exploratory model experiments, not biological accuracy measurements. The shuffled control permutes edge destinations, preserving source-associated weights/signs and destination in-degree multiplicity before sparse duplicate aggregation. It does not preserve cell-type or spatial structure. Use the same seed/settings for paired comparisons.

## Next scientific milestone

The body is still an engineered readout, so the proposed scientific advance has **not** been achieved yet. The next bounded experiment is to choose one leg/circuit with verified muscle assignments and physiological recordings, fit cell dynamics on a training subset, then predict held-out activation/silencing responses. Replace the leg oscillator with a biomechanical joint and real proprioceptive encoding. Only then expand to coordinated six-leg walking, sensory navigation, and eventually flight.

No autonomous learning, hunger, consciousness, biological fidelity, or novelty is claimed by this prototype.

## Circuit atlas (new)

Open `/atlas` on the same local server. It reads the same `data/circuit.json` the simulator uses: all 807 cells and 21,161 measured edges, with no extra download.

Three views share one side panel:

- **Circuit:** the four layers with layer-to-layer synapse totals. The 398 VNC interneurons are shown as their own layer; the cell counts earlier in this README cover only descending, motor and sensory cells.
- **Layer:** groups inside a layer (cell type, or developmental hemilineage for interneurons) and their strongest links. Selecting a group shows all of its inputs (blue) and outputs (red).
- **Anatomy:** cells at their annotated soma positions, dorsal view, anterior up, fly's left on the left. Drag to pan, scroll to zoom, and shift-drag or use "Select region" to summarize any rectangle. Clicking a cell draws its measured partners and lists them by synapse count.

Every panel reports how the simulator treats the selected cells and flags known modeling problems: non-leg-nerve motor neurons pooled into leg stride, cells whose unclear transmitter gives them zero weight, glutamate treated as inhibitory, and estimated positions. The URL hash records the selected cell, group or layer so a view can be reopened.

Positions are measured soma locations except for the 91 leg sensory neurons (somata in the legs) and 8 other cells without an annotated soma. These are drawn as rings at the synapse-weighted centroid of their partners, a display convenience rather than anatomy. The scale bar assumes 8 nm isotropic voxels. Neuron morphology and synapse locations are not shown; they live in the separate MaleCNS segmentation data. The odor-learning circuit is not in the atlas yet because `build_learning.py` does not export soma positions.

```sh
.venv/bin/python -m unittest -v test_atlas.py
node --test atlas-core.test.js
```

## Neuron workbench (new)

Open `/neuron`, or use "Open in neuron workbench" on any cell in the atlas. Any cell in either circuit can be opened by body ID; "Random Kenyon cell" picks one from the learning circuit. The workbench loads the cell's measured presynaptic partners and synapse counts from `/api/neuron`, then simulates the membrane in the browser (`neuron-core.js`).

- **Measured:** input partners, synapse counts, output partner count, and for locomotion cells the predicted transmitter that sets each input's sign (ACh +1, GABA and glutamate −1, unclear 0, as in the network model). PAM inputs to Kenyon cells are listed but not simulated.
- **Assumed:** two compartments (a passive dendrite that receives all synapses, coupled to an active spike initiation zone, with the soma off the path as in unipolar insect neurons); squid Hodgkin–Huxley kinetics as a placeholder for Drosophila channels; inputs firing at a set rate with random timing; conductance per spike scaled by log(1 + synapses). The learning dataset has no transmitter predictions, so every PN input is treated as excitatory.

The default strength (0.065) was chosen so a single claw of Kenyon cell 57729 rarely fires the cell and all seven usually do, matching the qualitative observation that Kenyon cells need several coincident claws. It is not fitted to recordings. For Kenyon cells, the workbench can feed the exact scalar and one-hot number codes used by the numerosity experiment.

## Numerosity experiment (new)

Open `/numerosity`. The task is choosing the larger of two numbers, 1 to 9, with some numbers never shown in training. Numbers are encoded on the 314 measured PNs: every number switches on exactly 30 PNs, so total activity carries no size cue. The **scalar code** (an HTM-style scalar encoder over PNs in a random order) gives neighboring numbers overlapping PNs; the **one-hot code** gives each number disjoint PNs. The measured PN→KC wiring and 95th-percentile threshold form the Kenyon cell code, and a plastic pooled readout over the measured KC→MBON edges, gated by PAM→KC synapse totals, learns a value per stimulus. Each trial both options receive feedback: +1 for the larger, −1 for the smaller. With chosen-only feedback, values drift toward "good whenever I picked it" and no ordering forms.

Saved benchmark (`data/numerosity-benchmark.json`, five seeds, P(choose the larger), chance 50%), for pairs where both numbers were held out:

| Held out | Scalar code | One-hot code | Scalar, shuffled PN→KC |
|---|---|---|---|
| 2, 5, 8 (interpolate) | 100% ± 0 | 48% ± 2 | 100% ± 0 |
| 4, 5, 6 (middle block) | 61% ± 22 | 49% ± 3 | 63% ± 18 |
| 8, 9 (extrapolate) | 10% ± 5 | 53% ± 2 | 12% ± 9 |

**Findings.** Overlapping codes let the readout interpolate between trained numbers; without overlap, held-out numbers stay at chance. Wide gaps weaken interpolation. Extrapolation fails systematically and is reversed: 9 shares fewer PNs with trained large numbers than 8 does, so it looks smaller. This is similarity-based generalization, not a learned rule of order. Shuffled PN→KC wiring performs as well as the measured wiring, so these results show nothing specific to the measured connectome. "One held-out number" accuracy is partly an artifact when held-out numbers are mid-range, because untrained values sit near 0, the midpoint of the ±1 outcomes.

```sh
.venv/bin/python numerosity.py --benchmark
.venv/bin/python -m unittest -v test_numerosity.py test_neuron.py
node --test neuron-core.test.js
```

The next planned step is a successor ("counting") test with an HTM-style sequence memory on Kenyon cells, labeled as an added assumption because the fly mushroom body has no established equivalent of distal-segment sequence memory.

## Data quality layer (new)

`build_quality.py` folds two more official MaleCNS tables into the lab: body statistics (synapse totals for every body, 780 MB) and per-synapse transmitter predictions (one row per pre-synapse, 2.7 GB). It streams only the rows for the lab's cells and writes `data/quality.json` with, for every cell in both circuits:

- **Coverage:** the share of the cell's input and output connections that the lab's subset contains. A cell with low input coverage is simulated from a small slice of what drives it.
- **Per-synapse transmitter evidence:** mean probability per transmitter, the share of the cell's synapses whose most likely transmitter is each one, and sign probabilities under the models' rule (acetylcholine +, GABA and glutamate −, others 0).

The atlas and neuron workbench show both for every cell and flag low coverage, mixed evidence and cells whose aggregate transmitter is "unclear" but whose synapses mostly agree. In the workbench, inputs in the learning circuit take their sign from the MaleCNS per-body consensus transmitter instead of being assumed excitatory.

**First results (MaleCNS v1.0).** Input coverage is low throughout the locomotion subset: descending neurons 0.5–2% (their drive comes from the brain), VNC interneurons median 15%, motor neurons 10%, leg sensory cells 4.5%. In the learning subset, Kenyon cells receive a median 27% of their input from the selected PNs and MBONs 55% from the selected KCs; PN and PAM inputs are not in the subset at all. Per-synapse predictions agree with the consensus for every non-motor locomotion cell, give 303 of 314 PNs as cholinergic (11 GABA/glutamate, mostly ventral-tract PNs), MBONs as 50 acetylcholine, 26 glutamate and 21 GABA, and PAM cells as dopamine. They fail systematically for two classes: 4,060 of 4,064 Kenyon cells come out "dopamine" and motor neurons mostly "histamine" or "acetylcholine", contradicting published physiology (cholinergic Kenyon cells, glutamatergic motor neurons). The consensus transmitter therefore remains the model's transmitter, and disagreements are flagged rather than applied. The network models themselves are unchanged in this step; transmitter predictions come from EM appearance, and receptor identity, including glutamate's real sign, is still not measured.

```sh
cd raw-data && B=https://storage.googleapis.com/flyem-male-cns/v1.0/connectome-data/flat-connectome && curl -fL -o body-stats.feather $B/body-stats-male-cns-v1.0-minconf-0.5.feather && curl -fL -o tbar-neurotransmitters.feather $B/tbar-neurotransmitters-male-cns-v1.0.feather && cd .. && .venv/bin/python build_quality.py raw-data
.venv/bin/python -m unittest -v test_quality.py
```

## Brain regions (new)

`build_regions.py` asks neuPrint (dataset `male-cns:v1.0`) for the region profile (`roiInfo`) of every traced neuron: how many of its input connections (post) and output connections (downstream) fall in each primary region. It writes `data/regions.json` with per-region totals, a region-to-region flow matrix and a region profile for every lab cell. Open `/regions` for the matrix: hover any cell for the connection (flow, neuron count, shares in both directions and the reverse flow), click a region to pin it in the panel. Regions are ordered by neuPrint's region hierarchy by default. Atlas cell panels and the neuron workbench show each cell's top input and output regions.

**First result.** All six lab descending neurons receive their input mainly in the premotor regions LAL, VES, IPS and SPS and in GNG: DNa02 34% LAL and 20% VES; DNa01 36–40% VES and 26–28% GNG. DNg13 sends its output to the legs on the opposite side. These regions are what a brain input to the locomotion model would need to represent.

Flow from region A to region B sums, over all traced neurons, the fraction of each neuron's input received in A times its output connections in B. Synapses sit inside one region, so this routes influence through neurons; it is not a count of synapses between regions. Only primary regions are used, since super- and sub-regions overlap and would double count.

Requires a personal neuPrint token, kept in the environment and never committed (the repository is public):

```sh
export NEUPRINT_APPLICATION_CREDENTIALS='<token from your neuPrint account page>'
.venv/bin/pip install -r requirements-regions.txt && .venv/bin/python build_regions.py
.venv/bin/python -m unittest -v test_regions.py
```

## 3D view and Neuroglancer links (new)

Open `/3d`. It draws neurons as official MaleCNS meshes and brain regions as translucent official region shapes, inside faint brain and VNC outlines. When a neuron's mesh cannot be loaded, it is drawn from its official skeleton instead and marked "skeleton". Presets cover the steering descending neurons, Kenyon cell 57729 and the regions where the descending neurons take their input. The page state is in the URL (`/3d?cells=10360,523769&regions=LAL(R),VES(R)`).

Data comes from the public bucket `gs://flyem-male-cns`, the same source as the Neuroglancer view, through a whitelisted, cached proxy in the lab server (`gcs.py`, `/api/gcs/...`): only mesh, skeleton and region layers are allowed, paths are validated, files are capped at 25 MB and cached in git-ignored `data/gcs-cache/`. three.js r128 is bundled in `vendor/` (MIT licence included), so the page does not depend on a CDN.

Every atlas cell, workbench cell and region panel also has **View in 3D** and **Neuroglancer ↗** links. The Neuroglancer link opens the official viewer with the cell selected and centred on its soma, together with brain and VNC outlines; region links filter the viewer's region list by name.

All layers the viewer uses are in Neuroglancer's legacy precomputed mesh format (confirmed by `tools/probe_ng.py`, results in `data/ng-probe.json`), and region labels in the bucket match the lab's region names. Full-resolution neuron meshes can be very large (DNa02's single fragment exceeds 25 MB), so `meshes.py` assembles each mesh once on the server, simplifies it by vertex clustering to at most 250,000 triangles for neurons and 60,000–80,000 for regions and outlines, and caches only the simplified version; `/api/mesh/<layer>/<id>` serves it. Clustering keeps the overall shape and branching but loses fine surface detail. The first view of a large neuron takes a while; later views are immediate. Hover a neuron's status to see how far it was simplified. Run the one-time probe on a machine with internet access to confirm the formats of the layers the viewer uses:

```sh
.venv/bin/python tools/probe_ng.py
node --test ng.test.js && .venv/bin/python -m unittest -v test_gcs.py
```

## Sparse expansion for continual learning (new)

Open `/expansion`. This tests the mushroom body's design as a machine-learning component. Handwritten 8×8 digits (UCI, bundled as `data/digits.npz`) arrive a few classes at a time, and each model is tested on all classes seen so far with no task label (class-incremental learning). The fly-inspired model expands pixels through the measured MaleCNS projection-neuron → Kenyon-cell wiring (pixels drive projection neurons through a fixed random assignment), keeps the 5% most driven Kenyon cells, and learns with an associative readout that strengthens only the correct class's connections from active cells. Controls: shuffled wiring, random sparse wiring with the same per-cell in-degree, dense random projection, expansion without sparsity, the associative readout on raw pixels, an error-driven readout on the sparse code, softmax regression on pixels, and a backpropagation MLP with a similar parameter count. An MLP trained on all classes at once gives the upper bound.

Saved benchmark (`data/expansion-benchmark.json`, five seeds, final accuracy on all classes / forgetting):

| Model | Digits | Digits, 5 per class | Two-shape classes |
|---|---|---|---|
| Measured PN→KC, 5% active, associative | 89% / 5% | 80% / 6% | 80% / 8% |
| Shuffled PN→KC, 5% active, associative | 88% / 5% | 79% / 7% | 80% / 7% |
| Dense random, 5% active, associative | 91% / 4% | 84% / 6% | 82% / 7% |
| Measured PN→KC, all active, associative | 89% / 5% | 81% / 7% | 74% / 9% |
| Associative readout on raw pixels | 90% / 5% | 84% / 6% | 76% / 9% |
| Measured PN→KC, 5% active, error-driven | 32% / 84% | 33% / 77% | 24% / 93% |
| MLP, backpropagation | 19% / 100% | 19% / 93% | 22% / 91% |
| MLP trained on all classes at once | 98% | 86% | 98% |

**Findings.** The associative learning rule is what prevents catastrophic forgetting (about 5% instead of 84–100%), and it works as well on raw pixels as on the expansion; with five examples per class it approaches the all-at-once upper bound. Sparse expansion helps only when classes have several shapes (two-shape classes: 80–82% versus 74–76% without sparsity or expansion). The measured wiring performs like shuffled and random wiring, slightly behind; about 6% of measured Kenyon cells receive no projection-neuron input and never fire. What transfers to machine learning is the learning rule and the sparse code, not the specific connectome.

```sh
.venv/bin/python expansion.py --benchmark
.venv/bin/python -m unittest -v test_expansion.py
```

## Theory: does the maths predict the experiment? (new)

Open `/theory`. `theory.py` states the sparse-expansion model formally and tests four predictions against simulation, with the "supported" thresholds fixed in the code before running.

The model: a fixed expansion `z(x) = top-k indicator of Wx` with sparsity `f = k/N`; an associative readout `w_c = Σ z(x_i)` over class-c examples, predicting `argmax_c ⟨z, w_c⟩ / ‖w_c‖`. Expanding the score shows the classifier is a kernel sum, `⟨z(x), w_c⟩ = Σ_i K(x, x_i)` with `K` the overlap of two sparse codes, which is what the predictions rest on.

- **P1 — why the error-driven rule forgets. Refuted as first stated, then replaced.** The original prediction was that forgetting grows with cross-task code overlap. Across 14 sparsity and wiring settings, r = 0.27. Comparing classes by normalised weights with no bias term at test time cuts forgetting by 0.17 on average and up to 0.29, so recency bias, not interference on shared units, dominates. The associative readout normalises by construction and forgets 0.05.
- **P2 — an optimal sparsity for classes with several shapes. Supported.** Accuracy on two-shape classes peaks at f = 0.01 (dense random) and f = 0.02 (measured), both interior to the swept range, while single-digit classes are flat in f.
- **P3 — overlap as a function of input angle. Supported.** For a Gaussian expansion the expected overlap is the joint upper-tail probability of a bivariate normal with correlation cos θ divided by f, which equals `f^tan²(θ/2)` to leading order and exactly at 90°. Simulated dense codes match the exact curve (RMSE 0.028). The measured wiring deviates (RMSE 0.098, bias −0.074): its weights are non-negative and its in-degrees uneven, so it is not the Gaussian the theory assumes.
- **P4 — capacity. Supported for moderate sparsity, breaks when very sparse.** Score statistics measured with 10 synthetic classes predict accuracy out to 400 classes with mean error 0.017 at f = 0.2 and 0.049 at f = 0.05, but 0.102 at f = 0.01, always optimistic. The cause is visible in the data: wrong-class score skew is +0.71 at f = 0.01, +0.25 at 0.05 and −0.01 at 0.2, so the Gaussian assumption fails exactly where the prediction does.

```sh
.venv/bin/python theory.py            # rewrites data/theory-benchmark.json
.venv/bin/python -m unittest -v test_theory.py
```

## Three flies in one arena (new)

Open `/arena`. Two agents labelled male and one labelled female walk in a shared 2D space, each running the same measured 807-cell locomotion circuit with the dynamics of `model.py`, coupled by hand-designed senses.

**What is measured:** each agent's circuit, its dynamics and its transmitter signs. **What is engineered:** odour falling off with distance, a two-antenna concentration comparison, seeing another fly as a bearing plus apparent size, who is attracted to whom, the courtship song rule, and the drive levels. There is no female connectome in this lab: MaleCNS is one male fly and the selected subset contains none of the sexually dimorphic courtship circuitry, so "female" labels engineered emissions and responses, not a different brain. There is no collision physics, so agents can overlap.

Each agent is drawn to scale as a fly: a 2.5 mm body with abdomen, thorax, head and antennae, six legs animated by that agent's own leg oscillators (the same engineered gait as the motion lab, driven by its motor pools), and wings that fold back except when a male extends one to sing. Faint haloes show the odour field the agents sense, and the view fits itself to the trajectories.

Scenarios: **rivalry** (both males approach her odour; each raises his drive on hearing a rival sing), **courtship** (only one male is attracted), **food** (a patch emits odour), **threat** (a looming threat at 4 s that everyone flees). Sensory controls remove vision, odour or song, or make the agents ignore each other entirely.

Saved benchmark (`data/arena-benchmark.json`, two seeds per scenario and condition):

| | All senses | No vision | No odour | Ignoring each other |
|---|---|---|---|---|
| Rivalry, closest male–female approach | 2.0 mm | 11.1 mm | 10.7 mm | 13.5 mm |
| Courtship, closest male–female approach | 3.8 mm | 11.1 mm | 11.2 mm | 13.5 mm |
| Food, closest approach to another fly | 4.9 mm | 7.3 mm | 10.7 mm | 13.5 mm |
| Mean path walked | 120–170 mm | 37–60 mm | 61 mm | 61 mm |

**Findings.** Both senses now matter: removing either odour or vision leaves the agents wandering at 11 mm separation, against 2 mm intact, and the threat scenario is unchanged by any sensory control because the threat response overrides them. Song is now brief and intermittent, because at walking speed the males pass the female rather than settling beside her.

**Bug found while building this.** The motion lab's target mode steered away from its target: a positive descending bias turns the body readout clockwise, but the bearing-to-bias mapping assumed the opposite. `model.steer()` now holds the correct sign, shared by the motion lab and the arena. Target mode reaches within 1.5 mm of its target over 20 s instead of wandering off; `test_arena.py` guards it.

```sh
.venv/bin/python arena.py --scenario rivalry
.venv/bin/python arena.py --benchmark        # about 5 minutes
.venv/bin/python -m unittest -v test_arena.py
```

### Walking speed, calibrated

The gait readout used to map a pooled motor rate through two arbitrary constants (`stride = motor_hz / 50`, `speed = 6 mm/s × stride`, step frequency capped at 5 Hz), which put the default at 1.2 mm/s and the ceiling below a real fly's walking speed. `model.gait()` now calibrates against reported *Drosophila* walking: step frequency scales with motor output up to 12 Hz, stride length runs 1.1–2.0 mm, and speed is their product; `model.turn_rate()` derives yaw from the difference between the two sides' leg speeds across a 1 mm track. The motion lab reports speed, step frequency and stride length beside those reported ranges. These are calibration choices, not measurements: there is still no muscle model, load or inertia.

**The circuit's operating band is narrow.** Below about 2.1 of descending drive the motor pools are silent; above about 3.0 they saturate. That fits cells receiving only 10–15% of their real input here, and it means the subset behaves close to a switch. The arena therefore maps a behavioural command in [0, 1] into that measured band rather than feeding drive in raw.

**Calibration cost something.** At about 13 mm/s the arena's proportional steering overshoots, so one male usually reaches her (1.5–2 mm) while the other stalls several millimetres away, instead of both reliably making contact. Before calibration they made contact, but only because every agent crawled at roughly 1 mm/s. A grid search over drive, approach speed and steering gain found no setting giving both realistic speed and reliable contact, so the limit is the controller, not the speed.

### Gait quality, measured

The arena measures duty factor, steps per leg per second and a tripod index (how much legs within a tripod agree, minus agreement across tripods) and shows them against reported ranges. **The measured subset does not produce tripod coordination:** the tripod index is about −0.02, and several legs are almost never lifted, because trochanter depressor motor neurons outnumber levators 14 to 2. Speed (13 mm/s) and duty factor (0.50–0.79) land in the reported ranges; the coordination does not. The drawing uses a relative stance rule (a leg is planted when it is lower than its own recent average) so legs visibly step, while the metrics report both that rule and the absolute one.

### Legs driven joint by joint

MaleCNS motor neuron types name the muscle each cell drives, so `bodyplan.py` groups them into the antagonist pairs of each leg joint: coxa protractors against retractors, trochanter levator against depressors, tibia extensor against flexors, tarsus levator against depressors. Each joint angle then follows `rest + span * tanh((agonist Hz − antagonist Hz) / scale)`, and the drawn legs are posed from those angles rather than from one oscillator phase. A leg counts as planted when its trochanter is depressed.

All six legs can drive their coxa, trochanter and tibia. Tarsus levator motor neurons are annotated only for the front legs, so four tarsi cannot be driven, and 59 motor neurons have no muscle name or no leg assignment and stay unused; `/arena` reports both.

**Body motion** offers two modes. "From pooled motor rates" is the lab's original readout. "From the planted feet" derives translation and turning from the movement of planted feet in the body frame, so body motion follows the joints rather than a stride average. Both are readouts of motor activity: there is no muscle model, tendon, load or inertia.

A fly's compound eyes cannot move, so there is nothing to drive there. Head and antennal movement exist (neck and antennal motor neurons) but neither group is in this subset, so head, eyes and antennae stay fixed.

Correction to an earlier note: the 59 motor neurons that leave through accessory nerves rather than leg nerves are, by their muscle names (trochanter flexor, sternotrochanter, pleural promotor and similar), leg-muscle motor neurons. Pooling them into leg stride is reasonable; only the 8 through the abdominal nerve remain unclear. The atlas flag now says this.

### 3D arena

Open `/arena3d` for the same arena run drawn as jointed 3D bodies: thorax, abdomen, head with fixed compound eyes, antennae, wings and six legs built from coxa, femur, tibia and tarsus segments at the lengths in `bodyplan.py`. Each fly stands on its own legs: the body height is set from the lowest planted foot rather than fixed.

**Brain-driven parts:** coxa, trochanter and tibia joints on all six legs, and the tarsus on the front legs, posed from the motor neurons of each joint's muscles. The panel lists every joint angle in degrees, with ▪ marking planted legs.

**Hand-driven parts:** head yaw and pitch, antennae, abdomen bend and both wings, with sliders. These have no motor neurons in this subset, so nothing in the simulation drives them; a wing can also extend automatically while a male sings. Compound eyes cannot move in a fly at all, so they are fixed to the head.

### How the neurons are modelled

The modules use different levels of detail, and only the workbench is biophysical:

| Module | Neuron model |
|---|---|
| Locomotion network and arena (`model.py`) | Leaky integrate-and-fire: 20 ms membrane constant, threshold 1, reset 0, 2 ms refractory, current-based synapses with a 10 ms exponential trace, Gaussian noise |
| Neuron workbench (`neuron-core.js`) | Hodgkin–Huxley with squid kinetics, two compartments, conductance-based synapses (0 mV excitatory, −80 mV inhibitory) |
| Odour learning (`learning.py`), numerosity, expansion | Firing-rate models, no membrane |
| Vision network (`vision.py`) | Discrete-time recurrent tanh units, one or two states per cell, trained by backpropagation through time |

What this leaves out of real fly neurons: graded (non-spiking) transmission, which many fly neurons use; conductance-based synapses in the network models, so no reversal potentials or shunting inhibition; dendritic compartments, although skeletons are now available; cell-type-specific time constants and spike-frequency adaptation; and Drosophila channel kinetics (para, Shaker, Shab) rather than squid.

### Two neuron models

`model.py` now offers a second neuron model, selectable in the motion lab and via `neuron` on `/api/run`:

- **`lif` (default):** leaky integrate-and-fire, 20 ms membrane constant, threshold 1, 2 ms refractory, current-based synapses.
- **`adex`:** adaptive exponential integrate-and-fire with conductance-based synapses (Brette & Gerstner). Adds a spike-initiation nonlinearity, spike-frequency adaptation, and synapses as conductances with reversal potentials (0 mV excitatory, −75 mV inhibitory), so inhibition shunts instead of subtracting a fixed current. The membrane is integrated in 0.1 ms substeps inside each 1 ms network step. Parameters are standard cortical values, not Drosophila measurements, with the drive and conductance scalings calibrated so both models fire at the same rate at the default drive (9.2 versus 9.6 Hz).

**Finding: the two models disagree about gain.** Raising descending drive from 2.0 to 4.0 makes the integrate-and-fire network escalate (motor pools 9 → 82 Hz, saturated 79% of the time), while the adaptive network holds motor output near 10 Hz with almost no saturation. Descending rates still rise with drive under both (15 → 25 Hz for the adaptive model), so the clamp is downstream: shunting inhibition and adaptation give the VNC gain control. That also means the narrow "usable band" the arena works around is a property of the integrate-and-fire model rather than of the connectome. Which behaviour is right is unknown: neither set of parameters is fitted to fly recordings.

The arena still uses the integrate-and-fire model, because its drive-to-speed mapping depends on that escalation.

## Body studio (new)

`/body` shows the fly at inspection size with every joint exposed: six legs with coxa, trochanter, tibia and tarsus controls, wings with an animated stroke, halteres, head yaw and pitch, antennae, proboscis and abdomen, plus part selection by click or menu, visibility, exploded view and model size. `flymesh.js` holds the procedural body: segmented abdomen, faceted compound eyes, veined wings, bristles, antennae with aristae, halteres and proboscis, with leg segments at the lengths in `bodyplan.py`.

"Drive legs from the circuit" runs the arena simulation and poses each leg from the motor neurons of that joint's own muscles, with planted feet marked and every joint angle listed in degrees. The panel labels which parts are brain-driven and which are not: head, antennae, proboscis, abdomen, wings and halteres have no motor neurons in this subset, and a fly's compound eyes cannot move at all. The body stands on its lowest foot, so it sits on the floor at any pose. Segment proportions and the mesh are drawn by eye; there is no muscle model, load or inertia.

## Synaptic conductances and receptor kinetics (new)

`/synapse` connects measured synapse counts to receptor conductances, which neither the locomotion network (current-based synapses) nor the workbench (generic conductances on one cell) does. For any cell in the circuit, each measured presynaptic partner opens a conductance with transmitter-specific kinetics: acetylcholine on a nicotinic cation channel (0 mV, 0.3/5 ms), GABA on GABA-A (−70 mV, 0.5/8 ms), glutamate on a glutamate-gated chloride channel (−70 mV, 0.5/10 ms, the insect inhibitory receptor) and histamine on a histamine-gated chloride channel (−80 mV, 0.3/6 ms). Conductances sum into one passive compartment with an optional −45 mV threshold.

Two ways of using the transmitter predictions:

- **Consensus per cell** uses each partner's consensus transmitter, the same call the network model makes.
- **Split by measured probabilities** divides each partner's conductance across receptors in proportion to the per-transmitter probabilities in `data/quality.json`, so prediction uncertainty reaches the membrane. It also recovers cells whose consensus label sits outside the receptor set but whose probabilities include one: for DNa02 that is 2 of 297 synapses, and conductance reaches four receptor types instead of three.

**What conductances change.** The shunting test runs the same excitatory drive with and without the inhibitory conductances. For DNa02 at double excitation the depolarisation falls from 2.6 mV to 0.4 mV, a ratio of 0.16: an inhibitory conductance shrinks the excitatory response itself, which a current-based synapse cannot do. DNa02 also stays silent under its measured input, since 24 of its 45 inputs are inhibitory against 19 cholinergic ones.

Measured: which cells connect, how many synapses, and the transmitter predictions with probabilities. Assumed: every kinetic constant, 0.03 nS per synapse, membrane capacitance and leak, threshold and presynaptic rates. No morphology, no cable structure, no receptor subtypes, no desensitisation; a transmitter prediction is not a receptor identification.

```sh
.venv/bin/python -m unittest -v test_synapse.py
```

## Navigation: heading, home vector and steering (new)

`/navigation` runs the fly's published navigation algorithm as a model: a ring of heading cells holds a compass bump, rotation cells shift it with turning, long-range inhibition keeps a single bump, vector cells accumulate the path travelled, goal cells hold a direction, and steering cells compare goal against heading and drive the descending neurons. The fly wanders for 12 s, then tries to get back to where it started.

Results with five seeds (`data/navigation-benchmark.json`), at the default compass drift of 0.15 rad/s:

| Condition | Closest to home | Got home | Compass error |
|---|---|---|---|
| Intact | 5.5 mm | 80% | 4.6° |
| No rotation cells | 93.0 mm | 0% | 31.8° |
| No long-range inhibition | 71.8 mm | 0% | 69.4° |
| No vector cells | 92.8 mm | 0% | 5.1° |
| Shuffled column map | 74.0 mm | 0% | 71.4° |
| Random goal | 93.0 mm | 0% | 5.1° |

Each control removes one mechanism and homing stops, for a different reason each time: without rotation the compass no longer tracks turning, without inhibition the bump spreads, without vector cells nothing is accumulated to return along, a shuffled column map breaks the correspondence between cells and directions, and a random goal leaves steering intact but aimed wrongly. Compass drift sets the difficulty: with no drift the integration is exact and homing is perfect, at 0.3 rad/s the fly lands 11 mm out, at 0.6 rad/s 22 mm out.

**What is assumed.** That these cell types play these roles (from the navigation literature), the ring dynamics and their gains, the drift, the conversion from steering activity to turning, and the movement model. The bump rotation is implemented as an exact circular shift rather than the discretised gradient version, because at 16 columns the attractor pins the latter; the mechanism is the same, the numerics are not.

**Measured connectivity.** `build_centralcomplex.py` extracts the central-complex navigation types (EPG, PEN, PEG, Delta7, PFN, FC2, hDelta, PFL, ER, ExR) and the measured edges among them and onto the lab's descending targets, writing `data/central-complex.json`. The probe on the real annotations found 1,247 traced cells of these types (456 PFN, 282 ER, 189 hDelta, 92 FC2, 50 EPG, 50 PFL, 42 PEN, 42 Delta7, 26 ExR, 18 PEG).

Column identity comes from the instance suffix, which the probe made legible: `EPG(PB08)_R2` is bridge glomerulus R2 (the parenthetical is a type code), `PFNv(PB05)_L4_C3` is glomerulus L4 with fan-shaped-body column C3, `Delta7(PB15)_L3R6_R` spans two glomeruli with soma side R, and `FC2A_C2_L` carries only a body column. Glomeruli R1–R8 map to ring positions 0–7 and L1–L8 to 8–15; cells labelled only by a body column spread over the same ring; cells spanning several glomeruli take the circular mean. Ring neurons (`ER5_L`, `ExR1_R`) have no column, which is correct: they carry cues, not compass positions.

In measured mode the rotation strength is read off the wiring rather than assumed: the model takes the difference between the two rotation cell classes' mean column offsets, so wiring that shifts the compass oppositely by side gives a gain of about one column and homing works, while wiring that shifts both sides the same way gives a gain of zero and the compass cannot track turning at all, failing exactly like removing the rotation cells. That is the test: the connectome decides whether the algorithm runs.

Once the file exists, `/navigation` reports the measured column offsets between functional groups and, separately, the offsets of each rotation cell type by side. That split matters: pooling the two rotation classes cancels their opposite shifts to zero, while the per-type breakdown shows whether the measured wiring really shifts the bump in opposite directions by side, which is what the algorithm needs for a turn to move the compass correctly. The page states plainly that the offsets are measured while the functional grouping and the rotation interpretation are assumptions.

```sh
.venv/bin/python build_centralcomplex.py raw-data --probe    # what the annotations hold
.venv/bin/python build_centralcomplex.py raw-data            # writes data/central-complex.json
.venv/bin/python navigation.py --benchmark
.venv/bin/python -m unittest -v test_navigation.py
```

## Spectra: is the measured wiring special? (new)

`/spectrum` asks the structural version of the question the task experiments kept answering with "no": do the eigenvalues of the measured weight matrix differ from randomisations with the same statistics? `spectral.py` compares six randomisations (destinations permuted, weights permuted among edges, targets redrawn, inhibitory cells permuted, fully random, and random wiring matched to the measured share of reciprocal pairs) under three normalisations.

**The lab's default normalisation caps the answer.** Incoming absolute weights sum to one per cell, so the spectral radius cannot exceed 1. A measured value near 1 means the matrix nearly attains its bound, not that the fly sits at the edge of chaos.

Locomotion subset, incoming-L1 normalisation:

| Condition | Radius | Reciprocity |
|---|---|---|
| Measured | 0.976 | 28.1% |
| Weights permuted among edges | 0.985 | 28.1% |
| Inhibitory cells permuted | 0.875 | 28.1% |
| Destinations permuted | 0.254 | 5.3% |
| Targets redrawn | 0.274 | 3.1% |
| Fully random | 0.265 | 3.3% |
| Random, reciprocity matched | 0.266 | 30.6% |

**What this says.** The near-unit radius follows the measured *topology*: it survives shuffling weights (0.985) and largely survives permuting which cells are inhibitory (0.875), but collapses to about 0.26 under anything that moves edges. Reciprocity alone does not explain it, since random wiring matched on reciprocity still gives 0.266. The gap survives the synapse-count threshold control and input-coverage filtering.

**But it does not survive changing the normalisation.** Scaled so the largest singular value is 1, the measured radius is 0.184 against 0.157–0.230 for the randomisations, with fully random highest. So the effect is a property of row-normalised measured topology, not a scale-free property of the graph.

**Where it comes from.** The leading eigenvector is concentrated, not global: its participation ratio is 0.019 against 0.609 when shuffled, and its mass sits on five SNpp39 sensory cells plus a couple of motor neurons. Those cells receive all of their measured input from each other, which under row normalisation makes a stochastic sub-block with an eigenvalue of 1 by construction. 619 of 794 cells with input are fed only from within their own strongly connected component, a consequence of how the subsets were cut.

**Other graphs.** The mushroom-body subset is essentially feedforward: measured radius about 0, while rewiring creates loops and pushes it to 1. The optic-lobe patch gives exactly 1 for every condition, because its cells' inputs lie entirely inside the patch.

### A subset selected for observability

`build_coverage_subset.py` answers the objection above by building a subset the other way round. Coverage is circular, since a cell's input coverage depends on which other cells are present, so it grows outward: seed with the lab's existing subsets, read every edge into them, measure each cell's coverage as the share of its true input synapses (from `body-stats.feather`) arriving from cells *inside* the subset, add the strongest missing upstream partners of the worst-covered cells up to a budget, then keep only cells that reach the target coverage. Dropping a cell removes its edges and lowers its neighbours' coverage, so the keep-set is iterated until stable, which means cells propped up by poorly observed neighbours fall out too.

The result is smaller and better observed than the type-selected subsets, and it is the matrix on which the spectral question can actually be asked: if measured still differs from its randomisations here, under a normalisation that does not bound the radius, the difference is about the wiring rather than about the cut. `/spectrum` lists it as "Coverage-selected subset" once built.

```sh
.venv/bin/python build_coverage_subset.py raw-data --target 0.5 --budget 3000
.venv/bin/python spectral.py --benchmark
```

Two passes over `weights.feather`, a few minutes each. Lower `--target` if too few cells survive; the run prints the seed coverage, how far it grew and how many cells reached the target.

**Conclusion for the deep-learning idea.** The spectral signature is real but it belongs to the lab's matrices, not to the fly: subset selection decides which cells look self-contained, and row normalisation turns those into unit eigenvalues. Using the connectome as an initialisation prior on the strength of "it sits near the edge of stability" would be building on an artefact. A fair test needs a subset chosen without a synapse threshold, a normalisation that does not bound the radius, and cells with high input coverage; `build_coverage_subset.py` above provides the first and third, and the `global_scale` normalisation the second.

```sh
.venv/bin/python spectral.py --benchmark
.venv/bin/python -m unittest -v test_spectral.py
```

## Where inhibition sits (new)

`/inhibition` follows up the one spectral effect that survived the observability control. On the coverage-selected subset the measured network is less excitable than matched random wiring, and moving which cells are inhibitory changes that substantially, from only about 6% of cells:

| Placement | Natural spectral radius |
|---|---|
| Measured | 33.4 |
| Random | 39.9 |
| Random, out-degree matched | 41.1 |
| Highest-throughput cells | 40.0 |
| Lowest-throughput cells | 47.4 |
| No inhibition | 47.4 |

Inhibitory cells are not the highest-degree ones (mean out-degree 127 against 194 for excitatory cells), but they carry more synapses each and 19% of all outgoing synapses in the subset.

**Does the placement compute anything?** `inhibition.py` uses the measured graph as a fixed recurrent network and trains only a linear readout, the standard reservoir test, keeping the number of inhibitory cells identical across conditions and moving only which cells they are. Tasks: memory capacity (reconstructing a random input stream at increasing delays) and delayed integration.

Rescaled to a common spectral radius, every placement scores the same (memory 8.2–8.3, integration at ceiling): with gain equalised, placement does nothing. Under one shared scale factor, so gain differences survive, the differences are small but consistent, and the right statistic is the per-seed difference, since the seed fixes which cells the subsample contains:

| Comparison | Memory difference | Seeds favouring measured | t |
|---|---|---|---|
| measured − random | +0.206 ± 0.175 | 14 of 15 | 4.6 |
| measured − no inhibition | +0.131 ± 0.133 | 14 of 15 | 3.8 |
| measured − lowest-throughput | +0.129 ± 0.134 | 14 of 15 | 3.8 |
| measured − out-degree matched | +0.093 ± 0.118 | 13 of 15 | 3.0 |
| measured − highest-throughput | +0.020 ± 0.101 | 9 of 15 | 0.8 |

**What this supports, and what it does not.** The measured placement beats random placement by about 3% of memory capacity, consistently across seeds, and beats having no inhibition at all by a similar margin. But it ties with simply putting inhibition on the highest-throughput cells, so the result is better described as "inhibition belongs on high-throughput cells" than as anything specific to this connectome. The effect is small, it disappears once gain is equalised, and it rests on transmitter predictions rather than measured receptors.

```sh
.venv/bin/python inhibition.py --benchmark --seeds 5
.venv/bin/python -m unittest -v test_inhibition.py
```

## Memory: the mushroom body, rebuilt for it (new, in progress)

Olfactory associative memory in the fly lives in the mushroom body, and its mechanism is well enough known to model honestly. An odour activates a sparse set of Kenyon cells; their axons run through about fifteen compartments, each with its own dopamine neurons delivering a teaching signal and its own output neurons reading out. When an odour's Kenyon cells fire together with a compartment's dopamine neurons, the Kenyon-cell-to-output synapses in that compartment weaken, and that weakened set of synapses is the memory. Presenting the odour again drives that compartment's output neuron less, tipping the balance of the output population toward approach or avoidance. The wiring decides where a memory can be written; the weights decide what it says. A connectome cannot show memory contents, since short-term memory changes synaptic strength rather than synapse count, but it can show exactly where memory can be written.

The lab's learning subset cannot support that model: it has no PPL1 dopamine neurons (the aversive teachers), no APL or DPM, and no loops from output neurons back to dopamine neurons or to each other, which extinction and second-order conditioning need. `build_mushroombody.py` extracts the complete circuit: Kenyon cells, output neurons, PAM and PPL dopamine neurons, DPM, APL, the lab's projection neurons, and the 200 strongest downstream partners of the output neurons, where memory leaves the mushroom body on its way to behaviour. Kenyon-cell-to-Kenyon-cell edges are summarised as a total rather than stored, since there are very many.

Compartments are parsed from instance names: `MBON01(y5B'2a)` has dendrites in γ5 and β′2a, `MBON11(y1pedc>a/B)` has dendrites in γ1-pedunculus and an axon to the α/β lobes, and `PAM07(y4<y1y2)` teaches in γ4 while receiving input in γ1 and γ2. Adjacent compartments are split correctly (`y1y2` is γ1 and γ2), β is written both `B` and `b` (`MBON26(b'2d)`), and the pedunculus both `pedc` and `ped` (`PPL101(y1ped)`).

The probe on the real annotations found every cell type the memory model needs: 4,064 Kenyon cells, 97 output neurons, 316 PAM reward-type dopamine neurons, 16 PPL1 aversive dopamine neurons (PPL101–PPL108, one per hemisphere each), 2 APL and 2 DPM. The 8 PPL2 cells are dopaminergic but innervate the calyx and lateral horn, so they get their own role and never count as compartment teachers.

Some cells carry no compartment in their name (PPL107 and PPL108 among the aversive teachers). They are placed from the data: within a compartment, the dopamine neurons and the output neurons contact the same stretch of Kenyon-cell axons, so the overlap between a cell's Kenyon-cell partners and each compartment's identifies where it works. Such assignments are recorded as inferred, with their overlap score, so they are never mistaken for annotations. The output is a compartment table listing which dopamine neurons teach and which output neurons read in each compartment.

```sh
.venv/bin/python build_mushroombody.py raw-data --probe    # what the annotations hold
.venv/bin/python build_mushroombody.py raw-data            # writes data/mushroom-body.json
.venv/bin/python -m unittest -v test_mushroombody.py
```

### The memory model

`memory.py` and `/memory` run the established account on that circuit, with the measured wiring deciding where memory can be written:

- **Encoding:** an odour's projection-neuron pattern drives Kenyon cells through the measured wiring, with about 5% left active (the APL's role, here a threshold).
- **Writing:** where active Kenyon cells coincide with dopamine in a compartment, their synapses onto that compartment's output neurons are depressed. Reward reaches the PAM cells and punishment the PPL1 cells, and each dopamine cell teaches only in the compartments it measurably innervates.
- **Reading:** an output neuron's valence follows from its compartment's teachers (punishment-taught compartments hold approach-promoting outputs), and behaviour is the change in approach drive relative to the untrained state.
- **Keeping:** a labile trace forms on every pairing and fades within hours; a consolidated trace forms only with spaced training and fades over days.
- **Extinction:** re-exposure without reinforcement lets output neurons drive dopamine neurons through the measured MBON-to-DAN loops, writing an opposing trace. Recovery is measured against the same fly trained identically but never re-exposed, so ordinary forgetting cancels out.

Experiments: forgetting after massed and spaced training for both reinforcers, extinction with and without the loops, compartment-by-compartment ablation (the direct test of "is memory everywhere?"), and controls with the loops cut, Kenyon-cell-to-output destinations permuted (which scrambles compartment membership) and no consolidation.

The model is validated on a small synthetic mushroom body with the published layout (`tests_support/mbsynth.py`): punishment teaches avoidance and reward approach; spaced training leaves a trace at 96 hours where massed training leaves none; blocking a punishment-taught compartment weakens aversive memory while blocking a reward-taught one does not; extinction happens only with the loops intact and fades back after a day, as a separate memory should. Those are checks that the model behaves correctly, not findings about the fly.

### Results on the measured circuit

The measured mushroom body has all fifteen compartments once sub-compartment labels are merged (γ1, γ1p and γ1pedc are all γ1; β′2a, β′2m and β′2p are all β′2), with 326 dopamine cells teaching in them and 87 of 93 output neurons assigned a valence from their compartment's teachers: 50 approach-promoting and 37 avoidance-promoting. PPL107 and PPL108 make only 13 and 1 synapses onto Kenyon cells, so they are not compartment teachers at all, and the calyx-innervating PPL2 cells are excluded by design.

**Is memory everywhere?** No, and not in one place either. Blocking plasticity one compartment at a time after spaced training:

| Aversive memory, lost when blocked | | Appetitive memory, lost when blocked | |
|---|---|---|---|
| α2 (PPL105) | 21% | β′1 (PAM13, PAM14) | 39% |
| α′1 (PPL103) | 19% | β′2 (PAM02, 03, 05, 06, 15) | 22% |
| γ2 (PPL103) | 17% | α1 (PAM11) | 17% |
| α′3 (PPL104) | 14% | γ5 (PAM01, PAM15) | 12% |
| α′2 (PPL105) | 9% | γ4 (PAM07, PAM08) | 11% |
| γ1 (PPL101, PPL102) | 8% | β2 (PAM03, PAM04) | 6% |
| | | β1 (PAM09, PAM10) | 4% |

Each memory is spread across a handful of compartments set by where its teachers innervate, no single compartment holds more than 39%, and the two sets do not overlap: aversive memory lives in the six compartments the PPL1 cells teach, appetitive memory in the seven the PAM cells teach. That separation comes from the measured innervation, not from the model.

Blocking some aversive compartments strengthens appetitive memory (α′3 by 20%, α2 by 11%): during reward training the output neurons drive the PPL1 cells through the measured loops, writing an opposing trace, and removing it helps. Cutting the loops entirely raises appetitive memory from 0.22 to 0.30 while leaving aversive memory unchanged.

**Where the model and the fly disagree.** γ1, taught by PPL101, is the compartment most strongly required for short-term aversive memory in fly experiments, but here it carries only 8%. And with the measured loops, this model produces no extinction (effect −0.001), where flies extinguish readily; the synthetic circuit extinguished only because its loop was built to. Both point at the same limitation: output neurons are weighted only by their Kenyon-cell input, loop effects are reduced to excitation onto dopamine cells, and the real extinction pathways, which involve specific output-to-dopamine connections and their signs, are not captured by that simplification.

**Spacing works for both reinforcers:** aversive memory from spaced training is −0.033 at 96 hours against 0 after massed training, and appetitive +0.018 against 0. Scrambling compartment membership (Kenyon-cell-to-output destinations permuted) changes memory only modestly (aversive −0.40 against −0.43), because the teaching signal is gated by the output neuron's compartment rather than the Kenyon cell's.

Assumed throughout: the learning rule, rates and time constants, the consolidation rule, PAM as rewarding and PPL1 as punishing (the real fly has exceptions), the valence rule, random odour codes and the readout.

```sh
.venv/bin/python memory.py
.venv/bin/python -m unittest -v test_memory.py
```

## Vision to walking (in progress)

Goal: a simulated fly that sees and steers. Moving scenes go through a pretrained connectome-constrained eye model ([flyvis](https://pypi.org/project/flyvis/), Lappalainen et al., Nature 2024), its output cell types reach the lab's steering descending neurons through measured MaleCNS pathways, and the existing locomotion model walks.

Milestones:

1. **Bridge (this iteration).** `build_vision.py` finds, in the full MaleCNS tables, how each flyvis output cell type (T4, T5, Tm, TmY and others) can reach DNa01, DNa02 and DNg13 in up to three hops. It works on (cell type, side) groups because flyvis models types, not cells. Influence along a path is the product of input fractions (synapses from the previous group / all synapses onto the next, including untyped inputs). It ranks anatomical routes; it does not predict activity. Output: `data/vision-bridge.json`.
2. **Eye responses.** Run a pretrained flyvis ensemble on standard stimuli (rotating gratings, translating patterns, looming discs) for each eye, and save per-type, per-column responses. Runs locally; flyvis downloads its pretrained models on first use.
3. **Open loop.** Map eye responses through the bridge into left/right descending drive for the locomotion model, and test whether rotation to one side produces turning in the compensating direction and looming produces a stopping or avoidance response. Controls: blinded eye, bridge with shuffled type identities, sign-flipped bridge.
4. **Closed loop.** Render the scene from the simulated fly's position every step, so its own walking changes what it sees.

**Bridge result (MaleCNS, `data/vision-bridge.json`).** Horizontal motion subtypes (T4a/T5a, then T4b/T5b) reach DNa02 on the same side far more strongly than vertical ones (c, d). Direct eye-to-DN connections are essentially absent; about 83% of the summed influence runs over three hops, mostly through visual projection neurons (LLPC1, LPC1, LT51, LC10, LPLC4, HS) into central steering regions. DNa02 receives about four times more visual influence from its own side, with a GABAergic route from the opposite side, a push-pull arrangement to test in simulation. Summed visual influence is a few percent of DNa02's input by this measure.

**Eye responses (milestone 2).** `vision_eye.py` runs pretrained flyvis models (ensemble `flow/0000`, first five by default) on gratings drifting in eight directions, a dark looming disc and full-field flashes, one stimulus at a time, and saves per-type mean and central-column responses to `data/eye-responses.json`. flyvis simulates one eye; the other eye's response to direction θ is taken from the mirrored direction. Direction angles are in the model's hex plane, and front-to-back is defined by T4a's preferred direction in the trained model. The script checks that T4b and T5b prefer the opposite direction, T5a the same, and T4c/T4d the orthogonal directions, opposite to each other. An untrained network fails these checks, so they discriminate. Install and run (downloads PyTorch and, on first run, the pretrained models into git-ignored `flyvis-data/`):

```sh
.venv/bin/pip install -r requirements-vision.txt && .venv/bin/python vision_eye.py
```

**Eye result (pretrained flyvis, five models, `data/eye-responses.json`).** Six of eight direction checks pass. The horizontal motion detectors that dominate the route to DNa02 behave as anatomy predicts: T4b and T5b prefer roughly the opposite direction to T4a (149° and 176° away) and T5a nearly the same (24°). The vertical pair is less clean: T4c and T4d are 129° apart instead of about 180°, and T5c is barely direction selective (index 0.15), so its preferred angle is not meaningful. Eight test directions 45° apart and responses averaged over all 721 columns make these angles coarse.

Caveats known in advance: flyvis was built from optic-lobe connectomes of other flies (FIB25/FIB19), while the bridge uses MaleCNS; the bridge's signs come from majority transmitter predictions; tangential and looming pathways are summarized at the type level; and the locomotion body is still the engineered readout.

```sh
.venv/bin/python build_vision.py /path/to/raw-data
.venv/bin/python -m unittest -v test_vision.py test_vision_eye.py
MALECNS_TEST_FLYVIS=1 .venv/bin/python -m unittest test_vision_eye.FlyvisSmokeTest   # slow, needs flyvis
```

## Odor-learning lab (new)

Open `/learning` on the same local server. The movement page links to it. This is a **separate trial-based rate model**, not a learning module already controlling the walking simulator. Its Y-maze replay illustrates discrete decisions, not a physics trajectory. It uses an additional measured MaleCNS subset: 314 antennal-lobe projection neurons (PNs), 4,064 Kenyon cells (KCs), 97 mushroom-body output neurons (MBONs), and 316 PAM dopamine neurons. The 194,246 retained edges are exactly the observed PN→KC, KC→MBON and PAM→KC connections among selected Traced annotations. Recurrent connections and other cell classes are omitted.

Protocol at defaults: 40 baseline choices; 80 acquisition choices with one odor rewarded; 40 unrewarded probe choices; a 30-step abstract memory delay followed by 40 unrewarded choices; 80 reversal choices with the other odor rewarded; and 40 final unrewarded choices. Both choices are presented on each trial. Locations are counterbalanced within every phase. Side is used only to display the chosen odor in the arena and is never passed to the value function. Reward labels are unavailable to encoding and choice until after the choice has been sampled. Actual reward is delivered only for a correct training choice. Probes never update weights, even after a non-rewarded choice. The delay explicitly decays weights once before the delayed-memory probe.

The mechanism is deliberately simplified and fully supplied by us:

- Abstract odors A/B activate random 20% subsets of the measured PN cells. These are not identified chemical odors, receptor-tuning curves, or a spatial smell simulation. Four noisy training exemplars and four distinct noisy test exemplars per odor are generated from the same templates (Gaussian standard deviation 0.035, clipped to [0,1]).
- PN→KC weights are `log(1+synapse_count)`, normalized by incoming sum. A 95th-percentile threshold sparsifies each KC response. The threshold and rate model are assumptions; no spiking dynamics are simulated in this module.
- Existing KC→MBON edges carry positive base magnitude `b_e = log(1+count_e)` and an initially zero learned offset `d_e`. For each odor, feature `f_e = KC_activity[pre(e)] * b_e`, normalized so nonzero features sum to one. Modeled odor value is `sum(d_e * f_e)`, an assumed pooled change in MBON response relative to initialization. This does not infer biological MBON approach/avoidance valence or transmitter signs.
- Probability of choosing A is `sigmoid((value_A-value_B)/temperature)`. Sampling uses a reproducible random seed. Choice probability and sampled choice fraction are shown separately; neither is a general intelligence score.
- PAM gating per KC is the normalized log of the total observed PAM→KC synapse count. This is an anatomical eligibility proxy, not a simulation of dopamine dynamics, release, receptors, or mushroom-body compartments.
- For a chosen training odor, error is `reward - predicted_value`. On existing active KC→MBON edges, `d += learning_rate * error * f * gate / sum(f² * gate)`, clipped to [-0.95,3]. Effective edge magnitude is `base * (1+d)`. No new anatomical edges are created. This generic normalized prediction-error rule is not fitted to physiology and is not a claimed implementation of a particular published plasticity model.
- Before delayed testing only, `d *= retention ** delay`. Delay steps are abstract, not seconds. Changing retention demonstrates this assumed decay rule rather than a recovered biological forgetting process.

Controls disable plasticity, silence KCs, remove training rewards, disable the PAM gate, or shuffle PN→KC destinations. Shuffling preserves PN source-associated weights and KC incoming edge multiplicity before aggregation, but changes pairing and normalized weights. It is not a complete degree/weight/cell-type matched biological null. All controls use the same task and stochastic choice seed. The saved benchmark varies seeds 7–11 and alternates initially rewarded odors A/B; its default settings are independent of current UI settings.

**Initial finding:** the constructed model learns and reverses its preference, and the shuffled network does too. Mechanism-disabling controls remain at modeled chance probability. These outcomes establish the implemented associative-learning capability and control behavior; they do not show the measured connectome is necessary or superior. The untrained value readout is zero by construction. Software tests asserting learning are regression checks, not independent evidence for biological fidelity. Nominal Wilson intervals in JSON summarize sampled choices within each block and are not uncertainty intervals for biological accuracy.

Commands:

```sh
.venv/bin/python build_learning.py /path/to/raw-data
.venv/bin/python learning.py --benchmark
.venv/bin/python learning.py --seed 7
.venv/bin/python -m unittest discover -p 'test_*.py' -v
```

`data/learning-circuit.json` contains the measured cells and edges plus source provenance. `data/learning-benchmark.json` contains 30 default experiments (six conditions × five seeds). A single exported experiment includes the trial log, probabilities, sampled choices, rewards, prediction errors, per-trial update norm, phase weight hashes, and largest final synaptic-gain changes. Full weight arrays are not exported; source, seed and parameters permit deterministic recomputation in this implementation. Do not interpret the largest changes as anatomically validated engrams.

Biological motivation (these sources do not validate this implementation):

- [Learning with reinforcement prediction errors in a model of the Drosophila mushroom body](https://www.nature.com/articles/s41467-021-22592-4), Nature Communications (2021).
- [Dopaminergic mechanism underlying reward-encoding of punishment omission during reversal learning in Drosophila](https://www.nature.com/articles/s41467-021-21388-w), Nature Communications (2021). This studies a specific aversive reversal mechanism; our appetitive task does not reproduce it.
- [Dopamine-mediated interactions between short- and long-term memory dynamics](https://www.nature.com/articles/s41586-024-07819-w), Nature (2024).

Next integration step: connect the learned odor-value output to explicit odor sampling and a downstream steering interface, while keeping measured pathways, assumed readouts, and actual locomotor physics distinguishable. Biological validation still requires held-out neural/behavioral recordings, calibrated receptor/compartment dynamics and stronger null models.

## Math classroom: addition versus memorization

Open `/math` on the same server (linked from the odor lab). This module asks whether a **constructed supervised classifier** on measured pathway support can learn addition examples. It is not a test of a biological fruit fly and is not connected to the walking or odor-choice controller.

Task: operands 0–9, with 19 answer classes (0–18). All 100 ordered operand pairs are enumerated. Ten unordered off-diagonal pairs, including both orders, are withheld: 20 test questions and 80 training questions. Diagonal pairs and the edge pairs (0,1)/(8,9) are protected from withholding so all answer classes have training examples. This is a constrained interpolation split, not a test of arbitrary numbers. A different seed changes the split and artificial encodings.

Each operand slot uses its own half of the actual PN population. Each digit selects a random 20% of that half, with no magnitude, sum, or arithmetic feature. Measured PN→KC weights and the top-5% sparsification produce input features. Actual MBON cells are randomly assigned to 19 answer pools. Initial KC→MBON edge magnitudes are aggregated per KC and answer pool. Plastic gain offsets are shared across existing edges from a KC to members of an answer pool; absent edge support remains absent. This grouping and all answer labels are artificial.

For a problem, feature `F[k,c] = KC_activity[k] * sum(log(1+count[k,MBON]))` over MBONs assigned to class c. Each output column is L2-normalized per example without using labels. Logits are `sum_k gain_offset[k,c] * F[k,c]`. Softmax produces answer probabilities. Zero offsets initially produce uniform probabilities; a fixed independent tiny tie-breaker picks top-1 answers for reporting. It is not a learned bias.

During teaching, supervised cross-entropy errors update gains with `learning_rate * F * (one_hot_teacher - probability) * PAM_gate`. Gains are clipped to [-0.95,8]. These bounds and learning parameters are engineering choices, not fitted physiological measurements. Teacher labels are computed by the experiment harness; the sum is never passed into the encoder or predictor. Every fifth epoch the frozen model is evaluated on training and test questions. **Do not select settings against this visible test curve and then call it an untouched validation set.** Subsequent model development requires a new independent test set.

Controls: no learning, fixed shuffled training labels, shuffled PN→KC destinations, and silenced KCs. Shuffled labels preserve the training label histogram. The interface distinguishes true addition accuracy from agreement with the supplied teacher labels. All comparisons use the same operand split and independent random streams for structure/encoding and teaching order. No-learning predictions are unchanged; test evaluation never updates weights. The test-weight hashes and supported-gain audit are included in exports.

**Initial fixed-default result (100 epochs, learning rate 0.15):** for seeds 7, 8 and 9, the measured pathway answers all 80 taught questions correctly and none of the 20 withheld questions correctly. The shuffled pathway has the same top-1 accuracies. Shuffled-label training closely fits the arbitrary taught labels. This is evidence of memorization by this supplied classifier and encoding, not acquisition of a general addition rule, a limitation of the biological fly, or superiority of the real connectome. We did not tune parameters to improve test performance after seeing these results.

The uniform-random expected accuracy is 1/19 (5.3%); the most-frequent-training-answer baseline is also shown for each test split. Twenty questions and three synthetic seeds are a small exploratory evaluation, not a biological or statistical proof.

```sh
.venv/bin/python arithmetic.py             # save one default lesson
.venv/bin/python arithmetic.py --benchmark # 5 conditions × 3 seeds
.venv/bin/python -m unittest discover -p 'test_*.py' -v
```

Saved results: `data/math-example.json`, `data/math-benchmark.json`. Browser exports include all 100 predictions and answer probabilities, split assignments, teacher labels for taught questions, learning curves, exact settings, anatomical source provenance, output-pool assignments, and audits. A quiz in the UI looks up that frozen model's prediction for the selected pair; it does not retrain or secretly compute the correct answer as a prediction.

## Inside the circuit: anatomy, membrane and sparse representations

Open `/explore`. Search actual body IDs or cell types, zoom/pan the directed graph, click a partner to follow its connections, and inspect incoming/outgoing synapse counts. The loaded graph contains **5,598 neurons and 215,407 directed edges**, the union of the existing motor and learning subsets. It is not the complete CNS: omitted cells and their connections are outside this viewer. Graph positions and moving dots are schematic; animation does not infer spike times, conduction delays or excitatory/inhibitory effects from anatomy.

The shape panel fetches the selected neuron's official coarse SWC centerline from the [MaleCNS v1.0 public release](https://male-cns.janelia.org/download/). Body 10360 is bundled for offline viewing; others download on selection and cache locally. The API returns the source URL and SHA-256. Original 8-nm coordinate units are converted to micrometers. Drag rotates the 3D projection; scroll zooms it. Centerlines do not supply synapse locations, branch identity, membrane properties or validated electrical compartments.

The membrane tab illustrates why a neuron can be more than a weighted sum: nonlinear voltage-dependent sodium activation/inactivation and potassium activation generate action potentials, recovery and temporal effects. `membrane.py` integrates classical Hodgkin–Huxley equations for a single membrane patch (Cm=1 µF/cm², gNa=120, gK=36, gLeak=0.3 mS/cm²; reversal potentials +50, −77, −54.387 mV). Input is applied from 10 to 40 ms. These are **generic squid-axon parameters, not measured fly physiology**; selecting another cell only changes the contextual identity. No morphology-based cable model or subcellular molecular reconstruction is claimed. See [NEURON's HH mechanism documentation](https://www.neuron.yale.edu/neuron/static/new_doc/modelspec/programmatic/mechanisms/mech.html).

A separate illustrative dendritic coincidence gate adds 4 µA/cm² if at least four of eight context bits are present. It is a supplied threshold rule, not measured computation of the selected neuron and not coupled to the math model. Blocking sodium channels eliminates the default model's spikes; opposing current can suppress them. Playback shows modeled voltage, ionic currents and channel gates, with schematic particles.

### SDR mathematics experiment

The third tab implements overlapping binary scalar codes, sparse KC winners and overlap visualization. It is **SDR-inspired, not Monty or a complete HTM implementation**. [Monty](https://github.com/thousandbrainsproject/tbp.monty) is a sensorimotor framework; no Monty runtime is installed. [Fergal Byrne's 2015 paCLA paper](https://arxiv.org/html/1509.08255v2) motivates sparse winners and contextual coincidence; temporal memory, learned dendritic segments and cortical columns are not implemented here. [Sean Pedersen's SDR discussion](https://seanpedersen.github.io/posts/sparse-distributed-representations/) provides further context. Neighboring display pixels are not necessarily neighboring biological cells.

Each operand uses its own half of the 314 actual PN cells, with eight active bits. Scalar windows shift by two positions per digit, so neighboring numbers overlap by six bits. Random SDRs match activity count; permuted scalar codes scramble digit order. Measured normalized PN→KC connectivity produces scores; the top 2% of 4,064 KCs with positive scores become binary winners. The same 80/20 mirrored-pair split as `/math` is used, with fresh seeds 101–103. The original plastic classifier uses the same measured KC→MBON support and artificial answer pools, now with these sparse features.

Two explicitly engineered numerical readouts accompany that classifier. The KC ridge decoder sums each KC's normalized answer-pool-supported features, normalizes the resulting KC vector, and fits a linear numerical target. The direct-input ridge baseline instead uses normalized operand SDRs, bypassing the connectome entirely. Both center features and targets using training rows only and solve ridge regression with fixed lambda=0.1. Predictions are rounded to the nearest integer and clipped to 0–18 for accuracy; raw predictions and test MAE are exported. These are single closed-form supervised fits, independent of teaching epoch count except epochs=0, which disables them. They do not use the classifier's local plasticity rule. Labels of unseen pairs never enter fitting.

**Exploratory fixed-setting results:** the scalar-SDR plastic classifier scores 0% on unseen pairs. The alternative KC ridge decoder scores 90%, 75%, and 90% (mean 85%); random SDR KC decoders score 20%, 25%, and 35%. Scalar SDRs with shuffled wiring score 75%, 70%, and 75%. Direct-input scalar and random-SDR baselines both score 100%. Therefore, success depends strongly on supplied encoding/readout assumptions, and this task does not establish that the biological connectome is necessary or superior. The direct numerical readout has an additive inductive bias suited to this task. These are small-digit interpolation tests, not general mathematical reasoning or evidence of fly arithmetic. Decoder comparisons were added during exploratory development; results are not a preregistered or independently replicated study.

Run `python sdr_math.py` to reproduce the 15 saved experiments (five representation/control combinations × three seeds). Each benchmark also stores both decoder baselines. `data/sdr-example.json` stores a full default run; the UI exports settings, all predictions, input codebooks, example active KC body IDs, split and weight audits. The membrane, math, odor-learning and movement models remain separate demonstrations. Integrating them into a physiological embodied agent would require additional measured dynamics and validation.

Verification: 34 unit tests cover existing behavior plus anatomy integrity, skeleton units, membrane responses and channel blocking, sparse-code overlap, deterministic controls and test-label isolation from decoder fitting.

## A connectome as a trainable visual network

Open `/vision`. This experiment trains the recurrent network itself, including weights on allowed anatomical edges and small computations inside each node. It is separate from the earlier arithmetic readouts and does not drive the embodied fly.

`build_vision_patch.py` extracts 209 traced right-side visual neurons across 19 assigned optic-lobe columns from the official MaleCNS annotations. The patch is centered at assigned hex coordinate (19,20), using radius 2 under `max(abs(dq),abs(dr),abs(dq-dr))`. Included types are L1/L2/L3, Mi1/Mi4/Mi9, C2/C3 and Tm1/Tm2/Tm9. All 1,421 measured directed edges between selected cells are retained, with original synapse counts and table provenance. The raw weights file is streamed batch by batch. T4/T5 and exterior cells/connections are omitted; this is not a complete biological motion pathway. Assigned columns provide an anatomical spatial index; their projection onto a planar hexagonal sensor grid is an engineering approximation, not calibrated eye optics.

Inputs are sampled intensities from 10-frame synthetic sequences of a translating, textured Gaussian disk, sometimes expanding. Each sequence independently draws position, velocity, initial radius, growth, texture phase/frequency and sensor noise. Nineteen scalar samples are scaled to [-1,1] and injected into corresponding L1/L2/L3 cells, bypassing photoreceptor dynamics. The targets are two image-plane velocity components (scaled by 0.075 for training) and a binary synthetic looming-hazard flag. The flag is 1 when growth >0.025 and the disk's extrapolated Gaussian radius four frames after the observed sequence exceeds its distance to the central sensor plus 0.45 field units. This is a defined image-space proxy, not collision physics, metric depth or 3D perception. The encoder receives no target labels, simulator velocity or growth parameters.

For node i with one or two state components, the supplied update is:

`h_i(t) = 0.5 h_i(t-1) + 0.5 tanh(sum_j W_ij h_j(t-1) + h_i(t-1) U_i + G_i x_column(i)(t) + b_i)`

The shared scalar weight W_ij transmits each state component along an allowed directed edge. U_i is a learned 1×1 or 2×2 local matrix; G_i and b_i are learned gains and biases. G_i is constrained to zero outside L1/L2/L3. Local state retention/mixing is an added dynamical assumption, including at cells without measured anatomical self-edges. A learned linear readout pools the final Tm1/Tm2/Tm9 states into two velocities and a hazard logit. Biological transmitter signs, synaptic dynamics, dendrites, spiking and ion channels are not reconstructed. Synapse counts are displayed as anatomy but deliberately do not initialize electrical strengths.

Four models see identical data and minibatch orders:

- Measured graph, one state per cell: 2,070 trainable parameters.
- Measured graph, two states per cell: 3,134 parameters.
- Rewired graph, two states per cell: 3,134 parameters. Directed double-edge swaps preserve each cell's incoming/outgoing degree and actual self-loops. Local parameters and input/output assignments match the real graph. Initial incoming-weight norms are matched node by node. The null does not preserve spatial or cell-type connectivity, and no mixing-time guarantee is claimed.
- Conventional dense recurrent network: 45 hidden units and 3,108 parameters. Learned mixing of all 19 input samples, dense recurrence, scalar local memory and the same activation/leak rule. This is a parameter-matched baseline for the two-state graph, not a match in state count, computation, input/readout geometry or wall time.

Training uses explicit NumPy backpropagation through all 10 frames and Adam (learning rate 0.004, beta1 0.9, beta2 0.999, epsilon 1e-8, global gradient norm capped at 2). Loss is mean squared error over the two normalized velocities plus 0.5 binary cross-entropy for hazard. Each run uses 256 training, 64 validation and 128 test sequences from independent RNG streams; the same three partitions are shared across its four models. Defaults use 35 epochs and batches of 32. Validation curves are displayed every five epochs but do not choose checkpoints. Test metrics are evaluated before and after training, not used by the optimizer. Epochs=0 is available as a no-training control.

The interface replays the first 12 test clips without selecting favorable examples. Predictions are final full-sequence outputs and remain fixed during replay; only sensory input and node states change frame by frame. Click a neuron, choose it from the selector, or follow an incoming source to inspect the actual saved state trajectory and its numerical update terms. Rewired edges are explicitly synthetic and have no displayed anatomical synapse count. Dense hidden units have no biological body ID. All graph animations display recurrent state variables, not electrical recordings.

**Fixed-default results across seeds 401–403:** average velocity RMSE is 0.0175 for measured scalar nodes, 0.0172 for measured two-state nodes, 0.0185 for rewired two-state nodes and 0.0157 for the dense recurrent baseline, in field units/frame. The training-mean baseline is about 0.0429. Hazard accuracy is 91.4%, 90.9%, 91.4% and 92.2%, respectively. Hazard is uncommon in these test samples (10–15% prevalence); always predicting no hazard already achieves 85–90% accuracy, so raw accuracy must be interpreted alongside the displayed probability Brier score and baseline. These small exploratory runs establish learnability of the supplied architecture on this synthetic task, not a statistically demonstrated advantage of the connectome or richer per-neuron models.

Removing all intercellular weights after training raises measured-graph motion RMSE to about 0.043, near the constant baseline. This shows the constructed graph models use their communication pathways; it does not show the particular biological topology is necessary. We restore the weights afterward and verify their hash. Tests also verify analytical gradients against finite differences, degree preservation, absent-edge constraints, evaluation without mutation, saved-checkpoint prediction reproduction and the displayed per-node state equation.

Reproduce:

```sh
OPENBLAS_NUM_THREADS=1 .venv/bin/python build_vision_patch.py /path/to/raw-data
OPENBLAS_NUM_THREADS=1 .venv/bin/python vision.py --benchmark
OPENBLAS_NUM_THREADS=1 .venv/bin/python -m unittest discover -p 'test_*.py' -q
```

The saved default full run is `data/vision-example.json`; summary comparisons are in `data/vision-benchmark.json`. Model weights for all three seeds and four architectures are stored as `data/vision-<seed>-<kind>.npz`. Browser retraining returns a new experiment without replacing the saved benchmark; use Export to keep it. JSON exports include all test targets/predictions, anatomical circuit/provenance, graph learned-edge weights, node-local parameters, displayed activity, validation curves, settings and audit hashes. The NPZ files also contain full input and output readout weights. Deterministic recomputation is supported within the same numerical environment; NumPy/BLAS version changes may cause small differences.

Related research: [Connectome-constrained networks predict neural activity across the fly visual system](https://www.nature.com/articles/s41586-024-07939-3). This prototype is independently implemented, does not use FlyVis weights, and has not been validated against biological recordings. Future steps would be a larger complete motion pathway, receptor/cell-type constraints, stronger matched nulls, unseen scene families and an embodied navigation task.

Current verification: **41 unit tests pass**, including all earlier modules and the visual-network checks.

### Trainable compound-eye layer

Two further architectures put a trainable eye in front of the measured two-state graph. Each column first passes through a photoreceptor stage with shared adaptation, `y_t = a·y_(t−1) + (1−a)·x_t` and output `g·(x_t − k·y_t)`, then a learned column-to-column projection into L1/L2/L3. **Retinotopic** (`eye_graph`) lets each column mix only with its six hex neighbours, like neural superposition; **unrestricted** (`eye_dense`) lets any column mix with any other. Both are created after all other random draws and start as an exact identity, so untrained they equal the measured graph, and all their other parameters match it. Gradients of the eye parameters are checked against finite differences in `test_vision_patch.py`. “Eye reset” re-evaluates each trained model with the eye returned to identity.

Three seeds, 35 epochs (means; lower is better except accuracy):

| Architecture | Parameters | Motion RMSE | Direction error | Hazard accuracy | Hazard Brier |
|---|---|---|---|---|---|
| Measured graph · 2 states | 3,134 | 0.0172 | 16.5° | 90.9% | 0.064 |
| + trainable eye, retinotopic | 3,240 | 0.0172 | 16.4° | 92.4% | 0.057 |
| + trainable eye, unrestricted | 3,498 | 0.0156 | 13.7° | 92.4% | 0.061 |
| Rewired graph · 2 states | 3,134 | 0.0185 | 18.8° | 91.4% | 0.070 |
| Dense recurrent baseline | 3,108 | 0.0157 | 14.5° | 92.2% | 0.067 |

Always predicting "no hazard" scores 86.7% accuracy (Brier 0.115). The retinotopic eye gives the best hazard calibration with 106 extra parameters and no motion benefit; the unrestricted eye gives the best motion, matching the dense network. In both, the learned photoreceptor adaptation stays near zero (`k` between −0.09 and 0.14) while gain rises to about 1.3 and each column mixes roughly 0.09 of each allowed neighbour, so the benefit comes from spatial pooling, not temporal adaptation. Resetting the eye after training hurts both (retinotopic motion RMSE 0.029, unrestricted hazard Brier 0.34), showing the downstream network adapts to the eye it trained with. Three seeds are too few to call any of these differences reliable.
