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

The project is versioned with git and published with `tools/setup_repo.sh` (default `merolaagi/malecns-lab`, private; set `MALECNS_VISIBILITY=public` to change). Each new iteration arrives as a `malecns-lab*.zip` in `~/Downloads`. `tools/sync.sh` applies the newest one to `~/Sites/malecns-lab`, removes files the iteration dropped, leaves untracked local files such as `raw-data/` and `.venv` alone, runs the full test suite, commits with the top entry of `CHANGES.md` as the message, pushes, and moves the zip to `~/Downloads/malecns-lab-applied/`. It refuses to run over uncommitted local edits and commits nothing if tests fail. It also skips a zip older than the committed version and archives leftover older downloads. After each successful sync the lab server is restarted in the background (`tools/lab.sh start|stop|restart|status|open|logs`; set `MALECNS_NO_RESTART=1` to skip). `tools/autosync.sh install` runs all of this automatically whenever a new zip lands in Downloads (macOS launchd agent, log in `~/Library/Logs/malecns-lab-autosync.log`); `tools/autosync.sh watch` does the same in a Terminal tab if macOS blocks background access to Downloads. Paths can be overridden with `MALECNS_PROJECT` and `MALECNS_DOWNLOADS`.

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

The atlas and neuron workbench show both for every cell and flag low coverage, mixed evidence and cells whose aggregate transmitter is "unclear" but whose synapses mostly agree. In the workbench, PN inputs to Kenyon cells take their sign from the per-synapse dominant transmitter instead of being assumed excitatory. The network models themselves are unchanged in this step; transmitter predictions come from EM appearance, and receptor identity, including glutamate's real sign, is still not measured.

```sh
cd raw-data && B=https://storage.googleapis.com/flyem-male-cns/v1.0/connectome-data/flat-connectome && curl -fL -o body-stats.feather $B/body-stats-male-cns-v1.0-minconf-0.5.feather && curl -fL -o tbar-neurotransmitters.feather $B/tbar-neurotransmitters-male-cns-v1.0.feather && cd .. && .venv/bin/python build_quality.py raw-data
.venv/bin/python -m unittest -v test_quality.py
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
