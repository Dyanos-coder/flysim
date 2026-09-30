# FlySim

Interactive browser simulation of fruit flies (*Drosophila melanogaster*) whose
behavior is driven by a model of their whole brain -- one connectome brain per fly.

- **Body & physics:** [FlyGym](https://github.com/NeLy-EPFL/flygym) / NeuroMechFly on MuJoCo.
- **Brain:** the FlyWire v783 connectome (138,639 neurons, 15M connections) as the
  leaky integrate-and-fire model of [Shiu et al., Nature 2024](https://github.com/philshiu/Drosophila_brain_model),
  re-implemented in numba (event-driven, active-neuron list) to run in real time on a CPU.
- **Walking:** CPG network driving leg steps recorded from a real fly, steered by a
  two-sided descending signal (left, right).
- **Rendering:** Three.js in the browser, streamed over a WebSocket.

## What the brain does

The world turns what the fly senses into Poisson input on sensory neurons and reads
descending / motor neurons back out. These links emerge from the wiring:

| You do | Sensory neurons | Neurons read out | Fly does |
|---|---|---|---|
| Sugar drop at the proboscis or feet | sugar GRNs | proboscis MNs (CB0911, CB0871) | stops, extends proboscis, eats the drop |
| Bitter drop | bitter GRNs | (proboscis MNs stay silent) | no feeding |
| Threat ball / thrown ball | LPLC2 (per eye) | giant fiber DNp01, DNa02 (contralateral) | takes off away from it |
| Touch the head / body | bristle mechanosensory | DNg grooming DNs | grooming |
| Vinegar / fruit juice nearby | vinegar-sensitive ORNs | — | (olfactory circuits light up) |

Nobody tells the fly what to do. Like a real fly's, its sensory and visual
projection neurons carry ongoing activity (`brain_link.SPONTANEOUS_HZ`); through
the wiring it makes the walking neurons fluctuate on their own: DNp09 fires in
bouts (forward walking), DNa02/DNa01 steer, MDN would drive backward walking.
The fly walks, stops and turns by itself (`scripts/spontaneous.py`,
`docs/autonomie.md`).

The neurons decide whether and which way; the stepping pattern (recorded steps),
takeoff, flight and grooming movements are animated (the ventral nerve cord is
not in FlyWire; flight has no aerodynamics). The model doesn't encode where an
odor comes from, so the fly finds food by exploring.

The right-hand panel is a live map of the selected fly's brain: every neuron at
its FlyWire position, colored by super-class, flashing when it spikes.

## GPU brains

With an NVIDIA GPU and CuPy (`cupy-cuda13x` plus the pip CUDA runtime/NVRTC, no
CUDA toolkit needed), all brains run together on the GPU (`flysim/brain_gpu.py`):
one copy of the wiring, one row of state per fly, two CUDA kernels per 0.5 ms
step recorded as a CUDA graph. It matches the CPU brain (r = 0.998 on the sugar
test, `scripts/validate_gpu.py`) and runs 6 always-active brains at ~1.4x real
time on an RTX 3050 laptop GPU. Without a usable GPU the server falls back to
the CPU brain. With several flies the limit is then MuJoCo physics on the CPU.

## Several flies

Each fly has its own MuJoCo simulation and its own brain (the 15M-connection
wiring is shared in memory, the neural state is not). Physics runs on one thread
per fly (MuJoCo releases the GIL) in 10 ms chunks; between chunks the world
exchanges what flies share: food, balls, bumping, and vision -- a fly flying at
another one looms on its LPLC2 neurons and can set off its escape, so one
startled fly can scatter the group. Walking neighbors don't trigger escapes.
The simulation starts with one fly; "Ajouter une mouche" adds more (up to 6)
while it runs, each built in the background with its own body and brain.

## Model corrections (flysim/neurons.py)

The raw v783 connectivity runs away into self-sustaining activity; two fixes:
- Dale's principle: one sign per neuron (literature transmitter, else prediction).
- Antennal lobe local neurons: uncertain-transmitter LNs treated as GABAergic,
  cholinergic LNs' chemical output removed (they act through gap junctions).

## Validation

`scripts/validate_brain.py` reruns Shiu et al.'s sugar experiment on FlyWire v630
and compares per-neuron rates with their Brian2 model: r = 0.999 (slope 1.01) at
dt = 0.1 ms, r = 0.998 with rates ~10% high at the dt = 0.5 ms used live.

## Run

```sh
uv sync
uv run python -m flysim.server            # --flies N at start (default 1), --no-brain, --cpu-brain
```

Then open http://localhost:8000. Data files go in `data/` (not versioned): from
[Shiu et al.](https://github.com/philshiu/Drosophila_brain_model) `Completeness_783.csv`,
`Connectivity_783.parquet`; from [flywire_annotations](https://github.com/flyconnectome/flywire_annotations)
`Supplemental_file1_neuron_annotations.tsv` saved as `neuron_annotations.tsv`.

## Layout

| Path | Role |
|---|---|
| `flysim/brain.py` | LIF network engine (numba) |
| `flysim/connectome.py` | Loads the connectome (cached as .npz) |
| `flysim/neurons.py` | Named neuron groups from annotations, sign corrections |
| `flysim/brain_link.py` | Brain thread, sensory drive in, firing rates out |
| `flysim/fly.py` | One fly: own physics, senses, brain → behavior, flight, grooming |
| `flysim/world.py` | Shared world: flies, food, balls, bumping, parallel stepping |
| `flysim/locomotion.py` | Vectorized CPG walking controller |
| `flysim/export.py` | Serializes MuJoCo geoms/poses for the browser |
| `flysim/server.py` | Physics thread + FastAPI/WebSocket streaming |
| `web/` | Three.js frontend |
| `scripts/` | Validation, probing and behavior tests |

## Roadmap

1. ✅ Sandbox: one physical fly, walking, push it with the mouse
2. ✅ Brain: FlyWire connectome driving feeding, escape, turning, grooming
3. ✅ Live brain map; odor search, food consumption, throwing, flight (scripted)
4. ✅ Several flies interacting (shared food, startle cascades)
5. Courtship and (scripted) life cycle
