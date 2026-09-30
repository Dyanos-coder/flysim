# FlySim

Interactive browser simulation of a fruit fly (*Drosophila melanogaster*) whose
behavior is driven by a model of its whole brain.

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

The neurons decide whether and which way; the takeoff, flight and grooming movements
are scripted motor programs (the ventral nerve cord is not in FlyWire). Two things
are scripted reflexes outside the brain model and labeled as such in the UI:
walking up an odor gradient (no descending neuron in the model encodes odor side)
and flight itself. Walking without odor is a user command: the model has no
spontaneous locomotor drive.

The right-hand panel is a live map of the brain: every neuron at its FlyWire
position, colored by super-class, flashing when it spikes.

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
uv run python -m flysim.server            # add --no-brain for physics only
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
| `flysim/world.py` | Fly + arena, senses, brain → behavior, push/items/threat |
| `flysim/locomotion.py` | Vectorized CPG walking controller |
| `flysim/export.py` | Serializes MuJoCo geoms/poses for the browser |
| `flysim/server.py` | Physics thread + FastAPI/WebSocket streaming |
| `web/` | Three.js frontend |
| `scripts/` | Validation, probing and behavior tests |

## Roadmap

1. ✅ Sandbox: one physical fly, walking, push it with the mouse
2. ✅ Brain: FlyWire connectome driving feeding, escape, turning, grooming
3. ✅ Live brain map; odor search, food consumption, throwing, flight (scripted)
4. Several flies interacting
5. Courtship and (scripted) life cycle
