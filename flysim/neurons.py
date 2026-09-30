"""Named neuron groups of the FlyWire v783 brain, from the FlyWire annotations
(github.com/flyconnectome/flywire_annotations, Supplemental file 1).

Groups are sets of neuron indices in the connectome's order, split by side
where it matters (left/right) so the body can feed lateralized stimuli in and
read lateralized commands out.
"""

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from flysim.connectome import DATA_DIR, Connectome

ANNOTATIONS = DATA_DIR / "neuron_annotations.tsv"

# Olfactory receptor neuron classes (by glomerulus) for two stock odors.
# Vinegar activates DM1 (Or42b), DM4 (Or59b), DP1m (Ir64a) and VM2 (Or43b);
# geosmin, a mould odor flies avoid, activates DA2 (Or56a) exclusively.
ODOR_ORNS = {
    "vinegar": ["ORN_DM1", "ORN_DM4", "ORN_DP1m", "ORN_VM2"],
    "geosmin": ["ORN_DA2"],
}


@dataclass
class NeuronGroups:
    by_name: dict[str, np.ndarray] = field(default_factory=dict)
    positions: np.ndarray | None = None  # (n, 3) soma/anchor positions in nm, NaN if unknown
    cell_type: np.ndarray | None = None  # (n,) object
    super_class: np.ndarray | None = None  # (n,) object
    side: np.ndarray | None = None  # (n,) object: "left" / "right" / "center" / None

    def __getitem__(self, name: str) -> np.ndarray:
        return self.by_name[name]

    def names(self) -> list[str]:
        return list(self.by_name)


def _read_annotations(path: Path):
    import pyarrow.csv as pc

    table = pc.read_csv(
        path,
        parse_options=pc.ParseOptions(delimiter="\t"),
        convert_options=pc.ConvertOptions(
            column_types={"root_id": "int64"},
            strings_can_be_null=True,
            include_columns=[
                "root_id", "pos_x", "pos_y", "pos_z", "super_class", "cell_class",
                "cell_sub_class", "cell_type", "side", "top_nt", "top_nt_conf", "known_nt",
            ],
        ),
    )
    return {name: table.column(name).to_numpy(zero_copy_only=False) for name in table.column_names}


INHIBITORY_NT = ("gaba", "glutamate", "histamine")


def _neuron_sign(known_nt, top_nt) -> int:
    """+1 / -1 for one neuron, or 0 if unknown. Literature-known transmitter
    first (e.g. "gaba, nitric oxide"; "gaba-negative" is not a positive
    identification), else the neuron-level prediction."""
    if known_nt:
        tokens = [t.strip().lower() for t in known_nt.replace(";", ",").split(",")]
        tokens = [t for t in tokens if t and not t.endswith("-negative")]
        if any(t in INHIBITORY_NT for t in tokens):
            return -1
        if any(t in ("acetylcholine", "dopamine", "serotonin", "octopamine") for t in tokens):
            return 1
    if top_nt:
        return -1 if top_nt.lower() in INHIBITORY_NT else 1
    return 0


def apply_dale_signs(connectome: Connectome, path: Path = ANNOTATIONS) -> int:
    """Enforce one sign per presynaptic neuron (Dale's principle) in place.

    The connectivity file signs each connection from per-synapse transmitter
    predictions, so some neurons end up with mixed or wrong-signed outputs --
    e.g. GABAergic antennal-lobe local neurons marked excitatory, which makes the
    whole network run away. Returns the number of connections whose sign changed.
    """
    a = _read_annotations(path)
    index = {int(r): i for i, r in enumerate(connectome.root_ids)}
    sign = np.zeros(connectome.n_neurons, dtype=np.int32)
    for r, known, top in zip(a["root_id"], a["known_nt"], a["top_nt"]):
        i = index.get(int(r))
        if i is not None:
            sign[i] = _neuron_sign(known, top)
    pre_sign = sign[connectome.pre]
    fixed = np.where(pre_sign != 0, np.abs(connectome.signed_counts) * pre_sign, connectome.signed_counts)
    changed = int(np.count_nonzero(fixed != connectome.signed_counts))
    connectome.signed_counts = fixed.astype(np.int32)
    return changed


def apply_antennal_lobe_fixes(connectome: Connectome, path: Path = ANNOTATIONS) -> tuple[int, int]:
    """Stop the antennal lobe from locking into runaway activity. In place.

    Even with Dale's principle, any olfactory input drives the AL local neurons
    (ALLNs) into a self-sustaining high-activity state that swamps the brain.
    Two corrections, both about how this LIF model represents ALLNs:
    - ALLNs with no literature transmitter and a low-confidence prediction
      (< 0.5, mostly "serotonin"/"dopamine" at ~0.3) are treated as GABAergic,
      the transmitter of most AL local neurons.
    - Cholinergic excitatory LNs excite projection neurons mainly through gap
      junctions (Yaksi & Wilson 2010), which this model lacks; their chemical
      output is removed rather than letting it act as strong excitation.
    With both, a lateral odor gives a clean ipsilateral DNa02 steering signal
    and activity decays after the odor stops. Returns (n_resigned, n_silenced).
    """
    a = _read_annotations(path)
    index = {int(r): i for i, r in enumerate(connectome.root_ids)}
    uncertain = np.zeros(connectome.n_neurons, dtype=bool)
    cholinergic = np.zeros(connectome.n_neurons, dtype=bool)
    for r, cls, known, conf in zip(a["root_id"], a["cell_class"], a["known_nt"], a["top_nt_conf"]):
        i = index.get(int(r))
        if i is None or cls != "ALLN":
            continue
        if known and "acetylcholine" in known.lower():
            cholinergic[i] = True
        elif not known and (conf is None or conf != conf or conf < 0.5):
            uncertain[i] = True
    counts = connectome.signed_counts
    m = uncertain[connectome.pre]
    counts[m] = -np.abs(counts[m])
    counts[cholinergic[connectome.pre]] = 0
    return int(uncertain.sum()), int(cholinergic.sum())


def load(connectome: Connectome, path: Path = ANNOTATIONS) -> NeuronGroups:
    a = _read_annotations(path)
    n = connectome.n_neurons
    index = {int(r): i for i, r in enumerate(connectome.root_ids)}
    rows = np.array([index.get(int(r), -1) for r in a["root_id"]])
    ok = rows >= 0

    def column(name, fill):
        out = np.full(n, fill, dtype=object)
        out[rows[ok]] = a[name][ok]
        return out

    sup, cls, sub = column("super_class", None), column("cell_class", None), column("cell_sub_class", None)
    ctype, side = column("cell_type", None), column("side", None)
    pos = np.full((n, 3), np.nan)
    pos[rows[ok]] = np.stack([a["pos_x"], a["pos_y"], a["pos_z"]], axis=1)[ok].astype(float)

    groups = NeuronGroups(positions=pos, cell_type=ctype, super_class=sup, side=side)

    def add(name, mask, lateral=True):
        groups.by_name[name] = np.flatnonzero(mask)
        if lateral:
            for s in ("left", "right"):
                groups.by_name[f"{name}_{s}"] = np.flatnonzero(mask & (side == s))

    # --- sensory inputs
    add("sugar", (cls == "gustatory") & np.isin(sub, ["sugar", "sugar/low_salt"]))
    add("bitter", (cls == "gustatory") & (sub == "bitter"))
    add("water", (cls == "gustatory") & (sub == "water"))
    for odor, types in ODOR_ORNS.items():
        add(f"odor_{odor}", np.isin(ctype, types))
    add("looming", ctype == "LPLC2")
    add("touch", (cls == "mechanosensory") & (sub == "grooming"))
    add("photoreceptors", (sup == "sensory") & (cls == "visual"))
    add("visual_projection", sup == "visual_projection")
    add("sensory_nonvisual", (sup == "sensory") & (cls != "visual") & (cls != "gustatory"))
    add("gustatory", (sup == "sensory") & (cls == "gustatory"))
    add("head_bristle", (cls == "mechanosensory") & (sub == "head bristle"))

    # --- outputs
    add("giant_fiber", ctype == "DNp01")
    add("turn_dna02", ctype == "DNa02")
    add("turn_dna01", ctype == "DNa01")
    add("forward_dnp09", ctype == "DNp09")
    add("backward_mdn", ctype == "MDN")
    add("proboscis_mn", sub == "proboscis_motor_neuron", lateral=False)
    add("descending", sup == "descending")
    add("motor", sup == "motor", lateral=False)
    return groups
