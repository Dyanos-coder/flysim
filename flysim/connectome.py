"""Load the FlyWire connectome in the format of Shiu et al.'s brain model repo.

Files (from github.com/philshiu/Drosophila_brain_model):
- completeness csv: index = FlyWire root ids; row order defines neuron indices.
- connectivity parquet: one row per connected pair with Presynaptic_Index,
  Postsynaptic_Index and "Excitatory x Connectivity" (signed synapse count).

The first load is cached as a compressed .npz next to the parquet.
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np

DATA_DIR = Path(__file__).resolve().parent.parent / "data"


@dataclass
class Connectome:
    root_ids: np.ndarray  # (n,) int64 FlyWire root ids, in neuron index order
    pre: np.ndarray  # (e,) int32
    post: np.ndarray  # (e,) int32
    signed_counts: np.ndarray  # (e,) int32

    @property
    def n_neurons(self) -> int:
        return len(self.root_ids)

    def index_of(self, root_ids) -> np.ndarray:
        """Neuron indices for FlyWire root ids (raises if any is missing)."""
        lookup = {int(r): i for i, r in enumerate(self.root_ids)}
        return np.array([lookup[int(r)] for r in root_ids], dtype=np.int64)


def load(
    completeness_csv: Path = DATA_DIR / "Completeness_783.csv",
    connectivity_parquet: Path = DATA_DIR / "Connectivity_783.parquet",
) -> Connectome:
    cache = connectivity_parquet.with_suffix(".npz")
    if cache.exists() and cache.stat().st_mtime >= connectivity_parquet.stat().st_mtime:
        z = np.load(cache)
        return Connectome(z["root_ids"], z["pre"], z["post"], z["signed_counts"])

    import pyarrow.csv
    import pyarrow.parquet

    ids_table = pyarrow.csv.read_csv(completeness_csv)
    root_ids = ids_table.column(0).to_numpy().astype(np.int64)
    table = pyarrow.parquet.read_table(
        connectivity_parquet,
        columns=["Presynaptic_Index", "Postsynaptic_Index", "Excitatory x Connectivity"],
    )
    c = Connectome(
        root_ids=root_ids,
        pre=table.column(0).to_numpy().astype(np.int32),
        post=table.column(1).to_numpy().astype(np.int32),
        signed_counts=table.column(2).to_numpy().astype(np.int32),
    )
    np.savez_compressed(
        cache, root_ids=c.root_ids, pre=c.pre, post=c.post, signed_counts=c.signed_counts
    )
    return c
