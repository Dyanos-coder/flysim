"""Download the data files the brain needs (not stored in the repository).

The server calls `ensure()` on startup, so a fresh clone just works. To fetch
them ahead of time (or the extra files used by the validation scripts):

    uv run python -m flysim.data                 # brain data (~135 MB)
    uv run python -m flysim.data --validation    # + FlyWire v630 reference (~90 MB)
"""

import argparse
import os
import sys
import time
import urllib.request
from pathlib import Path

from flysim.connectome import DATA_DIR

SHIU = "https://raw.githubusercontent.com/philshiu/Drosophila_brain_model/main"
ANNOTATIONS = "https://raw.githubusercontent.com/flyconnectome/flywire_annotations/main/supplemental_files"

# path in data/ -> source URL
BRAIN_FILES = {
    "Completeness_783.csv": f"{SHIU}/Completeness_783.csv",
    "Connectivity_783.parquet": f"{SHIU}/Connectivity_783.parquet",
    "neuron_annotations.tsv": f"{ANNOTATIONS}/Supplemental_file1_neuron_annotations.tsv",
}
VALIDATION_FILES = {
    "v630/completeness_630.csv": f"{SHIU}/2023_03_23_completeness_630_final.csv",
    "v630/connectivity_630.parquet": f"{SHIU}/2023_03_23_connectivity_630_final.parquet",
    "v630/sugarR.parquet": f"{SHIU}/results/example/sugarR.parquet",
}


def _download(url: str, dest: Path):
    """Download to a .part file, then rename: an interrupted download never
    leaves a truncated file behind."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_name(dest.name + ".part")
    with urllib.request.urlopen(url) as response, open(part, "wb") as out:
        total = int(response.headers.get("Content-Length") or 0)
        done, last = 0, 0.0
        while chunk := response.read(1 << 16):
            out.write(chunk)
            done += len(chunk)
            now = time.monotonic()
            if now - last > 0.5:
                last = now
                pct = f"{100 * done / total:3.0f} %" if total else ""
                print(f"\r  {dest.name}: {done / 1e6:6.1f} / {total / 1e6:.1f} Mo {pct}", end="", flush=True)
    os.replace(part, dest)
    print(f"\r  {dest.name}: {done / 1e6:.1f} Mo OK" + " " * 20, flush=True)


def _safe_console():
    """A Windows console in a legacy code page can't print every character;
    never let a message crash the program."""
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(errors="replace")


def ensure(validation: bool = False) -> None:
    """Download whichever data files are missing."""
    _safe_console()
    files = dict(BRAIN_FILES)
    if validation:
        files.update(VALIDATION_FILES)
    missing = {name: url for name, url in files.items() if not (DATA_DIR / name).exists()}
    if not missing:
        return
    print(f"Téléchargement des données du cerveau dans {DATA_DIR} (une seule fois)...")
    for name, url in missing.items():
        try:
            _download(url, DATA_DIR / name)
        except OSError as e:
            sys.exit(f"\nÉchec du téléchargement de {name} ({e}).\n"
                     f"Vérifie ta connexion, ou télécharge-le à la main depuis {url}\n"
                     f"et place-le dans {DATA_DIR / name}")


def main():
    parser = argparse.ArgumentParser(description="Télécharge les données de FlySim")
    parser.add_argument("--validation", action="store_true",
                        help="ajoute les fichiers de référence des scripts de validation (FlyWire v630)")
    ensure(validation=parser.parse_args().validation)
    print("Données prêtes.")


if __name__ == "__main__":
    main()
