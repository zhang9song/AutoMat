"""
Relax a structure with MatterSim:
  • If --file <path.xyz|path.cif> is given, read that file;
  • otherwise fall back to an ASE‑built Si diamond test cell.
The relaxed structure is written to  relaxed.cif
"""

import argparse
import numpy as np
from ase.build import bulk
from ase.io import read, write
from mattersim.forcefield.potential import MatterSimCalculator
from mattersim.applications.relax import Relaxer
import warnings
# ignore mattersim.forcefield.potential inner FutureWarning
warnings.filterwarnings(
    "ignore",
    category=FutureWarning,
    module=r"mattersim\.forcefield\.potential"
)


def axis_with_smallest_span(atoms):
    """
    返回坐标跨度（max - min）最小的轴索引 (0=x, 1=y, 2=z)。
    若有并列最小，取第一个出现的。
    """
    spans = atoms.positions.ptp(axis=0)  # ptp = max - min
    return int(np.argmin(spans))


def add_noise_on_axis(atoms, noise_scale=0.05):
    """
    在跨度最小的轴添加随机噪声。
    参数:
        atoms       : ASE Atoms 对象
        noise_scale : 噪声幅度 (Å)
    """
    axis = axis_with_smallest_span(atoms)
    atoms.positions[:, axis] += noise_scale * np.random.randn(len(atoms))
    return axis, atoms


# ------------------ CLI ------------------
parser = argparse.ArgumentParser(
    description="Relax CIF/XYZ structure with MatterSim + BFGS"
)
parser.add_argument("-f", "--file",
    default="/home/aiprogram/output_final.cif",
    metavar="STRUCTURE.{xyz,cif}",
    help="input structure file (XYZ or CIF)",
)
parser.add_argument("-n", "--steps",
    type=int,
    default=300,
    help="max optimisation steps (default 500)",
)
parser.add_argument("--noise",
    type=float,
    default=0.04,
    help="random noise amplitude in Å (default 0.05 Å)",
)
args = parser.parse_args()


# ------------------ Load structure ------------------
if args.file:
    try:
        atoms = read(args.file)  # ASE auto‑detects format by extension
        axis, atoms = add_noise_on_axis(atoms, noise_scale=args.noise)
        print(
            f"Loaded structure from {args.file}  (atoms={len(atoms)}) "
            f"→ added noise on axis {['x','y','z'][axis]}"
        )
    except Exception as err:
        raise SystemExit(f"Cannot read {args.file}: {err}")
else:
    atoms = bulk("Si", "diamond", a=5.43)
    axis, atoms = add_noise_on_axis(atoms, noise_scale=0.1)
    print(
        "No --file supplied → using perturbed Si diamond test cell "
        f"(noise on axis {['x','y','z'][axis]})"
    )

# ------------------ Calculator & Relaxer ------------------
atoms.calc = MatterSimCalculator(
    load_path="MatterSim-v1.0.0-5M.pth",
    device="cuda",
)

relaxer = Relaxer(
    optimizer="BFGS",        # 可改为 "FIRE" 等
    filter="ExpCellFilter",
    constrain_symmetry=False,
)

converged, relaxed_atoms = relaxer.relax(atoms, steps=args.steps)

# ------------------ Output ------------------
write("relaxed.cif", relaxed_atoms)
print("Relaxation complete → saved to  relaxed.cif")