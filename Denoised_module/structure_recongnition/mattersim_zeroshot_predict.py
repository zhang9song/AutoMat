import torch
from ase.build import bulk
from ase.io import read, write
from ase.units import GPa
from mattersim.forcefield import MatterSimCalculator
import warnings
# ignore mattersim.forcefield.potential inner FutureWarning
warnings.filterwarnings(
    "ignore",
    category=FutureWarning,
    module=r"mattersim\.forcefield\.potential"
)


device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Running MatterSim on {device}")


input_file = '/home/aiprogram/relaxed.cif'
# ------------------ Load structure ------------------
if input_file:
    try:
        atoms = read(input_file)     # ASE auto-detects format by extension
        print(f"Loaded structure from {input_file}  (atoms={len(atoms)})")
    except Exception as err:
        raise SystemExit(f"Cannot read {input_file}: {err}")
else:
    # initialize the structure of silicon
    atoms = bulk("Si", "diamond", a=5.43)
    # perturb the structure
    atoms.positions += 0.1 * np.random.randn(len(atoms), 3)
    print("No --file supplied → using perturbed Si diamond test cell")

atoms.calc = MatterSimCalculator(load_path="MatterSim-v1.0.0-5M.pth", device=device)

print(f"Energy (eV)                 = {atoms.get_potential_energy()}")
print(f"Energy per atom (eV/atom)   = {atoms.get_potential_energy()/len(atoms)}")
print(f"Forces of first atom (eV/A) = {atoms.get_forces()[0]}")
print(f"Stress[0][0] (eV/A^3)       = {atoms.get_stress(voigt=False)[0][0]}")
print(f"Stress[0][0] (GPa)          = {atoms.get_stress(voigt=False)[0][0] / GPa}")