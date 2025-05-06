#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
agent_pipeline_ds.py  (updated)

STEM big‑image → patch inference → template match → structure reconstruction →
primitive‑cell reduction → MatterSim relaxation.

改动要点
--------

1. **`inference_large_image_cv2_pil`** 现在显式 **return**
   重建结果的 PNG 路径，脚本直接使用其返回值来确定 `recon_png`。

2. **当用户没有给元素组成**
   * 在选出最终 label 后，自动用 label 文件名里的 *material_id*
     去 `metadata_csv` 里查元素集合，供 `process_image()` 使用。
   * 同时 `refine_top1()` 保持“用户优先”的过滤逻辑：
     – 若用户给了元素 → 用用户元素过滤；
     – 否则直接取 `top_matches[0]`（最快）。

其余流程保持不变。
"""

import os, re, json, shutil, time, warnings
import cv2
from pathlib import Path
from typing import Dict, Any, List, Optional

import numpy as np
import torch
from ase.io import read, write
from ase.units import GPa
from pymatgen.core import Structure

# ===== 请确认以下自定义模块可 import =====
from test_large_image_model_batch import inference_large_image_cv2_pil
from structure_paired_reconstruction.batch_structure_paired import match_one, load_metadata, extract_material_id
from structure_paired_reconstruction.batch_atoms_analysis import process_image, shrink_once
from mattersim.forcefield          import MatterSimCalculator
from mattersim.applications.relax  import Relaxer
# ========================================

warnings.filterwarnings("ignore", category=FutureWarning, module=r"mattersim")

# ---------- 辅助函数 ---------- #
_ELEMENT_PATTERN = re.compile(r'(?:elements?|元素)\s*[:=]\s*([A-Za-z,\s]+)', re.I)


def shrink_or_window(sup_cif, out_cif_path, MAX_NUM_ITER=4, MAX_ATOMS_NUM=50):
    cur = Structure.from_file(str(sup_cif))
    for step in range(1, MAX_NUM_ITER):
        if len(cur) <= MAX_ATOMS_NUM:
            print(f"≤{MAX_ATOMS_NUM} atoms，停止。")
            break
        nxt = shrink_once(cur)
        if nxt is None or len(nxt) >= len(cur):
            print("已无法进一步缩减。")
            break
        print(f"Step {step}: {len(cur)} → {len(nxt)} atoms")
        cur = nxt
    cur.to(filename=str(out_cif_path))
    
    
def parse_elements_from_text(text: str) -> Optional[List[str]]:
    m = _ELEMENT_PATTERN.search(text)
    if not m:
        return None
    return [e.strip().capitalize() for e in re.split(r'[,\s]+', m.group(1)) if e.strip()]


def span_min_axis(atoms):
    return int(np.argmin(atoms.positions.ptp(axis=0)))


def refine_top1(top_matches, user_elements, metadata):
    """
    若 `user_elements` 给定 → 用它们过滤；否则直接返回 top‑1
    """
    if not user_elements:
        return top_matches[0][0]

    target = set(user_elements)
    filtered = []
    for name, dist in top_matches:
        mid   = extract_material_id(name)
        elems = metadata.get(mid, set())
        if elems == target:
            filtered.append((name, dist))
    return (filtered or top_matches)[0][0]


# ---------- 主流程 ---------- #
def run_agent_pipeline(
    image_path      : str,
    user_message    : str,
    *,
    work_root       : str,
    weight_path     : str,
    label_dir       : str,
    metadata_csv    : str,
    max_atoms       : int  = 50,
    max_shrink_iter : int  = 4,
    relax_steps     : int  = 500,
    noise_amp       : float= 0.05,
    device = "cuda" if torch.cuda.is_available() else "cpu"
) -> Dict[str, Any]:

    t0       = time.time()
    img_p    = Path(image_path).expanduser().resolve(strict=True)
    root     = Path(work_root).expanduser().resolve()
    d_recon  = root / "01_recon"
    d_label  = root / "02_label"
    d_cif    = root / "03_recon_cif"
    d_relax  = root / "04_relax"
    for d in (d_recon, d_label, d_cif, d_relax):
        d.mkdir(parents=True, exist_ok=True)

    # === 1. PATCH 推理重建 ===
    print(f"\n=== [1] PATCH infer  {img_p.name} ===")
    recon_png = inference_large_image_cv2_pil(
        str(img_p), weight_path,
        crop_size=128, stride=64, batch_size=32, device=device
    )
    # === 1. PATCH 推理重建 ===
    print(f"\n=== [1] PATCH infer  {img_p.name} ===")
    recon_arr = inference_large_image_cv2_pil(
        str(img_p), weight_path,
        crop_size=128, stride=64, batch_size=32, device=device
        )
    recon_png = d_recon / f"{img_p.stem}_recon.png"     # ① 定义保存路径
    cv2.imwrite(str(recon_png), recon_arr)  # recon 是 uint8 numpy(H×W)
    print("Reconstructed image saved:", recon_png)      # ③ 后续继续用 Path

    # === 2. 模板匹配 ===
    print(f"\n=== [2] Template matching ===")
    metadata = load_metadata(Path(metadata_csv))
    user_elems = parse_elements_from_text(user_message)
    if user_elems:
        print("User‑provided elements:", user_elems)
    else:
        print("⚠️No element info in message – will infer from label/CSV")

    top_matches = match_one(recon_png, Path(label_dir), topk=3, min_area=5, max_dist=None, bin_width=5.0)
    best_label_name = refine_top1(top_matches, user_elems, metadata)

    src_label = Path(label_dir) / best_label_name
    dst_label = d_label / best_label_name
    shutil.copy2(src_label, dst_label)
    print("Chosen label:", best_label_name)

    # 若用户没给元素 → 用 label 的 material_id 去 CSV 查
    if not user_elems:
        mid = extract_material_id(best_label_name)
        user_elems = sorted(metadata.get(mid, []))
        if user_elems:
            print(f"Inferred elements from CSV: {user_elems}")
        else:
            print("⚠️  Cannot infer elements from CSV – process_image will use default logic")

    # === 3. 图像 → CIF & 缩减 ===
    print(f"\n=== [3] Image→CIF & shrink (≤{max_atoms} atoms) ===")
    
    # 3‑1  从 label PNG 得到 Atoms
    atoms = process_image(dst_label, user_elems)        # ← 仍然只返回 atoms
    if atoms is None or len(atoms) == 0:
        raise RuntimeError("process_image failed to detect atoms!")
    
    # 3‑2  保存 super‑cell CIF
    mid_match = re.search(r"(2dm-\d+)", dst_label.stem, re.I)
    mid = mid_match.group(1).lower() if mid_match else f"tmp-{uuid.uuid4().hex[:6]}"
    cif_super = d_cif / f"{mid}_reconstructed.cif"
    write(cif_super, atoms, format="cif", wrap=False)
    print("Super‑cell CIF:", cif_super)
    
    # 3‑3  调你已有的 shrink_or_window → output_final.cif
    cif_final = d_cif / "output_final.cif"
    shrink_or_window(cif_super, cif_final)                  # ★ 你的现成函数
    print(f"Reduced cell saved: {cif_final}")


    # === 4. MatterSim Relax ===
    print(f"\n=== [4] MatterSim relaxation (device={device}) ===")

    atoms_relax = read(cif_final)
    axis_min    = span_min_axis(atoms_relax)
    atoms_relax.positions[:, axis_min] += noise_amp * np.random.randn(len(atoms_relax))

    atoms_relax.calc = MatterSimCalculator(load_path="MatterSim-v1.0.0-5M.pth", device=device)
    relaxer = Relaxer(optimizer="BFGS", filter=None, constrain_symmetry=False)
    converged, atoms_relaxed = relaxer.relax(atoms_relax, steps=relax_steps)

    cif_relaxed = d_relax / "relaxed.cif"
    write(cif_relaxed, atoms_relaxed)

    E   = atoms_relaxed.get_potential_energy()
    F0  = atoms_relaxed.get_forces()[0]
    sxx = atoms_relaxed.get_stress(voigt=False)[0][0]

    print(f"Converged: {converged}")
    print(f"Energy (eV)               = {E:.6f}")
    print(f"Energy per atom (eV/atom) = {E/len(atoms_relaxed):.6f}")
    print(f"First‑atom force (eV/Å)   = {F0}")
    print(f"Stress xx (GPa)           = {sxx/GPa:.6f}")
    print(f"Total elapsed             = {time.time()-t0:.1f} s")

    return {
        "relaxed_cif"      : str(cif_relaxed),
        "energy_eV"        : float(E),
        "energy_per_atom"  : float(E/len(atoms_relaxed)),
        "force_first_atom" : F0.tolist(),
        "stress_xx_GPa"    : float(sxx/GPa),
        "converged"        : bool(converged),
    }


# ---------- CLI ---------- #
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="DeepSeek‑V3 STEM→Property agent")
    parser.add_argument("--image",      default='/home/aiprogram/project/yaotian/phase_structure_reconstruction/data_generation/large_data_test/img/orthogonal_2dm-26_supercell_12x12x1_dose10000_sampling0.1_iDPC_V3.png', help="big STEM image path or attachment")
    parser.add_argument("--workdir",    default='/home/aiprogram/project/yaotian/phase_structure_reconstruction/baseline', help="working root dir")
    parser.add_argument("--weights",    default='/home/aiprogram/project/yaotian/phase_structure_reconstruction/MOE_model_weights/moe_model.ckpt', help="patch‑inference weight")
    parser.add_argument("--label-dir",  default='/home/aiprogram/project/yaotian/phase_structure_reconstruction/data_generation/label', help="template label directory (*.png)")
    parser.add_argument("--meta-csv",   default='/home/aiprogram/project/yaotian/phase_structure_reconstruction/baseline/property.csv', help="material_id‑elements csv")
    parser.add_argument("--user-text",  default="请帮我分析这张图，元素: Al,Sb，剂量 30k",    help="original user message text")
    args = parser.parse_args()

    info = run_agent_pipeline(
        image_path   = args.image,
        user_message = args.user_text,
        work_root    = args.workdir,
        weight_path  = args.weights,
        label_dir    = args.label_dir,
        metadata_csv = args.meta_csv,
    )

    print("\n=== Pipeline Finished ===")
    print(json.dumps(info, indent=4))
