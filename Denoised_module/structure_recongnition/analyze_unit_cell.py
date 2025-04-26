#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
根据输入超胞 CIF 结构，反复缩减：
  1) 先尝试 spglib primitive；
  2) 若仍较大，再用整数因子折叠 motif；
  3) 若二者都失败，退出。
直到无法继续缩减或原子数 ≤ 50。
结果写到  output_final.cif
"""
import sys, numpy as np
from pymatgen.core import Structure, Lattice


def try_primitive(struct):
    for tol in (0.1, 0.25):
        try:
            cand = struct.get_primitive_structure(tolerance=tol)
        except Exception:
            continue
        if len(cand) < len(struct):
            return cand
    return None

def try_motif(struct, max_atoms=200):
    frac, species = struct.frac_coords, [s.specie for s in struct]
    best_u, best_f = None, None
    for a in range(2,13):
        for b in range(2,13):
            for c in (1,2,3,4):
                if a*b*c > len(struct): continue
                folded = np.mod(frac * [a,b,c], 1)
                for dec in (4,3,2):
                    uniq, clash = {}, False
                    for sp,fc in zip(species, np.round(folded,dec)):
                        k = tuple(fc)
                        if k in uniq and uniq[k]!=sp: clash=True; break
                        uniq[k]=sp
                    if clash: continue
                    if best_u is None or len(uniq)<len(best_u):
                        best_u, best_f = uniq, (a,b,c)
                if best_u and len(best_u)<=max_atoms: break
            if best_u and len(best_u)<=max_atoms: break
        if best_u and len(best_u)<=max_atoms: break
    if best_u and len(best_u)<len(struct):
        # 构造新的 lattice (方向不强行保持，与现有脚本一致)
        new_mat = struct.lattice.matrix.copy()
        new_mat[0] /= best_f[0]
        new_mat[1] /= best_f[1]
        new_mat[2] /= best_f[2]
        return Structure(Lattice(new_mat),
                         list(best_u.values()),
                         list(best_u.keys()))
    return None

def shrink_once(struct):
    new = try_primitive(struct)
    if new: return new
    return try_motif(struct)

# ------------------------------------------------------------------
# 主流程：循环缩减
# ------------------------------------------------------------------
cif_in = '/home/aiprogram/project/yaotian/phase_structure_reconstruction/structure_recongnition/orthogonal_2dm-767_supercell_12x12x1_dose60000_sampling0.1_iDPC_V3_reconstructed.cif'
cur = Structure.from_file(cif_in)
print(f"起始原子数: {len(cur)}")

for step in range(1, 4):          # 最多 14 次，防止死循环
    if len(cur) <= 50:
        print("≤50 atoms，停止。")
        break
    nxt = shrink_once(cur)
    if nxt is None or len(nxt) >= len(cur):
        print("已无法进一步缩减。")
        break
    print(f"Step {step}:  {len(cur)} → {len(nxt)} atoms")
    cur = nxt

cur.to(filename="output_final.cif")
print(f"最终结构 output_final.cif (atoms = {len(cur)})")
