#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
extract_best_periodic_region.py

从一个 2D CIF 结构中自动提取一个 200×200 Å 的最佳周期性区域，
评分准则（优先级从高到低）：
 1. 周期性（FFT 谱中离散峰能量占比）
 2. 分布均匀性（最近邻距离变异系数）
 3. 原子密度与全局平均密度偏差

输出 shrink.cif，包括该区域内所有原子，晶胞尺寸为 200×200×原始 c。
"""

import numpy as np
from ase import io
from ase.data import atomic_numbers
from ase.io import write
from ase import Atoms
from ase.neighborlist import neighbor_list, natural_cutoffs
from scipy.spatial import cKDTree
import os


# -------- 用户配置 --------
INPUT_CIF   = "/home/aiprogram/best_reconstructed_image_reconstructed.cif"  # 输入CIF 文件路径
OUTPUT_CIF  = "shrink.cif"              # 输出最佳区域CIF
WIN_SIZE    = 15.0    # 窗口边长 (Å)
COARSE_STEP = 5.0     # 粗扫描步长 (Å)
FINE_STEP   = 1.0      # 细扫描步长 (Å)
GRID_RES    = 0.2      # 频域分析网格分辨率 (Å)
# --------------------------

def load_structure(path):
    """加载 ASE Atoms，对应2D结构，只取xy坐标。"""
    atoms = io.read(path)
    pos_xy = atoms.get_positions()[:, :2]
    a, b, c = atoms.cell.lengths()
    return atoms, pos_xy, (a, b, c)

def periodic_score(subset_xy, win_size, grid_res):
    """
    周期性评分：将原子投影到网格后做 FFT，
    取除 DC 外前若干峰能量 / 总谱能量。
    """
    # 构建网格
    nx = int(np.ceil(win_size/grid_res))
    ny = int(np.ceil(win_size/grid_res))
    grid = np.zeros((ny, nx), float)
    # 原子到网格索引
    idx = ((subset_xy) / grid_res).astype(int)
    idx[:,0] = np.clip(idx[:,0], 0, nx-1)
    idx[:,1] = np.clip(idx[:,1], 0, ny-1)
    grid[idx[:,1], idx[:,0]] = 1.0
    # FFT 谱
    F = np.fft.fft2(grid)
    P = np.abs(F)**2
    P[0,0] = 0.0
    flat = P.ravel()
    # 选前 10 个峰
    if flat.size <= 10:
        top = flat
    else:
        top = flat[np.argpartition(flat, -10)[-10:]]
    return top.sum() / (flat.sum() + 1e-12)

def uniform_score(subset_xy):
    """
    均匀性评分：最近邻距离的变异系数 CV = sigma/mu，
    返回 1 - CV，裁剪到 [0,1]。
    """
    tree = cKDTree(subset_xy)
    d, _ = tree.query(subset_xy, k=2)
    nn = d[:,1]
    mu, sigma = nn.mean(), nn.std()
    if mu < 1e-6:
        return 0.0
    cv = sigma/mu
    return max(0.0, 1 - cv)

def density_score(N, area, avg_density):
    """
    密度评分：1 - |ρ_region - ρ_avg|/ρ_avg，裁剪到 [0,1]。
    """
    rho = N/area
    diff = abs(rho - avg_density)/max(avg_density,1e-12)
    return max(0.0, 1 - diff)

def evaluate_region(x0, y0, pos_xy, total_area, avg_density):
    """计算窗口(x0,y0)->(x0+WIN_SIZE,y0+WIN_SIZE)的三项评分。"""
    mask = ((pos_xy[:,0]>=x0)&(pos_xy[:,0]<x0+WIN_SIZE) &
            (pos_xy[:,1]>=y0)&(pos_xy[:,1]<y0+WIN_SIZE))
    subset = pos_xy[mask]
    N = len(subset)
    if N == 0:
        return None  # 空区跳过
    # 周期性
    p_score = periodic_score(subset - [x0,y0], WIN_SIZE, GRID_RES)
    # 均匀性
    u_score = uniform_score(subset)
    # 密度
    d_score = density_score(N, WIN_SIZE*WIN_SIZE, avg_density)
    return (p_score, u_score, d_score)

def find_best_window(pos_xy, cell_ab, total_area):
    """粗细两阶段滑窗，返回最佳窗口左下角坐标及评分。"""
    avg_density = len(pos_xy)/total_area
    x_min, y_min = pos_xy.min(axis=0)
    x_max_start = pos_xy[:,0].max() - WIN_SIZE
    y_max_start = pos_xy[:,1].max() - WIN_SIZE

    best = ((-1,-1), (-1.0,-1.0,-1.0))
    # 粗扫描
    xs = np.arange(x_min, x_max_start+1e-6, COARSE_STEP)
    ys = np.arange(y_min, y_max_start+1e-6, COARSE_STEP)
    for x in xs:
        for y in ys:
            scores = evaluate_region(x, y, pos_xy, total_area, avg_density)
            if scores is None: continue
            if scores > best[1]:
                best = ((x,y), scores)

    # 细扫描
    (cx, cy), base_scores = best
    xs = np.arange(max(x_min, cx-COARSE_STEP),
                   min(x_max_start, cx+COARSE_STEP)+1e-6, FINE_STEP)
    ys = np.arange(max(y_min, cy-COARSE_STEP),
                   min(y_max_start, cy+COARSE_STEP)+1e-6, FINE_STEP)
    for x in xs:
        for y in ys:
            scores = evaluate_region(x, y, pos_xy, total_area, avg_density)
            if scores is None: continue
            if scores > best[1]:
                best = ((x,y), scores)

    return best

def extract_and_write(atoms, pos_xy, best_xy, output_cif):
    """根据 best_xy 提取子区域原子，重设晶胞并写出 shrink.cif"""
    x0, y0 = best_xy
    mask = ((pos_xy[:,0]>=x0)&(pos_xy[:,0]<x0+WIN_SIZE) &
            (pos_xy[:,1]>=y0)&(pos_xy[:,1]<y0+WIN_SIZE))
    sub = atoms[mask]             # ASE Atoms 支持布尔索引
    # 平移坐标到 (0,0)
    new_pos = sub.get_positions()
    new_pos[:,0] -= x0
    new_pos[:,1] -= y0
    sub.set_positions(new_pos)
    # 新晶胞
    a_new = [WIN_SIZE, 0, 0]
    b_new = [0, WIN_SIZE, 0]
    c_new = atoms.cell[2]  # 保留原 c 方向
    sub.set_cell([a_new, b_new, c_new])
    sub.set_pbc((False, False, False))
    sub = merge_close_atoms(sub)
    write(output_cif, sub, format="cif")
    print(f"最佳区域导出至 {OUTPUT_CIF}，包含 {len(sub)} 个原子。")


def merge_close_atoms(atoms: Atoms, min_dist: float = 1.0) -> Atoms:
    """合并距离 < *min_dist* Å 的原子（含跨 PBC）。

    规则：
      • 簇内元素一致 → 质心；
      • 元素不同     → 保留 Z 最大元素，坐标取该原子原位。
    """
    if len(atoms) == 0:
        return atoms.copy()

    # 1️⃣  构建近邻对（ASE neighbor_list 支持 PBC）
    idx_i, idx_j = neighbor_list("ij", atoms, cutoff=min_dist * 0.999)  # 乘 0.999 避免边界浮点误差
    pairs = list(zip(idx_i, idx_j))

    # 2️⃣  并查集聚类
    parent = list(range(len(atoms)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(i, j):
        pi, pj = find(i), find(j)
        if pi != pj:
            parent[pj] = pi

    for i, j in pairs:
        union(i, j)

    clusters = {}
    for idx in range(len(atoms)):
        clusters.setdefault(find(idx), []).append(idx)

    # 3. 生成新原子列表
    pos = atoms.get_positions()
    syms = np.array(atoms.get_chemical_symbols())
    Z = np.array([atomic_numbers[s] for s in syms])
    cell = atoms.get_cell().array  # 3×3

    new_pos, new_sym = [], []
    for idxs in clusters.values():
        if len(idxs) == 1:
            i = idxs[0]
            new_pos.append(pos[i])
            new_sym.append(syms[i])
            continue
        sub_syms = syms[idxs]
        cart_coords = pos[idxs]
        if np.all(sub_syms == sub_syms[0]):
            merged_coord = _pbc_mean(cart_coords, cell)
            new_pos.append(merged_coord)
            new_sym.append(sub_syms[0])
        else:
            heavy_idx = idxs[np.argmax(Z[idxs])]
            new_pos.append(pos[heavy_idx])
            new_sym.append(syms[heavy_idx])

    merged = Atoms(symbols=list(new_sym), positions=new_pos)
    merged.set_cell(atoms.get_cell())
    merged.set_pbc(atoms.get_pbc())
    return merged


if __name__ == "__main__":
    # 1. 读结构
    atoms, pos_xy, cell_abc = load_structure(INPUT_CIF)
    a, b, c = cell_abc
    total_area = a * b

    # 2. 寻找最佳窗口
    (bx, by), (p,u,d) = find_best_window(pos_xy, (a,b), total_area)
    print(f"最佳窗口左下角：({bx:.1f}, {by:.1f}) 评分(P,U,D)=({p:.3f},{u:.3f},{d:.3f})")

    # 3. 提取并写出 shrink.cif
    extract_and_write(atoms, pos_xy, (bx, by), cell_abc)
