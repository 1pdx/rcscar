# -*- coding: utf-8 -*-
"""
DRI / RadarCart 风格 CSV：从 Raw 生成 Filtered、Spatial，多文件功率域合成 Combined，并绘图。

原理概要（据本目录样例反推）：
- Filtered：按距离 R（米）将雷达回波强度（默认取 RCS00；可选与 RCS01 在同一距离样本内做线性功率相加
  后再转 dB）分箱到 0.1 m 栅格上聚合，再按需施加距离维平滑。参考图/工程里常见：Savitzky–Golay（保峰）、高斯平滑、滑动平均、
  中值滤波（抑尖峰）、指数平滑（EMA）。
- Spatial：把主目标在传感器平面上的位置 (DX00, DY00) 映射到显示平面 (X, Y)。
  样例数据上全局仿射变换 X=a*DX+b*DY+c、Y=d*DX+e*DY+f 拟合残差约 0.13（任意单位）。
- Combined：对 M1/M2/M3 三条 Filtered 曲线在每一距离 R 上，先把 dB 转为线性功率、取算术平均，
  再转回 dB：RCS_comb = 10*log10(mean(10^(RCS_i/10)))（非对 dB 算术平均）。
- 圆周极坐标（按钮）：解析 Raw，极角=ViewAng，半径=RCS+20；灰散点为 RCS00–19 簇；
  蓝线为 RCS00 按 ViewAng 排序后滑动均值（窗 30）再连线；可勾选并指定一个 Spatial CSV，与 Raw 拟合曲线做双曲线对比；
  圆周图可设 RCS 标定 (dB)（半径 = RCS+标定+20）。
"""

from __future__ import annotations

import os
import re
import sys
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk

# Windows 常见黑体：避免图例中文缺字警告 / 方块字
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

try:
    from scipy import signal as scipy_signal
    from scipy.ndimage import gaussian_filter1d
    _HAVE_SCIPY = True
except ImportError:
    scipy_signal = None  # type: ignore
    gaussian_filter1d = None  # type: ignore
    _HAVE_SCIPY = False


# ---------------------------------------------------------------------------
# 仿射系数：由本目录 1_M1_S1_Raw / 1_M1_S1_Spatial 最小二乘拟合（主散射体 DX00,DY00）
_SPATIAL_AFFINE = {
    "x": (0.00095597, 0.04917567, 0.41222245),  # X = a*DX + b*DY + c
    "y": (0.00063747, 0.01344333, -0.70462078),
}

# 界面下拉框文案 → smooth_rcs_series 的 method 参数
SMOOTH_METHOD_BY_LABEL = {
    "无": "none",
    "滑动平均": "moving_average",
    "Savitzky-Golay": "savgol",
    "高斯": "gaussian",
    "中值": "median",
    "指数平滑 (EMA)": "ema",
}

# UI 与逻辑共用的滤波类型（字符串键）
SMOOTH_METHODS = (
    "none",
    "moving_average",
    "savgol",
    "gaussian",
    "median",
    "ema",
)


def _ensure_odd(win: int, upper: int) -> int:
    win = max(3, min(int(win), upper))
    if win % 2 == 0:
        win -= 1
    return max(3, win)


def smooth_rcs_series(
    z: np.ndarray,
    method: str,
    *,
    ma_window: int = 5,
    sg_window: int = 11,
    sg_poly: int = 3,
    gaussian_sigma: float = 1.5,
    median_kernel: int = 5,
    ema_alpha: float = 0.2,
) -> np.ndarray:
    """
    对距离轴上已排序的 RCS（dB）序列做一维平滑。参考图常用 Savitzky–Golay / 高斯抑噪保形。
    需在环境中安装 scipy（见 requirements.txt）。
    """
    z = np.asarray(z, dtype=float).copy()
    n = len(z)
    if n < 3 or method in ("none", "", "无"):
        return z

    if not _HAVE_SCIPY and method not in ("none", "moving_average", "ema"):
        raise RuntimeError("当前滤波类型需要 scipy，请执行: pip install scipy")

    m = np.isfinite(z)
    if not np.any(m):
        return z

    # --- 滑动平均（纯 numpy，可不依赖 scipy） ---
    def _moving_avg(arr: np.ndarray, k: int) -> np.ndarray:
        k = _ensure_odd(k, n if n % 2 == 1 else n - 1)
        if k > n:
            k = _ensure_odd(n - (0 if n % 2 == 1 else 1), n)
        pad = k // 2
        tmp = np.pad(arr, (pad, pad), mode="edge")
        cumsum = np.cumsum(np.insert(tmp, 0, 0))
        return (cumsum[k:] - cumsum[:-k]) / k

    # --- 指数平滑 ---
    def _ema(arr: np.ndarray, alpha: float) -> np.ndarray:
        alpha = float(np.clip(alpha, 0.01, 0.99))
        out = np.empty_like(arr)
        out[0] = arr[0]
        for i in range(1, len(arr)):
            out[i] = alpha * arr[i] + (1 - alpha) * out[i - 1]
        return out

    method = method.lower().strip()

    if method == "moving_average":
        return _moving_avg(z, ma_window)

    if method == "ema":
        return _ema(z, ema_alpha)

    assert scipy_signal is not None and gaussian_filter1d is not None

    if method == "savgol":
        w = _ensure_odd(sg_window, n if n % 2 == 1 else n - 1)
        p = int(np.clip(sg_poly, 1, w - 1))
        if w >= n:
            w = _ensure_odd(n - (0 if n % 2 == 1 else 1), n)
        if w < 3:
            return z
        p = min(p, w - 1)
        return scipy_signal.savgol_filter(z, window_length=w, polyorder=p, mode="nearest")

    if method == "gaussian":
        sigma = max(0.3, float(gaussian_sigma))
        return gaussian_filter1d(z, sigma=sigma, mode="nearest")

    if method == "median":
        k = _ensure_odd(median_kernel, n if n % 2 == 1 else n - 1)
        if k > n:
            k = _ensure_odd(n - (0 if n % 2 == 1 else 1), n)
        # scipy.signal.medfilt 对边界处理直观
        return scipy_signal.medfilt(z, kernel_size=k)

    return z


def loess_fit_curve(
    x: np.ndarray,
    y: np.ndarray,
    *,
    bandwidth_m: float = 2.0,
    min_points: int = 8,
    x_step_m: float = 0.1,
) -> tuple:
    """
    局部加权线性回归（LOESS/LOWESS 简化版），用于 RCS 曲线的非参数拟合。

    原理：
    - 在每一个查询点 xq 上，以高斯核 w = exp(-0.5 * (dx / bw)^2) 赋予邻域权重
    - 在带宽窗口内做加权最小二乘线性拟合 y = a*x + b
    - 不假设全局函数形式（与多项式拟合的本质区别）
    - 能自适应捕捉不同距离段上 RCS 的斜率变化

    参数：
        x, y: 观测数据点（已分箱/平滑后的距离与 RCS 值）
        bandwidth_m: 高斯带宽（米），控制平滑程度；越大曲线越平滑
        min_points: 有效权重窗口内最少点数，不足则返回 nan
        x_step_m: 输出栅格步长（米）

    返回：
        (xq, yq): 均匀栅格上的拟合值
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    if len(x) < min_points:
        return np.array([]), np.array([])

    # 在数据范围内生成均匀查询栅格
    x_min = float(np.nanmin(x))
    x_max = float(np.nanmax(x))
    nq = max(int((x_max - x_min) / float(x_step_m)) + 1, 2)
    xq = np.linspace(x_min, x_max, nq)

    bw = max(float(bandwidth_m), 1e-6)
    out = np.full_like(xq, np.nan, dtype=float)

    for i in range(int(xq.size)):
        xc = float(xq[i])
        dx = x - xc
        w = np.exp(-0.5 * (dx / bw) ** 2)
        mask = np.isfinite(w) & (w > 1e-6)
        if int(np.count_nonzero(mask)) < int(min_points):
            continue
        xm = x[mask]
        ym = y[mask]
        wm = w[mask]
        sw = float(np.sum(wm))
        if sw <= 1e-12:
            continue

        # 加权一阶最小二乘：y = a*x + b
        mx = float(np.sum(wm * xm) / sw)
        my = float(np.sum(wm * ym) / sw)
        x0 = xm - mx
        y0 = ym - my
        sxx = float(np.sum(wm * x0 * x0))
        if sxx <= 1e-12:
            out[i] = my
            continue
        sxy = float(np.sum(wm * x0 * y0))
        a = sxy / sxx
        b = my - a * mx
        out[i] = a * xc + b

    # 剔除输出中的 nan
    valid = np.isfinite(out)
    return xq[valid], out[valid]


def _rowwise_linear_power_rcs_db(df: pd.DataFrame, *, merge_rcs01: bool) -> np.ndarray:
    """
    同一原始行（同一采样时刻 / 同一 R）内 RCS（dB）序列。
    - merge_rcs01=False：仅用 RCS00。
    - merge_rcs01=True：对 RCS00、RCS01 转到线性功率相加后再取 dB：
      RCS_eff = 10*log10(10^(RCS00/10)+10^(RCS01/10))（与 Combined 多源合成思想一致）。
    """
    n = len(df)
    z00 = (
        pd.to_numeric(df["RCS00"], errors="coerce").to_numpy(dtype=float)
        if "RCS00" in df.columns
        else np.full(n, np.nan)
    )
    if not merge_rcs01:
        return z00
    z01 = (
        pd.to_numeric(df["RCS01"], errors="coerce").to_numpy(dtype=float)
        if "RCS01" in df.columns
        else np.full(n, np.nan)
    )
    p0 = np.where(np.isfinite(z00), 10.0 ** (z00 / 10.0), 0.0)
    p1 = np.where(np.isfinite(z01), 10.0 ** (z01 / 10.0), 0.0)
    psum = p0 + p1
    out = np.full(n, np.nan, dtype=float)
    ok = psum > 0.0
    out[ok] = 10.0 * np.log10(np.maximum(psum[ok], 1e-300))
    return out


def read_rrcs_csv(path: str) -> Tuple[np.ndarray, np.ndarray]:
    """读取 Combined / Filtered 等 R–RCS 表：自动定位 `R,RCS` 表头（元数据行数可能与 Raw 不一致）。"""
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        lines = f.readlines()
    start = 0
    for i, line in enumerate(lines):
        s = line.strip().lower()
        if s.startswith("r,") and "rcs" in s:
            start = i + 1
            break
    else:
        start = 5
    buf = "".join(lines[start:])
    from io import StringIO

    df = pd.read_csv(StringIO(buf), header=None, names=["R", "RCS"], engine="python")
    df.columns = [c.strip() for c in df.columns]
    r = pd.to_numeric(df["R"], errors="coerce").to_numpy()
    z = pd.to_numeric(df["RCS"], errors="coerce").to_numpy()
    m = np.isfinite(r) & np.isfinite(z)
    return r[m], z[m]


def read_combined_csv(path: str) -> Tuple[np.ndarray, np.ndarray]:
    """兼容旧名：与 :func:`read_rrcs_csv` 相同。"""
    return read_rrcs_csv(path)


PLOT_R_MIN = 5.0


def clean_rrcs_for_plot(
    r_vals: np.ndarray,
    rcs_vals: np.ndarray,
    *,
    min_r: float = PLOT_R_MIN,
) -> Tuple[np.ndarray, np.ndarray]:
    """清理绘图用 R–RCS 曲线：去掉无效值、R 小于阈值的点，并按 R 排序。"""
    r = np.asarray(r_vals, dtype=float)
    z = np.asarray(rcs_vals, dtype=float)
    m = np.isfinite(r) & np.isfinite(z) & (r >= float(min_r))
    if not np.any(m):
        return np.array([]), np.array([])

    r = r[m]
    z = z[m]
    order = np.argsort(r, kind="mergesort")
    r = r[order]
    z = z[order]

    uniq_r, inv = np.unique(r, return_inverse=True)
    if len(uniq_r) != len(r):
        sum_z = np.zeros(len(uniq_r), dtype=float)
        cnt_z = np.zeros(len(uniq_r), dtype=float)
        np.add.at(sum_z, inv, z)
        np.add.at(cnt_z, inv, 1.0)
        r = uniq_r
        z = sum_z / np.maximum(cnt_z, 1.0)

    return r, z


def interp_rrcs_to_x(
    target_r: np.ndarray,
    source_r: np.ndarray,
    source_z: np.ndarray,
    *,
    min_r: float = PLOT_R_MIN,
) -> Tuple[np.ndarray, np.ndarray]:
    """把 source 曲线插值到 target 的 R 坐标上，用于与参考曲线横坐标对齐。"""
    tr = np.asarray(target_r, dtype=float)
    tr = tr[np.isfinite(tr) & (tr >= float(min_r))]
    if len(tr) == 0:
        return np.array([]), np.array([])
    tr = np.unique(np.sort(tr))

    sr, sz = clean_rrcs_for_plot(source_r, source_z, min_r=min_r)
    if len(sr) == 0:
        return tr, np.full_like(tr, np.nan, dtype=float)
    if len(sr) == 1:
        zi = np.full_like(tr, np.nan, dtype=float)
        zi[np.isclose(tr, sr[0], rtol=0.0, atol=1e-9)] = sz[0]
        return tr, zi

    return tr, np.interp(tr, sr, sz, left=np.nan, right=np.nan)


def read_spatial_csv(path: str) -> pd.DataFrame:
    """读取 Spatial 导出：定位 `X,Y,RCS,...` 表头后读入表格。"""
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        lines = f.readlines()
    start = 0
    for i, line in enumerate(lines):
        s = line.strip().lower()
        if s.startswith("x,") and "y" in s and "rcs" in s:
            start = i + 1
            break
    else:
        start = 6
    buf = "".join(lines[start:])
    from io import StringIO

    df = pd.read_csv(StringIO(buf), header=None, engine="python")
    if df.shape[1] < 6:
        for c in range(df.shape[1], 6):
            df[c] = np.nan
    df = df.iloc[:, :6]
    df.columns = ["X", "Y", "RCS", "R", "VA", "IDX"]
    for c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def parse_meta_calibration_value(meta: Dict[str, str]) -> float:
    """从 Raw/Filtered 元数据行解析 Calibration 数值（dB）。"""
    cal_str = meta.get("Calibration", "0")
    cal_m = re.search(r"[-+]?\d*\.?\d+", cal_str)
    return float(cal_m.group(0)) if cal_m else 0.0


def _read_meta_lines(path: str) -> Dict[str, str]:
    meta: Dict[str, str] = {}
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for _ in range(5):
            line = f.readline()
            if not line:
                break
            line = line.strip()
            if "," in line:
                k, v = line.split(",", 1)
                meta[k.strip()] = v.strip()
    return meta


def read_raw_table(path: str) -> Tuple[Dict[str, str], pd.DataFrame]:
    meta = _read_meta_lines(path)
    df = pd.read_csv(path, skiprows=5)
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return meta, df


def collect_polar_scatter_va_rcs00_19(
    df: pd.DataFrame,
    *,
    rcs_radius_offset: float = 20.0,
    rcs_calibration_db: float = 0.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    灰色散点：每行同一 ViewAng 下，所有有限值 RCS00..RCS19。
    极角 = ViewAng（度→弧度），半径 = RCS(dBsm) + 标定 + rcs_radius_offset。
    """
    if "ViewAng" not in df.columns:
        raise ValueError("Raw 表缺少列 ViewAng")
    va = pd.to_numeric(df["ViewAng"], errors="coerce").to_numpy(dtype=float)
    m0 = np.isfinite(va)
    th0 = np.deg2rad(va)
    thetas: List[np.ndarray] = []
    rs: List[np.ndarray] = []
    for j in range(20):
        col = f"RCS{j:02d}"
        if col not in df.columns:
            break
        z = pd.to_numeric(df[col], errors="coerce").to_numpy(dtype=float)
        m = m0 & np.isfinite(z)
        if np.any(m):
            thetas.append(th0[m])
            rs.append(z[m] + float(rcs_calibration_db) + float(rcs_radius_offset))
    if not thetas:
        return np.array([]), np.array([])
    return np.concatenate(thetas), np.concatenate(rs)


def polar_primary_rcs00_line_smoothed(
    df: pd.DataFrame,
    *,
    rcs_radius_offset: float = 20.0,
    ma_window: int = 30,
    rcs_calibration_db: float = 0.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    蓝色曲线：每帧主反射体 RCS00 + ViewAng；先加标定 dB，再按 ViewAng 升序排列后，
    对 RCS00 沿该顺序做等权滑动均值（窗长 ma_window），半径 = 平滑后 RCS + offset，再连线。
    """
    if "ViewAng" not in df.columns or "RCS00" not in df.columns:
        raise ValueError("Raw 表需要列 ViewAng、RCS00")
    va = pd.to_numeric(df["ViewAng"], errors="coerce").to_numpy(dtype=float)
    z0 = pd.to_numeric(df["RCS00"], errors="coerce").to_numpy(dtype=float)
    m = np.isfinite(va) & np.isfinite(z0)
    va, z0 = va[m], z0[m]
    if len(va) == 0:
        return np.array([]), np.array([])
    z0 = z0 + float(rcs_calibration_db)
    o = np.argsort(va, kind="mergesort")
    va_s, z0_s = va[o], z0[o]
    k = max(1, int(ma_window))
    kernel = np.ones(k, dtype=float) / float(k)
    z_s = np.convolve(z0_s, kernel, mode="same")
    theta = np.deg2rad(va_s)
    r = z_s + float(rcs_radius_offset)
    return theta, r


def polar_spatial_rcs_line_smoothed(
    df_sp: pd.DataFrame,
    *,
    rcs_radius_offset: float = 20.0,
    ma_window: int = 30,
    rcs_calibration_db: float = 0.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Spatial：列 VA、RCS；先加标定，按 VA 升序后对 RCS 做与 Raw 相同窗长的滑动均值，再半径=值+offset。
    """
    if "VA" not in df_sp.columns or "RCS" not in df_sp.columns:
        raise ValueError("Spatial 表需要列 VA、RCS")
    va = pd.to_numeric(df_sp["VA"], errors="coerce").to_numpy(dtype=float)
    z = pd.to_numeric(df_sp["RCS"], errors="coerce").to_numpy(dtype=float)
    m = np.isfinite(va) & np.isfinite(z)
    va, z = va[m], z[m]
    if len(va) == 0:
        return np.array([]), np.array([])
    z = z + float(rcs_calibration_db)
    o = np.argsort(va, kind="mergesort")
    va_s, z_s = va[o], z[o]
    k = max(1, int(ma_window))
    kernel = np.ones(k, dtype=float) / float(k)
    z_sm = np.convolve(z_s, kernel, mode="same")
    theta = np.deg2rad(va_s)
    r = z_sm + float(rcs_radius_offset)
    return theta, r


def write_filtered_csv(
    out_path: str,
    meta: Dict[str, str],
    r_vals: np.ndarray,
    rcs_vals: np.ndarray,
    pad_nan_rows: int = 50,
) -> None:
    lines = [
        "Data Type,Filtered",
        f"Data File,{meta.get('Data File', '')}",
        f"Run Number,{meta.get('Run Number', '')}",
        f"Calibration,{meta.get('Calibration', '')}",
        "",
        "R,RCS",
    ]
    for _ in range(pad_nan_rows):
        lines.append("NaN,NaN")
    for r, z in zip(r_vals, rcs_vals):
        lines.append(f"{r:g},{z:g}")
    text = "\n".join(lines) + "\n"
    with open(out_path, "w", encoding="utf-8", newline="") as f:
        f.write(text)


def write_spatial_csv(
    out_path: str,
    meta: Dict[str, str],
    df_sp: pd.DataFrame,
) -> None:
    lines = [
        "Data Type,Spatial",
        f"Data File,{meta.get('Data File', '')}",
        f"Run Number,{meta.get('Run Number', '')}",
        f"Calibration,{meta.get('Calibration', '')}",
        "",
        "X,Y,RCS,R,VA,IDX",
    ]
    with open(out_path, "w", encoding="utf-8", newline="") as f:
        f.write("\n".join(lines) + "\n")
        df_sp.to_csv(f, header=False, index=False, lineterminator="\n")


def write_combined_csv(out_path: str, meta_files: List[Dict[str, str]], r_vals: np.ndarray, rcs_vals: np.ndarray) -> None:
    files_joined = ";".join(m.get("Data File", "") for m in meta_files) + ";"
    runs_joined = ";".join(str(m.get("Run Number", "")) for m in meta_files) + ";"
    cal = meta_files[0].get("Calibration", "") if meta_files else ""
    lines = [
        "Data Type,Combined",
        f"Data File,{files_joined}",
        f"Run Number,{runs_joined}",
        f"Calibration,{cal}",
        "",
        "R,RCS",
    ]
    for _ in range(50):
        lines.append("NaN,NaN")
    for r, z in zip(r_vals, rcs_vals):
        lines.append(f"{r:g},{z:g}")
    text = "\n".join(lines) + "\n"
    with open(out_path, "w", encoding="utf-8", newline="") as f:
        f.write(text)


def raw_to_filtered(
    df: pd.DataFrame,
    calibration: float,
    r_step: float = 0.1,
    pad_nan_rows: int = 50,
    smooth_method: str = "moving_average",
    ma_window: int = 5,
    sg_window: int = 11,
    sg_poly: int = 3,
    gaussian_sigma: float = 1.5,
    median_kernel: int = 5,
    ema_alpha: float = 0.2,
    use_rcs_field: str = "RCS00",
    apply_calibration: bool = False,
    merge_rcs01: bool = False,
    use_loess: bool = False,
    loess_bandwidth_m: float = 2.0,
    loess_min_points: int = 8,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    返回与导出文件对齐的距离栅格上的 RCS。

    流程（二选一）：
    - LOESS 启用：分箱 → LOESS 非参数拟合（跳过平滑，直接处理带噪分箱数据）
    - LOESS 关闭：分箱 → smooth_rcs_series（SG/高斯/EMA 等）

    LOESS 是非参数回归，自带抗噪能力，不需要前置平滑。
    """
    r_all = pd.to_numeric(df["R"], errors="coerce").to_numpy()
    if merge_rcs01:
        if "RCS00" not in df.columns:
            raise ValueError("merge_rcs01 需要列 RCS00")
        z_all = _rowwise_linear_power_rcs_db(df, merge_rcs01=True)
    else:
        if use_rcs_field not in df.columns:
            raise ValueError(f"列不存在: {use_rcs_field}")
        z_all = pd.to_numeric(df[use_rcs_field], errors="coerce").to_numpy()
    m = np.isfinite(r_all) & np.isfinite(z_all)
    r_all, z_all = r_all[m], z_all[m]
    if len(r_all) == 0:
        return np.array([]), np.array([])

    r_min = float(np.nanmin(r_all))
    r_max = float(np.nanmax(r_all))
    # 与样例一致：从 5 m 起至最大距离，步长 0.1（若最小距离 >5，则从 floor 开始）
    start = min(5.0, np.floor(r_min * 10) / 10)
    end = np.ceil(r_max * 10) / 10
    edges = np.arange(start, end + r_step * 0.5, r_step)
    centers = edges

    out_r: List[float] = []
    out_z: List[float] = []
    for rc in centers:
        sel = (r_all >= rc - r_step / 2) & (r_all < rc + r_step / 2)
        if not np.any(sel):
            continue
        val = float(np.nanmean(z_all[sel]))
        out_r.append(rc)
        out_z.append(val)

    out_r = np.asarray(out_r, dtype=float)
    out_z = np.asarray(out_z, dtype=float)

    # LOESS 与平滑互斥：LOESS 自带抗噪，直接处理分箱数据
    if use_loess and len(out_r) >= loess_min_points:
        out_r, out_z = loess_fit_curve(
            out_r, out_z,
            bandwidth_m=loess_bandwidth_m,
            min_points=loess_min_points,
            x_step_m=r_step,
        )
    else:
        out_z = smooth_rcs_series(
            out_z,
            smooth_method,
            ma_window=ma_window,
            sg_window=sg_window,
            sg_poly=sg_poly,
            gaussian_sigma=gaussian_sigma,
            median_kernel=median_kernel,
            ema_alpha=ema_alpha,
        )

    if apply_calibration:
        out_z = out_z + calibration

    return out_r, out_z


def raw_to_spatial(df: pd.DataFrame, merge_rcs01: bool = False) -> pd.DataFrame:
    a, b, cx = _SPATIAL_AFFINE["x"]
    d, e, fy = _SPATIAL_AFFINE["y"]
    rcs_series = _rowwise_linear_power_rcs_db(df, merge_rcs01=merge_rcs01)

    rows = []
    idx = 1
    for j, (_, row) in enumerate(df.iterrows()):
        dx = row.get("DX00")
        if pd.isna(dx):
            rows.append([np.nan, np.nan, np.nan, np.nan, np.nan, idx])
            idx += 1
            continue
        dy = float(row.get("DY00", np.nan))
        x = a * float(dx) + b * dy + cx
        y = d * float(dx) + e * dy + fy
        rcs = float(rcs_series[j])
        rr = row.get("R", np.nan)
        va = row.get("ViewAng", np.nan)
        rows.append([x, y, rcs, rr, va, idx])
        idx += 1

    out = pd.DataFrame(rows, columns=["X", "Y", "RCS", "R", "VA", "IDX"])
    return out


def combine_filtered_power_mean(
    list_r: List[np.ndarray],
    list_z: List[np.ndarray],
) -> Tuple[np.ndarray, np.ndarray]:
    """在多条曲线重叠的距离区间内，插值到统一 R 栅格后在线性功率域取平均再转 dB。"""
    if not list_r:
        return np.array([]), np.array([])
    r_min = max(float(np.min(r)) for r in list_r)
    r_max = min(float(np.max(r)) for r in list_r)
    if r_min >= r_max:
        r_min = float(min(np.min(r) for r in list_r))
        r_max = float(max(np.max(r) for r in list_r))

    step = 0.1
    grid = np.arange(np.ceil(r_min * 10) / 10, np.floor(r_max * 10) / 10 + step * 0.5, step)

    powers: List[np.ndarray] = []
    for r, z in zip(list_r, list_z):
        # 假定 R 单调可插值；若同一 R 出现多次，先按 R 排序聚合
        o = np.argsort(r)
        rs, zs = np.asarray(r)[o], np.asarray(z)[o]
        zi = np.interp(grid, rs, zs)
        powers.append(10 ** (zi / 10.0))

    mean_lin = np.mean(np.stack(powers, axis=0), axis=0)
    mean_db = 10 * np.log10(np.maximum(mean_lin, 1e-30))
    return grid, mean_db


class App(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title("DRI Raw → Filtered / Spatial / Combined")
        self.geometry("960x780")

        self.raw_paths: List[str] = []
        self.out_dir = tk.StringVar(value=os.getcwd())
        self.ref_combined_path = tk.StringVar(value="")
        self.ref_filtered_path = tk.StringVar(value="")
        self.ref_spatial_path = tk.StringVar(value="")
        self.save_filtered = tk.BooleanVar(value=False)
        self.save_spatial = tk.BooleanVar(value=False)
        self.save_combined = tk.BooleanVar(value=False)
        self._last_spatial_dfs: List[pd.DataFrame] = []
        self._plot_win: Optional[tk.Toplevel] = None
        self._plot_fig = None  # matplotlib.figure.Figure，避免弹窗销毁后悬空
        self._v_plot_ready = False
        self._v_all_r: List[np.ndarray] = []
        self._v_all_z: List[np.ndarray] = []
        self._v_rc = np.array([])
        self._v_zc = np.array([])
        self._v_labels = ["M1", "M2", "M3"]
        self._v_meta_calibration = 0.0
        self.plot_rcs_offset_db = tk.DoubleVar(value=0.0)
        self.polar_overlay_spatial = tk.BooleanVar(value=False)
        self.polar_compare_spatial_path = tk.StringVar(value="")
        self.polar_rcs_cal_db = tk.DoubleVar(value=0.0)

        frm = ttk.Frame(self, padding=8)
        frm.pack(fill=tk.BOTH, expand=True)

        ttk.Label(frm, text="选择 1～3 个 *_Raw.csv（顺序对应 M1→M3）").pack(anchor="w")
        bf = ttk.Frame(frm)
        bf.pack(fill=tk.X, pady=4)
        ttk.Button(bf, text="添加 Raw 文件…", command=self._pick_raw).pack(side=tk.LEFT)
        ttk.Button(bf, text="清空列表", command=self._clear_raw).pack(side=tk.LEFT, padx=6)

        self.listbox = tk.Listbox(frm, height=5)
        self.listbox.pack(fill=tk.X)

        ttk.Label(frm, text="输出目录").pack(anchor="w", pady=(8, 0))
        od = ttk.Frame(frm)
        od.pack(fill=tk.X)
        ttk.Entry(od, textvariable=self.out_dir).pack(side=tk.LEFT, fill=tk.X, expand=True)
        ttk.Button(od, text="浏览…", command=self._pick_out).pack(side=tk.LEFT, padx=4)

        ref_fr = ttk.LabelFrame(frm, text="参考对比（可选：与本次计算结果叠加）", padding=6)
        ref_fr.pack(fill=tk.X, pady=6)
        r1 = ttk.Frame(ref_fr)
        r1.pack(fill=tk.X)
        ttk.Button(r1, text="参考 Combined…", command=self._pick_ref_combined).pack(side=tk.LEFT)
        self.show_ref_combined = tk.BooleanVar(value=True)
        ttk.Checkbutton(r1, text="叠加 R–RCS 图", variable=self.show_ref_combined).pack(side=tk.LEFT, padx=10)
        ttk.Entry(ref_fr, textvariable=self.ref_combined_path).pack(fill=tk.X, pady=(2, 4))

        r2 = ttk.Frame(ref_fr)
        r2.pack(fill=tk.X)
        ttk.Button(r2, text="参考 Filtered…", command=self._pick_ref_filtered).pack(side=tk.LEFT)
        self.show_ref_filtered = tk.BooleanVar(value=False)
        ttk.Checkbutton(r2, text="叠加 R–RCS 图", variable=self.show_ref_filtered).pack(side=tk.LEFT, padx=10)
        ttk.Entry(ref_fr, textvariable=self.ref_filtered_path).pack(fill=tk.X, pady=(2, 4))

        r3 = ttk.Frame(ref_fr)
        r3.pack(fill=tk.X)
        ttk.Button(r3, text="参考 Spatial…", command=self._pick_ref_spatial).pack(side=tk.LEFT)
        self.show_ref_spatial = tk.BooleanVar(value=True)
        ttk.Checkbutton(r3, text="叠加 Spatial 散点图", variable=self.show_ref_spatial).pack(side=tk.LEFT, padx=10)
        ttk.Entry(ref_fr, textvariable=self.ref_spatial_path).pack(fill=tk.X, pady=(2, 0))

        opts = ttk.LabelFrame(frm, text="距离维滤波 / 平滑（分箱后作用于 RCS 序列；参考图常用 SG 或高斯）", padding=6)
        opts.pack(fill=tk.X, pady=6)

        self.smooth_label = tk.StringVar(value="Savitzky-Golay")
        ttk.Label(opts, text="算法").grid(row=0, column=0, sticky="w")
        _smooth_keys = list(SMOOTH_METHOD_BY_LABEL.keys())
        self.combo_smooth = ttk.Combobox(
            opts,
            textvariable=self.smooth_label,
            values=_smooth_keys,
            state="readonly",
            width=20,
        )
        self.combo_smooth.grid(row=0, column=1, sticky="w")

        self.ma_window = tk.IntVar(value=5)
        self.sg_window = tk.IntVar(value=11)
        self.sg_poly = tk.IntVar(value=3)
        self.gauss_sigma = tk.DoubleVar(value=1.5)
        self.median_kernel = tk.IntVar(value=5)
        self.ema_alpha = tk.DoubleVar(value=0.2)
        self.apply_cal = tk.BooleanVar(value=False)
        self.merge_rcs01 = tk.BooleanVar(value=False)
        self.smooth_combined_too = tk.BooleanVar(value=False)
        self.show_channels = tk.BooleanVar(value=True)
        # LOESS 非参数回归
        self.use_loess = tk.BooleanVar(value=False)
        self.loess_bandwidth = tk.DoubleVar(value=2.0)
        self.loess_min_points = tk.IntVar(value=8)
        self.loess_on_combined = tk.BooleanVar(value=True)

        ttk.Label(opts, text="滑动窗(奇数) MA/SG/中值").grid(row=1, column=0, sticky="w", pady=2)
        ttk.Spinbox(opts, from_=3, to=51, increment=2, textvariable=self.ma_window, width=6).grid(
            row=1, column=1, sticky="w"
        )
        ttk.Label(opts, text="SG 窗长(奇数)").grid(row=2, column=0, sticky="w", pady=2)
        ttk.Spinbox(opts, from_=5, to=51, increment=2, textvariable=self.sg_window, width=6).grid(
            row=2, column=1, sticky="w"
        )
        ttk.Label(opts, text="SG 多项式阶数").grid(row=2, column=2, sticky="e", padx=(12, 4))
        ttk.Spinbox(opts, from_=1, to=5, textvariable=self.sg_poly, width=6).grid(row=2, column=3, sticky="w")

        ttk.Label(opts, text="高斯 σ（采样点）").grid(row=3, column=0, sticky="w", pady=2)
        ttk.Spinbox(opts, from_=0.3, to=10.0, increment=0.1, textvariable=self.gauss_sigma, width=6).grid(
            row=3, column=1, sticky="w"
        )
        ttk.Label(opts, text="中值核长度(奇数)").grid(row=3, column=2, sticky="e", padx=(12, 4))
        ttk.Spinbox(opts, from_=3, to=31, increment=2, textvariable=self.median_kernel, width=6).grid(
            row=3, column=3, sticky="w"
        )

        ttk.Label(opts, text="EMA α").grid(row=4, column=0, sticky="w", pady=2)
        ttk.Spinbox(opts, from_=0.05, to=0.95, increment=0.05, textvariable=self.ema_alpha, width=6).grid(
            row=4, column=1, sticky="w"
        )
        ttk.Checkbutton(opts, text="对 Combined 曲线再次施加相同滤波", variable=self.smooth_combined_too).grid(
            row=4, column=2, columnspan=2, sticky="w"
        )

        ttk.Checkbutton(opts, text="RCS 加上 Calibration（一般保持关闭）", variable=self.apply_cal).grid(
            row=5, column=0, columnspan=2, sticky="w", pady=(4, 0)
        )
        ttk.Checkbutton(opts, text="绘制 M1/M2/M3 单通道", variable=self.show_channels).grid(
            row=5, column=2, columnspan=2, sticky="w", pady=(4, 0)
        )
        ttk.Checkbutton(
            opts,
            text="Filtered / Spatial / Combined：RCS00+RCS01 线性功率合成（同采样行）",
            variable=self.merge_rcs01,
        ).grid(row=6, column=0, columnspan=4, sticky="w", pady=(4, 0))

        # ── LOESS 非参数回归 ──
        loess_fr = ttk.LabelFrame(opts, text="LOESS 非参数拟合（局部加权线性回归，启用后替代上方平滑）", padding=6)
        loess_fr.grid(row=7, column=0, columnspan=4, sticky="ew", pady=(6, 0))
        ttk.Checkbutton(
            loess_fr,
            text="启用 LOESS 拟合（跳过平滑，直接对分箱数据做局部加权回归）",
            variable=self.use_loess,
        ).grid(row=0, column=0, columnspan=3, sticky="w")
        ttk.Label(loess_fr, text="带宽 (m)：").grid(row=1, column=0, sticky="w", pady=2)
        ttk.Spinbox(
            loess_fr,
            from_=0.2,
            to=10.0,
            increment=0.1,
            textvariable=self.loess_bandwidth,
            width=6,
        ).grid(row=1, column=1, sticky="w")
        ttk.Label(loess_fr, text="最少点数：").grid(row=1, column=2, sticky="e", padx=(12, 4))
        ttk.Spinbox(
            loess_fr,
            from_=3,
            to=50,
            increment=1,
            textvariable=self.loess_min_points,
            width=6,
        ).grid(row=1, column=3, sticky="w")
        ttk.Checkbutton(
            loess_fr,
            text="Combined 曲线也施加 LOESS",
            variable=self.loess_on_combined,
        ).grid(row=2, column=0, columnspan=3, sticky="w", pady=(2, 0))

        disp = ttk.LabelFrame(opts, text="绘图：拟合曲线 RCS 垂直平移（不改变已保存 CSV）", padding=4)
        disp.grid(row=8, column=0, columnspan=4, sticky="ew", pady=(8, 0))
        ttk.Label(disp, text="平移量 (dB)").grid(row=0, column=0, sticky="w")
        ttk.Spinbox(
            disp,
            from_=-50.0,
            to=50.0,
            increment=0.1,
            textvariable=self.plot_rcs_offset_db,
            width=10,
        ).grid(row=0, column=1, sticky="w", padx=4)
        ttk.Button(disp, text="填入元数据标定", command=self._apply_meta_calibration_to_plot_offset).grid(
            row=0, column=2, padx=(8, 4)
        )
        ttk.Button(disp, text="更新图表", command=self._refresh_plot_if_ready).grid(row=0, column=3, sticky="w")

        self.stats_var = tk.StringVar(value="")
        ttk.Label(frm, textvariable=self.stats_var, foreground="gray").pack(anchor="w")

        save_fr = ttk.LabelFrame(frm, text="保存 CSV（勾选后才会写入磁盘）", padding=6)
        save_fr.pack(fill=tk.X, pady=(4, 0))
        sf = ttk.Frame(save_fr)
        sf.pack(fill=tk.X)
        ttk.Checkbutton(sf, text="Filtered", variable=self.save_filtered).pack(side=tk.LEFT, padx=(0, 12))
        ttk.Checkbutton(sf, text="Spatial", variable=self.save_spatial).pack(side=tk.LEFT, padx=(0, 12))
        ttk.Checkbutton(sf, text="Combined", variable=self.save_combined).pack(side=tk.LEFT)

        run_fr = ttk.Frame(frm)
        run_fr.pack(fill=tk.X, pady=8)
        ttk.Button(run_fr, text="运行转换并绘图", command=self._run).pack(side=tk.LEFT)
        ttk.Button(run_fr, text="打开图表大窗口…", command=self._open_plot_popup_safe).pack(side=tk.LEFT, padx=12)
        ttk.Button(run_fr, text="圆周极坐标 (ViewAng–RCS)…", command=self._open_polar_va_rcs_plot).pack(
            side=tk.LEFT, padx=12
        )
        ttk.Label(
            run_fr,
            text="（改「绘图平移」后点「更新图表」无需重算；运行后会自动弹出图表）",
            foreground="gray",
        ).pack(side=tk.LEFT)

        polar_sp_fr = ttk.LabelFrame(frm, text="圆周极坐标：与 Spatial 文件对比", padding=6)
        polar_sp_fr.pack(fill=tk.X, pady=(6, 0))
        ps_row = ttk.Frame(polar_sp_fr)
        ps_row.pack(fill=tk.X)
        ttk.Checkbutton(
            ps_row,
            text="启用：与所选 Spatial 做双曲线对比（非自动推断路径）",
            variable=self.polar_overlay_spatial,
        ).pack(side=tk.LEFT)
        ttk.Button(ps_row, text="选择 Spatial CSV…", command=self._pick_polar_compare_spatial).pack(
            side=tk.LEFT, padx=10
        )
        ttk.Entry(polar_sp_fr, textvariable=self.polar_compare_spatial_path).pack(fill=tk.X, pady=(4, 0))

        polar_cal_fr = ttk.Frame(frm)
        polar_cal_fr.pack(fill=tk.X, pady=(0, 4))
        ttk.Label(polar_cal_fr, text="圆周极坐标 RCS 标定 (dB)：").pack(side=tk.LEFT)
        ttk.Spinbox(
            polar_cal_fr,
            from_=-50.0,
            to=50.0,
            increment=0.1,
            textvariable=self.polar_rcs_cal_db,
            width=10,
        ).pack(side=tk.LEFT, padx=4)
        ttk.Button(polar_cal_fr, text="圆周标定←元数据", command=self._apply_meta_to_polar_rcs_cal).pack(
            side=tk.LEFT, padx=8
        )

    def _apply_meta_to_polar_rcs_cal(self) -> None:
        v = float(self._v_meta_calibration)
        if self.raw_paths:
            try:
                meta = _read_meta_lines(self.raw_paths[0])
                v = parse_meta_calibration_value(meta)
            except Exception:
                pass
        self.polar_rcs_cal_db.set(v)

    def _pick_polar_compare_spatial(self) -> None:
        cur = self.polar_compare_spatial_path.get().strip()
        init_dir = os.path.dirname(cur) if cur and os.path.isdir(os.path.dirname(cur)) else ""
        if not init_dir:
            init_dir = os.path.dirname(self.raw_paths[0]) if self.raw_paths else self.out_dir.get().strip()
        if not init_dir or not os.path.isdir(init_dir):
            init_dir = os.getcwd()
        p = filedialog.askopenfilename(
            title="选择用于圆周极坐标对比的 Spatial CSV",
            initialdir=init_dir,
            filetypes=[("CSV", "*.csv"), ("All", "*.*")],
        )
        if p:
            self.polar_compare_spatial_path.set(p)

    def _open_polar_va_rcs_plot(self) -> None:
        """解析 Raw：极角=ViewAng，半径=RCS+标定+20；灰点=RCS00–19；蓝线=Raw RCS00 平滑；可选 Spatial 橙线对比。"""
        init_dir = os.path.dirname(self.raw_paths[0]) if self.raw_paths else self.out_dir.get().strip()
        if not init_dir or not os.path.isdir(init_dir):
            init_dir = os.getcwd()
        path = filedialog.askopenfilename(
            title="选择 Raw CSV 绘制 ViewAng–RCS 极坐标",
            initialdir=init_dir,
            filetypes=[("CSV", "*.csv"), ("All", "*.*")],
        )
        if not path:
            return
        cal = float(self.polar_rcs_cal_db.get())
        try:
            _meta, df = read_raw_table(path)
            self._v_meta_calibration = parse_meta_calibration_value(_meta)
            th_s, r_s = collect_polar_scatter_va_rcs00_19(
                df, rcs_radius_offset=20.0, rcs_calibration_db=cal
            )
            th_l, r_l = polar_primary_rcs00_line_smoothed(
                df, rcs_radius_offset=20.0, ma_window=30, rcs_calibration_db=cal
            )
        except Exception as e:
            messagebox.showerror("错误", str(e))
            return

        th_sl = np.array([])
        r_sl = np.array([])
        if self.polar_overlay_spatial.get():
            sp_path = self.polar_compare_spatial_path.get().strip()
            if not sp_path or not os.path.isfile(sp_path):
                messagebox.showwarning(
                    "提示",
                    "已勾选「与所选 Spatial 对比」，但未选择有效 Spatial 文件。\n"
                    "请点击「选择 Spatial CSV…」指定一个文件；本次仅绘制 Raw。",
                )
            else:
                try:
                    df_sp = read_spatial_csv(sp_path)
                    th_sl, r_sl = polar_spatial_rcs_line_smoothed(
                        df_sp,
                        rcs_radius_offset=20.0,
                        ma_window=30,
                        rcs_calibration_db=cal,
                    )
                except Exception as e:
                    messagebox.showwarning("Spatial", f"读取 Spatial 失败，已跳过对比：\n{e}")

        win = tk.Toplevel(self)
        win.title(f"极坐标 ViewAng–RCS — {os.path.basename(path)}")
        win.geometry("920x920")
        outer = ttk.Frame(win, padding=4)
        outer.pack(fill=tk.BOTH, expand=True)

        fig = plt.figure(figsize=(9.0, 9.0), dpi=100)
        ax = fig.add_subplot(111, projection="polar")
        if len(th_s):
            ax.scatter(
                th_s,
                r_s,
                s=8,
                c="0.55",
                alpha=0.35,
                edgecolors="none",
                label="Raw RCS00–RCS19（簇）",
                rasterized=True,
                zorder=1,
            )
        if len(th_l):
            ax.plot(
                th_l,
                r_l,
                "b-",
                lw=2.2,
                label="Raw：RCS00 滑动均值（窗=30，按 ViewAng）",
                zorder=3,
            )
        if len(th_sl):
            ax.plot(
                th_sl,
                r_sl,
                color="#ff7f0e",
                ls="--",
                lw=2.2,
                label="Spatial：RCS 滑动均值（窗=30，按 VA）",
                zorder=4,
            )

        mae_note = ""
        if len(th_l) and len(th_sl):
            r_sp_on_raw = np.interp(th_l, th_sl, r_sl, left=np.nan, right=np.nan)
            m = np.isfinite(r_sp_on_raw) & np.isfinite(r_l)
            if np.any(m):
                diff = r_l[m] - r_sp_on_raw[m]
                mae_note = f"  |  两曲线半径差 MAE = {float(np.mean(np.abs(diff))):.4f}（与 dB 差相同）"

        cal_note = f"标定 {cal:+.3f} dB  " if abs(cal) > 1e-9 else ""
        ax.set_title(
            f"{cal_note}半径 = RCS + 标定 + 20  |  极角 = ViewAng / VA（°→rad）{mae_note}",
            pad=16,
        )
        ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), fontsize=9)
        ax.grid(True, alpha=0.35)
        fig.tight_layout()

        canvas = FigureCanvasTkAgg(fig, master=outer)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        NavigationToolbar2Tk(canvas, outer)
        win.lift()

    def _apply_meta_calibration_to_plot_offset(self) -> None:
        self.plot_rcs_offset_db.set(float(self._v_meta_calibration))
        self._refresh_plot_if_ready()

    def _refresh_plot_if_ready(self) -> None:
        if not self._v_plot_ready:
            messagebox.showinfo("提示", "请先点击「运行转换并绘图」生成结果。")
            return
        self._open_plot_popup()

    def _destroy_plot_popup(self) -> None:
        if self._plot_win is None:
            return
        try:
            if self._plot_fig is not None:
                plt.close(self._plot_fig)
        except Exception:
            pass
        try:
            self._plot_win.destroy()
        except tk.TclError:
            pass
        self._plot_win = None
        self._plot_fig = None

    def _open_plot_popup_safe(self) -> None:
        if not self._v_plot_ready:
            messagebox.showinfo("提示", "请先点击「运行转换并绘图」生成结果。")
            return
        self._open_plot_popup()

    def _open_plot_popup(self) -> None:
        self._destroy_plot_popup()
        win = tk.Toplevel(self)
        self._plot_win = win
        win.title("DRI — RCS / Spatial 图表")
        try:
            win.state("zoomed")
        except tk.TclError:
            win.geometry("1280x960")
        win.protocol("WM_DELETE_WINDOW", self._destroy_plot_popup)

        outer = ttk.Frame(win, padding=4)
        outer.pack(fill=tk.BOTH, expand=True)

        fig, (ax, ax_spatial) = plt.subplots(
            2,
            1,
            figsize=(14, 9),
            dpi=100,
            gridspec_kw={"height_ratios": [2.2, 1.0], "hspace": 0.28},
        )
        self._plot_fig = fig
        self._fill_plot_axes(ax, ax_spatial)

        canvas = FigureCanvasTkAgg(fig, master=outer)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        NavigationToolbar2Tk(canvas, outer)
        win.lift()
        win.focus_force()

    def _fill_plot_axes(self, ax, ax_spatial) -> None:
        """根据最近一次运行缓存的数据绘制双坐标轴。"""
        all_r = self._v_all_r
        all_z = self._v_all_z
        rc = self._v_rc
        zc = self._v_zc
        labels = self._v_labels
        off = float(self.plot_rcs_offset_db.get())

        ax.clear()
        ax_spatial.clear()
        cmap = plt.cm.tab10.colors

        ref_c = self.ref_combined_path.get().strip()
        ref_combined_r: Optional[np.ndarray] = None
        ref_combined_z: Optional[np.ndarray] = None
        if self.show_ref_combined.get() and ref_c and os.path.isfile(ref_c):
            rr, zz = read_rrcs_csv(ref_c)
            ref_combined_r, ref_combined_z = clean_rrcs_for_plot(rr, zz)

        ref_f = self.ref_filtered_path.get().strip()
        ref_filtered_r: Optional[np.ndarray] = None
        ref_filtered_z: Optional[np.ndarray] = None
        if self.show_ref_filtered.get() and ref_f and os.path.isfile(ref_f):
            rf, zf = read_rrcs_csv(ref_f)
            ref_filtered_r, ref_filtered_z = clean_rrcs_for_plot(rf, zf)

        if self.show_channels.get():
            for i, (r, z, lab) in enumerate(zip(all_r, all_z, labels[: len(all_r)])):
                rp, zd = clean_rrcs_for_plot(r, np.asarray(z, dtype=float) + off)
                if len(rp):
                    ax.plot(
                        rp,
                        zd,
                        "-",
                        color=cmap[i % len(cmap)],
                        lw=1.6,
                        alpha=0.88,
                        label=f"{lab}（拟合）",
                    )

        zc_offset = np.asarray(zc, dtype=float) + off if len(zc) else np.asarray(zc, dtype=float)
        rc_plot, zc_plot = clean_rrcs_for_plot(rc, zc_offset)
        if ref_combined_r is not None and len(ref_combined_r) and len(rc_plot):
            rc_aligned, zc_aligned = interp_rrcs_to_x(ref_combined_r, rc_plot, zc_plot)
            m_aligned = np.isfinite(zc_aligned)
            rc_plot = rc_aligned[m_aligned]
            zc_plot = zc_aligned[m_aligned]
        if len(all_r) >= 2 and len(rc_plot):
            ax.plot(rc_plot, zc_plot, "k-", lw=2.4, label="Combined 拟合（功率平均）")

        if ref_combined_r is not None and ref_combined_z is not None and len(ref_combined_r):
            ax.plot(
                ref_combined_r,
                ref_combined_z,
                "--",
                color="C3",
                lw=2.0,
                alpha=0.95,
                label="参考 Combined",
            )

        if ref_filtered_r is not None and ref_filtered_z is not None and len(ref_filtered_r):
            ax.plot(
                ref_filtered_r,
                ref_filtered_z,
                ":",
                color="C2",
                lw=2.0,
                alpha=0.95,
                label="参考 Filtered",
            )

        stats_text = ""
        ref_path_mae = ""
        ref_rr: Optional[np.ndarray] = None
        ref_zz: Optional[np.ndarray] = None
        if ref_combined_r is not None and ref_combined_z is not None and len(ref_combined_r):
            ref_path_mae = ref_c
            ref_rr, ref_zz = ref_combined_r, ref_combined_z
        elif ref_filtered_r is not None and ref_filtered_z is not None and len(ref_filtered_r):
            ref_path_mae = ref_f
            ref_rr, ref_zz = ref_filtered_r, ref_filtered_z

        if ref_rr is not None and ref_zz is not None and len(ref_rr) and len(ref_zz) and len(rc_plot):
            r_eval, zfit_i = interp_rrcs_to_x(ref_rr, rc, zc_offset)
            _, zref_i = interp_rrcs_to_x(r_eval, ref_rr, ref_zz)
            m = np.isfinite(zref_i) & np.isfinite(zfit_i)
            if np.any(m):
                diff = zfit_i[m] - zref_i[m]
                mae = float(np.mean(np.abs(diff)))
                rms = float(np.sqrt(np.mean(diff**2)))
                tag = "Combined 参考" if ref_path_mae == ref_c else "Filtered 参考"
                off_note = f"，含绘图平移 {off:+.3f} dB" if abs(off) > 1e-9 else ""
                stats_text = (
                    f"拟合 Combined vs {tag} 重叠段（R≥{PLOT_R_MIN:g}，按参考横坐标）"
                    f" MAE = {mae:.4f} dB  |  RMS = {rms:.4f} dB{off_note}"
                )
            else:
                stats_text = f"参考与拟合距离轴（R≥{PLOT_R_MIN:g}）重叠不足，无法计算 MAE。"
        self.stats_var.set(stats_text)

        has_sp = bool(self._last_spatial_dfs)
        ref_sp_path = self.ref_spatial_path.get().strip()
        show_sp_overlay = self.show_ref_spatial.get() and ref_sp_path and os.path.isfile(ref_sp_path)

        if has_sp or show_sp_overlay:
            for i, df_sp in enumerate(self._last_spatial_dfs):
                xs = pd.to_numeric(df_sp["X"], errors="coerce").to_numpy()
                ys = pd.to_numeric(df_sp["Y"], errors="coerce").to_numpy()
                mxy = np.isfinite(xs) & np.isfinite(ys)
                if np.any(mxy):
                    lab = labels[i] if i < len(labels) else f"Ch{i+1}"
                    ax_spatial.plot(
                        xs[mxy],
                        ys[mxy],
                        "o",
                        ms=3.5,
                        alpha=0.55,
                        color=cmap[i % len(cmap)],
                        label=f"{lab} Spatial（计算）",
                    )
            if show_sp_overlay:
                try:
                    df_ref = read_spatial_csv(ref_sp_path)
                    xr = pd.to_numeric(df_ref["X"], errors="coerce").to_numpy()
                    yr = pd.to_numeric(df_ref["Y"], errors="coerce").to_numpy()
                    mr = np.isfinite(xr) & np.isfinite(yr)
                    if np.any(mr):
                        ax_spatial.plot(
                            xr[mr],
                            yr[mr],
                            "x",
                            ms=5,
                            color="black",
                            alpha=0.75,
                            label="参考 Spatial",
                        )
                except Exception:
                    pass
            ax_spatial.set_xlabel("X")
            ax_spatial.set_ylabel("Y")
            ax_spatial.set_title("Spatial（X–Y）")
            ax_spatial.grid(True, alpha=0.3)
            ax_spatial.set_aspect("equal", adjustable="datalim")
            if ax_spatial.get_legend_handles_labels()[0]:
                ax_spatial.legend(loc="best", fontsize=9)
            ax_spatial.set_visible(True)
        else:
            ax_spatial.set_visible(False)

        ax.set_xlabel("R (m)")
        ax.set_ylabel("RCS (dB)")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best", fontsize=10)
        fig = ax.figure
        fig.tight_layout()

    def _pick_raw(self) -> None:
        paths = filedialog.askopenfilenames(
            title="选择 Raw CSV",
            filetypes=[("CSV", "*.csv"), ("All", "*.*")],
        )
        for p in paths:
            if p not in self.raw_paths:
                self.raw_paths.append(p)
        self._refresh_list()

    def _clear_raw(self) -> None:
        self.raw_paths.clear()
        self._refresh_list()

    def _refresh_list(self) -> None:
        self.listbox.delete(0, tk.END)
        for p in self.raw_paths:
            self.listbox.insert(tk.END, p)

    def _pick_out(self) -> None:
        d = filedialog.askdirectory(title="输出目录")
        if d:
            self.out_dir.set(d)

    def _pick_ref_combined(self) -> None:
        p = filedialog.askopenfilename(
            title="选择参考 Combined CSV（如 1_Mx3_S1_Combined.csv）",
            filetypes=[("CSV", "*.csv"), ("All", "*.*")],
        )
        if p:
            self.ref_combined_path.set(p)

    def _pick_ref_filtered(self) -> None:
        p = filedialog.askopenfilename(
            title="选择参考 Filtered CSV",
            filetypes=[("CSV", "*.csv"), ("All", "*.*")],
        )
        if p:
            self.ref_filtered_path.set(p)

    def _pick_ref_spatial(self) -> None:
        p = filedialog.askopenfilename(
            title="选择参考 Spatial CSV",
            filetypes=[("CSV", "*.csv"), ("All", "*.*")],
        )
        if p:
            self.ref_spatial_path.set(p)

    def _current_smooth_method(self) -> str:
        label = self.smooth_label.get().strip()
        return SMOOTH_METHOD_BY_LABEL.get(label, "savgol")

    def _run(self) -> None:
        if not self.raw_paths:
            messagebox.showwarning("提示", "请先添加至少一个 Raw 文件。")
            return
        out_dir = self.out_dir.get().strip()
        os.makedirs(out_dir, exist_ok=True)

        labels = ["M1", "M2", "M3"]
        all_r: List[np.ndarray] = []
        all_z: List[np.ndarray] = []
        metas: List[Dict[str, str]] = []

        meth = self._current_smooth_method()
        ma_w = int(self.ma_window.get())
        if ma_w % 2 == 0:
            ma_w += 1
        sg_w = int(self.sg_window.get())
        if sg_w % 2 == 0:
            sg_w += 1
        med_k = int(self.median_kernel.get())
        if med_k % 2 == 0:
            med_k += 1

        try:
            self._last_spatial_dfs.clear()
            saved_parts: List[str] = []

            for i, path in enumerate(self.raw_paths[:3]):
                meta, df = read_raw_table(path)
                metas.append(meta)

                calibration = parse_meta_calibration_value(meta)
                if i == 0:
                    self._v_meta_calibration = calibration

                r_f, z_f = raw_to_filtered(
                    df,
                    calibration,
                    smooth_method=meth,
                    ma_window=ma_w,
                    sg_window=sg_w,
                    sg_poly=int(self.sg_poly.get()),
                    gaussian_sigma=float(self.gauss_sigma.get()),
                    median_kernel=med_k,
                    ema_alpha=float(self.ema_alpha.get()),
                    apply_calibration=self.apply_cal.get(),
                    merge_rcs01=self.merge_rcs01.get(),
                    use_loess=self.use_loess.get(),
                    loess_bandwidth_m=float(self.loess_bandwidth.get()),
                    loess_min_points=int(self.loess_min_points.get()),
                )
                all_r.append(r_f)
                all_z.append(z_f)

                safe = os.path.basename(path).replace("_Raw.csv", "").replace(".csv", "")
                df_sp = raw_to_spatial(df, merge_rcs01=self.merge_rcs01.get())
                self._last_spatial_dfs.append(df_sp)

                if self.save_filtered.get():
                    fp_f = os.path.join(out_dir, f"{safe}_Filtered.csv")
                    write_filtered_csv(fp_f, meta, r_f, z_f)
                    saved_parts.append(os.path.basename(fp_f))
                if self.save_spatial.get():
                    fp_s = os.path.join(out_dir, f"{safe}_Spatial.csv")
                    write_spatial_csv(fp_s, meta, df_sp)
                    saved_parts.append(os.path.basename(fp_s))

            rc = np.array([])
            zc = np.array([])
            if len(all_r) >= 2:
                rc, zc = combine_filtered_power_mean(all_r, all_z)
                # 平滑 与 LOESS 互斥
                if self.use_loess.get() and self.loess_on_combined.get() and len(rc) >= int(self.loess_min_points.get()):
                    rc, zc = loess_fit_curve(
                        rc, zc,
                        bandwidth_m=float(self.loess_bandwidth.get()),
                        min_points=int(self.loess_min_points.get()),
                        x_step_m=0.1,
                    )
                elif self.smooth_combined_too.get() and meth != "none" and len(zc):
                    zc = smooth_rcs_series(
                        zc,
                        meth,
                        ma_window=ma_w,
                        sg_window=sg_w,
                        sg_poly=int(self.sg_poly.get()),
                        gaussian_sigma=float(self.gauss_sigma.get()),
                        median_kernel=med_k,
                        ema_alpha=float(self.ema_alpha.get()),
                    )
                if self.save_combined.get():
                    fp_c = os.path.join(out_dir, "Combined_from_gui.csv")
                    write_combined_csv(fp_c, metas, rc, zc)
                    saved_parts.append(os.path.basename(fp_c))

            self._v_all_r = all_r
            self._v_all_z = all_z
            self._v_rc = rc
            self._v_zc = zc
            self._v_labels = labels
            self._v_plot_ready = True
            self._open_plot_popup()

            if saved_parts:
                messagebox.showinfo("完成", "已保存：\n" + "\n".join(saved_parts))
            else:
                messagebox.showinfo("完成", "未勾选保存项，仅完成内存中的计算与绘图。")

        except Exception as e:
            messagebox.showerror("错误", str(e))
            raise


def main() -> None:
    try:
        app = App()
        # Windows 下窗口常被 IDE 挡住；短暂置顶便于看见
        app.lift()
        app.attributes("-topmost", True)
        app.after(300, lambda: app.attributes("-topmost", False))
        app.mainloop()
    except tk.TclError as err:
        print(
            "Tkinter 无法创建窗口（常见于：解释器无 tk、SSH/无桌面、或运行方式未执行 main）。",
            err,
            file=sys.stderr,
        )
        raise


if __name__ == "__main__":
    main()
