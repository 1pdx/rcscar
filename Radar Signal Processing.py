"""
Radar Signal Processing.py

ARS40X Cluster 配置（RadarCfg）、RCS 采样曲线与拟合工具。
（已移除 CAN Object 0x60A/0x60B 解码与跟踪；RCS 管线使用 RcsMeasSample。）
"""

from __future__ import annotations

import math
import time
import warnings
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np
import can

try:
    from scipy.signal import savgol_filter as _savgol_filter
except ImportError:  # pragma: no cover
    _savgol_filter = None

# 距离-RCS 拟合方法：
# - "loess": 局部加权线性回归，能体现斜率变化且不产生高阶多项式振荡（默认）
# - "polyfit": 兼容旧版本的高阶多项式拟合
RCS_FIT_METHOD = "loess"
# polyfit 配置（仅当 RCS_FIT_METHOD="polyfit" 时使用）
RCS_POLYFIT_DEGREE = 20
RCS_POLYFIT_DEG_CAP = 10
RCS_FIT_POLY_SAMPLE_STEP_M = 0.1
# loess 配置（仅当 RCS_FIT_METHOD="loess" 时使用）
RCS_FIT_LOESS_BW_M = 2.0  # 带宽（m）；越大越平滑，越小越贴数据、越能体现斜率变化
RCS_FIT_LOESS_MIN_POINTS = 8  # 每个 xq 至少需要的有效权重点数
# 拟合前仅作用于 fit_curve 输入：沿距离排序后的平滑（不改落盘 rcs_raw）。
# Savitzky–Golay：目标窗口长度（偶数会自动减 1 为奇数）；局部一阶多项式。
RCS_CURVE_SG_WINDOW = 39
RCS_CURVE_SG_POLY = 1
# 未安装 scipy 时回退：抑峰中值 + 滑动平均
RCS_CURVE_PEAK_MEDIAN_WIN = 7
RCS_CURVE_PEAK_MA_WIN = 5
_POLYFIT_RANK_WARN = getattr(np, "RankWarning", None)
if _POLYFIT_RANK_WARN is None:
    from numpy.exceptions import RankWarning as _POLYFIT_RANK_WARN


def _odd_window_leq(k: int, n: int) -> int:
    """将窗口长度压到不超过 n，且为奇数；返回值 <3 表示不做中值/卷积类平滑。"""
    k = min(int(k), int(n))
    if k < 3:
        return 1
    if k % 2 == 0:
        k -= 1
    return k if k >= 3 else 1


def _median_filter_1d(y: np.ndarray, k: int) -> np.ndarray:
    y = np.asarray(y, dtype=float)
    n = int(y.size)
    k = _odd_window_leq(k, n)
    if k < 3:
        return y.copy()
    half = k // 2
    padded = np.pad(y, (half, half), mode="edge")
    out = np.empty(n, dtype=float)
    for i in range(n):
        out[i] = float(np.median(padded[i : i + k]))
    return out


def _moving_mean_1d(y: np.ndarray, k: int) -> np.ndarray:
    y = np.asarray(y, dtype=float)
    n = int(y.size)
    k = _odd_window_leq(k, n)
    if k < 3:
        return y.copy()
    half = k // 2
    pad = np.pad(y, (half, half), mode="edge")
    kernel = np.ones(k, dtype=float) / float(k)
    return np.convolve(pad, kernel, mode="valid").astype(float)


def _ensure_odd_sg_window(win: int, upper: int) -> int:
    """与 dri_pipeline_gui._ensure_odd 相同：奇数窗口且不超过 upper。"""
    win = max(3, min(int(win), int(upper)))
    if win % 2 == 0:
        win -= 1
    return max(3, win)


def _savgol_smooth_rcs_db_series(y: np.ndarray) -> np.ndarray:
    """
    对沿距离已排序的 RCS（dB）序列做 Savitzky–Golay：窗口 RCS_CURVE_SG_WINDOW、polyorder=RCS_CURVE_SG_POLY（一阶）。
    """
    y = np.asarray(y, dtype=float).copy()
    n = int(y.size)
    if n < 3 or _savgol_filter is None:
        return y
    # scipy：window_length 为不超过 n 的正奇数
    max_odd = n if (n % 2 == 1) else (n - 1)
    if max_odd < 3:
        return y
    w = _ensure_odd_sg_window(int(RCS_CURVE_SG_WINDOW), max_odd)
    if w < 3:
        return y
    p = int(np.clip(int(RCS_CURVE_SG_POLY), 1, w - 1))
    return np.asarray(_savgol_filter(y, window_length=w, polyorder=p, mode="nearest"), dtype=float)


def _peak_smooth_rcs_series(y: np.ndarray) -> np.ndarray:
    """沿已按 x 排序的序列平滑 RCS，再送入 LOESS / polyfit。优先 SG（窗口见 RCS_CURVE_SG_WINDOW），无 scipy 时抑峰中值+滑动平均。"""
    y = np.asarray(y, dtype=float)
    if y.size <= 1:
        return y.copy()
    if _savgol_filter is not None:
        return _savgol_smooth_rcs_db_series(y)
    ys = y.copy()
    w_med = int(RCS_CURVE_PEAK_MEDIAN_WIN)
    if w_med >= 3:
        ys = _median_filter_1d(ys, w_med)
    w_ma = int(RCS_CURVE_PEAK_MA_WIN)
    if w_ma >= 3:
        ys = _moving_mean_1d(ys, w_ma)
    return ys


def _loess_linear_predict(
    x: np.ndarray,
    y: np.ndarray,
    xq: np.ndarray,
    *,
    bandwidth_m: float,
    min_points: int,
) -> np.ndarray:
    """
    局部加权线性回归（LOESS/LOWESS 的简化版）：在每个查询点 xq 上用高斯权重做一次加权线性拟合。
    - 不依赖 scipy
    - 能较好体现斜率随距离变化
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    xq = np.asarray(xq, dtype=float)
    out = np.full_like(xq, np.nan, dtype=float)
    bw = max(float(bandwidth_m), 1e-6)

    for i in range(int(xq.size)):
        xc = float(xq[i])
        dx = x - xc
        w = np.exp(-0.5 * (dx / bw) ** 2)
        mask = np.isfinite(w) & np.isfinite(x) & np.isfinite(y) & (w > 1e-6)
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
    return out


# 距离-RCS 绘图/拟合过滤：仅影响分箱/拟合输入点，不影响原始记录落盘内容
RCS_FILTER_X_MIN_M = 4.0
RCS_FILTER_X_MAX_M = 60.0
RCS_FILTER_ABS_Y_MAX_M = 1.2
# 前方距离分箱宽度（m），用于拟合/直线拟合前的聚合
RCS_DIST_BIN_M = 0.1
# 同一分箱内若 |Δt| ≤ 该值（s），视为同一雷达周期内多反射点，RCS 按功率叠加：
#   P_total = Σ 10^(RCS/10)，RCS_total = 10×log10(P_total)（与 RCS00+RCS01 合并一致）
RCS_BIN_INCOHERENT_SUM_MAX_TIME_SPAN_S = 0.08

# RCS 录制：绘图/拟合曲线的 RCS 平滑；落盘 rcs_raw 仍为瞬时
RCS_POINT_RCS_EMA_ALPHA = 0.35


# ========================= 0) SocketCAN init + RadarCfg =========================

def setup_socketcan(iface: str, bitrate: int) -> None:
    """Best-effort socketcan setup."""
    import subprocess

    subprocess.run(["ip", "link", "set", iface, "down"], check=False)
    subprocess.run(["ip", "link", "set", iface, "type", "can", "bitrate", str(int(bitrate))], check=True)
    subprocess.run(["ip", "link", "set", iface, "up"], check=True)


# ARS40X RadarCfg：Cluster 扩展输出（0x600 / 0x701）
RADAR_CFG_PAYLOAD_CLUSTER_EXT = bytes.fromhex("F8000000109C0000")


def send_radar_cfg_payload(bus: can.BusABC, sensor_id: int = 0, payload: Optional[bytes] = None) -> None:
    """下发 RadarCfg 原始 8 字节载荷（默认 Cluster）。"""
    cfg_id = 0x200 + int(sensor_id) * 0x10
    data = payload if payload is not None else RADAR_CFG_PAYLOAD_CLUSTER_EXT
    msg = can.Message(arbitration_id=cfg_id, is_extended_id=False, data=data)
    for _ in range(20):
        bus.send(msg)
        time.sleep(0.05)


def send_radar_cfg_cluster(bus: can.BusABC, sensor_id: int = 0) -> None:
    """切换为 Cluster 输出（0x600 Cluster_0_Status + 0x701 Cluster_1_General）。"""
    send_radar_cfg_payload(bus, sensor_id, RADAR_CFG_PAYLOAD_CLUSTER_EXT)


def print_ars40x_terminal_mode_commands(sensor_id: int = 0) -> None:
    """在终端打印 RadarCfg Cluster 切换命令（cansend）。"""
    cid = 0x200 + int(sensor_id) * 0x10
    ch = RADAR_CFG_PAYLOAD_CLUSTER_EXT.hex().upper()
    print("[ARS40X RadarCfg] Cluster 模式 (0x600/0x701):")
    print(f"  cansend can0 {cid:X}#{ch}")


def init_radar_cluster_output(
    iface: str,
    bitrate: int,
    sensor_id: int = 0,
    send_count: int = 20,
    verify_timeout_s: float = 1.2,
    retries: int = 3,
) -> None:
    """切换雷达为 Cluster 输出，并确认总线上出现 0x600 / 0x701。"""
    cfg_id = 0x200 + int(sensor_id) * 0x10
    msg = can.Message(
        arbitration_id=cfg_id, is_extended_id=False, data=RADAR_CFG_PAYLOAD_CLUSTER_EXT
    )

    for _ in range(int(retries)):
        bus = can.interface.Bus(channel=iface, interface="socketcan", bitrate=int(bitrate))
        try:
            for _ in range(int(send_count)):
                bus.send(msg)
                time.sleep(0.05)

            off = int(sensor_id) * 0x10
            id_stat = 0x600 + off
            id_gen = 0x701 + off
            t0 = time.time()
            while time.time() - t0 < float(verify_timeout_s):
                m = bus.recv(timeout=0.1)
                if m is None or m.is_extended_id:
                    continue
                aid = int(m.arbitration_id)
                if aid in (id_stat, id_gen):
                    return
        finally:
            try:
                bus.shutdown()
            except Exception:
                pass
        time.sleep(0.2)

    raise RuntimeError(
        "Radar cluster output init failed: no 0x600/0x701 cluster frames seen after cfg retries."
    )


# ========================= 1) RCS 几何样本（圆周段关联等；非 CAN Object） =========================


def combine_rcs_db_incoherent_sum(rcs_db_values: Sequence[float]) -> Optional[float]:
    """
    同一帧内多个散射点在 dBsm 下的非相干功率线性叠加（绘图/解析与 DRI 一致）：
        P_total = Σ 10^(RCS_i / 10)
        RCS_total = 10 × log10(P_total)
    两项时即 RCS00 与 RCS01 的合并；也用于分箱绘图时同一雷达周期内多点合并。
    """
    vals: List[float] = []
    for v in rcs_db_values:
        try:
            fv = float(v)
        except (TypeError, ValueError):
            continue
        if math.isfinite(fv):
            vals.append(fv)
    if not vals:
        return None
    if len(vals) == 1:
        return float(vals[0])
    p_lin = 0.0
    for v in vals:
        p_lin += 10.0 ** (v / 10.0)
    if p_lin <= 0.0 or not math.isfinite(p_lin):
        return None
    return float(10.0 * math.log10(p_lin))


@dataclass
class RcsMeasSample:
    oid: int
    x: float
    y: float
    vx: float
    vy: float
    dyn: int
    rcs_db: float
    t: float
    x_raw: float = float("nan")
    y_raw: float = float("nan")
    rcs_kf_db: float = float("nan")

    @property
    def rng(self) -> float:
        return float(math.hypot(self.x, self.y))

    def xy_raw(self) -> Tuple[float, float]:
        if math.isfinite(self.x_raw) and math.isfinite(self.y_raw):
            return (float(self.x_raw), float(self.y_raw))
        return (float(self.x), float(self.y))


# 兼容旧引用名
ObjMeas = RcsMeasSample


# ========================= 2) RCS collection =========================

@dataclass
class CurvePoint:
    """RCS 采样点：几何 + RCS；x/y/rcs_filt 供绘图与拟合。"""

    t: float
    x_raw: float
    y_raw: float
    r_raw: float
    x: float
    y: float
    rcs_raw: float
    rcs_filt: float


class RcsRunRecorder:
    def __init__(self, max_segments: int = 0, dist_bin_m: float = RCS_DIST_BIN_M):
        # Keep max_segments only for backward compatibility; recording is now continuous.
        self.max_segments = int(max_segments) if max_segments is not None else 0
        self.dist_bin_m = max(float(dist_bin_m), 1e-3)
        self._rcs_plot_ema: Optional[float] = None
        self.reset()

    def reset(self):
        self.oid_hint: Optional[int] = None
        self.segments: List[List[CurvePoint]] = []
        self._cur: List[CurvePoint] = []
        self._ended = False
        self._rcs_plot_ema = None

    @property
    def oid(self) -> Optional[int]:
        return self.oid_hint

    @oid.setter
    def oid(self, value: Optional[int]) -> None:
        self.oid_hint = int(value) if value is not None else None

    def arm(self, oid_hint: int):
        self.reset()
        self.oid_hint = int(oid_hint)

    def ended(self) -> bool:
        return self._ended

    def add_point(self, m: RcsMeasSample):
        if self._ended:
            return
        xr, yr = m.xy_raw()
        r_slant_raw = float(math.hypot(xr, yr))
        xf, yf = float(m.x), float(m.y)
        rcs_r = float(m.rcs_db)
        # 绘图/拟合用 RCS：优先 rcs_kf_db；否则 EMA
        if math.isfinite(getattr(m, "rcs_kf_db", float("nan"))):
            rcs_f = float(getattr(m, "rcs_kf_db"))
        else:
            a = float(RCS_POINT_RCS_EMA_ALPHA)
            if self._rcs_plot_ema is None:
                self._rcs_plot_ema = rcs_r
            else:
                self._rcs_plot_ema = (1.0 - a) * float(self._rcs_plot_ema) + a * rcs_r
            rcs_f = float(self._rcs_plot_ema)
        pt = CurvePoint(
            t=float(m.t),
            x_raw=xr,
            y_raw=yr,
            r_raw=r_slant_raw,
            x=xf,
            y=yf,
            rcs_raw=rcs_r,
            rcs_filt=rcs_f,
        )
        self.oid_hint = int(m.oid)
        self._cur.append(pt)

    def finalize(self):
        if self._cur:
            self.segments = [list(self._cur)]
            self._cur = []
        self._ended = True

    def point_count(self) -> int:
        return len(self._all_points(include_live=True))

    def _all_points(self, include_live: bool = True) -> List[CurvePoint]:
        points: List[CurvePoint] = []
        for seg in self.segments:
            points.extend(seg)
        if include_live and self._cur:
            points.extend(self._cur)
        return points

    def _distance_mean_xy(
        self,
        x_min: Optional[float] = None,
        x_max: Optional[float] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        points = self._all_points(include_live=True)
        if not points:
            return np.asarray([], dtype=float), np.asarray([], dtype=float)

        # 绘图/拟合：距离用 x,y；RCS 用 rcs_filt
        x = np.asarray([p.x for p in points], dtype=float)
        y_lat = np.asarray([p.y for p in points], dtype=float)
        y = np.asarray([p.rcs_filt for p in points], dtype=float)
        t = np.asarray([p.t for p in points], dtype=float)
        mask = np.isfinite(x) & np.isfinite(y) & np.isfinite(y_lat) & np.isfinite(t)
        mask &= (x >= float(RCS_FILTER_X_MIN_M)) & (x <= float(RCS_FILTER_X_MAX_M))
        mask &= (np.abs(y_lat) <= float(RCS_FILTER_ABS_Y_MAX_M))
        if x_min is not None and x_max is not None:
            mask &= (x >= float(x_min)) & (x <= float(x_max))
        x = x[mask]
        y = y[mask]
        t = t[mask]
        if x.size == 0:
            return np.asarray([], dtype=float), np.asarray([], dtype=float)

        bin_ids = np.round(x / self.dist_bin_m).astype(np.int64)
        uniq_bins = np.unique(bin_ids)
        x_mean = []
        y_mean = []
        t_same = float(RCS_BIN_INCOHERENT_SUM_MAX_TIME_SPAN_S)
        for bid in uniq_bins:
            mbin = bin_ids == bid
            xs_b = x[mbin]
            ys_b = y[mbin]
            ts_b = t[mbin]
            x_mean.append(float(np.mean(xs_b)))
            if ys_b.size <= 1:
                y_mean.append(float(ys_b[0]))
            elif float(np.max(ts_b) - np.min(ts_b)) <= t_same:
                cr = combine_rcs_db_incoherent_sum(ys_b.tolist())
                y_mean.append(float(cr) if cr is not None else float(np.mean(ys_b)))
            else:
                y_mean.append(float(np.mean(ys_b)))

        x_out = np.asarray(x_mean, dtype=float)
        y_out = np.asarray(y_mean, dtype=float)
        order = np.argsort(x_out)
        return x_out[order], y_out[order]

    def fitted_curve(self, grid: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        x_min = float(np.min(grid)) if grid.size else None
        x_max = float(np.max(grid)) if grid.size else None
        fit = self.fit_curve(x_min=x_min, x_max=x_max)
        if fit is None:
            return grid, np.full_like(grid, np.nan, dtype=float)
        x_fit, y_fit = fit
        y_out = np.full_like(grid, np.nan, dtype=float)
        if x_fit.size == 1:
            if grid.size:
                idx = int(np.argmin(np.abs(grid - float(x_fit[0]))))
                y_out[idx] = float(y_fit[0])
            return grid, y_out

        span_mask = (grid >= float(x_fit[0])) & (grid <= float(x_fit[-1]))
        if np.any(span_mask):
            y_out[span_mask] = np.interp(grid[span_mask], x_fit, y_fit)
        return grid, y_out

    def fit_curve(
        self,
        x_min: Optional[float] = None,
        x_max: Optional[float] = None,
    ) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        x, y = self._distance_mean_xy(x_min=x_min, x_max=x_max)
        if x.size == 0:
            return None
        if x.size == 1:
            return np.asarray([float(x[0])], dtype=float), np.asarray([float(y[0])], dtype=float)
        y = _peak_smooth_rcs_series(y)
        span = float(np.max(x) - np.min(x))
        if span <= 1e-9:
            return np.asarray([float(x[0])], dtype=float), np.asarray([float(np.mean(y))], dtype=float)

        lo = float(np.min(x))
        hi = float(np.max(x))
        if x_min is not None:
            lo = max(lo, float(x_min))
        if x_max is not None:
            hi = min(hi, float(x_max))
        if hi <= lo:
            lo, hi = float(np.min(x)), float(np.max(x))
        if str(RCS_FIT_METHOD).strip().lower() == "polyfit":
            deg = min(int(RCS_POLYFIT_DEGREE), int(RCS_POLYFIT_DEG_CAP), int(x.size) - 1)
            deg = max(1, deg)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", _POLYFIT_RANK_WARN)
                coef = np.polyfit(x.astype(float), y.astype(float), deg, rcond=1e-12)
            step = max(float(RCS_FIT_POLY_SAMPLE_STEP_M), 1e-4)
            n = max(2, int(np.ceil((hi - lo) / step)) + 1)
            xs = np.linspace(lo, hi, n, dtype=float)
            ys = np.polyval(coef, xs)
            return xs, ys.astype(float)

        # 默认：LOESS 局部加权线性回归，体现斜率变化且不产生高阶振荡
        step = max(float(RCS_FIT_POLY_SAMPLE_STEP_M), 1e-4)
        n = max(2, int(np.ceil((hi - lo) / step)) + 1)
        xs = np.linspace(lo, hi, n, dtype=float)
        ys = _loess_linear_predict(
            x.astype(float),
            y.astype(float),
            xs,
            bandwidth_m=float(RCS_FIT_LOESS_BW_M),
            min_points=int(RCS_FIT_LOESS_MIN_POINTS),
        )
        return xs, ys.astype(float)

    def fit_line(self) -> Optional[Tuple[float, float]]:
        x, y = self._distance_mean_xy()
        if x.size < 2:
            return None
        coef = np.polyfit(x, y, 1)
        return float(coef[0]), float(coef[1])


@dataclass
class AssocLock:
    active: bool = False
    last_oid: Optional[int] = None
    last_t: float = 0.0

    r_est: float = 0.0
    y_est_leftpos: float = 0.0
    rcs_est: float = 0.0

    x_last: float = 0.0
    y_last_leftpos: float = 0.0

    y0_leftpos: float = 0.0
    r0: float = 0.0

    lost_s: float = 0.0

    @property
    def armed(self):
        return self.active

    @property
    def y_right(self) -> float:
        return -float(self.y_est_leftpos)

    @property
    def y0_right(self) -> float:
        return -float(self.y0_leftpos)

    def reset(self):
        self.active = False
        self.last_oid = None
        self.last_t = 0.0
        self.r_est = 0.0
        self.y_est_leftpos = 0.0
        self.rcs_est = 0.0
        self.x_last = 0.0
        self.y_last_leftpos = 0.0
        self.y0_leftpos = 0.0
        self.r0 = 0.0
        self.lost_s = 0.0

    def arm_from_meas(self, m: RcsMeasSample):
        self.active = True
        self.last_oid = m.oid
        self.last_t = m.t
        self.r_est = m.rng
        self.y_est_leftpos = m.y
        self.rcs_est = m.rcs_db
        self.x_last = m.x
        self.y_last_leftpos = m.y
        self.y0_leftpos = m.y
        self.r0 = m.rng
        self.lost_s = 0.0

    def arm_from(self, m: RcsMeasSample):
        self.arm_from_meas(m)

    def disarm(self):
        self.reset()

    def step(self, candidates: List[RcsMeasSample], now: float, cmd_speed_mps: float, hold_s: float = 1.2) -> Optional[RcsMeasSample]:
        if not self.active:
            return None

        dt = max(1e-3, now - self.last_t)
        r_pred = self.r_est
        y_pred = self.y_est_leftpos

        base_gate_r = 1.6
        base_gate_y = 1.6
        gate_r = base_gate_r + max(0.0, cmd_speed_mps) * dt * 4.0
        gate_y = base_gate_y

        anchor_gate_y = 4.0
        gate_rcs = 8.0

        best: Optional[RcsMeasSample] = None
        best_cost = 1e9

        for m in candidates:
            r = m.rng
            dy = m.y - y_pred
            dr = r - r_pred

            if abs(dr) > gate_r:
                continue
            if abs(dy) > gate_y:
                continue
            if abs(m.y - self.y0_leftpos) > anchor_gate_y:
                continue

            drcs = (m.rcs_db - self.rcs_est)
            cost = (dr / gate_r) ** 2 + (dy / gate_y) ** 2 + 0.15 * (drcs / gate_rcs) ** 2
            if self.last_oid is not None and m.oid == self.last_oid:
                cost *= 0.85
            if cost < best_cost:
                best_cost = cost
                best = m

        if best is None:
            self.lost_s += dt
            if self.lost_s <= hold_s:
                return None

            gate_r2 = min(12.0, gate_r * 3.0 + 4.0)
            best2 = None
            best_cost2 = 1e9
            for m in candidates:
                if abs(m.y - self.y0_leftpos) > anchor_gate_y:
                    continue
                r = m.rng
                dr = r - r_pred
                if abs(dr) > gate_r2:
                    continue
                dy = m.y - y_pred
                drcs = (m.rcs_db - self.rcs_est)
                cost = (dr / gate_r2) ** 2 + 0.7 * (dy / (gate_y * 1.8)) ** 2 + 0.10 * (drcs / gate_rcs) ** 2
                if best2 is None or cost < best_cost2:
                    best2 = m
                    best_cost2 = cost

            if best2 is None:
                self.reset()
                return None
            best = best2

        alpha = 0.35
        self.r_est = (1 - alpha) * self.r_est + alpha * best.rng
        self.y_est_leftpos = (1 - alpha) * self.y_est_leftpos + alpha * best.y
        self.rcs_est = (1 - alpha) * self.rcs_est + alpha * best.rcs_db

        self.x_last = best.x
        self.y_last_leftpos = best.y

        self.last_oid = best.oid
        self.last_t = best.t
        self.lost_s = 0.0
        return best

    def associate(self, candidates: List[RcsMeasSample], now_t: float) -> Optional[RcsMeasSample]:
        return self.step(candidates, now_t, cmd_speed_mps=0.3, hold_s=1.2)
