"""
Radar Signal Processing.py

Radar processing utilities for ARS408 object output (0x60A/0x60B).
UI removed; this module focuses on decoding, tracking, and stable target snapshots.
"""

from __future__ import annotations

import math
import time
import threading
from dataclasses import dataclass, field
from collections import deque
from typing import Dict, List, Optional, Tuple

import numpy as np
import can


# ========================= 0) SocketCAN init + RadarCfg =========================

def setup_socketcan(iface: str, bitrate: int) -> None:
    """Best-effort socketcan setup."""
    import subprocess

    subprocess.run(["ip", "link", "set", iface, "down"], check=False)
    subprocess.run(["ip", "link", "set", iface, "type", "can", "bitrate", str(int(bitrate))], check=True)
    subprocess.run(["ip", "link", "set", iface, "up"], check=True)


def send_radar_cfg_object(bus: can.BusABC, sensor_id: int = 0) -> None:
    """Switch radar to objects output: payload 08 00 00 00 08 00 00 00."""
    cfg_id = 0x200 + int(sensor_id) * 0x10
    payload = bytes.fromhex("08 00 00 00 08 00 00 00")
    msg = can.Message(arbitration_id=cfg_id, is_extended_id=False, data=payload)
    for _ in range(20):
        bus.send(msg)
        time.sleep(0.05)


def init_radar_object_output(
    iface: str,
    bitrate: int,
    sensor_id: int = 0,
    send_count: int = 20,
    verify_timeout_s: float = 1.2,
    retries: int = 3,
) -> None:
    """Initialize radar object output and verify 0x60A/0x60B are seen."""
    cfg_id = 0x200 + int(sensor_id) * 0x10
    payload = bytes.fromhex("08 00 00 00 08 00 00 00")
    msg = can.Message(arbitration_id=cfg_id, is_extended_id=False, data=payload)

    for _ in range(int(retries)):
        bus = can.interface.Bus(channel=iface, interface="socketcan", bitrate=int(bitrate))
        try:
            for _ in range(int(send_count)):
                bus.send(msg)
                time.sleep(0.05)

            t0 = time.time()
            while time.time() - t0 < float(verify_timeout_s):
                m = bus.recv(timeout=0.1)
                if m is None or m.is_extended_id:
                    continue
                if int(m.arbitration_id) in (0x60A, 0x60B):
                    return
        finally:
            try:
                bus.shutdown()
            except Exception:
                pass
        time.sleep(0.2)

    raise RuntimeError("Radar object output init failed: no 0x60A/0x60B seen after cfg retries.")


# ========================= 1) Object decode (0x60B) =========================

def extract_motorola_u(data: bytes, start: int, length: int) -> int:
    """Motorola(big-endian) bit extraction (DBC @0)."""
    if length <= 0:
        return 0
    if len(data) != 8:
        data = data.ljust(8, b"\x00")[:8]
    byte = start // 8
    bit = start % 8  # 0=LSB, 7=MSB
    val = 0
    for _ in range(length):
        if not (0 <= byte < 8):
            break
        b = (data[byte] >> bit) & 0x1
        val = (val << 1) | int(b)
        if bit == 0:
            byte += 1
            bit = 7
        else:
            bit -= 1
    return val


def _phys_m(data: bytes, start: int, length: int, offset: float, res: float) -> float:
    return float(extract_motorola_u(data, start, length) * res + offset)


@dataclass
class ObjMeas:
    oid: int
    x: float
    y: float
    vx: float
    vy: float
    dyn: int
    rcs_db: float
    t: float

    @property
    def rng(self) -> float:
        return float(math.hypot(self.x, self.y))

    @property
    def y_right(self) -> float:
        return -float(self.y)


def decode_60B(data: bytes) -> ObjMeas:
    oid = extract_motorola_u(data, 7, 8)
    x = _phys_m(data, 15, 13, -500.0, 0.2)
    y = _phys_m(data, 18, 11, -204.6, 0.2)
    vx = _phys_m(data, 39, 10, -128.0, 0.25)
    vy = _phys_m(data, 45, 9, -64.0, 0.25)
    dyn = extract_motorola_u(data, 50, 3)
    rcs = _phys_m(data, 63, 8, -64.0, 0.5)
    return ObjMeas(oid=int(oid), x=float(x), y=float(y), vx=float(vx), vy=float(vy), dyn=int(dyn), rcs_db=float(rcs), t=time.time())


# ========================= 2) Tracker + clutter filtering =========================

@dataclass
class TrackState:
    oid: int
    last: Optional[ObjMeas] = None
    last_update: float = 0.0
    age: int = 0
    miss: int = 0

    x_hist: deque = field(default_factory=lambda: deque(maxlen=20))
    y_hist: deque = field(default_factory=lambda: deque(maxlen=20))
    rcs_hist: deque = field(default_factory=lambda: deque(maxlen=20))

    x_ema: Optional[float] = None
    y_ema: Optional[float] = None
    rcs_ema: Optional[float] = None

    stable_until: float = 0.0

    def update(self, m: ObjMeas, ema_alpha: float = 0.25):
        self.last = m
        self.last_update = m.t
        self.age += 1
        self.miss = 0

        self.x_hist.append(m.x)
        self.y_hist.append(m.y)
        self.rcs_hist.append(m.rcs_db)

        if self.x_ema is None:
            self.x_ema = m.x
            self.y_ema = m.y
            self.rcs_ema = m.rcs_db
        else:
            self.x_ema = (1.0 - ema_alpha) * self.x_ema + ema_alpha * m.x
            self.y_ema = (1.0 - ema_alpha) * self.y_ema + ema_alpha * m.y
            self.rcs_ema = (1.0 - ema_alpha) * self.rcs_ema + ema_alpha * m.rcs_db

    def step_miss(self):
        self.miss += 1

    def stability_score(self) -> float:
        if self.last is None:
            return 0.0
        age_term = min(1.0, self.age / 12.0)

        px = float(np.std(self.x_hist)) if len(self.x_hist) > 3 else 0.0
        py = float(np.std(self.y_hist)) if len(self.y_hist) > 3 else 0.0
        pos_std = math.hypot(px, py)
        pos_term = 1.0 / (1.0 + 0.8 * pos_std)

        pr = float(np.std(self.rcs_hist)) if len(self.rcs_hist) > 3 else 0.0
        rcs_term = 1.0 / (1.0 + 0.2 * pr)

        dyn_bonus = 1.10 if self.last.dyn in (1, 3, 7) else 1.0
        return float(age_term * pos_term * rcs_term * dyn_bonus)

    def is_stable_now(self, min_age: int = 5, min_score: float = 0.30) -> bool:
        return self.age >= int(min_age) and self.stability_score() >= float(min_score)

    def is_display_stable(
        self,
        now: float,
        hold_s: float = 0.8,
        min_age: int = 5,
        min_score: float = 0.30,
    ) -> bool:
        if self.is_stable_now(min_age=min_age, min_score=min_score):
            self.stable_until = max(self.stable_until, now + hold_s)
            return True
        return now <= self.stable_until


class ObjectTracker:
    """Decode objects and maintain TrackState set."""

    def __init__(self, iface: str, bitrate: int, roi_front_abs: float = 80.0, roi_lat_abs: float = 20.0):
        self.iface = iface
        self.bitrate = int(bitrate)
        self.roi_front_abs = float(roi_front_abs)
        self.roi_lat_abs = float(roi_lat_abs)

        self.bus = can.interface.Bus(channel=self.iface, interface="socketcan", bitrate=self.bitrate)

        self._lock = threading.Lock()
        self._stop = False
        self._th = threading.Thread(target=self._loop, daemon=True)

        self._cycle: Dict[int, ObjMeas] = {}
        self.tracks: Dict[int, TrackState] = {}

        self._th.start()

    def stop(self):
        self._stop = True
        try:
            self._th.join(timeout=1.0)
        except Exception:
            pass
        try:
            self.bus.shutdown()
        except Exception:
            pass

    def _loop(self):
        while not self._stop:
            msg = self.bus.recv(timeout=0.1)
            if msg is None or msg.is_extended_id:
                continue
            cid = int(msg.arbitration_id)

            if cid == 0x60B and len(msg.data) == 8:
                m = decode_60B(bytes(msg.data))
                if not (abs(m.x) <= self.roi_front_abs and abs(m.y) <= self.roi_lat_abs):
                    continue
                with self._lock:
                    self._cycle[m.oid] = m
                continue

            if cid == 0x60A:
                with self._lock:
                    self._finalize()
                continue

    def _finalize(self):
        now = time.time()
        present = set(self._cycle.keys())

        for oid, tr in list(self.tracks.items()):
            if oid not in present:
                tr.step_miss()

        for oid, m in self._cycle.items():
            tr = self.tracks.get(oid)
            if tr is None:
                tr = TrackState(oid=oid)
                self.tracks[oid] = tr
            tr.update(m)

        stale = [oid for oid, tr in self.tracks.items() if now - tr.last_update > 1.5]
        for oid in stale:
            del self.tracks[oid]

        self._cycle = {}

    def snapshot_tracks(self) -> List[TrackState]:
        with self._lock:
            return list(self.tracks.values())


class RadarObjectTracker:
    """Provide stable target snapshots for UI."""

    def __init__(
        self,
        iface: str,
        bitrate: int,
        roi_front_abs: float = 80.0,
        roi_lat_abs: float = 20.0,
        stable_min_age: int = 5,
        stable_min_score: float = 0.30,
        stable_hold_s: float = 0.8,
        front_only: bool = True,
    ) -> None:
        self.tracker = ObjectTracker(iface=iface, bitrate=bitrate, roi_front_abs=roi_front_abs, roi_lat_abs=roi_lat_abs)
        self.stable_min_age = int(stable_min_age)
        self.stable_min_score = float(stable_min_score)
        self.stable_hold_s = float(stable_hold_s)
        self.front_only = bool(front_only)

    def stop(self) -> None:
        self.tracker.stop()

    def get_targets_snapshot(self, min_score: Optional[float] = None) -> List[ObjMeas]:
        now = time.time()
        tracks = self.tracker.snapshot_tracks()
        results: List[ObjMeas] = []
        score_th = self.stable_min_score if min_score is None else float(min_score)
        for tr in tracks:
            if tr.last is None:
                continue
            if self.front_only and tr.last.x < 0.0:
                continue
            if not tr.is_display_stable(
                now,
                hold_s=self.stable_hold_s,
                min_age=self.stable_min_age,
                min_score=score_th,
            ):
                continue
            if tr.age < self.stable_min_age or tr.stability_score() < score_th:
                continue
            results.append(tr.last)
        return results

    def get_best_stable_target(self, min_score: Optional[float] = None) -> Optional[ObjMeas]:
        targets = self.get_targets_snapshot(min_score=min_score)
        if not targets:
            return None
        best = None
        best_score = -1.0
        track_map = {tr.oid: tr for tr in self.tracker.snapshot_tracks() if tr.last is not None}
        for t in targets:
            tr = track_map.get(t.oid)
            score = tr.stability_score() if tr is not None else 0.0
            if score > best_score:
                best_score = score
                best = t
        return best


# ========================= 3) RCS collection =========================

@dataclass
class CurvePoint:
    t: float
    x: float
    y: float
    r_raw: float
    rcs_raw: float
    rcs_filt: float


class RcsRunRecorder:
    def __init__(self, max_segments: int = 10):
        self.max_segments = int(max_segments)
        self.reset()

    def reset(self):
        self.oid_hint: Optional[int] = None
        self.segments: List[List[CurvePoint]] = []
        self._cur: List[CurvePoint] = []
        self._last_front: Optional[float] = None
        self._dir: int = 0
        self._ended = False
        self._rcs_ema: Optional[float] = None

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

    def add_point(self, m: ObjMeas):
        if self._ended:
            return
        r = m.rng
        front = m.x
        if self._rcs_ema is None:
            self._rcs_ema = m.rcs_db
        else:
            self._rcs_ema = 0.85 * self._rcs_ema + 0.15 * m.rcs_db
        pt = CurvePoint(t=m.t, x=m.x, y=m.y, r_raw=r, rcs_raw=m.rcs_db, rcs_filt=float(self._rcs_ema))

        self.oid_hint = int(m.oid)

        if self._last_front is None:
            self._last_front = front
            return

        d_front = front - self._last_front
        self._last_front = front

        th = 0.25
        new_dir = 0
        if d_front > th:
            new_dir = +1
        elif d_front < -th:
            new_dir = -1

        min_pts = 25
        if new_dir != 0 and new_dir != self._dir:
            if self._dir == -1 and len(self._cur) >= min_pts:
                self.segments.append(self._cur)
                if len(self.segments) >= self.max_segments:
                    self._ended = True
                    self._cur = []
                    self._dir = new_dir
                    return
            self._cur = []
            self._dir = new_dir
            if self._dir == -1:
                self._cur.append(pt)
            return

        if self._dir == 0 and new_dir != 0:
            self._dir = new_dir
            if self._dir == -1:
                self._cur.append(pt)
            return

        if self._dir == -1:
            self._cur.append(pt)

    def finalize(self):
        if self._cur:
            self.segments.append(self._cur)
            self._cur = []
        self.segments = self.segments[:self.max_segments]
        self._ended = True

    def raw_text(self) -> str:
        lines = ["# segment_idx	t(s)	x(m)	y(m)	rcs(dBsm)"]
        for si, seg in enumerate(self.segments):
            for p in seg:
                lines.append(f"{si}	{p.t:.6f}	{p.x:.3f}	{p.y:.3f}	{p.rcs_raw:.3f}")
        return "".join(lines)

    def fitted_curve(self, grid: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        x_min = float(np.min(grid)) if grid.size else None
        x_max = float(np.max(grid)) if grid.size else None
        fit = self.fit_curve(x_min=x_min, x_max=x_max)
        if fit is None:
            return grid, np.full_like(grid, np.nan, dtype=float)
        coeffs, x_mean, x_scale = fit
        x_scale = max(x_scale, 1e-6)
        xn = (grid - x_mean) / x_scale
        return grid, np.polyval(coeffs, xn)

    def fit_curve(
        self,
        x_min: Optional[float] = None,
        x_max: Optional[float] = None,
    ) -> Optional[Tuple[np.ndarray, float, float]]:
        if not self.segments:
            return None
        xs = []
        ys = []
        for seg in self.segments:
            for p in seg:
                xs.append(p.x)
                ys.append(p.rcs_filt)
        min_pts = 8
        if len(xs) < min_pts:
            return None
        x = np.asarray(xs, dtype=float)
        y = np.asarray(ys, dtype=float)
        if x_min is not None and x_max is not None:
            mask = (x >= x_min) & (x <= x_max)
            x = x[mask]
            y = y[mask]
        if len(x) < min_pts:
            return None
        if np.ptp(x) < 1.0:
            return None

        order = np.argsort(x)
        x = x[order]
        y = y[order]

        if len(x) > 200:
            idx = np.linspace(0, len(x) - 1, 200).astype(int)
            x = x[idx]
            y = y[idx]

        x_mean = float(np.mean(x))
        x_scale = float(np.std(x))
        if x_scale < 1e-6:
            return None

        deg = 3
        if len(x) < 8:
            deg = 2
        if len(x) < deg + 1:
            return None

        coeffs = np.polyfit((x - x_mean) / x_scale, y, deg)
        return coeffs, x_mean, x_scale

    def fit_line(self) -> Optional[Tuple[float, float]]:
        if not self.segments:
            return None
        xs = []
        ys = []
        for seg in self.segments:
            for p in seg:
                xs.append(p.x)
                ys.append(p.rcs_filt)
        if len(xs) < 10:
            return None
        x = np.asarray(xs, dtype=float)
        y = np.asarray(ys, dtype=float)
        if np.ptp(x) < 1.0:
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

    def arm_from_meas(self, m: ObjMeas):
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

    def arm_from(self, m: ObjMeas):
        self.arm_from_meas(m)

    def disarm(self):
        self.reset()

    def step(self, candidates: List[ObjMeas], now: float, cmd_speed_mps: float, hold_s: float = 1.2) -> Optional[ObjMeas]:
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

        best: Optional[ObjMeas] = None
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

    def associate(self, candidates: List[ObjMeas], now_t: float) -> Optional[ObjMeas]:
        return self.step(candidates, now_t, cmd_speed_mps=0.3, hold_s=1.2)
