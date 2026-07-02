#!/usr/bin/env python
# -*- coding: utf-8 -*-

from collections import OrderedDict
from typing import Dict, List, Optional

import numpy as np

# ======================== RCS 参考曲线接口 ========================

RCS_REFERENCE_CLASSES = OrderedDict(
    [
        ("pedistrain", "假人"),
        ("toddler", "幼儿"),
        ("car", "气球车"),
        ("evt_balloon_car", "EVT气球车"),
        ("bicycle", "自行车"),
        ("electric motor", "电动车"),
    ]
)

RCS_REFERENCE_ANGLES = [f"{angle}度" for angle in range(0, 360, 30)]


def get_rcs_reference_options():
    """提供给主 UI 的下拉选项（类别 + 角度）。"""
    return list(RCS_REFERENCE_CLASSES.keys()), list(RCS_REFERENCE_ANGLES)


def get_rcs_reference_labels() -> Dict[str, str]:
    """类别名称的中文显示（可选）。"""
    return dict(RCS_REFERENCE_CLASSES)


def _limits_pedistrain():
    x_limit = np.arange(3, 39, 2)
    y_limit_down = np.array(
        [
            -14.5,
            -12.5,
            -11,
            -10,
            -9,
            -8.5,
            -8,
            -7.5,
            -6.5,
            -6,
            -5.75,
            -5.5,
            -5.25,
            -5.2,
            -5.1,
            -5.0,
            -5.0,
            -5.0,
        ]
    )
    y_limit_up = y_limit_down + 10
    z1 = np.polyfit(x_limit, y_limit_down, 3)
    z2 = np.polyfit(x_limit, y_limit_up, 3)
    p1 = np.poly1d(z1)
    p2 = np.poly1d(z2)
    y11 = []
    y12 = []
    t = np.arange(3, 52, 2)
    for val in t:
        if val <= 33:
            y11.append(p1(val))
            y12.append(p2(val))
        else:
            y11.append(-5)
            y12.append(5)
    return {
        "x": t,
        "lower": np.array(y11),
        "upper": np.array(y12),
        "xlim": (0.0, 60.0),
        "ylim": (-25.0, 20.0),
    }


def _limits_toddler():
    # Digitized from the supplied toddler reference figure. The two dashed
    # boundary curves are sampled by eye. Match the other reference products:
    # use a cubic fit for the rising part, then blend into the far-range plateau.
    x_limit = np.asarray(
        [2.0, 5.0, 8.0, 10.0, 15.0, 20.0, 25.0, 30.0],
        dtype=float,
    )
    y_limit_down = np.asarray(
        [-20.8, -18.7, -16.6, -15.1, -12.5, -10.3, -9.3, -9.0],
        dtype=float,
    )
    y_limit_up = np.asarray(
        [-10.8, -8.5, -6.5, -5.0, -2.0, -0.4, 0.7, 1.0],
        dtype=float,
    )
    p_down = np.poly1d(np.polyfit(x_limit, y_limit_down, 3))
    p_up = np.poly1d(np.polyfit(x_limit, y_limit_up, 3))

    def hermite_blend(
        x: float,
        x0: float,
        x1: float,
        y0: float,
        y1: float,
        m0: float,
        m1: float,
    ) -> float:
        u = (float(x) - float(x0)) / (float(x1) - float(x0))
        u = max(0.0, min(1.0, u))
        h00 = 2.0 * u**3 - 3.0 * u**2 + 1.0
        h10 = u**3 - 2.0 * u**2 + u
        h01 = -2.0 * u**3 + 3.0 * u**2
        h11 = u**3 - u**2
        dx = float(x1) - float(x0)
        return h00 * y0 + h10 * dx * m0 + h01 * y1 + h11 * dx * m1

    plateau_start_x = 25.0
    plateau_end_x = 34.0
    plateau_lower = -9.0
    plateau_upper = 1.0
    p_down_deriv = np.polyder(p_down)
    p_up_deriv = np.polyder(p_up)
    start_lower = float(p_down(plateau_start_x))
    start_upper = float(p_up(plateau_start_x))
    transition_span = plateau_end_x - plateau_start_x

    def monotone_start_slope(raw_slope: float, y0: float, y1: float) -> float:
        secant = (float(y1) - float(y0)) / transition_span
        if abs(secant) <= 1e-12:
            return 0.0
        if raw_slope * secant <= 0.0:
            return 0.0
        return float(np.sign(secant) * min(abs(float(raw_slope)), 3.0 * abs(secant)))

    start_lower_slope = monotone_start_slope(
        float(p_down_deriv(plateau_start_x)),
        start_lower,
        plateau_lower,
    )
    start_upper_slope = monotone_start_slope(
        float(p_up_deriv(plateau_start_x)),
        start_upper,
        plateau_upper,
    )

    y11 = []
    y12 = []
    t = np.arange(2.0, 50.0 + 1e-9, 0.1, dtype=float)
    for val in t:
        if val <= plateau_start_x:
            y11.append(float(p_down(val)))
            y12.append(float(p_up(val)))
        elif val < plateau_end_x:
            y11.append(
                hermite_blend(
                    val,
                    plateau_start_x,
                    plateau_end_x,
                    start_lower,
                    plateau_lower,
                    start_lower_slope,
                    0.0,
                )
            )
            y12.append(
                hermite_blend(
                    val,
                    plateau_start_x,
                    plateau_end_x,
                    start_upper,
                    plateau_upper,
                    start_upper_slope,
                    0.0,
                )
            )
        else:
            y11.append(plateau_lower)
            y12.append(plateau_upper)
    return {
        "x": t,
        "lower": np.asarray(y11, dtype=float),
        "upper": np.asarray(y12, dtype=float),
        "xlim": (0.0, 60.0),
        "ylim": (-25.0, 6.0),
    }


def _limits_car():
    x_limit = np.arange(5, 40, 2)
    y_limit_down = np.array(
        [-3, -1.5, 0.5, 1.5, 3, 4, 5.5, 6.5, 7.5, 8, 8.5, 9, 9.5, 9.8, 10, 10, 10, 10]
    )
    y_limit_up = y_limit_down + 12
    z1 = np.polyfit(x_limit, y_limit_down, 3)
    z2 = np.polyfit(x_limit, y_limit_up, 3)

    p1 = np.poly1d(z1)
    p2 = np.poly1d(z2)

    y11 = []
    y12 = []

    t = np.arange(5, 52, 2)
    for val in t:
        if val <= 33:
            y11.append(p1(val))
            y12.append(p2(val))
        else:
            y11.append(10)
            y12.append(22)
    return {
        "x": t,
        "lower": np.array(y11),
        "upper": np.array(y12),
        "xlim": (0.0, 60.0),
        "ylim": (-5.0, 30.0),
    }


def _limits_evt_balloon_car():
    # Source: EVT气球车毫米波雷达 RCS 反射率范围要求（表3）
    # Distances (m): 5, 10, 20, 30, 40
    # Two rows in the source image are labeled as upper/lower but appear swapped numerically.
    # We normalize by taking lower=min(row_a,row_b), upper=max(row_a,row_b) per distance.
    xp = np.asarray([5.0, 10.0, 20.0, 30.0, 40.0], dtype=float)
    row_a = np.asarray([-3.0, 2.0, 8.0, 10.0, 10.0], dtype=float)
    row_b = np.asarray([10.0, 13.0, 20.0, 22.0, 23.0], dtype=float)
    lower_pts = np.minimum(row_a, row_b)
    upper_pts = np.maximum(row_a, row_b)

    t = np.arange(5.0, 52.0, 2.0, dtype=float)
    lower = np.interp(t, xp, lower_pts, left=float(lower_pts[0]), right=float(lower_pts[-1]))
    upper = np.interp(t, xp, upper_pts, left=float(upper_pts[0]), right=float(upper_pts[-1]))

    ylim_min = float(np.min(lower)) - 5.0
    ylim_max = float(np.max(upper)) + 5.0
    return {
        "x": t,
        "lower": lower.astype(float),
        "upper": upper.astype(float),
        "xlim": (0.0, 60.0),
        "ylim": (ylim_min, ylim_max),
    }


def _limits_bicycle(angle: str):
    x_limit = np.array(
        [1, 2, 3.4, 5, 7, 8.6, 10.3, 11, 13.7, 14.2, 17, 17.3, 20, 21.5, 27.4, 29.7, 31.3, 37, 40]
    )
    base_down = np.array(
        [-10, -8.5, -6.8, -5, -3.5, -2.6, -1.7, -1.5, -0.1, 0.1, 1.5, 1.6, 3.0, 3.3, 4.5, 5, 5.2, 6.5, 7]
    )
    base_up = base_down + 10

    # Overall y shift down by 10 for bicycle reference curves.
    base_down = base_down - 10.0
    base_up = base_up - 10.0

    z0_low = np.polyfit(x_limit, base_down, 3)
    z0_up = np.polyfit(x_limit, base_up, 3)
    p0_low = np.poly1d(z0_low)
    p0_up = np.poly1d(z0_up)

    y_limit_down = base_down.copy()
    y_limit_up = base_up.copy()

    z1 = np.polyfit(x_limit, y_limit_down, 3)
    z2 = np.polyfit(x_limit, y_limit_up, 3)

    p1 = np.poly1d(z1)
    p2 = np.poly1d(z2)

    y11 = []
    y12 = []

    def smoothstep(u: float) -> float:
        return u * u * (3.0 - 2.0 * u)

    t = np.arange(1, 53, 1)
    start_x = 40.0
    end_x = 52.0

    for val in t:
        if angle == "0度" or val <= start_x:
            y11.append(p1(val))
            y12.append(p2(val))
        else:
            ratio = (float(val) - start_x) / (end_x - start_x)
            ratio = max(0.0, min(1.0, ratio))
            w = smoothstep(ratio)
            target_lower = float(p0_low(val))
            target_upper = float(p0_up(val))
            y11.append(float(p1(val)) * (1.0 - w) + target_lower * w)
            y12.append(float(p2(val)) * (1.0 - w) + target_upper * w)

    ylim = (-25.0, 15.0) if angle == "90度" else (-35.0, 10.0)
    return {
        "x": t,
        "lower": np.array(y11),
        "upper": np.array(y12),
        "xlim": (0.0, 60.0),
        "ylim": ylim,
    }


def _limits_electric_motor(angle: str):
    return _limits_bicycle(angle)


def get_rcs_reference_limits(obj_class, angle) -> Optional[Dict[str, np.ndarray]]:
    """
    返回指定目标类别/角度的参考上下限曲线数据。
    返回格式:
        {
            "x": np.ndarray,
            "lower": np.ndarray,
            "upper": np.ndarray,
            "xlim": (xmin, xmax),
            "ylim": (ymin, ymax),
        }
    如果类别不支持则返回 None。
    """
    if not obj_class:
        return None
    _class = str(obj_class).strip()
    ang = str(angle).strip() if angle else ""
    if not ang:
        ang = "0度"

    if _class == "pedistrain":
        return _limits_pedistrain()
    if _class == "toddler":
        return _limits_toddler()
    if _class == "car":
        return _limits_car()
    if _class == "evt_balloon_car":
        return _limits_evt_balloon_car()
    if _class == "bicycle":
        return _limits_bicycle(ang)
    if _class == "electric motor":
        return _limits_electric_motor(ang)

    return None
