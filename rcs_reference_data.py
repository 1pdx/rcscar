#!/usr/bin/env python
# -*- coding: utf-8 -*-

from collections import OrderedDict
from typing import Dict, List, Optional

import numpy as np

# ======================== RCS 参考曲线接口 ========================

RCS_REFERENCE_CLASSES = OrderedDict(
    [
        ("pedistrain", "假人"),
        ("car", "气球车"),
        ("bicycle", "自行车"),
        ("electric motor", "电动车"),
    ]
)

RCS_REFERENCE_ANGLES = ["0度", "45度", "90度", "135度", "180度", "270度"]


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
        "xlim": (0.0, 50.0),
        "ylim": (-25.0, 20.0),
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
        "xlim": (0.0, 50.0),
        "ylim": (-5.0, 30.0),
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
        "xlim": (0.0, 50.0),
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
    if _class == "car":
        return _limits_car()
    if _class == "bicycle":
        return _limits_bicycle(ang)
    if _class == "electric motor":
        return _limits_electric_motor(ang)

    return None
