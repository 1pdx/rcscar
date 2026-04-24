"""
star_trajectory_planner.py

星型测量轨迹规划（独立于 UI）。

设计目标：
- ENU 校准后在平面系选择目标点，以目标点为参考定义角度零方向
- 角度定义：以「轨迹原点(锚点) -> 目标点」为 0°，向左(逆时针)为正
- 每个角度生成可重复往返的直线：第一条直线由原点与 0°方向确定；后续角度的直线由上一条直线
  以“目标点”为圆心旋转得到（例如 +30°），因此“起点”会随角度在目标点周围旋转
- 角度切换时，插入外部规划的平滑过渡（如三次 Bezier，呈 S 形），落到下一直线内侧端点前的衔接点，
  再短直道贴入内侧端点后倒车回该角测量起点
- 轨迹点密度默认对齐圆周运动：≈0.08m/点
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, List, Optional, Sequence, Tuple

# 与 main_ui.py 里的 SegmentRange 一致：
# (start_idx, end_idx, speed_sign, rcs_start, speed_mps, accel_dist_m, decel_dist_m)
SegmentRange = Tuple[int, int, int, bool, float, float, float]


@dataclass(frozen=True)
class StarMeasurementSpec:
    # angle_cycles: [(angle_deg, cycles)] 角度以“左侧为正(逆时针)”定义
    angle_cycles: List[Tuple[int, int]]
    # 距离目标点最近距离（m）：星型测量的“面对目标物”往返以该距离作为最近点
    inner_radius_m: float
    # 固定直线长度（m）
    line_length_m: float
    speed_mps: float
    accel_dist_m: float
    decel_dist_m: float
    # 点密度（对齐圆周 densify：0.08m/点）
    max_step_m: float = 0.08
    # 角度切换过渡：是否插入过渡段
    enable_transitions: bool = True
    # 过渡曲线（外部 Bezier/S 形）终点落在「内侧端点（靠目标）」沿测量方向之前本距离处，
    # 终点位姿航向已与下一直线一致；随后短直道驶入内侧端点，再倒车回该角测量起点。
    transition_heading_align_reserve_m: float = 4.0


@dataclass
class StarMeasurementPlan:
    task_names: List[str]
    local_points: List[Tuple[float, float]]
    ranges: List[SegmentRange]
    range_task_names: List[Optional[str]]
    transition_count: int


def _wrap_angle_rad(a: float) -> float:
    v = float(a)
    while v > math.pi:
        v -= 2.0 * math.pi
    while v <= -math.pi:
        v += 2.0 * math.pi
    return v


def _rotate_point_about(
    p: Tuple[float, float],
    center: Tuple[float, float],
    angle_rad_ccw: float,
) -> Tuple[float, float]:
    """绕 center 旋转 angle_rad_ccw（数学方向：逆时针为正）。"""
    px, py = float(p[0]), float(p[1])
    cx, cy = float(center[0]), float(center[1])
    dx, dy = px - cx, py - cy
    ca = math.cos(float(angle_rad_ccw))
    sa = math.sin(float(angle_rad_ccw))
    return (cx + dx * ca - dy * sa, cy + dx * sa + dy * ca)


def _sample_line(p0: Tuple[float, float], p1: Tuple[float, float], max_step_m: float) -> List[Tuple[float, float]]:
    x0, y0 = float(p0[0]), float(p0[1])
    x1, y1 = float(p1[0]), float(p1[1])
    dist = math.hypot(x1 - x0, y1 - y0)
    if dist <= 1e-9:
        return [(x0, y0), (x1, y1)]
    step = max(0.02, float(max_step_m))
    n = max(2, int(math.ceil(dist / step)) + 1)
    pts: List[Tuple[float, float]] = []
    for i in range(n):
        t = i / (n - 1)
        pts.append((x0 + (x1 - x0) * t, y0 + (y1 - y0) * t))
    return pts


def _append_segment(
    plan_points: List[Tuple[float, float]],
    plan_ranges: List[SegmentRange],
    range_task_names: List[Optional[str]],
    segment_points: Sequence[Tuple[float, float]],
    *,
    speed_sign: int,
    speed_mps: float,
    accel_dist_m: float,
    decel_dist_m: float,
    rcs_start: bool,
    task_name: Optional[str],
) -> None:
    if len(segment_points) < 2:
        return
    if not plan_points:
        plan_points.extend((float(x), float(y)) for x, y in segment_points)
        start_idx = 0
        end_idx = len(plan_points) - 1
    else:
        same_start = math.hypot(
            plan_points[-1][0] - float(segment_points[0][0]),
            plan_points[-1][1] - float(segment_points[0][1]),
        ) <= 1e-6
        start_idx = len(plan_points) - 1 if same_start else len(plan_points)
        append_slice = segment_points[1:] if same_start else segment_points
        plan_points.extend((float(x), float(y)) for x, y in append_slice)
        end_idx = len(plan_points) - 1
    if end_idx <= start_idx:
        return
    plan_ranges.append(
        (
            int(start_idx),
            int(end_idx),
            int(-1 if int(speed_sign) < 0 else 1),
            bool(rcs_start),
            float(speed_mps),
            float(accel_dist_m),
            float(decel_dist_m),
        )
    )
    range_task_names.append(task_name)


def build_star_measurement_plan(
    *,
    target_point_xy: Tuple[float, float],
    origin_xy: Tuple[float, float],
    spec: StarMeasurementSpec,
    # 过渡段规划器：输入起止位姿(含航向)，返回 transition_points, transition_ranges
    # 允许为 None（不插入过渡段）
    transition_planner: Optional[
        Callable[
            [Tuple[float, float, float], Tuple[float, float, float], float, float],
            Tuple[List[Tuple[float, float]], List[SegmentRange]],
        ]
    ] = None,
    transition_speed_mps: Optional[float] = None,
    transition_close_threshold_m: float = 1.0,
    standoff_margin_m: float = 1.6,
) -> StarMeasurementPlan:
    angles = [(int(a) % 360, max(1, int(c))) for a, c in list(spec.angle_cycles or [])]
    if not angles:
        raise ValueError("请至少勾选一个测量角度。")
    line_len = float(spec.line_length_m)
    if line_len < 0.2:
        raise ValueError("直线长度过短，至少保留 0.2 m 的测量距离。")

    cx, cy = float(target_point_xy[0]), float(target_point_xy[1])
    ox, oy = float(origin_xy[0]), float(origin_xy[1])
    base_dx = cx - ox
    base_dy = cy - oy
    if math.hypot(base_dx, base_dy) < 0.05:
        raise ValueError("目标点距离轨迹原点过近，无法稳定定义 0° 基准方向。")
    base_heading = math.atan2(base_dy, base_dx)

    plan_points: List[Tuple[float, float]] = []
    plan_ranges: List[SegmentRange] = []
    range_task_names: List[Optional[str]] = []
    task_names: List[str] = []
    transition_count = 0

    trans_speed = float(transition_speed_mps) if transition_speed_mps is not None else max(0.05, min(float(spec.speed_mps), 0.75 * float(spec.speed_mps) + 0.08))
    # 过渡段起始“先往前带一点”，增大有效曲率半径，减少切换瞬间的急转。
    lead_out_m = max(1.2, min(6.0, 0.12 * float(line_len)))

    def angle_label(a: int) -> str:
        return f"星型{int(a) % 360}°"

    def cycle_label(a: int, i: int) -> str:
        return f"星型{int(a) % 360}°_第{int(i)}次"

    def ray_endpoints_from_origin(heading: float) -> Tuple[Tuple[float, float], Tuple[float, float]]:
        # 兼容旧实现：保留接口名，但星型测量的“面对目标物往返”应以目标点为参考，
        # 每条直线的最近点为 inner_radius，前进段为“外侧 -> 靠近目标”的方向。
        ux, uy = math.cos(heading), math.sin(heading)
        inner_r = max(0.2, float(spec.inner_radius_m))
        # heading 指向目标的方向；因此 outer->inner 的方向为 +u
        outer = (cx - (inner_r + line_len) * ux, cy - (inner_r + line_len) * uy)
        inner = (cx - inner_r * ux, cy - inner_r * uy)
        return outer, inner

    prev_end_pose: Optional[Tuple[float, float, float]] = None
    prev_angle_deg: Optional[int] = None
    prev_start_pt: Optional[Tuple[float, float]] = None
    prev_end_pt: Optional[Tuple[float, float]] = None
    prev_heading: Optional[float] = None

    for idx_angle, (angle_deg, cycles) in enumerate(angles):
        # “左侧为正(逆时针)”与数学航向 CCW 为正一致：heading = base + angle
        heading = _wrap_angle_rad(base_heading + math.radians(float(angle_deg)))
        if prev_start_pt is None or prev_end_pt is None or prev_angle_deg is None:
            # 第一条：以原点出发的 0°/指定角度直线
            start_pt, end_pt = ray_endpoints_from_origin(heading)
        else:
            # 后续：把上一条直线以“目标点”为圆心旋转得到
            # 角度定义为“向左(逆时针)为正”，与旋转函数的 CCW 正方向一致
            delta_ccw_deg = float(int(angle_deg) - int(prev_angle_deg))
            delta_ccw_rad = math.radians(delta_ccw_deg)
            start_pt = _rotate_point_about(prev_start_pt, (cx, cy), delta_ccw_rad)
            end_pt = _rotate_point_about(prev_end_pt, (cx, cy), delta_ccw_rad)

        task_names.append(angle_label(angle_deg) if cycles == 1 else f"{angle_label(angle_deg)}x{cycles}")

        # 角度切换过渡：相邻两条直线的“起点”之间，生成曲率不突变的过渡曲线连接
        if (
            spec.enable_transitions
            and transition_planner is not None
            and prev_end_pose is not None
            and math.hypot(prev_end_pose[0] - start_pt[0], prev_end_pose[1] - start_pt[1]) > 0.05
        ):
            # 过渡段：Bezier 直达「下一直线内侧端点」前的衔接点（航向已对齐直线），
            # 再短直道到内侧端点，最后倒车回该角外侧起点（与测量往返起点一致）。
            dx2 = float(end_pt[0]) - float(start_pt[0])
            dy2 = float(end_pt[1]) - float(start_pt[1])
            d2 = math.hypot(dx2, dy2)
            if d2 <= 1e-6:
                ux2, uy2 = math.cos(heading), math.sin(heading)
            else:
                ux2, uy2 = dx2 / d2, dy2 / d2
            reserve_m = max(0.0, float(spec.transition_heading_align_reserve_m))
            # 留出尾段直道：Bezier 不贴到内侧端点，便于末端航向与位置同时到位
            min_tail = max(0.35, min(2.0, 0.05 * d2)) if d2 > 1e-6 else 0.35
            reserve_m = min(reserve_m, max(0.0, d2 - min_tail))
            along_m = d2 - reserve_m
            if along_m <= 0.05 and d2 > 1e-6:
                along_m = max(0.05, d2 * 0.5)
                reserve_m = d2 - along_m
            blend_end = (
                float(start_pt[0]) + along_m * ux2,
                float(start_pt[1]) + along_m * uy2,
            )

            # 先沿上一条直线“往前带一段”，再开始过渡曲线
            lead_pose = prev_end_pose
            if prev_start_pt is not None and prev_end_pt is not None:
                dx = float(prev_end_pt[0]) - float(prev_start_pt[0])
                dy = float(prev_end_pt[1]) - float(prev_start_pt[1])
                d = math.hypot(dx, dy)
                if d > 1e-6:
                    ux, uy = dx / d, dy / d
                    prev_line_heading = math.atan2(dy, dx)
                    lead = min(float(lead_out_m), 0.45 * d)
                    lead_pt = (float(prev_start_pt[0]) + lead * ux, float(prev_start_pt[1]) + lead * uy)
                    if math.hypot(lead_pt[0] - float(prev_end_pose[0]), lead_pt[1] - float(prev_end_pose[1])) > 0.05:
                        _append_segment(
                            plan_points,
                            plan_ranges,
                            range_task_names,
                            _sample_line((float(prev_end_pose[0]), float(prev_end_pose[1])), lead_pt, float(spec.max_step_m)),
                            speed_sign=1,
                            speed_mps=float(trans_speed),
                            accel_dist_m=float(spec.accel_dist_m),
                            decel_dist_m=float(spec.decel_dist_m),
                            rcs_start=False,
                            task_name=None,
                        )
                        # 过渡曲线起点航向贴合上一条直线切向，避免原地大角度变向
                        lead_pose = (float(lead_pt[0]), float(lead_pt[1]), float(prev_line_heading))
            trans_pts, trans_ranges = transition_planner(
                lead_pose,
                (blend_end[0], blend_end[1], float(heading)),
                trans_speed,
                # 放宽 close_threshold，让规划器更倾向于生成“慢慢贴合”的大半径曲线
                float(max(transition_close_threshold_m, 1.2 * lead_out_m, 0.30 * line_len)),
            )
            for r in trans_ranges:
                s0, e0, sgn, rcs, sp, ad, dd = r
                _append_segment(
                    plan_points,
                    plan_ranges,
                    range_task_names,
                    trans_pts[int(s0) : int(e0) + 1],
                    speed_sign=int(sgn),
                    speed_mps=float(sp),
                    accel_dist_m=float(ad),
                    decel_dist_m=float(dd),
                    rcs_start=bool(rcs),
                    task_name=None,
                )
            if trans_ranges:
                transition_count += 1

            # 过渡终点 → 内侧端点（靠目标）：航向已在 blend_end 与直线一致，短直道贴入终点
            lp0, lp1 = float(plan_points[-1][0]), float(plan_points[-1][1])
            tail_m = math.hypot(float(end_pt[0]) - lp0, float(end_pt[1]) - lp1)
            if tail_m > 0.05:
                ad_align = min(float(spec.accel_dist_m), max(0.15, 0.35 * tail_m))
                dd_align = min(float(spec.decel_dist_m), max(0.25, 0.55 * tail_m))
                _append_segment(
                    plan_points,
                    plan_ranges,
                    range_task_names,
                    _sample_line((lp0, lp1), end_pt, float(spec.max_step_m)),
                    speed_sign=1,
                    speed_mps=float(trans_speed),
                    accel_dist_m=ad_align,
                    decel_dist_m=dd_align,
                    rcs_start=False,
                    task_name=None,
                )

            # 在内侧端点停稳后，倒车回该角测量起点（外侧），再开始该角度的往返测量
            lp0, lp1 = float(plan_points[-1][0]), float(plan_points[-1][1])
            if math.hypot(float(start_pt[0]) - lp0, float(start_pt[1]) - lp1) > 0.05:
                _append_segment(
                    plan_points,
                    plan_ranges,
                    range_task_names,
                    _sample_line((lp0, lp1), start_pt, float(spec.max_step_m)),
                    speed_sign=-1,
                    speed_mps=float(trans_speed),
                    accel_dist_m=float(spec.accel_dist_m),
                    decel_dist_m=float(spec.decel_dist_m),
                    rcs_start=False,
                    task_name=None,
                )

        for c in range(1, cycles + 1):
            # 往：origin -> end（触发 RCS）
            _append_segment(
                plan_points,
                plan_ranges,
                range_task_names,
                _sample_line(start_pt, end_pt, float(spec.max_step_m)),
                speed_sign=1,
                speed_mps=float(spec.speed_mps),
                accel_dist_m=float(spec.accel_dist_m),
                decel_dist_m=float(spec.decel_dist_m),
                rcs_start=True,
                task_name=cycle_label(angle_deg, c),
            )

            # 返：end -> start（返回该直线起点）
            _append_segment(
                plan_points,
                plan_ranges,
                range_task_names,
                _sample_line(end_pt, start_pt, float(spec.max_step_m)),
                speed_sign=-1,
                speed_mps=float(spec.speed_mps),
                accel_dist_m=float(spec.accel_dist_m),
                decel_dist_m=float(spec.decel_dist_m),
                rcs_start=False,
                task_name=angle_label(angle_deg),
            )

        prev_end_pose = (float(start_pt[0]), float(start_pt[1]), float(heading))
        prev_angle_deg = int(angle_deg)
        prev_start_pt = (float(start_pt[0]), float(start_pt[1]))
        prev_end_pt = (float(end_pt[0]), float(end_pt[1]))
        prev_heading = float(heading)

    # 转成局部坐标：平移到锚点 (origin_xy)
    local_points = [(px - ox, py - oy) for px, py in plan_points]
    return StarMeasurementPlan(
        task_names=task_names,
        local_points=local_points,
        ranges=plan_ranges,
        range_task_names=range_task_names,
        transition_count=int(transition_count),
    )
