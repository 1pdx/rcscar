import argparse
import math
import os
import re
import sys
import statistics
from collections import defaultdict
from dataclasses import dataclass
from typing import List, Optional, Sequence


@dataclass(frozen=True)
class Series:
    x: List[float]
    rcs: List[float]
    label: str


def _is_nan(v: float) -> bool:
    return isinstance(v, float) and math.isnan(v)


def _to_float(s: str) -> Optional[float]:
    try:
        v = float(s.strip())
    except Exception:
        return None
    if _is_nan(v):
        return None
    return v


def _read_text(path: str) -> str:
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()


def _split_ws(line: str) -> List[str]:
    return [t for t in re.split(r"\s+", line.strip()) if t]


def load_txt_auto(path: str, label: str) -> Series:
    """
    支持两类 txt：
    1) 你现在这种：首行以 # 开头，列名包含 x(...) 与 rcs(...)
    2) 通用两列：x  rcs（空格/制表符分隔）
    """
    xs: List[float] = []
    ys: List[float] = []

    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        header_tokens: Optional[List[str]] = None
        x_idx: Optional[int] = None
        rcs_idx: Optional[int] = None

        for raw in f:
            line = raw.strip()
            if not line:
                continue

            if header_tokens is None and line.startswith("#"):
                header_tokens = _split_ws(line.lstrip("#").strip())
                lower = [t.lower() for t in header_tokens]
                for i, t in enumerate(lower):
                    if x_idx is None and re.fullmatch(r"x(\(.*\))?", t):
                        x_idx = i
                    if rcs_idx is None and t.startswith("rcs"):
                        rcs_idx = i
                continue

            if line.startswith("#"):
                continue

            parts = _split_ws(line)
            if len(parts) < 2:
                continue

            if x_idx is not None and rcs_idx is not None and len(parts) > max(x_idx, rcs_idx):
                x = _to_float(parts[x_idx])
                rcs = _to_float(parts[rcs_idx])
            else:
                x = _to_float(parts[0])
                rcs = _to_float(parts[1])

            if x is None or rcs is None:
                continue
            xs.append(x)
            ys.append(rcs)

    if not xs:
        raise ValueError(f"未能从 TXT 解析出任何数据: {path}")
    return Series(xs, ys, label=label)


def load_csv_combined(path: str, label: str) -> Series:
    """
    针对你现在的 Combined CSV：
    - 前面有多行元信息
    - 数据从包含 'R,RCS' 的行之后开始（两列）
    """
    text = _read_text(path)
    lines = text.splitlines()

    start = None
    for i, line in enumerate(lines):
        if re.search(r"^\s*R\s*,\s*RCS\s*$", line, flags=re.IGNORECASE):
            start = i + 1
            break
    if start is None:
        # 兜底：找任何包含 RCS 的两列表头
        for i, line in enumerate(lines):
            if "rcs" in line.lower() and "," in line:
                start = i + 1
                break
    if start is None:
        raise ValueError(f"未找到 CSV 表头（例如 'R,RCS'）: {path}")

    xs: List[float] = []
    ys: List[float] = []
    for raw in lines[start:]:
        line = raw.strip()
        if not line:
            continue
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 2:
            continue
        x = _to_float(parts[0])
        rcs = _to_float(parts[1])
        if x is None or rcs is None:
            continue
        xs.append(x)
        ys.append(rcs)

    if not xs:
        raise ValueError(f"未能从 CSV 解析出任何数据: {path}")
    return Series(xs, ys, label=label)


def detect_format(path: str, fmt_arg: str) -> str:
    if fmt_arg != "auto":
        return fmt_arg
    ext = os.path.splitext(path)[1].lower()
    if ext in {".txt", ".tsv", ".dat", ".log"}:
        return "txt"
    if ext in {".csv"}:
        return "csv"
    # 兜底：看内容
    head = _read_text(path)[:4096]
    if "R,RCS" in head or "," in head:
        return "csv"
    return "txt"


def load_series(path: str, fmt: str, label: str) -> Series:
    if fmt == "csv":
        return load_csv_combined(path, label=label)
    if fmt == "txt":
        return load_txt_auto(path, label=label)
    raise ValueError(f"不支持的格式: {fmt}")


def apply_offset(series: Series, offset: float) -> Series:
    if offset == 0:
        return series
    return Series(series.x, [y + offset for y in series.rcs], label=series.label)


def calibration_label(name: str, offset: float) -> str:
    name = (name or "").strip()
    if name:
        return f"{name} 标定值={offset:g}"
    return f"标定值={offset:g}"


def clip_by_x(series: Series, x_min: float, x_max: float) -> Series:
    xs: List[float] = []
    ys: List[float] = []
    for x, y in zip(series.x, series.rcs):
        if x < x_min or x > x_max:
            continue
        xs.append(x)
        ys.append(y)
    if not xs:
        raise ValueError(f"{series.label} 在 x∈[{x_min},{x_max}] 区间内无有效数据")
    return Series(xs, ys, label=series.label)


def filter_by_rcs_range(series: Series, rcs_min: Optional[float], rcs_max: Optional[float]) -> Series:
    if rcs_min is None and rcs_max is None:
        return series
    xs: List[float] = []
    ys: List[float] = []
    for x, y in zip(series.x, series.rcs):
        if rcs_min is not None and y < rcs_min:
            continue
        if rcs_max is not None and y > rcs_max:
            continue
        xs.append(x)
        ys.append(y)
    if not xs:
        raise ValueError(f"{series.label} 在 rcs 范围过滤后无有效数据")
    return Series(xs, ys, label=series.label)


def remove_outliers_mad(series: Series, mad_k: Optional[float]) -> Series:
    """
    用 Median Absolute Deviation (MAD) 做鲁棒异常点剔除：
    保留 |y - median(y)| <= mad_k * 1.4826 * MAD 的点。
    mad_k=None 或 <=0 表示不剔除。
    """
    if mad_k is None or mad_k <= 0:
        return series
    if len(series.rcs) < 10:
        return series

    med = statistics.median(series.rcs)
    abs_dev = [abs(y - med) for y in series.rcs]
    mad = statistics.median(abs_dev)
    if mad == 0:
        return series

    sigma = 1.4826 * mad
    thr = mad_k * sigma

    xs: List[float] = []
    ys: List[float] = []
    for x, y in zip(series.x, series.rcs):
        if abs(y - med) <= thr:
            xs.append(x)
            ys.append(y)
    if not xs:
        raise ValueError(f"{series.label} 异常值剔除后无有效数据（可调大 mad_k 或关闭）")
    return Series(xs, ys, label=series.label)


def lowpass_moving_average(series: Series, window: int) -> Series:
    """
    简单低通滤波：移动平均（window 越大越平滑）。
    先按 x 排序，再对 rcs 做滑动窗口均值。
    """
    if window <= 1:
        return series
    pairs = sorted(zip(series.x, series.rcs), key=lambda t: t[0])
    xs = [p[0] for p in pairs]
    ys = [p[1] for p in pairs]
    n = len(ys)
    if n < 3:
        return series

    w = min(window, n)
    half = w // 2

    out_y: List[float] = []
    for i in range(n):
        a = max(0, i - half)
        b = min(n, i + half + 1)
        out_y.append(sum(ys[a:b]) / (b - a))

    return Series(xs, out_y, label=f"{series.label}（低通MA{w}）")


def bin_series_uniform(series: Series, bin_width: float) -> Series:
    """
    按固定宽度对 x 分箱，箱内 rcs 取平均；输出 x 为箱中心 (k+0.5)*bin_width。
    """
    if bin_width <= 0:
        raise ValueError("分箱宽度 bin_width 必须 > 0")
    gx: dict[int, List[float]] = defaultdict(list)
    gy: dict[int, List[float]] = defaultdict(list)
    for x, y in zip(series.x, series.rcs):
        k = int(math.floor(x / bin_width + 1e-12))
        gx[k].append(x)
        gy[k].append(y)
    if not gx:
        raise ValueError(f"{series.label} 分箱后无数据")
    xs_out: List[float] = []
    ys_out: List[float] = []
    for k in sorted(gx.keys()):
        y_mean = sum(gy[k]) / len(gy[k])
        x_center = (k + 0.5) * bin_width
        xs_out.append(x_center)
        ys_out.append(y_mean)
    return Series(xs_out, ys_out, label=series.label)


def preprocess_text_series(
    series: Series,
    x_min: float,
    x_max: float,
    text_x_min: float,
    rcs_min: Optional[float],
    rcs_max: Optional[float],
    mad_k: Optional[float],
) -> Series:
    # TXT：额外丢弃 x<text_x_min（默认 5m）；低通在 SG 之后再做，不在此处
    s = clip_by_x(series, x_min=max(x_min, text_x_min), x_max=x_max)
    s = filter_by_rcs_range(s, rcs_min=rcs_min, rcs_max=rcs_max)
    s = remove_outliers_mad(s, mad_k=mad_k)
    return s


def smooth_series_savgol(
    series: Series, window_length: int, polyorder: int
) -> Series:
    """
    Savitzky-Golay 平滑（对 rcs 沿 x 排序后滤波），窗长须为奇数且 ≥ 阶数+1。
    """
    try:
        import numpy as np
        from scipy.signal import savgol_filter
    except Exception as e:
        raise RuntimeError(
            "TXT 平滑需要 numpy 与 scipy。请先安装：pip install numpy scipy"
        ) from e

    pairs = sorted(zip(series.x, series.rcs), key=lambda t: t[0])
    xs = np.array([p[0] for p in pairs], dtype=float)
    ys = np.array([p[1] for p in pairs], dtype=float)
    n = int(ys.size)
    if n < 3:
        return Series(xs.tolist(), ys.tolist(), label=series.label)

    w = min(int(window_length), n)
    if w % 2 == 0:
        w -= 1
    if w < 3:
        return Series(xs.tolist(), ys.tolist(), label=series.label)

    po = int(polyorder)
    if po >= w:
        po = w - 1
    if po < 0:
        po = 0

    ys_f = savgol_filter(ys, window_length=w, polyorder=po, mode="interp")
    return Series(
        xs.tolist(),
        ys_f.tolist(),
        label=f"{series.label}（SG 窗={w} 阶={po}）",
    )


def finalize_text_series(
    s_after_offset: Series,
    name_for_legend: str,
    offset: float,
    do_fit: bool,
    bin_width: float,
    sg_window: int,
    sg_polyorder: int,
    lowpass_window: int,
) -> Series:
    """
    TXT 标定后流程：0.1m（可调）分箱 → Savitzky-Golay 平滑（可选，默认窗 51、一阶）→ 低通（拟合后，window<=1 关闭）。
    若未勾选拟合：仅分箱，不做 SG 与低通。
    """
    legend = calibration_label(name_for_legend, offset)
    binned = bin_series_uniform(s_after_offset, bin_width)
    if not do_fit:
        return Series(binned.x, binned.rcs, label=legend)
    sg = smooth_series_savgol(
        binned, window_length=sg_window, polyorder=sg_polyorder
    )
    smoothed = lowpass_moving_average(sg, window=lowpass_window)
    return Series(smoothed.x, smoothed.rcs, label=legend)


def fit_curve_poly(series: Series, degree: int, num_points: int = 400) -> Series:
    """
    对 (x, rcs) 做多项式拟合，并生成一条平滑曲线用于对比。
    """
    if degree < 1:
        raise ValueError("拟合阶数 degree 必须 >= 1")
    if num_points < 50:
        num_points = 50

    try:
        import numpy as np
    except Exception as e:
        raise RuntimeError("缺少 numpy。请先安装：pip install numpy") from e

    pairs = sorted(zip(series.x, series.rcs), key=lambda t: t[0])
    xs = np.array([p[0] for p in pairs], dtype=float)
    ys = np.array([p[1] for p in pairs], dtype=float)

    if xs.size < degree + 1:
        raise ValueError(f"{series.label} 数据点太少，无法做 {degree} 阶拟合（至少需要 {degree+1} 个点）")

    # 高阶拟合容易数值病态，先把 x 映射到 [-1, 1] 区间再拟合，稳定性更好
    x_min = float(xs.min())
    x_max = float(xs.max())
    x_mid = 0.5 * (x_min + x_max)
    x_scale = 0.5 * (x_max - x_min) if x_max != x_min else 1.0
    xs_n = (xs - x_mid) / x_scale

    coeff = np.polyfit(xs_n, ys, deg=degree)
    poly = np.poly1d(coeff)

    x_new = np.linspace(x_min, x_max, num_points)
    x_new_n = (x_new - x_mid) / x_scale
    y_new = poly(x_new_n)

    return Series(x_new.tolist(), y_new.tolist(), label=f"{series.label}（拟合{degree}阶）")


def _configure_matplotlib_chinese_fallback(plt) -> None:
    # Windows 常见中文字体：微软雅黑/宋体/黑体等。找不到时 matplotlib 会自动继续回退。
    plt.rcParams["font.sans-serif"] = [
        "Microsoft YaHei",
        "SimHei",
        "SimSun",
        "Arial Unicode MS",
        "DejaVu Sans",
    ]
    plt.rcParams["axes.unicode_minus"] = False


def plot(series1: Series, series2: Series, out: Optional[str], show: bool) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception as e:
        raise RuntimeError(
            "缺少 matplotlib。请先安装：pip install matplotlib"
        ) from e

    _configure_matplotlib_chinese_fallback(plt)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(series1.x, series1.rcs, marker="o", markersize=2, linewidth=1, label=series1.label)
    ax.plot(series2.x, series2.rcs, marker="x", markersize=2, linewidth=1, label=series2.label)

    ax.set_xlabel("x")
    ax.set_ylabel("rcs")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()

    if out:
        fig.savefig(out, dpi=160)
    if show or not out:
        plt.show()


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="将两种格式的数据画在同一坐标轴下对比（横坐标 x，纵坐标 rcs）。"
    )
    p.add_argument("--ui", action="store_true", help="打开简单 UI 界面")
    p.add_argument("--file1", required=True, help="数据文件1（CSV 或 TXT）")
    p.add_argument("--file2", required=True, help="数据文件2（CSV 或 TXT）")
    p.add_argument("--format1", choices=["auto", "csv", "txt"], default="auto", help="文件1格式")
    p.add_argument("--format2", choices=["auto", "csv", "txt"], default="auto", help="文件2格式")
    p.add_argument("--label1", default="DRI", help="曲线1名称（默认 DRI）")
    p.add_argument("--label2", default="LZ", help="曲线2名称（默认 LZ）")
    p.add_argument(
        "--biao-ding-zhi",
        "--offset",
        dest="offset",
        type=float,
        default=-6.71,
        help="标定值：对文件2的 rcs 做纵坐标偏移（rcs += offset，默认 -6.71）",
    )
    p.add_argument("--x-min", type=float, default=0.0, help="参与对比/拟合的最小 x（默认 0）")
    p.add_argument("--x-max", type=float, default=50.0, help="参与对比/拟合的最大 x（默认 50）")
    p.add_argument("--text-x-min", type=float, default=5.0, help="TXT 的最小 x（剔除 x<该值，默认 5）")
    p.add_argument("--text-rcs-min", type=float, default=-80.0, help="TXT 的 rcs 最小值（超出剔除，默认 -80）")
    p.add_argument("--text-rcs-max", type=float, default=80.0, help="TXT 的 rcs 最大值（超出剔除，默认 80）")
    p.add_argument(
        "--text-mad-k",
        type=float,
        default=6.0,
        help="TXT 异常值剔除强度（MAD 阈值倍数，0 或负数表示关闭；默认 6）",
    )
    p.add_argument(
        "--text-bin-width",
        type=float,
        default=0.1,
        help="TXT x 方向分箱宽度（米，默认 0.1）",
    )
    p.add_argument(
        "--text-lowpass-window",
        type=int,
        default=5,
        help="TXT SG 后低通滤波窗口（移动平均，<=1 关闭；默认 5；越大越平滑）",
    )
    p.add_argument(
        "--text-sg-window",
        type=int,
        default=51,
        help="TXT 分箱后对 rcs 做 Savitzky-Golay 的窗长（须为奇数，默认 51；点少时自动缩短）",
    )
    p.add_argument(
        "--text-sg-order",
        type=int,
        default=1,
        help="TXT Savitzky-Golay 多项式阶数（默认 1）",
    )
    p.add_argument(
        "--fit2",
        action="store_true",
        help="对文件2（通常是 TXT）的 0-50m 数据先拟合成曲线再对比",
    )
    p.add_argument(
        "--fit-degree",
        type=int,
        default=20,
        help="文件2为 CSV 时多项式拟合阶数（默认 20）；TXT 使用 SG，见 --text-sg-window / --text-sg-order",
    )
    p.add_argument("--fit-points", type=int, default=400, help="拟合曲线生成点数（默认 400）")
    p.add_argument("--out", default=None, help="输出图片路径（如 out.png）")
    p.add_argument("--show", action="store_true", help="强制弹窗显示图像（即使指定了 --out）")
    return p


def run_ui() -> int:
    try:
        import tkinter as tk
        from tkinter import filedialog, messagebox, ttk
    except Exception as e:
        raise RuntimeError("你的 Python 环境缺少 tkinter（一般是自带的）。") from e

    root = tk.Tk()
    root.title("RCS 数据对比（简单UI）")
    root.geometry("780x400")

    last_text_curve: Optional[Series] = None
    last_text_curve_source: Optional[str] = None

    file1_var = tk.StringVar(value="")
    file2_var = tk.StringVar(value="")
    out_var = tk.StringVar(value="")
    label1_var = tk.StringVar(value="DRI")
    label2_var = tk.StringVar(value="LZ")
    offset_var = tk.StringVar(value="-6.71")
    x_min_var = tk.StringVar(value="0")
    x_max_var = tk.StringVar(value="50")
    text_x_min_var = tk.StringVar(value="5")
    text_bin_width_var = tk.StringVar(value="0.1")
    text_rcs_min_var = tk.StringVar(value="-80")
    text_rcs_max_var = tk.StringVar(value="80")
    text_mad_k_var = tk.StringVar(value="6")
    text_lp_window_var = tk.StringVar(value="5")
    text_sg_window_var = tk.StringVar(value="51")
    text_sg_order_var = tk.StringVar(value="1")
    fit2_var = tk.BooleanVar(value=True)
    fit_degree_var = tk.StringVar(value="20")

    fmt1_var = tk.StringVar(value="auto")
    fmt2_var = tk.StringVar(value="auto")
    show_var = tk.BooleanVar(value=True)

    def pick_file(target: tk.StringVar) -> None:
        path = filedialog.askopenfilename(
            title="选择数据文件",
            filetypes=[
                ("数据文件", "*.csv *.txt *.tsv *.dat *.log"),
                ("CSV", "*.csv"),
                ("TXT", "*.txt"),
                ("所有文件", "*.*"),
            ],
        )
        if path:
            target.set(path)

    def pick_out() -> None:
        path = filedialog.asksaveasfilename(
            title="保存图片到",
            defaultextension=".png",
            filetypes=[("PNG", "*.png"), ("JPG", "*.jpg"), ("所有文件", "*.*")],
        )
        if path:
            out_var.set(path)

    def run_plot() -> None:
        nonlocal last_text_curve, last_text_curve_source

        f1 = file1_var.get().strip()
        f2 = file2_var.get().strip()
        if not f1 or not os.path.exists(f1):
            messagebox.showerror("错误", "请先选择有效的 文件1。")
            return
        if not f2 or not os.path.exists(f2):
            messagebox.showerror("错误", "请先选择有效的 文件2。")
            return

        try:
            offset = float(offset_var.get().strip() or "-6.71")
        except Exception:
            messagebox.showerror("错误", "标定值必须是数字（例如 -6.71 或 0）。")
            return

        try:
            x_min = float(x_min_var.get().strip() or "0")
            x_max = float(x_max_var.get().strip() or "50")
            if x_max <= x_min:
                raise ValueError
        except Exception:
            messagebox.showerror("错误", "x 范围不合法，请确保 x_max > x_min（例如 0 和 50）。")
            return

        try:
            fit_degree = int(fit_degree_var.get().strip() or "20")
        except Exception:
            messagebox.showerror("错误", "拟合阶数必须是整数（例如 20 或 12）。")
            return

        try:
            text_rcs_min = float(text_rcs_min_var.get().strip() or "-80")
            text_rcs_max = float(text_rcs_max_var.get().strip() or "80")
            if text_rcs_max <= text_rcs_min:
                raise ValueError
        except Exception:
            messagebox.showerror("错误", "TXT rcs 范围不合法，请确保 max > min（例如 -80 和 80）。")
            return

        try:
            text_x_min = float(text_x_min_var.get().strip() or "5")
        except Exception:
            messagebox.showerror("错误", "TXT 起始 x 必须是数字（例如 5）。")
            return

        try:
            text_bin_width = float(text_bin_width_var.get().strip() or "0.1")
            if text_bin_width <= 0:
                raise ValueError
        except Exception:
            messagebox.showerror("错误", "TXT 分箱宽度必须是正数（例如 0.1）。")
            return

        try:
            text_mad_k = float(text_mad_k_var.get().strip() or "6")
        except Exception:
            messagebox.showerror("错误", "TXT 异常值剔除 mad_k 必须是数字（例如 6，或 0 关闭）。")
            return

        try:
            text_lp_window = int(text_lp_window_var.get().strip() or "5")
        except Exception:
            messagebox.showerror("错误", "TXT 低通窗口必须是整数（例如 5，或 1 关闭）。")
            return

        try:
            text_sg_window = int(text_sg_window_var.get().strip() or "51")
            if text_sg_window < 3:
                raise ValueError
        except Exception:
            messagebox.showerror(
                "错误",
                "TXT Savitzky-Golay 窗长必须是整数且 ≥3（默认 51；奇数会自动对齐）。",
            )
            return

        try:
            text_sg_order = int(text_sg_order_var.get().strip() or "1")
            if text_sg_order < 0:
                raise ValueError
        except Exception:
            messagebox.showerror("错误", "TXT Savitzky-Golay 阶数必须是非负整数（默认 1）。")
            return

        try:
            fmt1 = detect_format(f1, fmt1_var.get())
            fmt2 = detect_format(f2, fmt2_var.get())
            s1 = load_series(f1, fmt1, label=label1_var.get().strip() or "DRI")
            s2 = load_series(f2, fmt2, label=label2_var.get().strip() or "LZ")

            # 0-50m（默认）范围裁剪；TXT 还会做：超范围剔除 + 异常值剔除；标定后：分箱→拟合→低通
            s1 = clip_by_x(s1, x_min=x_min, x_max=x_max)
            if fmt2 == "txt":
                s2 = preprocess_text_series(
                    s2,
                    x_min=x_min,
                    x_max=x_max,
                    text_x_min=text_x_min,
                    rcs_min=text_rcs_min,
                    rcs_max=text_rcs_max,
                    mad_k=text_mad_k,
                )
            else:
                s2 = clip_by_x(s2, x_min=x_min, x_max=x_max)

            s2_shifted = apply_offset(s2, offset)
            name2 = s2.label
            if fmt2 == "txt":
                s2_shifted = finalize_text_series(
                    s2_shifted,
                    name_for_legend=name2,
                    offset=offset,
                    do_fit=fit2_var.get(),
                    bin_width=text_bin_width,
                    sg_window=text_sg_window,
                    sg_polyorder=text_sg_order,
                    lowpass_window=text_lp_window,
                )
                last_text_curve = s2_shifted
                last_text_curve_source = f2
            else:
                s2_shifted = Series(s2_shifted.x, s2_shifted.rcs, label=calibration_label(name2, offset))
                if fit2_var.get():
                    s2_shifted = fit_curve_poly(s2_shifted, degree=fit_degree)
                    s2_shifted = Series(
                        s2_shifted.x, s2_shifted.rcs, label=calibration_label(name2, offset)
                    )
                last_text_curve = None
                last_text_curve_source = None

            out_path = out_var.get().strip() or None
            plot(s1, s2_shifted, out=out_path, show=show_var.get())
            if out_path:
                messagebox.showinfo("完成", f"已生成图片：\n{out_path}")
        except Exception as e:
            messagebox.showerror("运行失败", str(e))

    def export_text_curve() -> None:
        """
        导出 TEXT（文件2为 txt）在图中使用的曲线点 (x, rcs)。
        若未生成过图或文件2不是 txt，会提示。
        """
        nonlocal last_text_curve, last_text_curve_source

        if last_text_curve is None:
            messagebox.showwarning("提示", "尚未生成 TEXT 拟合曲线。请先选择 TXT 作为文件2并点击“生成对比图”。")
            return

        base = "text_curve"
        if last_text_curve_source:
            base = os.path.splitext(os.path.basename(last_text_curve_source))[0] + "_curve"

        save_path = filedialog.asksaveasfilename(
            title="导出曲线点到 TXT",
            defaultextension=".txt",
            initialfile=f"{base}.txt",
            filetypes=[("TXT", "*.txt"), ("所有文件", "*.*")],
        )
        if not save_path:
            return

        try:
            with open(save_path, "w", encoding="utf-8") as f:
                f.write("# x(m)\trcs\n")
                for x, y in zip(last_text_curve.x, last_text_curve.rcs):
                    f.write(f"{x:.6f}\t{y:.6f}\n")
            messagebox.showinfo("完成", f"已导出：\n{save_path}")
        except Exception as e:
            messagebox.showerror("导出失败", str(e))

    def add_row(row: int, title: str, var: tk.StringVar, pick_cmd=None) -> None:
        tk.Label(root, text=title, width=10, anchor="e").grid(row=row, column=0, padx=10, pady=8, sticky="e")
        ent = tk.Entry(root, textvariable=var)
        ent.grid(row=row, column=1, columnspan=3, padx=8, pady=8, sticky="we")
        if pick_cmd is not None:
            tk.Button(root, text="选择…", command=pick_cmd, width=10).grid(
                row=row, column=4, padx=10, pady=8, sticky="w"
            )

    root.columnconfigure(1, weight=1)
    root.columnconfigure(2, weight=1)
    root.columnconfigure(3, weight=1)

    add_row(0, "文件1", file1_var, pick_cmd=lambda: pick_file(file1_var))
    add_row(1, "文件2", file2_var, pick_cmd=lambda: pick_file(file2_var))

    tk.Label(root, text="格式1", width=10, anchor="e").grid(row=2, column=0, padx=10, pady=8, sticky="e")
    ttk.Combobox(root, textvariable=fmt1_var, values=["auto", "csv", "txt"], width=10, state="readonly").grid(
        row=2, column=1, padx=8, pady=8, sticky="w"
    )
    tk.Label(root, text="格式2", width=10, anchor="e").grid(row=2, column=2, padx=10, pady=8, sticky="e")
    ttk.Combobox(root, textvariable=fmt2_var, values=["auto", "csv", "txt"], width=10, state="readonly").grid(
        row=2, column=3, padx=8, pady=8, sticky="w"
    )

    tk.Label(root, text="名称1", width=10, anchor="e").grid(row=3, column=0, padx=10, pady=8, sticky="e")
    tk.Entry(root, textvariable=label1_var).grid(row=3, column=1, padx=8, pady=8, sticky="we")
    tk.Label(root, text="名称2", width=10, anchor="e").grid(row=3, column=2, padx=10, pady=8, sticky="e")
    tk.Entry(root, textvariable=label2_var).grid(row=3, column=3, padx=8, pady=8, sticky="we")

    tk.Label(root, text="标定值", width=10, anchor="e").grid(row=4, column=0, padx=10, pady=8, sticky="e")
    tk.Entry(root, textvariable=offset_var).grid(row=4, column=1, padx=8, pady=8, sticky="w")
    tk.Label(root, text="(rcs += 标定值，作用于文件2)", anchor="w").grid(
        row=4, column=2, columnspan=3, padx=8, pady=8, sticky="w"
    )

    tk.Label(root, text="x范围", width=10, anchor="e").grid(row=5, column=0, padx=10, pady=8, sticky="e")
    tk.Entry(root, textvariable=x_min_var, width=10).grid(row=5, column=1, padx=8, pady=8, sticky="w")
    tk.Label(root, text="~", anchor="center").grid(row=5, column=1, padx=72, pady=8, sticky="w")
    tk.Entry(root, textvariable=x_max_var, width=10).grid(row=5, column=1, padx=92, pady=8, sticky="w")
    tk.Label(root, text="(默认 0~50m；TXT 50m 之后剔除)", anchor="w").grid(
        row=5, column=2, columnspan=3, padx=8, pady=8, sticky="w"
    )

    tk.Checkbutton(root, text="对文件2先拟合/平滑成曲线再对比", variable=fit2_var).grid(
        row=6, column=1, padx=8, pady=6, sticky="w"
    )
    tk.Label(root, text="CSV 拟合阶", anchor="e").grid(row=6, column=2, padx=8, pady=6, sticky="e")
    tk.Entry(root, textvariable=fit_degree_var, width=6).grid(row=6, column=3, padx=8, pady=6, sticky="w")

    tk.Label(root, text="TXT rcs", width=10, anchor="e").grid(row=7, column=0, padx=10, pady=8, sticky="e")
    tk.Entry(root, textvariable=text_rcs_min_var, width=10).grid(row=7, column=1, padx=8, pady=8, sticky="w")
    tk.Label(root, text="~", anchor="center").grid(row=7, column=1, padx=72, pady=8, sticky="w")
    tk.Entry(root, textvariable=text_rcs_max_var, width=10).grid(row=7, column=1, padx=92, pady=8, sticky="w")
    tk.Label(root, text="(超范围剔除)", anchor="w").grid(row=7, column=2, padx=8, pady=8, sticky="w")

    tk.Label(root, text="TXT 分箱", width=10, anchor="e").grid(row=8, column=0, padx=10, pady=8, sticky="e")
    tk.Entry(root, textvariable=text_bin_width_var, width=10).grid(row=8, column=1, padx=8, pady=8, sticky="w")
    tk.Label(root, text="(米，默认 0.1；先分箱再 SG 再低通)", anchor="w").grid(
        row=8, column=2, columnspan=3, padx=8, pady=8, sticky="w"
    )

    tk.Label(root, text="TXT SG", width=10, anchor="e").grid(row=9, column=0, padx=10, pady=8, sticky="e")
    tk.Label(root, text="窗长", anchor="e").grid(row=9, column=1, padx=8, pady=8, sticky="w")
    tk.Entry(root, textvariable=text_sg_window_var, width=8).grid(row=9, column=1, padx=52, pady=8, sticky="w")
    tk.Label(root, text="阶数", anchor="e").grid(row=9, column=2, padx=8, pady=8, sticky="e")
    tk.Entry(root, textvariable=text_sg_order_var, width=6).grid(row=9, column=3, padx=8, pady=8, sticky="w")
    tk.Label(root, text="(Savgol；默认 51 / 1)", anchor="w").grid(row=9, column=4, padx=8, pady=8, sticky="w")

    tk.Label(root, text="TXT 起始x", width=10, anchor="e").grid(row=10, column=0, padx=10, pady=8, sticky="e")
    tk.Entry(root, textvariable=text_x_min_var, width=10).grid(row=10, column=1, padx=8, pady=8, sticky="w")
    tk.Label(root, text="(剔除 x<该值，默认 5m)", anchor="w").grid(row=10, column=2, columnspan=3, padx=8, pady=8, sticky="w")

    tk.Label(root, text="TXT 异常", width=10, anchor="e").grid(row=11, column=0, padx=10, pady=8, sticky="e")
    tk.Label(root, text="mad_k", anchor="e").grid(row=11, column=1, padx=8, pady=8, sticky="w")
    tk.Entry(root, textvariable=text_mad_k_var, width=6).grid(row=11, column=1, padx=58, pady=8, sticky="w")
    tk.Label(root, text="SG后低通", anchor="e").grid(row=11, column=2, padx=8, pady=8, sticky="e")
    tk.Entry(root, textvariable=text_lp_window_var, width=6).grid(row=11, column=3, padx=8, pady=8, sticky="w")
    tk.Label(root, text="(窗口<=1 关闭)", anchor="w").grid(row=11, column=4, padx=8, pady=8, sticky="w")

    add_row(12, "输出图", out_var, pick_cmd=pick_out)

    tk.Checkbutton(root, text="弹窗显示图像", variable=show_var).grid(
        row=13, column=1, padx=8, pady=6, sticky="w"
    )

    btns = tk.Frame(root)
    btns.grid(row=14, column=0, columnspan=5, pady=16)
    tk.Button(btns, text="生成对比图", command=run_plot, width=16).pack(side="left", padx=10)
    tk.Button(btns, text="导出拟合曲线TXT", command=export_text_curve, width=18).pack(side="left", padx=10)
    tk.Button(btns, text="退出", command=root.destroy, width=10).pack(side="left", padx=10)

    root.mainloop()
    return 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    # 无参数直接运行时：打开 UI
    if argv is None and len(sys.argv) == 1:
        return run_ui()

    # 支持：python compare_rcs.py --ui（避免必填参数校验）
    arg_list = list(sys.argv[1:] if argv is None else argv)
    if "--ui" in arg_list:
        return run_ui()

    args = build_arg_parser().parse_args(argv)

    fmt1 = detect_format(args.file1, args.format1)
    fmt2 = detect_format(args.file2, args.format2)

    s1 = load_series(args.file1, fmt1, label=args.label1)
    s2 = load_series(args.file2, fmt2, label=args.label2)

    # 同时把两份数据裁剪到相同 x 区间便于对比；TXT 额外做：超范围剔除 + 异常值剔除；标定后：分箱→SG→低通
    s1 = clip_by_x(s1, x_min=args.x_min, x_max=args.x_max)
    if fmt2 == "txt":
        s2 = preprocess_text_series(
            s2,
            x_min=args.x_min,
            x_max=args.x_max,
            text_x_min=args.text_x_min,
            rcs_min=args.text_rcs_min,
            rcs_max=args.text_rcs_max,
            mad_k=args.text_mad_k,
        )
    else:
        s2 = clip_by_x(s2, x_min=args.x_min, x_max=args.x_max)

    s2_shifted = apply_offset(s2, args.offset)
    name2 = s2.label
    if fmt2 == "txt":
        s2_shifted = finalize_text_series(
            s2_shifted,
            name_for_legend=name2,
            offset=args.offset,
            do_fit=args.fit2,
            bin_width=args.text_bin_width,
            sg_window=args.text_sg_window,
            sg_polyorder=args.text_sg_order,
            lowpass_window=args.text_lowpass_window,
        )
    else:
        s2_shifted = Series(s2_shifted.x, s2_shifted.rcs, label=calibration_label(name2, args.offset))
        if args.fit2:
            s2_shifted = fit_curve_poly(s2_shifted, degree=args.fit_degree, num_points=args.fit_points)
            s2_shifted = Series(s2_shifted.x, s2_shifted.rcs, label=calibration_label(name2, args.offset))

    plot(s1, s2_shifted, out=args.out, show=args.show)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

