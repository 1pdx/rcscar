# renamed: rcs_workflow.py
# role: RCS 采集、解析、处理、绘图和导出。
# contains: 距离-RCS/圆周-RCS 采集流程、Cluster Raw CSV 解析、RCS00/01 功率合并、曲线拟合、参考限线、Matplotlib 绘制、图片/CSV 导出。
# also contains: 圆周航向角优先/时间平铺兜底的极角生成、静态圆周测试、RCS 目标关联和重锁记录。
# notes: 轨迹分段何时触发 RCS 在 trajectory_workflow.py；这里只负责 RCS 数据本身。
# -*- coding: utf-8 -*-

from ui_shared import *
from ui_shared import _peak_smooth_rcs_series, _rcs_ref_mod


MainWindow = None  # 主入口回填真实 MainWindow，供 mixin 内静态方法引用。


class MainWindowRcsMixin:
    def _on_rcs_plot_mode_changed(self, _index: int) -> None:
        self._rcs_plot_mode = self._get_rcs_plot_mode()
        self._draw_rcs()


    def _set_rcs_plot_calibration_db(self, value: float, *, redraw: bool = False) -> None:
        self._rcs_plot_calibration_db = float(value)
        spin = getattr(self, "rcs_calibration_spin", None)
        if spin is not None:
            spin.blockSignals(True)
            spin.setValue(float(value))
            spin.blockSignals(False)
        if redraw:
            self._draw_rcs()


    def _on_rcs_plot_calibration_changed(self, value: float) -> None:
        self._set_rcs_plot_calibration_db(float(value), redraw=True)


    def _get_rcs_plot_mode(self) -> str:
        if hasattr(self, "rcs_plot_mode_combo") and self.rcs_plot_mode_combo is not None:
            mode = self.rcs_plot_mode_combo.currentData()
            if isinstance(mode, str) and mode:
                return mode
        return str(getattr(self, "_rcs_plot_mode", "distance") or "distance")


    def _select_rcs_plot_mode(self, mode: str) -> None:
        """切换 RCS 绘图模式（距离-RCS / 圆周-RCS），并同步内部状态。"""
        m = str(mode or "").strip()
        if m not in {"distance", "orbit"}:
            return
        self._rcs_plot_mode = m
        combo = getattr(self, "rcs_plot_mode_combo", None)
        if combo is None:
            return
        for i in range(combo.count()):
            if combo.itemData(i) == m:
                combo.blockSignals(True)
                combo.setCurrentIndex(i)
                combo.blockSignals(False)
                return


    def _ensure_rcs_axes(self, plot_mode: str) -> None:
        projection = "polar" if plot_mode == "orbit" else "cartesian"
        if projection == getattr(self, "_rcs_ax_projection", "") and getattr(self, "rcs_ax", None) is not None:
            return
        figure = self.rcs_canvas.figure
        figure.clf()
        if projection == "polar":
            self.rcs_ax = figure.add_subplot(111, projection="polar")
            figure.subplots_adjust(bottom=0.10, top=0.90)
        else:
            self.rcs_ax = figure.add_subplot(111)
            figure.subplots_adjust(bottom=0.18)
        self._rcs_ax_projection = projection


    def _append_rcs_orbit_sample(
        self,
        point: Optional[CurvePoint],
        pose: Optional[PoseSolution] = None,
    ) -> None:
        if point is None or self.target_point is None:
            return
        resolved_pose = pose if pose is not None else get_robot_pose()
        if resolved_pose is None:
            return

        dx = float(resolved_pose.x) - float(self.target_point[0])
        dy = float(resolved_pose.y) - float(self.target_point[1])
        orbit_radius_m = math.hypot(dx, dy)
        if orbit_radius_m <= 1e-3:
            return

        self._rcs_orbit_samples.append(
            {
                "timestamp": float(point.t),
                "angle_rad": float(math.atan2(dy, dx)),
                "orbit_radius_m": float(orbit_radius_m),
                "rcs_raw": float(point.rcs_raw),
                "rcs_filt": float(point.rcs_filt),
                "car_x_m": float(resolved_pose.x),
                "car_y_m": float(resolved_pose.y),
            }
        )
        if len(self._rcs_orbit_samples) > 12000:
            self._rcs_orbit_samples = self._rcs_orbit_samples[-12000:]


    @staticmethod
    def _finite_row_float(r: Dict[str, float], *keys: str) -> Optional[float]:
        for key in keys:
            if key not in r:
                continue
            try:
                v = float(r[key])
            except (TypeError, ValueError):
                continue
            if math.isfinite(v):
                return v
        return None


    @staticmethod
    def _orbit_angles_from_heading_rows(
        rows: List[Dict[str, float]],
    ) -> Optional[Tuple[np.ndarray, float, Optional[bool]]]:
        angles: List[float] = []
        for r in rows:
            yaw = MainWindow._finite_row_float(r, "yaw_rad")
            if yaw is None:
                heading_deg = MainWindow._finite_row_float(r, "heading_deg", "Heading")
                if heading_deg is not None:
                    yaw = math.radians(float(heading_deg))
            if yaw is None:
                return None
            yaw_deg = math.degrees(float(yaw))
            azimuth_deg = (90.0 - yaw_deg) % 360.0
            angles.append(math.radians(azimuth_deg))

        if len(angles) < 2:
            return None

        angle_arr = np.asarray(angles, dtype=float)
        if not np.all(np.isfinite(angle_arr)):
            return None
        span_arr = np.unwrap(angle_arr)
        span = float(np.max(span_arr) - np.min(span_arr))
        if span < math.radians(2.0):
            return None

        return np.mod(angle_arr, 2.0 * math.pi).astype(float), float(math.degrees(span)), None


    @staticmethod
    def _orbit_angles_from_time_rows(rows: List[Dict[str, float]]) -> np.ndarray:
        t_arr = np.asarray([float(r["t"]) for r in rows], dtype=float)
        span_t = float(t_arr[-1] - t_arr[0])
        if span_t <= 1e-12:
            return np.linspace(0.0, 2.0 * math.pi, t_arr.size, dtype=float, endpoint=True)
        return (t_arr - t_arr[0]) / span_t * (2.0 * math.pi)


    def _build_orbit_polar_series_from_rows(
        self, rows: List[Dict[str, float]]
    ) -> Optional[Dict[str, Any]]:
        """极径为 RCS(dBsm)，极角优先用惯导 yaw 换算的 0-360°方位角；无效时回退到时间平铺。

        当前圆周图直接以真实 dBsm 作为极径；保留 floor/radii 字段仅兼容旧缓存结构。
        相邻时间点 RCS 跳变达到 ORBIT_RCS_ADJACENT_JUMP_REJECT_DB 时剔除当前点。
        若数据来自 Cluster Raw 重放，rcs_filt 应与直线段相同：
        先行内 RCS00+RCS01 线性功率合并（见 _parse_cluster_rcs_csv / _cluster_rcs_rowwise_rcs00_rcs01_linear_power）。
        """
        if len(rows) < 2:
            return None

        def _row_ok(r: Dict[str, float]) -> bool:
            try:
                t = float(r["t"])
                v = float(r["rcs_filt"])
                return bool(np.isfinite(t) and np.isfinite(v))
            except (KeyError, TypeError, ValueError):
                return False

        rows_sorted = sorted((r for r in rows if _row_ok(r)), key=lambda r: float(r["t"]))
        if len(rows_sorted) < 2:
            return None

        filtered_rows: List[Dict[str, float]] = []
        last_v: Optional[float] = None
        jump_thr = float(ORBIT_RCS_ADJACENT_JUMP_REJECT_DB)
        for r in rows_sorted:
            v = float(r["rcs_filt"])
            if last_v is not None and abs(v - last_v) >= jump_thr:
                continue
            filtered_rows.append(r)
            last_v = v
        if len(filtered_rows) < 2:
            return None

        values = np.asarray([float(r["rcs_filt"]) for r in filtered_rows], dtype=float)
        heading_angles = MainWindow._orbit_angles_from_heading_rows(filtered_rows)
        if heading_angles is not None:
            angles, span_deg, clockwise = heading_angles
            angle_source = "heading"
        else:
            angles = MainWindow._orbit_angles_from_time_rows(filtered_rows)
            span_deg = 360.0
            clockwise = None
            angle_source = "time"
        fit_angles, fit_values = self._bin_orbit_rcs_by_angle(
            angles,
            values,
            ORBIT_RCS_ANGLE_BIN_DEG,
        )

        rmin = float(np.min(values))
        rmax = float(np.max(values))
        tick_step = max(float(ORBIT_RCS_RADIAL_TICK_STEP_DB), 1e-6)
        tick_start = math.floor(rmin / tick_step) * tick_step
        tick_end = math.ceil(rmax / tick_step) * tick_step
        if tick_end <= tick_start:
            tick_end = tick_start + tick_step
        tick_values = np.arange(tick_start, tick_end + tick_step * 0.5, tick_step, dtype=float)
        floor = tick_start - max(0.5, tick_step * 0.1)
        radii = values - floor
        tick_positions = tick_values - floor
        r_top = float(np.max(tick_positions)) + tick_step * 0.2
        return {
            "mode": "fixed_radius_time",
            "angles": angles,
            "radii": radii,
            "values": values,
            "tick_values": tick_values,
            "tick_positions": tick_positions,
            "r_top": r_top,
            "span_deg": span_deg,
            "value_min": rmin,
            "value_max": rmax,
            "r_floor": floor,
            "radial_tick_step_db": tick_step,
            "raw_point_count": len(rows_sorted),
            "filtered_point_count": len(filtered_rows),
            "jump_reject_db": jump_thr,
            "orbit_angle_by_time": angle_source == "time",
            "orbit_angle_source": angle_source,
            "orbit_clockwise": clockwise,
            "fit_angles": fit_angles,
            "fit_values": fit_values,
            "angle_bin_deg": float(ORBIT_RCS_ANGLE_BIN_DEG),
            "fit_point_count": int(fit_angles.size),
        }


    @staticmethod
    def _bin_orbit_rcs_by_angle(
        angles: np.ndarray,
        values: np.ndarray,
        bin_deg: float = ORBIT_RCS_ANGLE_BIN_DEG,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """按平铺后的角度分箱，箱内 RCS 使用 dBsm 算术平均。"""
        a = np.asarray(angles, dtype=float)
        v = np.asarray(values, dtype=float)
        if a.size == 0 or v.size == 0:
            return np.asarray([], dtype=float), np.asarray([], dtype=float)
        n = min(int(a.size), int(v.size))
        a = a[:n]
        v = v[:n]
        mask = np.isfinite(a) & np.isfinite(v)
        if int(np.count_nonzero(mask)) < 2:
            return np.asarray([], dtype=float), np.asarray([], dtype=float)
        a = a[mask]
        v = v[mask]

        two_pi = 2.0 * math.pi
        try:
            bin_deg_eff = float(bin_deg)
        except (TypeError, ValueError):
            bin_deg_eff = 1.0
        if not math.isfinite(bin_deg_eff):
            bin_deg_eff = 1.0
        bin_deg_eff = max(bin_deg_eff, 1e-3)
        n_bins = max(1, int(math.ceil(360.0 / bin_deg_eff)))
        bin_w = two_pi / float(n_bins)
        # 这里保留 [0, 2π] 的铺展语义；终点 2π 归入最后一个角度箱，不回绕到 0。
        a = np.clip(a, 0.0, np.nextafter(two_pi, 0.0))
        bin_ids = np.floor(a / bin_w).astype(np.int64)
        bin_ids = np.clip(bin_ids, 0, n_bins - 1)

        by_bin: Dict[int, List[float]] = defaultdict(list)
        for bid, val in zip(bin_ids.tolist(), v.tolist()):
            fv = float(val)
            if math.isfinite(fv):
                by_bin[int(bid)].append(fv)
        if not by_bin:
            return np.asarray([], dtype=float), np.asarray([], dtype=float)

        out_angles: List[float] = []
        out_values: List[float] = []
        for bid in sorted(by_bin.keys()):
            vals = by_bin[int(bid)]
            if not vals:
                continue
            out_angles.append((float(bid) + 0.5) * bin_w)
            out_values.append(float(np.mean(np.asarray(vals, dtype=float))))
        return np.asarray(out_angles, dtype=float), np.asarray(out_values, dtype=float)


    @staticmethod
    def _is_orbit_rcs_polar_csv(path: Path) -> bool:
        """是否为圆周极坐标采样 CSV（文件名含 __orbit_rcs__，或表头含极坐标列）。"""
        if path.suffix.lower() != ".csv":
            return False
        if "__orbit_rcs__" in path.name:
            return True
        try:
            with path.open(encoding="utf-8-sig", newline="") as f:
                reader = csv.reader(f)
                header = next(reader, None)
            if not header:
                return False
            h = {str(x).strip().lower() for x in header if str(x).strip()}
            need = {"t", "theta_rad", "rcs_raw", "rcs_filt", "oid"}
            x_ok = ("x_m" in h or "x" in h) and ("y_m" in h or "y" in h)
            return need.issubset(h) and x_ok
        except OSError:
            return False


    @staticmethod
    def _cluster_csv_has_dri_cluster_header(path: Path) -> bool:
        """是否为 Cluster Raw（Time + DX00…RCS19）表头。"""
        if path.suffix.lower() != ".csv":
            return False
        try:
            with path.open(encoding="utf-8-sig", newline="") as f:
                reader = csv.reader(f)
                for row in reader:
                    if not row:
                        continue
                    if row[0].strip() == "Time" and any((c or "").strip() == "DX00" for c in row):
                        return True
        except OSError:
            return False
        return False


    @staticmethod
    def _cluster_csv_data_row_count(path: Path) -> int:
        """统计 Cluster Raw 表头后的数据行数，用于判断分段 CSV 是否真正写入。"""
        if path.suffix.lower() != ".csv":
            return 0
        count = 0
        header_seen = False
        try:
            with path.open(encoding="utf-8-sig", newline="") as f:
                reader = csv.reader(f)
                for row in reader:
                    if not row:
                        continue
                    if not header_seen:
                        if row[0].strip() == "Time" and any((c or "").strip() == "DX00" for c in row):
                            header_seen = True
                        continue
                    if any(str(c or "").strip() for c in row):
                        count += 1
        except OSError:
            return 0
        return count


    @staticmethod
    def _is_orbit_cluster_raw_csv(path: Path) -> bool:
        """是否为圆周段 Cluster Raw CSV（如 *_orbit_cluster_segXX_Raw_*.csv）。"""
        if path.suffix.lower() != ".csv":
            return False
        marker_tokens = ("orbit_cluster", "_orbit_", "__orbit_rcs__", "圆周")
        name_has_marker = any(token in path.name.casefold() for token in marker_tokens)
        metadata_has_marker = False
        try:
            with path.open(encoding="utf-8-sig", newline="") as f:
                reader = csv.reader(f)
                for row in reader:
                    if not row:
                        continue
                    row_text = " ".join(str(c or "").strip() for c in row).casefold()
                    metadata_has_marker = metadata_has_marker or any(
                        token in row_text for token in marker_tokens
                    )
                    if row[0].strip() == "Time" and any((c or "").strip() == "DX00" for c in row):
                        return bool(name_has_marker or metadata_has_marker)
        except OSError:
            return False
        return False


    def _parse_orbit_rcs_csv(self, file_path: str) -> List[Dict[str, float]]:
        path = Path(file_path)
        rows: List[Dict[str, float]] = []
        with path.open(encoding="utf-8-sig", newline="") as f:
            reader = csv.DictReader(f)
            if not reader.fieldnames:
                raise ValueError("圆周RCS CSV 无表头或为空")
            lower = {str(k).strip().lower(): k for k in reader.fieldnames if k is not None}

            def _col(*names: str) -> Optional[str]:
                for n in names:
                    if n in lower:
                        return lower[n]
                return None

            c_t = _col("t", "time_s", "time")
            c_th = _col("theta_rad")
            c_rr = _col("rcs_raw")
            c_rf = _col("rcs_filt", "rcs_filtered")
            c_x = _col("x_m", "x")
            c_y = _col("y_m", "y")
            c_oid = _col("oid", "cluster_id")
            if not all([c_t, c_th, c_rr, c_rf, c_x, c_y, c_oid]):
                raise ValueError("圆周RCS CSV 表头缺少必需列 (t,theta_rad,rcs_raw,rcs_filt,x_m,y_m,oid)")
            for raw in reader:
                if not raw:
                    continue
                try:
                    def _get(key: str) -> str:
                        v = raw.get(key)
                        if v is None:
                            raise ValueError
                        return str(v).strip()

                    rows.append(
                        {
                            "t": float(_get(c_t)),
                            "theta_rad": float(_get(c_th)),
                            "rcs_raw": float(_get(c_rr)),
                            "rcs_filt": float(_get(c_rf)),
                            "x": float(_get(c_x)),
                            "y": float(_get(c_y)),
                            "oid": int(float(_get(c_oid))),
                        }
                    )
                except (ValueError, TypeError, KeyError):
                    continue
        if len(rows) < 2:
            raise ValueError("圆周RCS CSV 有效点不足")
        return rows


    def _get_rcs_orbit_plot_series(self) -> Optional[Dict[str, Any]]:
        if self._orbit_polar_cached_series is not None:
            return self._orbit_polar_cached_series
        if not self._rcs_orbit_samples:
            return None

        rows_like: List[Dict[str, float]] = []
        for i, sample in enumerate(self._rcs_orbit_samples):
            try:
                v = float(sample.get("rcs_filt", float("nan")))
            except (TypeError, ValueError):
                continue
            if not np.isfinite(v):
                continue
            try:
                t = float(sample.get("timestamp", float("nan")))
            except (TypeError, ValueError):
                t = float("nan")
            if not np.isfinite(t):
                t = float(i)
            rows_like.append({"t": t, "rcs_filt": v})
        if len(rows_like) >= 2:
            ser = self._build_orbit_polar_series_from_rows(rows_like)
            if ser is not None:
                return ser

        return None


    def _update_rcs_save_dir_button_tooltip(self) -> None:
        if hasattr(self, "btn_select_rcs_save_dir") and self.btn_select_rcs_save_dir is not None:
            self.btn_select_rcs_save_dir.setToolTip(
                f"从 {self._rcs_save_dir} 选择 Cluster Raw CSV；圆周-RCS 图由 Raw 经行内功率合并后优先按惯导航向绘制。"
            )


    def _on_select_rcs_save_dir(self) -> None:
        # Repurposed: choose a saved RCS data file and plot it.
        # Saved data is auto-written to recorded_data/rcs_data, so we do not change the save directory here.
        self._show_tool_dialog(self._rcs_viewer_dialog)
        if self._rcs_recording:
            QtWidgets.QMessageBox.information(
                self,
                "Cluster RCS 采集中",
                "请先等待当前 Cluster RCS 采集结束后再绘制历史数据。",
            )
            return

        file_paths, _ = QtWidgets.QFileDialog.getOpenFileNames(
            self,
            "选择已保存的 Cluster Raw CSV（可多选合并拟合，仅距离-RCS）",
            str(self._rcs_save_dir),
            "CSV 表格 (*.csv);;所有文件(*)",
        )
        if not file_paths:
            return
        # 多选合并：仅支持距离-RCS 文件，合并后只绘制一条拟合曲线
        if len(file_paths) >= 2:
            try:
                curve = self._build_merged_loaded_rcs_curve(file_paths, display_name=None)
            except Exception as exc:
                QtWidgets.QMessageBox.warning(self, "载入失败", f"合并读取RCS数据失败: {exc}")
                self._log(f"RCS合并绘图载入失败: {exc}")
                return
            ref_summary = self._format_rcs_reference_summary()
            plot_dlg = RcsPlotConfigDialog(
                str(file_paths[0]),
                ref_summary,
                default_custom_name=curve.display_name,
                default_calibration_db=self._rcs_plot_calibration_db,
                enable_curve_csv_export=True,
                parent=self,
            )
            if plot_dlg.exec_() != QtWidgets.QDialog.Accepted:
                return
            custom_name, cal_db, export_curve_csv = plot_dlg.values()
            self._rcs_plot_calibration_db = cal_db
            if custom_name:
                curve = replace(curve, display_name=custom_name)
            show_ref = bool(self._rcs_ref_class and self._rcs_ref_angle and self._rcs_ref_limits is not None)
            self._rcs_hide_reference_limits = not show_ref
            self._rcs_base_file_path = None
            self._rcs_curve_csv_source_paths = [str(p) for p in file_paths]
            self._loaded_rcs_curves = []
            self._orbit_polar_cached_series = None
            self.rcs_recorder.reset()
            self.rcs_recorder.segments = [list(seg) for seg in curve.segments]
            self.rcs_recorder._cur = []
            self.rcs_recorder._ended = True
            self.rcs_recorder.oid = None
            self._rcs_recording = False
            self._rcs_show_only_fitted = False
            self._rcs_orbit_samples = []
            self._rcs_active_segment_index = None
            self._rcs_active_path_name = None
            self._rcs_active_target_name = None
            self._rcs_fitted = curve.fitted
            self._rcs_target_name = curve.display_name
            self._rcs_segment_run_labels = curve.segment_run_labels
            self._rcs_segment_display_labels = curve.segment_display_labels
            for i in range(self.rcs_plot_mode_combo.count()):
                if self.rcs_plot_mode_combo.itemData(i) == "distance":
                    self.rcs_plot_mode_combo.blockSignals(True)
                    self.rcs_plot_mode_combo.setCurrentIndex(i)
                    self.rcs_plot_mode_combo.blockSignals(False)
                    break
            self._rcs_plot_mode = "distance"
            self._ensure_rcs_axes("distance")
            cal_note = f" | 标定{cal_db:+.2f}dB" if abs(float(cal_db)) > 1e-9 else ""
            csv_note = ""
            if export_curve_csv:
                _filtered_paths, combined_path = self._try_export_rcs_curve_csvs(
                    list(file_paths),
                    cal_db,
                )
                if combined_path is not None:
                    csv_note = f" | CSV={combined_path.name}"
            self.rcs_status_label.setText(
                f"RCS绘图: 已合并 {len(file_paths)} 个文件 | 总点数={curve.point_count}{cal_note}{csv_note}"
            )
            self._log(
                f"RCS合并绘图数据已载入: 文件数={len(file_paths)} | 总点数={curve.point_count}"
                f"{cal_note}{csv_note}"
            )
            self._draw_rcs()
            return

        file_path = str(file_paths[0])

        is_orbit_file = MainWindow._is_orbit_rcs_polar_csv(Path(file_path))
        if is_orbit_file:
            try:
                orows = self._parse_orbit_rcs_csv(file_path)
            except Exception as exc:
                QtWidgets.QMessageBox.warning(self, "载入失败", f"读取圆周RCS CSV 失败: {exc}")
                self._log(f"圆周RCS载入失败: {exc}")
                return
            ref_summary = self._format_rcs_reference_summary()
            stem = Path(file_path).stem
            plot_dlg = RcsPlotConfigDialog(
                file_path,
                ref_summary,
                default_custom_name=stem,
                default_calibration_db=self._rcs_plot_calibration_db,
                enable_curve_csv_export=False,
                parent=self,
            )
            if plot_dlg.exec_() != QtWidgets.QDialog.Accepted:
                return
            custom_name, cal_db, _export_curve_csv = plot_dlg.values()
            self._rcs_plot_calibration_db = cal_db
            self._orbit_polar_cached_series = self._build_orbit_polar_series_from_rows(orows)
            # 圆周图不叠加距离-RCS参考上下限
            self._rcs_hide_reference_limits = True
            self._rcs_base_file_path = str(file_path)
            self._rcs_curve_csv_source_paths = []
            self._loaded_rcs_curves = []
            self.rcs_recorder.reset()
            self._rcs_segment_run_labels = None
            self._rcs_segment_display_labels = None
            self.rcs_recorder._cur = []
            self.rcs_recorder._ended = True
            self.rcs_recorder.oid = None
            self._rcs_recording = False
            self._rcs_show_only_fitted = False
            self._rcs_orbit_samples = []
            self._rcs_fitted = None
            self._rcs_active_segment_index = None
            self._rcs_active_path_name = None
            self._rcs_active_target_name = None
            self._rcs_target_name = custom_name or stem
            self._rcs_segment_display_labels = None
            for i in range(self.rcs_plot_mode_combo.count()):
                if self.rcs_plot_mode_combo.itemData(i) == "orbit":
                    self.rcs_plot_mode_combo.blockSignals(True)
                    self.rcs_plot_mode_combo.setCurrentIndex(i)
                    self.rcs_plot_mode_combo.blockSignals(False)
                    break
            self._rcs_plot_mode = "orbit"
            self._ensure_rcs_axes("orbit")
            cal_note = f" | 标定{cal_db:+.2f}dB" if abs(float(cal_db)) > 1e-9 else ""
            self.rcs_status_label.setText(
                f"圆周RCS: 已载入 {Path(file_path).name} | 点数={len(orows)} | "
                f"角度分箱均值曲线 | 径向=RCS(dBsm){cal_note}"
            )
            self._log(
                f"圆周RCS CSV 已载入: {file_path} | 点数={len(orows)} | "
                f"平铺到2π后按{ORBIT_RCS_ANGLE_BIN_DEG:g}°分箱算术平均{cal_note}"
            )
            self._draw_rcs()
            return

        if MainWindow._is_orbit_cluster_raw_csv(Path(file_path)):
            try:
                orows = self._orbit_plot_rows_from_cluster_raw_path(file_path)
            except Exception as exc:
                QtWidgets.QMessageBox.warning(self, "载入失败", f"读取圆周 Cluster Raw 失败: {exc}")
                self._log(f"圆周 Cluster Raw 载入失败: {exc}")
                return
            if len(orows) < 2:
                QtWidgets.QMessageBox.warning(self, "载入失败", "圆周 Cluster Raw 有效 RCS 点不足")
                self._log(f"圆周 Cluster Raw 载入失败: 有效点不足 | 文件={file_path}")
                return
            ref_summary = self._format_rcs_reference_summary()
            stem = Path(file_path).stem
            plot_dlg = RcsPlotConfigDialog(
                file_path,
                ref_summary,
                default_custom_name=stem,
                default_calibration_db=self._rcs_plot_calibration_db,
                enable_curve_csv_export=False,
                parent=self,
            )
            if plot_dlg.exec_() != QtWidgets.QDialog.Accepted:
                return
            custom_name, cal_db, _export_curve_csv = plot_dlg.values()
            self._rcs_plot_calibration_db = cal_db
            self._orbit_polar_cached_series = self._build_orbit_polar_series_from_rows(orows)
            self._rcs_hide_reference_limits = True
            self._rcs_base_file_path = str(file_path)
            self._rcs_curve_csv_source_paths = []
            self._loaded_rcs_curves = []
            self.rcs_recorder.reset()
            self._rcs_segment_run_labels = None
            self.rcs_recorder._cur = []
            self.rcs_recorder._ended = True
            self.rcs_recorder.oid = None
            self._rcs_recording = False
            self._rcs_show_only_fitted = False
            self._rcs_orbit_samples = []
            self._rcs_fitted = None
            self._rcs_active_segment_index = None
            self._rcs_active_path_name = None
            self._rcs_active_target_name = None
            self._rcs_target_name = custom_name or stem
            self._select_rcs_plot_mode("orbit")
            self._ensure_rcs_axes("orbit")
            cal_note = f" | 标定{cal_db:+.2f}dB" if abs(float(cal_db)) > 1e-9 else ""
            angle_source = (
                str(self._orbit_polar_cached_series.get("orbit_angle_source", "time"))
                if self._orbit_polar_cached_series is not None
                else "time"
            )
            angle_text = "惯导方位角" if angle_source == "heading" else "时间平铺兜底"
            self.rcs_status_label.setText(
                f"圆周RCS: 已载入 {Path(file_path).name} | 点数={len(orows)} | "
                f"RCS00/RCS01功率合并 | 角度分箱均值曲线 | 径向=RCS(dBsm){cal_note}"
            )
            self._log(
                f"圆周 Cluster Raw 已载入: {file_path} | 点数={len(orows)} | "
                f"RCS00/RCS01功率合并 | {angle_text}后按{ORBIT_RCS_ANGLE_BIN_DEG:g}°分箱算术平均{cal_note}"
            )
            self._draw_rcs()
            return

        try:
            curve = self._build_loaded_rcs_curve(file_path, display_name=None)
        except Exception as exc:
            QtWidgets.QMessageBox.warning(self, "载入失败", f"读取RCS数据失败: {exc}")
            self._log(f"RCS绘图载入失败: {exc}")
            return

        ref_summary = self._format_rcs_reference_summary()
        plot_dlg = RcsPlotConfigDialog(
            file_path,
            ref_summary,
            default_custom_name=curve.display_name,
            default_calibration_db=self._rcs_plot_calibration_db,
            enable_curve_csv_export=True,
            parent=self,
        )
        if plot_dlg.exec_() != QtWidgets.QDialog.Accepted:
            return
        custom_name, cal_db, export_curve_csv = plot_dlg.values()
        self._rcs_plot_calibration_db = cal_db
        if custom_name:
            curve = replace(curve, display_name=custom_name)

        # Base plot: show reference limits if user has selected a reference product.
        show_ref = bool(self._rcs_ref_class and self._rcs_ref_angle and self._rcs_ref_limits is not None)
        self._rcs_hide_reference_limits = not show_ref
        self._rcs_base_file_path = str(curve.file_path)
        self._rcs_curve_csv_source_paths = [str(curve.file_path)]
        self._loaded_rcs_curves = []
        self._orbit_polar_cached_series = None
        if MainWindow._cluster_csv_has_dri_cluster_header(Path(curve.file_path)):
            try:
                orows = self._orbit_plot_rows_from_cluster_raw_path(str(curve.file_path))
                if len(orows) >= 2:
                    self._orbit_polar_cached_series = self._build_orbit_polar_series_from_rows(orows)
            except Exception:
                self._orbit_polar_cached_series = None
        self.rcs_recorder.reset()
        self.rcs_recorder.segments = [list(seg) for seg in curve.segments]
        self.rcs_recorder._cur = []
        self.rcs_recorder._ended = True
        self.rcs_recorder.oid = None
        self._rcs_recording = False
        self._rcs_show_only_fitted = False
        self._rcs_orbit_samples = []
        self._rcs_active_segment_index = None
        self._rcs_active_path_name = None
        self._rcs_active_target_name = None
        self._rcs_fitted = curve.fitted
        self._rcs_target_name = curve.display_name or Path(curve.file_path).stem
        self._rcs_segment_run_labels = curve.segment_run_labels
        self._rcs_segment_display_labels = curve.segment_display_labels

        for i in range(self.rcs_plot_mode_combo.count()):
            if self.rcs_plot_mode_combo.itemData(i) == "distance":
                self.rcs_plot_mode_combo.blockSignals(True)
                self.rcs_plot_mode_combo.setCurrentIndex(i)
                self.rcs_plot_mode_combo.blockSignals(False)
                break
        self._rcs_plot_mode = "distance"
        self._ensure_rcs_axes("distance")

        cal_note = f" | 标定{cal_db:+.2f}dB" if abs(float(cal_db)) > 1e-9 else ""
        csv_note = ""
        if export_curve_csv:
            _filtered_paths, combined_path = self._try_export_rcs_curve_csvs(
                [str(curve.file_path)],
                cal_db,
            )
            if combined_path is not None:
                csv_note = f" | CSV={combined_path.name}"
        self.rcs_status_label.setText(
            f"RCS绘图: 已载入 {Path(curve.file_path).name} | 点数={curve.point_count}{cal_note}{csv_note}"
        )
        self._log(
            f"RCS绘图数据已载入(无参考上下限): 文件={curve.file_path} | 点数={curve.point_count}"
            f"{cal_note}{csv_note}"
        )
        self._draw_rcs()


    def _compose_rcs_raw_filename(
        self,
        trajectory_name: Optional[str] = None,
        target_name: Optional[str] = None,
        segment_index: Optional[int] = None,
        timestamp: Optional[float] = None,
    ) -> str:
        if getattr(self, "_radial_measurement_spec", None) is not None:
            tn = str(trajectory_name or self._get_current_path_name() or "").strip()
            if self._parse_radial_rcs_task_name(tn) is not None:
                return f"{self._radial_rcs_segment_file_stem(tn)}.csv"
        traj_token = self._safe_filename_token(trajectory_name or self._get_current_path_name())
        target_token = self._safe_filename_token(target_name or self._get_selected_rcs_target_name())
        seg_token = f"_seg{int(segment_index):02d}" if segment_index is not None else ""
        stamp = time.strftime("%Y%m%d_%H%M%S", time.localtime(float(timestamp or time.time())))
        return f"{traj_token}__{target_token}{seg_token}__{stamp}.csv"


    def _resolve_rcs_save_dir(self, trajectory_name: Optional[str]) -> Path:
        """RCS 落盘目录。

        星型/径向测量：一次执行共用一个会话文件夹；每段 CSV 文件名为「X度第N次测量」样式（见 _format_radial_rcs_segment_label）。
        """
        base = Path(self._rcs_save_dir)
        if getattr(self, "_radial_measurement_spec", None) is not None:
            session_dir = self._ensure_radial_rcs_session_dir()
            if session_dir is not None:
                return session_dir
        return base


    @staticmethod
    def _write_curvepoints_cluster_rcs_csv(
        file_path: Path,
        segments: List[List[CurvePoint]],
        *,
        data_file_display: str,
        include_seg_idx: bool,
        run_number: int = 1,
        calibration=None,
    ) -> None:
        """与 ars40x_cluster_logger / DRI Raw 元数据一致；可选 SegIdx 列用于多段合并。"""
        base_cols = build_columns()
        headers = list(base_cols) + (["SegIdx"] if include_seg_idx else [])
        n_slots = (len(base_cols) - 9) // 3
        cal_s = format_cluster_csv_calibration(calibration)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        with file_path.open("w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["Data Type", "Raw"])
            w.writerow(["Data File", str(data_file_display)])
            w.writerow(["Run Number", int(run_number)])
            w.writerow(["Calibration", cal_s])
            w.writerow([])
            w.writerow(headers)
            for seg_idx, seg in enumerate(segments):
                for p in seg:
                    dx = float(p.x_raw)
                    dy = float(p.y_raw)
                    r_geom = round(math.hypot(dx, dy), 4)
                    view_deg = round(math.degrees(math.atan2(dy, dx)), 4)
                    row: List[Any] = [
                        round(float(p.t), 4),
                        r_geom,
                        view_deg,
                        "NaN",
                        "NaN",
                        "NaN",
                        "NaN",
                        "NaN",
                        "NaN",
                    ]
                    for i in range(n_slots):
                        if i == 0:
                            row += [round(dx, 2), round(dy, 2), round(float(p.rcs_raw), 2)]
                        else:
                            row += ["NaN", "NaN", "NaN"]
                    if include_seg_idx:
                        row.append(int(seg_idx))
                    w.writerow(row)


    def _find_current_locked_radar_target(self) -> Optional[Any]:
        if self.tracked_target_id is None:
            return None
        tracked_oid = int(self.tracked_target_id)
        now_ts = time.time()
        for target in list(getattr(self, "_latest_radar_targets", []) or []):
            if self._extract_radar_target_oid(target) != tracked_oid:
                continue
            if self._is_radar_target_fresh(target, now_ts=now_ts):
                return target
        return None


    def _set_static_orbit_controls_recording(self, active: bool) -> None:
        if hasattr(self, "btn_static_orbit_start"):
            self.btn_static_orbit_start.setEnabled(not active)
        if hasattr(self, "btn_static_orbit_finish"):
            self.btn_static_orbit_finish.setEnabled(active)
        if hasattr(self, "edit_static_orbit_name"):
            self.edit_static_orbit_name.setEnabled(not active)
        if hasattr(self, "spin_static_orbit_radius"):
            self.spin_static_orbit_radius.setEnabled(not active)


    def _set_rcs_plot_mode_without_signal(self, mode: str) -> None:
        for i in range(self.rcs_plot_mode_combo.count()):
            if self.rcs_plot_mode_combo.itemData(i) == mode:
                self.rcs_plot_mode_combo.blockSignals(True)
                self.rcs_plot_mode_combo.setCurrentIndex(i)
                self.rcs_plot_mode_combo.blockSignals(False)
                break
        self._rcs_plot_mode = mode
        self._ensure_rcs_axes(mode)


    def _static_orbit_rows_from_points(self, points: List[CurvePoint]) -> List[Dict[str, float]]:
        return [{"t": float(p.t), "rcs_filt": float(p.rcs_filt)} for p in points]


    def _append_static_orbit_rcs_sample(self, target: Any) -> None:
        if not self._static_orbit_rcs_active:
            return
        oid = self._extract_radar_target_oid(target)
        if oid is None or int(oid) != int(self._static_orbit_rcs_oid or -1):
            return
        try:
            rcs_db = float(getattr(target, "rcs_db", getattr(target, "rcs", 0.0)))
        except (TypeError, ValueError):
            return
        if not math.isfinite(rcs_db):
            return
        t_rel = max(0.0, time.time() - float(self._static_orbit_rcs_start_ts))
        if self._static_orbit_rcs_points:
            last_t = float(self._static_orbit_rcs_points[-1].t)
            if t_rel <= last_t + 1e-3:
                return
        radius_m = float(self._static_orbit_rcs_radius_m)
        self._static_orbit_rcs_points.append(
            CurvePoint(
                t=t_rel,
                x_raw=radius_m,
                y_raw=0.0,
                r_raw=radius_m,
                x=radius_m,
                y=0.0,
                rcs_raw=rcs_db,
                rcs_filt=rcs_db,
            )
        )
        if len(self._static_orbit_rcs_points) > 20000:
            self._static_orbit_rcs_points = self._static_orbit_rcs_points[-20000:]
        rows = self._static_orbit_rows_from_points(self._static_orbit_rcs_points)
        self._orbit_polar_cached_series = self._build_orbit_polar_series_from_rows(rows)
        self.rcs_status_label.setText(
            f"静态圆周RCS: 采集中 | id={oid} RCS={rcs_db:.1f}dBsm "
            f"点={len(self._static_orbit_rcs_points)} 半径={radius_m:.2f}m"
        )
        self._draw_rcs(live=True)


    def _save_static_orbit_rcs_recording(self) -> str:
        points = list(self._static_orbit_rcs_points)
        if len(points) == 1:
            p0 = points[0]
            points.append(
                CurvePoint(
                    t=float(p0.t) + 0.02,
                    x_raw=float(p0.x_raw),
                    y_raw=float(p0.y_raw),
                    r_raw=float(p0.r_raw),
                    x=float(p0.x),
                    y=float(p0.y),
                    rcs_raw=float(p0.rcs_raw),
                    rcs_filt=float(p0.rcs_filt),
                )
            )
        if len(points) < 2:
            raise ValueError("有效采样点不足，至少需要 2 个点")

        name = str(self._static_orbit_rcs_name or "静态圆周").strip() or "静态圆周"
        trajectory_name = self._get_current_path_name()
        save_dir = self._resolve_rcs_save_dir(trajectory_name)
        save_dir.mkdir(parents=True, exist_ok=True)
        name_tok = self._safe_filename_token(name)
        radius_tok = self._safe_filename_token(f"R{float(self._static_orbit_rcs_radius_m):.2f}m")
        oid_tok = self._safe_filename_token(f"ID{self._static_orbit_rcs_oid}")
        stamp = time.strftime("%Y%m%d_%H%M%S", time.localtime(time.time()))
        file_path = self._unique_output_path(
            save_dir / f"{name_tok}_static_orbit_cluster_{radius_tok}_{oid_tok}_{stamp}.csv"
        )
        self._write_curvepoints_cluster_rcs_csv(
            file_path,
            [points],
            data_file_display=str(file_path),
            include_seg_idx=False,
        )

        rows = self._static_orbit_rows_from_points(points)
        self._orbit_polar_cached_series = self._build_orbit_polar_series_from_rows(rows)
        self._rcs_orbit_samples = []
        self._loaded_rcs_curves = []
        self._rcs_fitted = None
        self._rcs_recording = False
        self._rcs_show_only_fitted = False
        self._rcs_base_file_path = str(file_path)
        self._rcs_hide_reference_limits = True
        self._rcs_target_name = name
        self._rcs_segment_run_labels = None
        self._rcs_segment_display_labels = None
        self.rcs_recorder.reset()
        self.rcs_recorder._cur = []
        self.rcs_recorder._ended = True
        self.rcs_recorder.oid = self._static_orbit_rcs_oid
        self._static_orbit_rcs_points = points
        return str(file_path)


    def _on_static_orbit_rcs_clicked(self) -> None:
        self._show_tool_dialog(self._rcs_viewer_dialog)
        self._set_rcs_plot_mode_without_signal("orbit")
        if hasattr(self, "edit_static_orbit_name"):
            self.edit_static_orbit_name.setFocus()


    def _on_static_orbit_rcs_start_clicked(self) -> None:
        if self._rcs_recording:
            QtWidgets.QMessageBox.information(
                self,
                "RCS采集中",
                "请先等待当前 RCS 采集结束后再开始静态圆周测试。",
            )
            return
        if self._static_orbit_rcs_active:
            return
        target = self._find_current_locked_radar_target()
        if target is None:
            QtWidgets.QMessageBox.warning(
                self,
                "静态圆周测试",
                "请先在雷达目标检查窗口锁定一个新鲜目标，再开始静态圆周测试。",
            )
            return
        oid = self._extract_radar_target_oid(target)
        if oid is None:
            QtWidgets.QMessageBox.warning(self, "静态圆周测试", "当前锁定目标 ID 无效。")
            return
        name = str(self.edit_static_orbit_name.text() if hasattr(self, "edit_static_orbit_name") else "").strip()
        if not name:
            name = f"静态圆周_目标ID{int(oid)}"
            if hasattr(self, "edit_static_orbit_name"):
                self.edit_static_orbit_name.setText(name)
        radius_m = float(
            self.spin_static_orbit_radius.value()
            if hasattr(self, "spin_static_orbit_radius")
            else ORBIT_RCS_NOMINAL_FORWARD_M
        )
        self._static_orbit_rcs_active = True
        self._static_orbit_rcs_points = []
        self._static_orbit_rcs_start_ts = time.time()
        self._static_orbit_rcs_name = name
        self._static_orbit_rcs_radius_m = radius_m
        self._static_orbit_rcs_oid = int(oid)
        self._orbit_polar_cached_series = None
        self._rcs_orbit_samples = []
        self._loaded_rcs_curves = []
        self._rcs_fitted = None
        self._rcs_base_file_path = None
        self._rcs_hide_reference_limits = True
        self._rcs_target_name = name
        self.rcs_recorder.reset()
        self.rcs_recorder.oid = int(oid)
        self._set_rcs_plot_mode_without_signal("orbit")
        self._set_static_orbit_controls_recording(True)
        self._append_static_orbit_rcs_sample(target)
        self._log(
            f"静态圆周RCS开始: 名称={name} | 半径={radius_m:.2f}m | 目标ID={int(oid)}"
        )


    def _on_static_orbit_rcs_finish_clicked(self) -> None:
        if not self._static_orbit_rcs_active:
            return
        self._static_orbit_rcs_active = False
        try:
            file_path = self._save_static_orbit_rcs_recording()
        except Exception as exc:
            QtWidgets.QMessageBox.warning(self, "静态圆周测试", f"保存失败: {exc}")
            self._log(f"静态圆周RCS保存失败: {exc}")
            self._set_static_orbit_controls_recording(False)
            return

        self._set_static_orbit_controls_recording(False)
        self._set_rcs_plot_mode_without_signal("orbit")
        self.rcs_status_label.setText(
            f"静态圆周RCS: 已结束并保存 | 点={len(self._static_orbit_rcs_points)} | {file_path}"
        )
        self._log(
            f"静态圆周RCS已保存: 名称={self._static_orbit_rcs_name} | "
            f"半径={self._static_orbit_rcs_radius_m:.2f}m | "
            f"点={len(self._static_orbit_rcs_points)} | 文件={file_path}"
        )
        self._draw_rcs()


    def _save_rcs_raw_snapshot(
        self,
        trajectory_name: Optional[str] = None,
        target_name: Optional[str] = None,
        segment_index: Optional[int] = None,
    ) -> Optional[str]:
        if self.rcs_recorder.point_count() <= 0:
            return None
        save_dir = self._resolve_rcs_save_dir(trajectory_name)
        save_dir.mkdir(parents=True, exist_ok=True)
        file_path = save_dir / self._compose_rcs_raw_filename(
            trajectory_name=trajectory_name,
            target_name=target_name,
            segment_index=segment_index,
        )
        segs = [list(s) for s in self.rcs_recorder.segments if s]
        if not segs and self.rcs_recorder._cur:
            segs = [list(self.rcs_recorder._cur)]
        if not segs:
            return None
        self._write_curvepoints_cluster_rcs_csv(
            file_path,
            segs,
            data_file_display=str(file_path),
            include_seg_idx=len(segs) > 1,
        )
        self._write_rcs_fitted_companion_file(file_path, self.rcs_recorder)
        return str(file_path)


    @staticmethod
    def _write_rcs_fitted_companion_file(raw_file_path: Path, recorder: RcsRunRecorder) -> Optional[str]:
        """在原始落盘旁写入拟合曲线采样 `*_fitted.csv`。"""
        try:
            fit_path = raw_file_path.with_name(f"{raw_file_path.stem}_fitted.csv")
            segments = [list(seg) for seg in getattr(recorder, "segments", []) if seg]
            cur = list(getattr(recorder, "_cur", []) or [])
            if cur:
                segments.append(cur)
            fit = MainWindow._fitted_curve_from_segments(segments)
            with fit_path.open("w", newline="", encoding="utf-8") as f:
                w = csv.writer(f)
                w.writerow(["x_m", "rcs_fit_dBsm"])
                if fit is not None:
                    xs, ys = fit
                    for a, b in zip(np.asarray(xs, dtype=float).ravel(), np.asarray(ys, dtype=float).ravel()):
                        if np.isfinite(a) and np.isfinite(b):
                            w.writerow([f"{float(a):.4f}", f"{float(b):.4f}"])
            return str(fit_path)
        except Exception:
            return None


    def _reset_rcs_trajectory_file_cache(self) -> None:
        self._rcs_saved_trajectory_files.clear()


    def _save_rcs_trajectory_snapshot(
        self,
        trajectory_name: Optional[str],
        target_name: Optional[str],
        segment_points: List[CurvePoint],
    ) -> Optional[str]:
        if not segment_points:
            return None

        resolved_trajectory_name = str(trajectory_name or self._get_current_path_name()).strip() or "轨迹"
        resolved_target_name = str(target_name or self._get_selected_rcs_target_name()).strip() or "未命名目标"
        cache_key = f"{resolved_trajectory_name}\n{resolved_target_name}"
        record = self._rcs_saved_trajectory_files.get(cache_key)

        save_dir = self._resolve_rcs_save_dir(resolved_trajectory_name)
        save_dir.mkdir(parents=True, exist_ok=True)
        if record is None:
            file_path = save_dir / self._compose_rcs_raw_filename(
                trajectory_name=resolved_trajectory_name,
                target_name=resolved_target_name,
                segment_index=None,
            )
            if file_path.exists():
                stem = file_path.stem
                suffix = file_path.suffix or ".csv"
                counter = 2
                while True:
                    candidate = file_path.with_name(f"{stem}_{counter:02d}{suffix}")
                    if not candidate.exists():
                        file_path = candidate
                        break
                    counter += 1
            record = AggregatedRcsFile(
                trajectory_name=resolved_trajectory_name,
                target_name=resolved_target_name,
                file_path=str(file_path),
                segments=[],
            )
            self._rcs_saved_trajectory_files[cache_key] = record

        record.segments.append(list(segment_points))
        fp = Path(record.file_path)
        self._write_curvepoints_cluster_rcs_csv(
            fp,
            record.segments,
            data_file_display=str(fp),
            include_seg_idx=True,
        )
        tmp_rec = RcsRunRecorder()
        tmp_rec.segments = [list(s) for s in record.segments]
        tmp_rec._ended = True
        self._write_rcs_fitted_companion_file(fp, tmp_rec)
        return str(record.file_path)


    def _guess_rcs_plot_defaults(
        self,
        file_path: str,
    ) -> Tuple[Optional[str], Optional[str], Optional[str]]:
        stem = Path(file_path).stem
        parts = stem.split("__")
        guessed_name = ""
        if len(parts) >= 2:
            guessed_name = re.sub(r"_seg\d+$", "", parts[1], flags=re.IGNORECASE)
        elif parts:
            guessed_name = parts[0]
        guessed_name = guessed_name.replace("_", " ").strip()

        guessed_class = None
        guessed_key = self._normalized_name_key(guessed_name)
        if guessed_key:
            for name in self._rcs_ref_class_options:
                if self._normalized_name_key(name) == guessed_key:
                    guessed_class = name
                    break

        guessed_angle = (
            self._rcs_ref_angle
            if self._rcs_ref_angle in self._rcs_ref_angle_options
            else None
        )
        custom_name = guessed_name or guessed_class or self._rcs_target_name or ""
        return guessed_class, guessed_angle, custom_name or None


    @staticmethod
    def _common_nonempty_value(values: List[Optional[str]]) -> Optional[str]:
        filtered = [str(value).strip() for value in values if str(value or "").strip()]
        if not filtered:
            return None
        first = filtered[0]
        return first if all(value == first for value in filtered[1:]) else None


    @staticmethod
    def _ensure_unique_rcs_display_names(
        file_paths: List[str],
        display_names: List[Optional[str]],
    ) -> List[str]:
        result: List[str] = []
        counts: Dict[str, int] = {}
        for index, file_path in enumerate(file_paths):
            raw_name = display_names[index] if index < len(display_names) else None
            base_name = str(raw_name or "").strip() or Path(file_path).stem
            key = base_name.casefold()
            counts[key] = counts.get(key, 0) + 1
            if counts[key] > 1:
                base_name = f"{base_name} ({counts[key]})"
            result.append(base_name)
        return result


    @staticmethod
    def _cluster_rcs_csv_slot_dx_rcs(
        row: List[str],
        col: Dict[str, int],
        slot_index: int,
    ) -> Optional[Tuple[float, float]]:
        """读取 DX/RCS 槽位；RCS 或 DX 任一无效则跳过该槽。"""
        dxk = f"DX{slot_index:02d}"
        rck = f"RCS{slot_index:02d}"
        if dxk not in col or rck not in col:
            return None
        try:
            dx = MainWindow._cluster_csv_cell_float(row[col[dxk]])
            rcs = MainWindow._cluster_csv_cell_float(row[col[rck]])
        except (ValueError, KeyError, IndexError):
            return None
        if not math.isfinite(dx) or not math.isfinite(rcs):
            return None
        return float(dx), float(rcs)


    @staticmethod
    def _filtered_rcs_csv_path(raw_path: Path) -> Path:
        return raw_path.with_name(f"{raw_path.stem}_Filtered.csv")


    @staticmethod
    def _build_filtered_rcs_rows_from_cluster_raw_path(
        raw_path: Path,
        calibration_db: float,
    ) -> Tuple[List[List[str]], List[Tuple[float, float]], int]:
        """
        Filtered CSV 数据：
        每帧按 abs(DX[i]-R) 取最近两个有效槽，记录 (DX, RCS+calibration)，再按 0.1m X 分箱算术平均。
        """
        metadata_rows: List[List[str]] = []
        bins: Dict[int, List[float]] = defaultdict(list)
        selected_count = 0
        with raw_path.open(encoding="utf-8-sig", newline="") as f:
            reader = csv.reader(f)
            header_cells: Optional[List[str]] = None
            col: Dict[str, int] = {}
            for row in reader:
                if row and row[0].strip() == "Time" and any(
                    (c or "").strip() == "DX00" for c in row
                ):
                    header_cells = [(c or "").strip() for c in row]
                    col = {name: idx for idx, name in enumerate(header_cells)}
                    break
                metadata_rows.append(list(row))
            if not header_cells:
                raise ValueError("Cluster RCS CSV：未找到表头行（含 Time、DX00）")
            for k in ("R", "DX00", "RCS00"):
                if k not in col:
                    raise ValueError(f"Cluster RCS CSV 缺少列「{k}」")
            max_ix = max(col.values())

            for raw_row in reader:
                if not raw_row or not any(str(c).strip() for c in raw_row):
                    continue
                row = list(raw_row)
                while len(row) <= max_ix:
                    row.append("")
                try:
                    range_val = MainWindow._cluster_csv_cell_float(row[col["R"]])
                except (ValueError, KeyError, IndexError):
                    continue
                if not math.isfinite(range_val):
                    continue

                candidates: List[Tuple[float, int, float, float]] = []
                for slot in range(int(MAX_CLUSTERS)):
                    parsed = MainWindow._cluster_rcs_csv_slot_dx_rcs(row, col, slot)
                    if parsed is None:
                        continue
                    dx, rcs = parsed
                    candidates.append(
                        (abs(float(dx) - float(range_val)), int(slot), float(dx), float(rcs))
                    )
                if not candidates:
                    continue

                candidates.sort(key=lambda item: (float(item[0]), int(item[1])))
                for _dist_err, _slot, dx, rcs in candidates[:2]:
                    bin_id = int(round(float(dx) * 10.0))
                    bins[bin_id].append(float(rcs) + float(calibration_db))
                    selected_count += 1

        if not bins:
            raise ValueError("Cluster RCS CSV 中没有可生成 Filtered 的有效帧")

        rows: List[Tuple[float, float]] = []
        for bin_id in range(min(bins.keys()), max(bins.keys()) + 1):
            values = bins.get(int(bin_id), [])
            r_val = float(bin_id) / 10.0
            if values:
                rows.append((r_val, float(sum(values)) / float(len(values))))
            else:
                rows.append((r_val, float("nan")))
        return metadata_rows, rows, selected_count


    @staticmethod
    def _prepare_rcs_two_column_metadata(
        metadata_rows: List[List[str]],
        output_path: Path,
        calibration_db: float,
    ) -> List[List[str]]:
        rows = [list(r) for r in metadata_rows]
        saw_data_file = False
        saw_calibration = False
        for row in rows:
            if not row:
                continue
            key = str(row[0] or "").strip().casefold()
            if key == "data file":
                while len(row) < 2:
                    row.append("")
                row[1] = str(output_path)
                saw_data_file = True
            elif key == "calibration":
                while len(row) < 2:
                    row.append("")
                row[1] = format_cluster_csv_calibration(calibration_db)
                saw_calibration = True

        insert_at = next(
            (
                i
                for i, row in enumerate(rows)
                if not row or not any(str(c).strip() for c in row)
            ),
            len(rows),
        )
        if not saw_data_file:
            rows.insert(insert_at, ["Data File", str(output_path)])
            insert_at += 1
        if not saw_calibration:
            rows.insert(insert_at, ["Calibration", format_cluster_csv_calibration(calibration_db)])

        if rows and any(str(c).strip() for c in rows[-1]):
            rows.append([])
        elif not rows:
            rows.append([])
        return rows


    @staticmethod
    def _write_rcs_two_column_csv(
        output_path: Path,
        metadata_rows: List[List[str]],
        columns: Tuple[str, str],
        rows: List[Tuple[float, float]],
        *,
        calibration_db: float,
        include_nan_rcs: bool,
        x_decimals: int,
    ) -> None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        meta = MainWindow._prepare_rcs_two_column_metadata(
            metadata_rows, output_path, calibration_db
        )
        with output_path.open("w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            for row in meta:
                w.writerow(row)
            w.writerow(list(columns))
            for x_val, rcs_val in rows:
                if not math.isfinite(float(x_val)):
                    continue
                if not math.isfinite(float(rcs_val)):
                    if not include_nan_rcs:
                        continue
                    rcs_text = "NaN"
                else:
                    rcs_text = f"{float(rcs_val):.4f}"
                w.writerow([f"{float(x_val):.{int(x_decimals)}f}", rcs_text])


    @staticmethod
    def _build_combined_rcs_rows_from_filtered_sets(
        filtered_sets: List[List[Tuple[float, float]]],
    ) -> List[Tuple[float, float]]:
        by_bin: Dict[int, List[float]] = defaultdict(list)
        for rows in filtered_sets:
            for r_val, rcs_val in rows:
                if not math.isfinite(float(r_val)) or not math.isfinite(float(rcs_val)):
                    continue
                by_bin[int(round(float(r_val) * 10.0))].append(float(rcs_val))
        if not by_bin:
            return []

        xs: List[float] = []
        ys: List[float] = []
        for bin_id in sorted(by_bin.keys()):
            vals = by_bin[int(bin_id)]
            if not vals:
                continue
            xs.append(float(bin_id) / 10.0)
            ys.append(float(sum(vals)) / float(len(vals)))

        x_arr = np.asarray(xs, dtype=float)
        y_arr = np.asarray(ys, dtype=float)
        if y_arr.size and _peak_smooth_rcs_series is not None:
            try:
                y_arr = np.asarray(_peak_smooth_rcs_series(y_arr), dtype=float)
            except Exception:
                pass
        out: List[Tuple[float, float]] = []
        for x_val, y_val in zip(x_arr.tolist(), y_arr.tolist()):
            if math.isfinite(float(x_val)) and math.isfinite(float(y_val)):
                out.append((float(x_val), float(y_val)))
        return out


    @staticmethod
    def _combined_rcs_csv_path(raw_paths: List[Path]) -> Optional[Path]:
        paths = [Path(p) for p in raw_paths if str(p).strip()]
        if not paths:
            return None
        first = paths[0]
        if len(paths) == 1:
            return first.with_name(f"{first.stem}_combined.csv")
        source_key = "\n".join(
            str(p.resolve()) if p.exists() else str(p) for p in paths
        )
        digest = hashlib.sha1(source_key.encode("utf-8")).hexdigest()[:8]
        return first.with_name(f"{first.stem}_combined_{len(paths)}files_{digest}.csv")


    def _export_rcs_filtered_and_combined_csvs(
        self,
        file_paths: List[str],
        calibration_db: float,
        combined_rows_override: Optional[List[Tuple[float, float]]] = None,
    ) -> Tuple[List[Path], Optional[Path]]:
        filtered_paths: List[Path] = []
        filtered_sets: List[List[Tuple[float, float]]] = []
        combined_metadata: Optional[List[List[str]]] = None
        raw_paths: List[Path] = []

        for file_path in file_paths:
            raw_path = Path(file_path)
            raw_paths.append(raw_path)
            metadata_rows, filtered_rows, _selected_count = (
                MainWindow._build_filtered_rcs_rows_from_cluster_raw_path(
                    raw_path, calibration_db
                )
            )
            filtered_path = MainWindow._filtered_rcs_csv_path(raw_path)
            MainWindow._write_rcs_two_column_csv(
                filtered_path,
                metadata_rows,
                ("X", "RCS"),
                filtered_rows,
                calibration_db=calibration_db,
                include_nan_rcs=True,
                x_decimals=1,
            )
            filtered_paths.append(filtered_path)
            filtered_sets.append(filtered_rows)
            if combined_metadata is None:
                combined_metadata = metadata_rows

        combined_rows = (
            list(combined_rows_override)
            if combined_rows_override is not None
            else MainWindow._build_combined_rcs_rows_from_filtered_sets(filtered_sets)
        )
        combined_path: Optional[Path] = None
        if combined_metadata is not None:
            combined_path = MainWindow._combined_rcs_csv_path(raw_paths)
        if combined_path is not None:
            MainWindow._write_rcs_two_column_csv(
                combined_path,
                combined_metadata,
                ("X", "RCS"),
                combined_rows,
                calibration_db=calibration_db,
                include_nan_rcs=False,
                x_decimals=1,
            )
        return filtered_paths, combined_path


    def _try_export_rcs_curve_csvs(
        self,
        file_paths: List[str],
        calibration_db: float,
    ) -> Tuple[List[Path], Optional[Path]]:
        try:
            green_rows = self._current_distance_rcs_green_curve_rows()
            filtered_paths, combined_path = self._export_rcs_filtered_and_combined_csvs(
                file_paths,
                calibration_db,
                combined_rows_override=green_rows if green_rows else None,
            )
        except Exception as exc:
            self._log(f"RCS曲线CSV生成失败: {exc}")
            QtWidgets.QMessageBox.warning(self, "RCS曲线CSV生成失败", str(exc))
            return [], None
        filtered_preview = ", ".join(p.name for p in filtered_paths[:4])
        if len(filtered_paths) > 4:
            filtered_preview += ", ..."
        self._log(
            "RCS曲线CSV已生成: "
            f"Filtered={filtered_preview or '无'}"
            + (f" | combined={combined_path}" if combined_path else "")
        )
        return filtered_paths, combined_path


    @staticmethod
    def _rcs_segment_file_labels(file_path: str, segment_count: int) -> List[str]:
        name = Path(str(file_path)).stem or "RCS数据"
        count = max(0, int(segment_count))
        if count <= 0:
            return []
        if count == 1:
            return [name]
        return [f"{name} #{idx}" for idx in range(1, count + 1)]


    @staticmethod
    def _rcs_legend_kwargs(export: bool = False) -> Dict[str, Any]:
        return {
            "fontsize": 8 if export else 7,
            "handlelength": 2.4 if export else 2.0,
            "labelspacing": 0.35,
            "borderaxespad": 0.35,
        }


    def _build_loaded_rcs_curve(
        self,
        file_path: str,
        display_name: Optional[str],
    ) -> LoadedRcsCurve:
        segments, segment_run_labels = self._parse_saved_rcs_raw_file(file_path)
        recorder = RcsRunRecorder()
        recorder.segments = [list(seg) for seg in segments]
        recorder._cur = []
        recorder._ended = True
        recorder.oid = None

        fitted = self._fitted_curve_from_segments(segments)
        fitted = self._clip_curve_by_distance(*fitted)
        label = str(display_name or "").strip() or Path(file_path).stem
        point_count = sum(len(seg) for seg in segments)
        return LoadedRcsCurve(
            file_path=str(file_path),
            display_name=label,
            segments=segments,
            fitted=fitted,
            point_count=point_count,
            segment_run_labels=segment_run_labels,
            segment_display_labels=self._rcs_segment_file_labels(file_path, len(segments)),
        )


    def _build_merged_loaded_rcs_curve(
        self,
        file_paths: List[str],
        display_name: Optional[str],
    ) -> LoadedRcsCurve:
        paths = [str(p) for p in file_paths if str(p).strip()]
        if not paths:
            raise ValueError("未选择可合并的RCS数据文件")
        merged_segments: List[List[CurvePoint]] = []
        merged_run_labels: List[int] = []
        merged_file_labels: List[str] = []
        run_base = 1
        for p in paths:
            if MainWindow._is_orbit_rcs_polar_csv(Path(p)) or MainWindow._is_orbit_cluster_raw_csv(Path(p)):
                raise ValueError("合并拟合仅支持距离-RCS，不支持圆周-RCS 数据")
            segs, _ = self._parse_saved_rcs_raw_file(p)
            merged_segments.extend(segs)
            merged_run_labels.extend(list(range(run_base, run_base + len(segs))))
            merged_file_labels.extend(self._rcs_segment_file_labels(p, len(segs)))
            run_base += len(segs)
        recorder = RcsRunRecorder()
        recorder.segments = [list(seg) for seg in merged_segments]
        recorder._cur = []
        recorder._ended = True
        recorder.oid = None
        fitted = self._fitted_curve_from_segments(merged_segments)
        fitted = self._clip_curve_by_distance(*fitted)
        label = str(display_name or "").strip() or f"合并({len(paths)}个文件)"
        point_count = sum(len(seg) for seg in merged_segments)
        # 用第一个文件名作为展示基准（真实来源为多文件合并）
        pseudo_path = str(paths[0])
        return LoadedRcsCurve(
            file_path=pseudo_path,
            display_name=label,
            segments=merged_segments,
            fitted=fitted,
            point_count=point_count,
            segment_run_labels=merged_run_labels,
            segment_display_labels=merged_file_labels,
        )


    def _load_saved_rcs_for_compare(
        self,
        file_paths: List[str],
        ref_class: Optional[str],
        ref_angle: Optional[str],
        display_names: List[Optional[str]],
    ) -> None:
        resolved_paths = [str(path) for path in file_paths if str(path).strip()]
        if not resolved_paths:
            raise ValueError("未选择可载入的RCS数据文件")

        resolved_names = self._ensure_unique_rcs_display_names(resolved_paths, display_names)
        loaded_curves = [
            self._build_loaded_rcs_curve(path, resolved_names[index])
            for index, path in enumerate(resolved_paths)
        ]

        self.rcs_recorder.reset()
        self.rcs_recorder._cur = []
        self.rcs_recorder._ended = True
        self.rcs_recorder.oid = None
        self._rcs_recording = False
        self._rcs_show_only_fitted = False
        self._rcs_orbit_samples = []
        self._orbit_polar_cached_series = None
        self._rcs_active_segment_index = None
        self._rcs_active_path_name = None
        self._rcs_active_target_name = None
        self._loaded_rcs_curves = loaded_curves if len(loaded_curves) > 1 else []
        self._rcs_curve_csv_source_paths = list(resolved_paths)

        if len(loaded_curves) == 1:
            curve = loaded_curves[0]
            self.rcs_recorder.segments = [list(seg) for seg in curve.segments]
            self._rcs_fitted = curve.fitted
            self._rcs_target_name = curve.display_name or Path(curve.file_path).stem
            self._rcs_segment_run_labels = curve.segment_run_labels
            self._rcs_segment_display_labels = curve.segment_display_labels
        else:
            self._rcs_fitted = None
            compare_title = self._common_nonempty_value([curve.display_name for curve in loaded_curves])
            if compare_title:
                compare_title = f"{compare_title} 对比"
            else:
                compare_title = f"RCS对比({len(loaded_curves)}组)"
            self._rcs_target_name = compare_title
            self._rcs_segment_run_labels = None
            self._rcs_segment_display_labels = None

        self._apply_rcs_reference_selection(ref_class, ref_angle, log_change=False)

        total_points = sum(curve.point_count for curve in loaded_curves)
        if len(loaded_curves) == 1:
            self.rcs_status_label.setText(
                f"RCS绘图: 已载入 {Path(loaded_curves[0].file_path).name} | 点数={loaded_curves[0].point_count}"
            )
            self._log(
                f"RCS绘图数据已载入: 文件={loaded_curves[0].file_path} | 点数={loaded_curves[0].point_count}"
                f" | 产品={self._rcs_ref_class or '未选择'}"
                f" | 角度={self._rcs_ref_angle or '未选择'}"
                f" | 名称={self._rcs_target_name or 'RCS'}"
            )
        else:
            joined_names = " / ".join(curve.display_name for curve in loaded_curves[:6])
            if len(loaded_curves) > 6:
                joined_names += " / ..."
            self.rcs_status_label.setText(
                f"RCS绘图: 已载入 {len(loaded_curves)} 组数据对比 | 总点数={total_points}"
            )
            self._log(
                f"RCS对比数据已载入: 数量={len(loaded_curves)} | 总点数={total_points}"
                f" | 产品={self._rcs_ref_class or '未选择'}"
                f" | 角度={self._rcs_ref_angle or '未选择'}"
                f" | 曲线={joined_names}"
            )

        self._draw_rcs()


    @staticmethod
    def _cluster_csv_cell_float(cell: str) -> float:
        s = str(cell).strip()
        if not s or s.upper() == "NAN":
            return float("nan")
        return float(s)


    @staticmethod
    def _group_cluster_dicts_for_individual_targets(
        clusters: List[Any],
        merge_radius_m: float,
    ) -> List[Tuple[int, float, float, float]]:
        """
        将同一帧内多个簇合并为「目标物」散点：
        - CAN 声明相同 Cluster_ID 的簇必合并；
        - 否则若平面距离 ≤ merge_radius_m（DX/DY，m）则视为同一物体合并。
        几何：DX/DY 取平均；RCS：非相干功率叠加（与 combine_rcs_db_incoherent_sum 一致）。
        返回 [(oid, dx, dy, rcs_db), ...]，oid 优先取组内最小 Cluster_ID，否则取簇下标最小值。
        """
        n = len(clusters)
        if n <= 0:
            return []

        parent = list(range(n))

        def find(i: int) -> int:
            while parent[i] != i:
                parent[i] = parent[parent[i]]
                i = parent[i]
            return i

        def union(i: int, j: int) -> None:
            ri, rj = find(i), find(j)
            if ri != rj:
                parent[rj] = ri

        def getxy(i: int) -> Tuple[float, float]:
            c = clusters[i]
            return float(c["DX"]), float(c["DY"])

        def get_cid(i: int) -> Optional[int]:
            c = clusters[i]
            if isinstance(c, dict) and "ClusterID" in c:
                try:
                    return int(c["ClusterID"])
                except (TypeError, ValueError):
                    return None
            return None

        r_m = max(float(merge_radius_m), 1e-6)
        r2 = r_m * r_m
        for i in range(n):
            xi, yi = getxy(i)
            ci = get_cid(i)
            for j in range(i + 1, n):
                cj = get_cid(j)
                same_id = ci is not None and cj is not None and ci == cj
                xj, yj = getxy(j)
                close = (xi - xj) ** 2 + (yi - yj) ** 2 <= r2
                if same_id or close:
                    union(i, j)

        by_root: Dict[int, List[int]] = defaultdict(list)
        for i in range(n):
            by_root[find(i)].append(i)

        out: List[Tuple[int, float, float, float]] = []
        for _root, idxs in by_root.items():
            idxs_s = sorted(int(k) for k in idxs)
            dxs = [float(clusters[k]["DX"]) for k in idxs_s]
            dys = [float(clusters[k]["DY"]) for k in idxs_s]
            rcs_list = [float(clusters[k]["RCS"]) for k in idxs_s]
            dx_m = float(sum(dxs)) / float(len(dxs))
            dy_m = float(sum(dys)) / float(len(dys))
            if combine_rcs_db_incoherent_sum is not None:
                rcs_m = combine_rcs_db_incoherent_sum(rcs_list)
            else:
                rcs_m = float(
                    10.0
                    * math.log10(sum(10.0 ** (float(v) / 10.0) for v in rcs_list))
                )
            if rcs_m is None:
                rcs_m = rcs_list[0]
            cids_ok = [get_cid(k) for k in idxs_s]
            cids_f = [x for x in cids_ok if x is not None]
            oid = int(min(cids_f)) if cids_f else int(min(idxs_s))
            out.append((oid, dx_m, dy_m, float(rcs_m)))

        out.sort(key=lambda row: (row[0], row[1], row[2]))
        return out


    @staticmethod
    def _parse_cluster_rcs_csv_row_slots(
        row: List[str],
        col: Dict[str, int],
        slot_index: int,
    ) -> Optional[Tuple[float, float, float]]:
        """读取 DX/DY/RCS 槽位；三者均有限值时返回 (dx, dy, rcs_db)，否则 None。"""
        dxk = f"DX{slot_index:02d}"
        dyk = f"DY{slot_index:02d}"
        rck = f"RCS{slot_index:02d}"
        for k in (dxk, dyk, rck):
            if k not in col:
                return None
        try:
            dx = MainWindow._cluster_csv_cell_float(row[col[dxk]])
            dy = MainWindow._cluster_csv_cell_float(row[col[dyk]])
            rcs = MainWindow._cluster_csv_cell_float(row[col[rck]])
        except (ValueError, KeyError, IndexError):
            return None
        if not all(math.isfinite(v) for v in (dx, dy, rcs)):
            return None
        return (float(dx), float(dy), float(rcs))


    @staticmethod
    def _cluster_rcs_curve_point_from_slot(
        t_val: float,
        seg_key: int,
        x_raw: float,
        y_raw: float,
        rcs_val: float,
    ) -> Tuple[int, CurvePoint]:
        r_slant = float(math.hypot(x_raw, y_raw))
        return (
            int(seg_key),
            CurvePoint(
                t=float(t_val),
                x_raw=float(x_raw),
                y_raw=float(y_raw),
                r_raw=float(r_slant),
                x=float(x_raw),
                y=float(y_raw),
                rcs_raw=float(rcs_val),
                rcs_filt=float(rcs_val),
            ),
        )


    @staticmethod
    def _cluster_rcs_rowwise_rcs00_rcs01_linear_power(
        row: List[str],
        col: Dict[str, int],
        t_val: float,
        seg_key: int,
    ) -> List[Tuple[int, CurvePoint]]:
        """
        与 DRI `dri_pipeline_gui._rowwise_linear_power_rcs_db(merge_rcs01=True)` 一致：
        同采样行内 RCS_eff = 10*log10(10^(RCS00/10)+10^(RCS01/10))；几何用主簇 DX00/DY00。
        若仅单槽有效则退化为该槽一点。
        """
        s0 = MainWindow._parse_cluster_rcs_csv_row_slots(row, col, 0)
        s1 = MainWindow._parse_cluster_rcs_csv_row_slots(row, col, 1)
        out: List[Tuple[int, CurvePoint]] = []
        if s0 is not None and s1 is not None:
            x0, y0, r0 = float(s0[0]), float(s0[1]), float(s0[2])
            _x1, _y1, r1 = float(s1[0]), float(s1[1]), float(s1[2])
            if all(math.isfinite(v) for v in (x0, y0, r0, r1)):
                if combine_rcs_db_incoherent_sum is not None:
                    rcs_m = combine_rcs_db_incoherent_sum([r0, r1])
                else:
                    psum = 10.0 ** (r0 / 10.0) + 10.0 ** (r1 / 10.0)
                    rcs_m = float(10.0 * math.log10(max(psum, 1e-300)))
                if rcs_m is None or not math.isfinite(rcs_m):
                    rcs_m = r0
                out.append(
                    MainWindow._cluster_rcs_curve_point_from_slot(
                        t_val, seg_key, x0, y0, float(rcs_m)
                    )
                )
                return out
        if s0 is not None:
            x0, y0, r0 = float(s0[0]), float(s0[1]), float(s0[2])
            if all(math.isfinite(v) for v in (x0, y0, r0)):
                out.append(
                    MainWindow._cluster_rcs_curve_point_from_slot(t_val, seg_key, x0, y0, r0)
                )
                return out
        if s1 is not None:
            x1, y1, r1 = float(s1[0]), float(s1[1]), float(s1[2])
            if all(math.isfinite(v) for v in (x1, y1, r1)):
                out.append(
                    MainWindow._cluster_rcs_curve_point_from_slot(t_val, seg_key, x1, y1, r1)
                )
        return out


    @staticmethod
    def _cluster_rcs_point_with_t(p: CurvePoint, t_new: float) -> CurvePoint:
        return CurvePoint(
            t=float(t_new),
            x_raw=float(p.x_raw),
            y_raw=float(p.y_raw),
            r_raw=float(p.r_raw),
            x=float(p.x),
            y=float(p.y),
            rcs_raw=float(p.rcs_raw),
            rcs_filt=float(p.rcs_filt),
        )


    @staticmethod
    def _cluster_rcs_points_time_relative(points: List[CurvePoint]) -> List[CurvePoint]:
        """与 DRI Raw 一致：段内 Time 从 0 秒起（相对首点时间）。"""
        if not points:
            return points
        t0 = float(points[0].t)
        return [
            MainWindow._cluster_rcs_point_with_t(p, float(p.t) - t0) for p in points
        ]


    @staticmethod
    def _cluster_rcs_points_drop_spatial_spikes(
        points: List[CurvePoint],
        max_dist_step_m: float,
        time_gap_reset_s: float,
    ) -> List[CurvePoint]:
        """
        剔除相邻帧间首目标几何突变点（如 DX/DY 对应 x 方向巨跳）。
        时间间隔 ≥ time_gap_reset_s 时视为新一段，不与前一点比幅值。
        """
        if len(points) <= 1:
            return list(points)
        out: List[CurvePoint] = [points[0]]
        last = points[0]
        for p in points[1:]:
            dt = float(p.t) - float(last.t)
            if not math.isfinite(dt):
                continue
            # 同一时间戳的多槽位（RCS00/RCS01/…）几何可相距较远，不得按「帧间突变」剔除
            if abs(dt) < 1e-9:
                out.append(p)
                last = p
                continue
            if dt >= float(time_gap_reset_s):
                out.append(p)
                last = p
                continue
            dist = math.hypot(
                float(p.x_raw) - float(last.x_raw),
                float(p.y_raw) - float(last.y_raw),
            )
            if dist <= float(max_dist_step_m):
                out.append(p)
                last = p
        return out


    @staticmethod
    def _finalize_cluster_rcs_curve_points(
        points: List[CurvePoint],
    ) -> List[CurvePoint]:
        """排序 → 去空间突变 → 相对时间（DRI 风格）。"""
        if not points:
            return points
        pts = sorted(points, key=lambda p: (float(p.t), float(p.x_raw)))
        pts = MainWindow._cluster_rcs_points_drop_spatial_spikes(
            pts,
            CLUSTER_RCS_PARSE_MAX_DIST_STEP_M,
            CLUSTER_RCS_PARSE_TIME_GAP_RESET_S,
        )
        return MainWindow._cluster_rcs_points_time_relative(pts)


    @staticmethod
    def _parse_cluster_rcs_csv(
        path: Path,
        *,
        merge_rcs00_rcs01_rowwise: bool = True,
        orbit_row_primary_only: bool = False,
    ) -> Tuple[List[List[CurvePoint]], List[int]]:
        """Cluster 0x701 落盘 CSV：每帧一行多槽位 DXii/DYii/RCSii。
        - merge_rcs00_rcs01_rowwise=True（默认）：与 DRI `dri_pipeline_gui._rowwise_linear_power_rcs_db`
          一致，同采样行 RCS00+RCS01 线性功率合成后只生成一点（几何取 DX00/DY00）；槽位 02 起仍逐点展开。
        - orbit_row_primary_only=True：仅保留每帧上述行内合并点（忽略 02+ 槽），供圆周-RCS 绘图。
        - merge_rcs00_rcs01_rowwise=False：每槽位各一点（旧行为）。
        整理：剔除首目标平面位置突变帧，Time 归一为段内相对秒。"""
        tagged: List[Tuple[int, CurvePoint]] = []
        with path.open(encoding="utf-8-sig", newline="") as f:
            reader = csv.reader(f)
            header_cells: Optional[List[str]] = None
            col: Dict[str, int] = {}
            for row in reader:
                if not row:
                    continue
                if row[0].strip() == "Time" and any((c or "").strip() == "DX00" for c in row):
                    header_cells = [(c or "").strip() for c in row]
                    col = {name: idx for idx, name in enumerate(header_cells)}
                    break
            if not header_cells:
                raise ValueError("Cluster RCS CSV：未找到表头行（含 Time、DX00）")
            req = ("Time", "DX00", "DY00", "RCS00")
            for k in req:
                if k not in col:
                    raise ValueError(f"Cluster RCS CSV 缺少列「{k}」")
            max_ix = max(col.values())
            use_seg = "SegIdx" in col

            for raw_row in reader:
                if not raw_row or not any(str(c).strip() for c in raw_row):
                    continue
                row = list(raw_row)
                while len(row) <= max_ix:
                    row.append("")
                try:
                    t_val = MainWindow._cluster_csv_cell_float(row[col["Time"]])
                except (ValueError, KeyError, IndexError):
                    continue
                if not math.isfinite(t_val):
                    continue

                seg_key = 0
                if use_seg:
                    try:
                        seg_key = int(round(float(MainWindow._cluster_csv_cell_float(row[col["SegIdx"]]))))
                    except (ValueError, OverflowError):
                        seg_key = 0

                row_any = False
                if merge_rcs00_rcs01_rowwise:
                    merged01 = MainWindow._cluster_rcs_rowwise_rcs00_rcs01_linear_power(
                        row, col, t_val, seg_key
                    )
                    for item in merged01:
                        tagged.append(item)
                        row_any = True
                    if not orbit_row_primary_only:
                        for slot in range(2, int(MAX_CLUSTERS)):
                            s = MainWindow._parse_cluster_rcs_csv_row_slots(row, col, slot)
                            if s is None:
                                continue
                            x_raw, y_raw, rcs_val = float(s[0]), float(s[1]), float(s[2])
                            if not all(math.isfinite(v) for v in (x_raw, y_raw, rcs_val)):
                                continue
                            row_any = True
                            tagged.append(
                                MainWindow._cluster_rcs_curve_point_from_slot(
                                    t_val, seg_key, x_raw, y_raw, rcs_val
                                )
                            )
                else:
                    for slot in range(int(MAX_CLUSTERS)):
                        s = MainWindow._parse_cluster_rcs_csv_row_slots(row, col, slot)
                        if s is None:
                            continue
                        x_raw, y_raw, rcs_val = float(s[0]), float(s[1]), float(s[2])
                        if not all(math.isfinite(v) for v in (x_raw, y_raw, rcs_val)):
                            continue
                        row_any = True
                        tagged.append(
                            MainWindow._cluster_rcs_curve_point_from_slot(
                                t_val, seg_key, x_raw, y_raw, rcs_val
                            )
                        )
                if not row_any:
                    continue

        if not tagged:
            raise ValueError(
                "Cluster RCS CSV 中没有可用的簇点（至少需要某一槽位 DXii/DYii/RCSii 均为有限值）"
            )

        if use_seg:
            by_seg: Dict[int, List[CurvePoint]] = defaultdict(list)
            for sk, pt in tagged:
                by_seg[int(sk)].append(pt)
            sorted_keys = sorted(by_seg.keys())
            segments: List[List[CurvePoint]] = []
            segment_run_labels: List[int] = []
            for run_i, sk in enumerate(sorted_keys, start=1):
                pts = sorted(by_seg[sk], key=lambda p: (float(p.t), float(p.x_raw)))
                if pts:
                    segments.append(MainWindow._finalize_cluster_rcs_curve_points(pts))
                    segment_run_labels.append(run_i)
            if not segments:
                raise ValueError("Cluster RCS CSV 中没有有效的分段数据")
            return segments, segment_run_labels

        points = [pt for _, pt in tagged]
        points.sort(key=lambda p: (float(p.t), float(p.x_raw)))
        return [MainWindow._finalize_cluster_rcs_curve_points(points)], [1]


    def _parse_saved_rcs_raw_file(self, file_path: str) -> Tuple[List[List[CurvePoint]], List[int]]:
        path = Path(file_path)
        if path.suffix.lower() != ".csv":
            raise ValueError("仅支持载入 .csv 格式的 Cluster RCS 数据")
        return MainWindow._parse_cluster_rcs_csv(path)


    def _orbit_plot_rows_from_cluster_raw_path(self, file_path: str) -> List[Dict[str, float]]:
        """行内 RCS00+RCS01 线性功率合并后每帧一点；若 Raw 含 Heading，则用于圆周极角。"""
        path = Path(file_path)
        rows: List[Dict[str, float]] = []
        with path.open(encoding="utf-8-sig", newline="") as f:
            reader = csv.reader(f)
            header_cells: Optional[List[str]] = None
            col: Dict[str, int] = {}
            for row in reader:
                if not row:
                    continue
                if row[0].strip() == "Time" and any((c or "").strip() == "DX00" for c in row):
                    header_cells = [(c or "").strip() for c in row]
                    col = {name: idx for idx, name in enumerate(header_cells)}
                    break
            if not header_cells:
                raise ValueError("Cluster RCS CSV：未找到表头行（含 Time、DX00）")
            for k in ("Time", "DX00", "DY00", "RCS00"):
                if k not in col:
                    raise ValueError(f"Cluster RCS CSV 缺少列「{k}」")
            max_ix = max(col.values())
            use_seg = "SegIdx" in col
            for raw_row in reader:
                if not raw_row or not any(str(c).strip() for c in raw_row):
                    continue
                row = list(raw_row)
                while len(row) <= max_ix:
                    row.append("")
                try:
                    t_val = MainWindow._cluster_csv_cell_float(row[col["Time"]])
                except (ValueError, KeyError, IndexError):
                    continue
                if not math.isfinite(t_val):
                    continue
                seg_key = 0
                if use_seg:
                    try:
                        seg_key = int(round(float(MainWindow._cluster_csv_cell_float(row[col["SegIdx"]]))))
                    except (ValueError, OverflowError):
                        seg_key = 0
                merged = MainWindow._cluster_rcs_rowwise_rcs00_rcs01_linear_power(
                    row, col, t_val, seg_key
                )
                if not merged:
                    continue
                _seg, p = merged[0]
                out = {"t": float(p.t), "rcs_filt": float(p.rcs_filt)}
                if "Heading" in col:
                    try:
                        h = MainWindow._cluster_csv_cell_float(row[col["Heading"]])
                    except (ValueError, KeyError, IndexError):
                        h = float("nan")
                    if math.isfinite(h):
                        out["heading_deg"] = float(h)
                if "circle_clockwise" in col:
                    try:
                        cw = MainWindow._cluster_csv_cell_float(row[col["circle_clockwise"]])
                    except (ValueError, KeyError, IndexError):
                        cw = float("nan")
                    if math.isfinite(cw):
                        out["orbit_clockwise"] = float(cw)
                rows.append(out)
        rows.sort(key=lambda r: float(r["t"]))
        if rows:
            t0 = float(rows[0]["t"])
            for r in rows:
                r["t"] = float(r["t"]) - t0
        return rows


    def _on_plot_saved_rcs_clicked(self) -> None:
        # Repurposed per request: "导入数据对比"
        # Based on the currently plotted base file, pick another file and overlay fitted curve(s).
        self._show_tool_dialog(self._rcs_viewer_dialog)
        if self._rcs_recording:
            QtWidgets.QMessageBox.information(
                self,
                "Cluster RCS 采集中",
                "请先等待当前 Cluster RCS 采集结束后再进行数据对比。",
            )
            return
        if not self._rcs_base_file_path:
            QtWidgets.QMessageBox.information(
                self,
                "尚未选择基础数据",
                "请先点击“选择数据绘RCS图”，载入一组基础数据后再导入对比数据。",
            )
            self._log("RCS对比已阻止: 尚未选择基础绘图数据文件")
            return

        new_path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self,
            "选择对比RCS数据（将与当前曲线叠加对比）",
            str(self._rcs_save_dir),
            "CSV (*.csv);;所有文件(*)",
        )
        if not new_path:
            return

        base_path = str(self._rcs_base_file_path)
        file_paths = [base_path, str(new_path)]
        default_names = [
            str(self._rcs_target_name or Path(base_path).stem).strip() or Path(base_path).stem,
            Path(new_path).stem,
        ]
        reference_summary = self._format_rcs_reference_summary()
        dialog = RcsCompareConfigDialog(
            file_paths=file_paths,
            reference_summary=reference_summary,
            default_display_names=default_names,
            default_calibration_db=self._rcs_plot_calibration_db,
            parent=self,
        )
        if dialog.exec_() != QtWidgets.QDialog.Accepted:
            self._log("RCS对比已取消")
            return
        display_names, cal_db = dialog.values()
        self._set_rcs_plot_calibration_db(cal_db)

        # Compare view: show reference limits only when a reference is selected and limits are available.
        show_ref = bool(self._rcs_ref_class and self._rcs_ref_angle and self._rcs_ref_limits is not None)
        self._rcs_hide_reference_limits = not show_ref
        try:
            self._load_saved_rcs_for_compare(
                file_paths,
                self._rcs_ref_class if show_ref else None,
                self._rcs_ref_angle if show_ref else None,
                display_names,
            )
        except Exception as exc:
            self._log(f"RCS对比失败: {exc}")
            QtWidgets.QMessageBox.warning(self, "RCS对比失败", f"载入RCS数据失败:\n{exc}")


    def _load_rcs_reference_options(self) -> None:
        self._rcs_ref_class_options = []
        self._rcs_ref_angle_options = []
        self._rcs_ref_labels = {}
        if _rcs_ref_mod is None:
            return
        try:
            if hasattr(_rcs_ref_mod, "get_rcs_reference_options"):
                classes, angles = _rcs_ref_mod.get_rcs_reference_options()
                self._rcs_ref_class_options = list(classes) if classes else []
                self._rcs_ref_angle_options = list(angles) if angles else []
            if hasattr(_rcs_ref_mod, "get_rcs_reference_labels"):
                labels = _rcs_ref_mod.get_rcs_reference_labels()
                if isinstance(labels, dict):
                    self._rcs_ref_labels = labels
        except Exception as exc:
            print(f"[main_ui] Failed to read rcs reference options: {exc}")


    def _has_rcs_reference_options(self) -> bool:
        return bool(self._rcs_ref_class_options) and bool(self._rcs_ref_angle_options)


    def _format_rcs_reference_summary(
        self,
        ref_class: Optional[str] = None,
        ref_angle: Optional[str] = None,
    ) -> str:
        if not self._has_rcs_reference_options():
            return "参考库未加载"
        resolved_class = str(ref_class or self._rcs_ref_class or "").strip()
        resolved_angle = str(ref_angle or self._rcs_ref_angle or "").strip()
        if not resolved_class or not resolved_angle:
            return "未选择"
        display_class = str(self._rcs_ref_labels.get(resolved_class, resolved_class)).strip()
        if display_class and display_class != resolved_class:
            return f"{display_class} ({resolved_class}) / {resolved_angle}"
        return f"{resolved_class} / {resolved_angle}"


    def _update_rcs_reference_summary_label(self) -> None:
        summary = self._format_rcs_reference_summary()
        if hasattr(self, "label_rcs_reference_summary") and self.label_rcs_reference_summary is not None:
            self.label_rcs_reference_summary.setText(f"RCS参考: {summary}")
        if hasattr(self, "btn_select_rcs_reference") and self.btn_select_rcs_reference is not None:
            self.btn_select_rcs_reference.setEnabled(self._has_rcs_reference_options())
        if hasattr(self, "btn_plot_saved_rcs") and self.btn_plot_saved_rcs is not None:
            self.btn_plot_saved_rcs.setEnabled(True)
            if self._has_rcs_reference_options():
                tooltip = (
                    f"导入数据对比 | 当前参考: {summary}"
                    if (self._rcs_ref_class and self._rcs_ref_angle)
                    else "导入数据对比：未选择参考产品（将不叠加参考上下限）"
                )
                self.btn_plot_saved_rcs.setToolTip(tooltip)
            else:
                self.btn_plot_saved_rcs.setToolTip("导入数据对比：参考库未加载（将不叠加参考上下限）")


    def _apply_rcs_reference_selection(
        self,
        ref_class: Optional[str],
        ref_angle: Optional[str],
        log_change: bool = True,
    ) -> None:
        resolved_class = str(ref_class or "").strip() or None
        resolved_angle = str(ref_angle or "").strip() or None
        if resolved_class is None:
            resolved_angle = None
        self._rcs_ref_class = resolved_class
        self._rcs_ref_angle = resolved_angle
        self._update_rcs_reference_summary_label()
        self._update_rcs_reference_limits()
        # 一旦用户选择了参考产品，就应立即显示上下限（除非确实没有可用 limits）。
        if self._rcs_ref_class is not None and self._rcs_ref_angle is not None and self._rcs_ref_limits is not None:
            self._rcs_hide_reference_limits = False
            self._draw_rcs()
        if not log_change:
            return
        if self._rcs_ref_class is not None and self._rcs_ref_angle is not None:
            self._log(f"RCS参考已选择: {self._format_rcs_reference_summary()}")
        else:
            self._log("RCS参考已清除")


    def _on_select_rcs_reference(self) -> None:
        if not self._has_rcs_reference_options():
            QtWidgets.QMessageBox.information(
                self,
                "参考库未加载",
                "当前没有可用的 RCS 参考产品和角度配置。",
            )
            self._log("选择RCS参考产品失败: 参考库未加载")
            return
        dialog = RcsReferenceConfigDialog(
            class_options=self._rcs_ref_class_options,
            angle_options=self._rcs_ref_angle_options,
            class_labels=self._rcs_ref_labels,
            default_class=self._rcs_ref_class,
            default_angle=self._rcs_ref_angle,
            parent=self,
        )
        if dialog.exec_() != QtWidgets.QDialog.Accepted:
            return
        ref_class, ref_angle = dialog.values()
        self._apply_rcs_reference_selection(ref_class, ref_angle)


    def _update_rcs_reference_limits(self) -> None:
        if self._rcs_ref_class is None or self._rcs_ref_angle is None:
            self._rcs_ref_limits = None
            self._draw_rcs()
            return
        if _rcs_ref_mod is None or not hasattr(_rcs_ref_mod, "get_rcs_reference_limits"):
            self._rcs_ref_limits = None
            self._draw_rcs()
            return
        try:
            limits = _rcs_ref_mod.get_rcs_reference_limits(self._rcs_ref_class, self._rcs_ref_angle)
        except Exception as exc:
            self._log(f"参考边界读取失败: {exc}")
            self._rcs_ref_limits = None
            self._draw_rcs()
            return
        self._rcs_ref_limits = limits
        if limits is None:
            self._log(f"未找到参考边界: {self._rcs_ref_class} {self._rcs_ref_angle}")
        self._draw_rcs()


    def _apply_rcs_plot_calibration_to_y(self, ys: np.ndarray) -> np.ndarray:
        c = float(getattr(self, "_rcs_plot_calibration_db", 0.0) or 0.0)
        y = np.asarray(ys, dtype=float)
        if c == 0.0:
            return y
        return y + c


    @staticmethod
    def _smooth_rcs_curve_y(ys: np.ndarray) -> np.ndarray:
        y = np.asarray(ys, dtype=float)
        if y.size <= 1:
            return y.copy()
        if _peak_smooth_rcs_series is not None:
            return np.asarray(_peak_smooth_rcs_series(y), dtype=float)

        a = float(RCS_STRAIGHT_EMA_ALPHA)
        if not np.isfinite(a) or a <= 0.0 or a >= 1.0:
            return y.copy()
        out = np.empty_like(y, dtype=float)
        out[0] = float(y[0])
        for i in range(1, int(y.size)):
            out[i] = a * float(y[i]) + (1.0 - a) * float(out[i - 1])
        return out


    @staticmethod
    def _single_measurement_curve_from_points(
        points: List[CurvePoint],
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        单次测量曲线：使用已按行内 RCS00/RCS01 功率叠加后的点，按距离分箱后做 SG 平滑。
        """
        if not points:
            return np.asarray([], dtype=float), np.asarray([], dtype=float)
        xs = np.asarray([float(p.x) for p in points], dtype=float)
        ys = np.asarray([float(p.rcs_filt) for p in points], dtype=float)
        ts = np.asarray([float(p.t) for p in points], dtype=float)
        mask = np.isfinite(xs) & np.isfinite(ys) & np.isfinite(ts)
        mask &= (xs >= float(RCS_STRAIGHT_X_MIN_M)) & (xs <= float(RCS_STRAIGHT_X_MAX_M))
        xs = xs[mask]
        ys = ys[mask]
        ts = ts[mask]
        if xs.size == 0:
            return np.asarray([], dtype=float), np.asarray([], dtype=float)

        order0 = np.argsort(xs)
        xs = xs[order0]
        ys = ys[order0]
        ts = ts[order0]
        if xs.size >= 2:
            keep = np.ones(xs.size, dtype=bool)
            last_idx = 0
            for j in range(1, xs.size):
                if not keep[last_idx]:
                    last_idx = j
                    continue
                if abs(float(ys[j]) - float(ys[last_idx])) > 10.0:
                    keep[j] = False
                else:
                    last_idx = j
            xs = xs[keep]
            ys = ys[keep]
            ts = ts[keep]
            if xs.size == 0:
                return np.asarray([], dtype=float), np.asarray([], dtype=float)

        bin_m = max(float(RCS_FIT_GRID_STEP_M), 1e-4)
        bin_ids = np.floor(xs / bin_m).astype(np.int64)
        xb: List[float] = []
        yb: List[float] = []
        t_gate = float(RCS_BIN_INCOHERENT_SUM_MAX_TIME_SPAN_S)
        for bid in np.unique(bin_ids):
            m = bin_ids == bid
            if not np.any(m):
                continue
            xb.append(float(np.mean(xs[m])))
            ys_b = ys[m]
            ts_b = ts[m]
            if ys_b.size <= 1:
                yb.append(float(ys_b[0]))
            elif float(np.max(ts_b) - np.min(ts_b)) <= t_gate:
                if combine_rcs_db_incoherent_sum is not None:
                    cr = combine_rcs_db_incoherent_sum(ys_b.tolist())
                    yb.append(float(cr) if cr is not None else float(np.mean(ys_b)))
                else:
                    p_sum = float(np.sum(10.0 ** (ys_b.astype(float) / 10.0)))
                    yb.append(float(10.0 * math.log10(max(p_sum, 1e-300))))
            else:
                yb.append(float(np.mean(ys_b)))

        x_out = np.asarray(xb, dtype=float)
        y_out = np.asarray(yb, dtype=float)
        if x_out.size == 0:
            return x_out, y_out
        order = np.argsort(x_out)
        x_out = x_out[order]
        y_out = MainWindow._smooth_rcs_curve_y(y_out[order])
        return x_out, y_out


    @staticmethod
    def _fuse_measurement_curves(
        curves: List[Tuple[np.ndarray, np.ndarray]],
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        多次测量最终曲线：每次先形成单条曲线，再按距离分箱剔除跨次异常点并 SG 平滑。
        """
        valid: List[Tuple[np.ndarray, np.ndarray]] = []
        for xs, ys in curves:
            x = np.asarray(xs, dtype=float)
            y = np.asarray(ys, dtype=float)
            n = min(int(x.size), int(y.size))
            if n <= 0:
                continue
            x = x[:n]
            y = y[:n]
            m = np.isfinite(x) & np.isfinite(y)
            if np.any(m):
                valid.append((x[m], y[m]))
        if not valid:
            return np.asarray([], dtype=float), np.asarray([], dtype=float)
        if len(valid) == 1:
            return valid[0][0].copy(), valid[0][1].copy()

        bin_m = max(float(RCS_FIT_GRID_STEP_M), 1e-4)
        thr_db = 10.0
        bin_to_xs: Dict[int, List[float]] = {}
        bin_to_ys: Dict[int, List[float]] = {}
        for xb, yb in valid:
            bids = np.floor(xb / bin_m).astype(np.int64)
            for bid, xk, yk in zip(bids.tolist(), xb.tolist(), yb.tolist()):
                bin_to_xs.setdefault(int(bid), []).append(float(xk))
                bin_to_ys.setdefault(int(bid), []).append(float(yk))

        x_all: List[float] = []
        y_all: List[float] = []
        for bid in sorted(bin_to_ys.keys()):
            ys_list = bin_to_ys.get(int(bid), [])
            xs_list = bin_to_xs.get(int(bid), [])
            if not ys_list or not xs_list:
                continue
            ys_arr = np.asarray(ys_list, dtype=float)
            xs_arr = np.asarray(xs_list, dtype=float)
            med = float(np.median(ys_arr))
            keep = np.abs(ys_arr - med) <= thr_db
            if not np.any(keep):
                continue
            x_all.append(float(np.mean(xs_arr[keep])))
            y_all.append(float(np.mean(ys_arr[keep])))

        x_out = np.asarray(x_all, dtype=float)
        y_out = np.asarray(y_all, dtype=float)
        if x_out.size == 0:
            return x_out, y_out
        order = np.argsort(x_out)
        x_out = x_out[order]
        y_out = MainWindow._smooth_rcs_curve_y(y_out[order])
        return x_out, y_out


    @staticmethod
    def _fitted_curve_from_segments(
        segments: List[List[CurvePoint]],
    ) -> Tuple[np.ndarray, np.ndarray]:
        curves = [
            MainWindow._single_measurement_curve_from_points(list(seg))
            for seg in segments
            if seg
        ]
        return MainWindow._fuse_measurement_curves(curves)


    @staticmethod
    def _clip_curve_by_distance(
        xs: np.ndarray,
        ys: np.ndarray,
        x_min: float = 0.0,
        x_max: float = RCS_STRAIGHT_X_MAX_M,
    ) -> Tuple[np.ndarray, np.ndarray]:
        x = np.asarray(xs, dtype=float)
        y = np.asarray(ys, dtype=float)
        if x.size == 0 or y.size == 0:
            return x, y
        n = min(x.size, y.size)
        x = x[:n]
        y = y[:n]
        mask = np.isfinite(x) & (x >= float(x_min)) & (x <= float(x_max))
        return x[mask], y[mask]


    def _plot_rcs_reference_limits(
        self,
        ax,
        *,
        show_labels: bool = True,
        linewidth: float = 2.0,
        alpha: float = 1.0,
    ) -> None:
        if bool(getattr(self, "_rcs_hide_reference_limits", False)):
            return
        limits = self._rcs_ref_limits
        if limits is None:
            return

        xs = limits.get("x")
        if xs is None:
            return
        xs = np.asarray(xs, dtype=float)
        mask = np.isfinite(xs) & (xs >= 0.0) & (xs <= RCS_MAX_DISTANCE_M)
        if not np.any(mask):
            return

        lower = limits.get("lower")
        upper = limits.get("upper")
        ref = limits.get("ref")
        ref_color = "#000000"
        if lower is not None:
            lower = np.asarray(lower, dtype=float)
            if lower.shape == xs.shape:
                ax.plot(
                    xs[mask],
                    lower[mask],
                    color=ref_color,
                    linewidth=linewidth,
                    alpha=alpha,
                    solid_capstyle="round",
                    solid_joinstyle="round",
                    antialiased=True,
                    label="下界" if show_labels else None,
                )
        if upper is not None:
            upper = np.asarray(upper, dtype=float)
            if upper.shape == xs.shape:
                ax.plot(
                    xs[mask],
                    upper[mask],
                    color=ref_color,
                    linewidth=linewidth,
                    alpha=alpha,
                    solid_capstyle="round",
                    solid_joinstyle="round",
                    antialiased=True,
                    label="上界" if show_labels else None,
                )
        if ref is not None:
            ref = np.asarray(ref, dtype=float)
            if ref.shape == xs.shape:
                ax.plot(
                    xs[mask],
                    ref[mask],
                    color=ref_color,
                    linewidth=linewidth,
                    alpha=alpha,
                    solid_capstyle="round",
                    solid_joinstyle="round",
                    antialiased=True,
                    label="参考值" if show_labels else None,
                )
        return

        if self._rcs_ref_limits is None:
            return

        xs = self._rcs_ref_limits.get("x")
        if xs is None:
            return
        xs = np.asarray(xs, dtype=float)
        mask = np.isfinite(xs) & (xs >= 0.0) & (xs <= RCS_MAX_DISTANCE_M)
        if not np.any(mask):
            return

        lower = self._rcs_ref_limits.get("lower")
        upper = self._rcs_ref_limits.get("upper")
        ref = self._rcs_ref_limits.get("ref")
        if lower is not None:
            lower = np.asarray(lower, dtype=float)
            if lower.shape == xs.shape:
                ax.plot(xs[mask], lower[mask], color="k", linewidth=2, label="下界")
        if upper is not None:
            upper = np.asarray(upper, dtype=float)
            if upper.shape == xs.shape:
                ax.plot(xs[mask], upper[mask], color="k", linewidth=2, label="上界")
        if ref is not None:
            ref = np.asarray(ref, dtype=float)
            if ref.shape == xs.shape:
                ax.plot(xs[mask], ref[mask], color="b", linewidth=2, label="参考值")


    @staticmethod
    def _thin_rcs_tick_labels(
        positions: np.ndarray,
        values: np.ndarray,
        max_ticks: int,
    ) -> Tuple[np.ndarray, np.ndarray]:
        if positions.size <= max_ticks or max_ticks <= 0:
            return positions, values
        indices = np.linspace(0, positions.size - 1, max_ticks, dtype=int)
        indices = np.unique(indices)
        return positions[indices], values[indices]


    def _get_rcs_plot_display_title(self, plot_mode: str, compact: bool = False) -> str:
        if compact:
            if plot_mode == "orbit":
                return "Orbit RCS"
            return "RCS Compare" if self._loaded_rcs_curves else "RCS"
        if plot_mode == "orbit":
            return f"{self._get_rcs_title()} | Orbit RCS"
        return self._get_rcs_title()


    def _rcs_preview_aspect_ratio(self) -> float:
        if self._get_rcs_plot_mode() == "orbit":
            return 1.0

        canvas = getattr(self, "rcs_canvas", None)
        if canvas is not None:
            try:
                width = float(canvas.width())
                height = float(canvas.height())
                if math.isfinite(width) and math.isfinite(height) and width > 0.0 and height > 0.0:
                    return max(0.5, min(2.5, width / height))
            except Exception:
                pass
        return 4.6 / 3.8


    def _create_rcs_export_figure(self) -> Figure:
        plot_mode = self._get_rcs_plot_mode()
        aspect = self._rcs_preview_aspect_ratio()
        export_width_in = 7.8
        figsize = (export_width_in, export_width_in / aspect)
        fig = Figure(figsize=figsize, dpi=140, facecolor="white")
        if plot_mode == "orbit":
            ax = fig.add_subplot(111, projection="polar")
        else:
            ax = fig.add_subplot(111)
        self._render_rcs_plot_on_axis(
            ax,
            plot_mode,
            live=self._rcs_recording,
            compact=False,
            export=True,
        )
        fig.tight_layout(rect=(0.04, 0.04, 0.98, 0.86), pad=1.0)
        return fig


    @staticmethod
    def _ema_smooth_rcs_array(y: np.ndarray, alpha: float) -> np.ndarray:
        y = np.asarray(y, dtype=float)
        if y.size <= 1:
            return y.copy()
        a = float(alpha)
        if not np.isfinite(a) or a <= 0.0 or a >= 1.0:
            return y.copy()
        out = np.empty_like(y, dtype=float)
        out[0] = float(y[0])
        for i in range(1, int(y.size)):
            out[i] = a * float(y[i]) + (1.0 - a) * float(out[i - 1])
        return out


    @staticmethod
    def _bin_rcs_curve_points_for_plot(points: List[CurvePoint]) -> Tuple[np.ndarray, np.ndarray]:
        """与距离-RCS图中每次测量曲线一致：0.1m 分箱、同周期功率合并、RCS跳变剔除、平滑。"""
        if not points:
            return np.asarray([], dtype=float), np.asarray([], dtype=float)
        xs = np.asarray([float(p.x) for p in points], dtype=float)
        ys = np.asarray([float(p.rcs_filt) for p in points], dtype=float)
        ts = np.asarray([float(p.t) for p in points], dtype=float)
        mask = np.isfinite(xs) & np.isfinite(ys) & np.isfinite(ts)
        mask &= (xs >= float(RCS_STRAIGHT_X_MIN_M)) & (xs <= float(RCS_STRAIGHT_X_MAX_M))
        xs = xs[mask]
        ys = ys[mask]
        ts = ts[mask]
        if xs.size == 0:
            return np.asarray([], dtype=float), np.asarray([], dtype=float)

        order0 = np.argsort(xs)
        xs = xs[order0]
        ys = ys[order0]
        ts = ts[order0]
        if xs.size >= 2:
            keep = np.ones(xs.size, dtype=bool)
            last_idx = 0
            for j in range(1, xs.size):
                if not keep[last_idx]:
                    last_idx = j
                    continue
                if abs(float(ys[j]) - float(ys[last_idx])) > 10.0:
                    keep[j] = False
                else:
                    last_idx = j
            xs = xs[keep]
            ys = ys[keep]
            ts = ts[keep]
            if xs.size == 0:
                return np.asarray([], dtype=float), np.asarray([], dtype=float)

        bin_m = 0.1
        bin_ids = np.floor(xs / float(bin_m)).astype(np.int64)
        uniq = np.unique(bin_ids)
        xb: List[float] = []
        yb: List[float] = []
        t_gate = float(RCS_BIN_INCOHERENT_SUM_MAX_TIME_SPAN_S)
        for bid in uniq:
            m = bin_ids == bid
            if not np.any(m):
                continue
            xb.append(float(np.mean(xs[m])))
            ys_b = ys[m]
            ts_b = ts[m]
            if ys_b.size <= 1:
                yb.append(float(ys_b[0]))
            elif float(np.max(ts_b) - np.min(ts_b)) <= t_gate:
                if combine_rcs_db_incoherent_sum is not None:
                    cr = combine_rcs_db_incoherent_sum(ys_b.tolist())
                    yb.append(float(cr) if cr is not None else float(np.mean(ys_b)))
                else:
                    yb.append(
                        float(
                            10.0
                            * math.log10(float(np.sum(10.0 ** (ys_b.astype(float) / 10.0))))
                        )
                    )
            else:
                yb.append(float(np.mean(ys_b)))
        x_out = np.asarray(xb, dtype=float)
        y_out = np.asarray(yb, dtype=float)
        order = np.argsort(x_out)
        x_out = x_out[order]
        y_out = y_out[order]
        if _peak_smooth_rcs_series is not None:
            y_out = np.asarray(_peak_smooth_rcs_series(y_out), dtype=float)
        else:
            y_out = MainWindow._ema_smooth_rcs_array(y_out, float(RCS_STRAIGHT_EMA_ALPHA))
        return x_out, y_out


    def _current_distance_rcs_green_curve_rows(self) -> List[Tuple[float, float]]:
        """返回当前距离-RCS图绿色融合曲线的 X/RCS 点；无绿色曲线时返回空列表。"""
        if self._get_rcs_plot_mode() != "distance" or getattr(self, "_loaded_rcs_curves", []):
            return []
        recorder = getattr(self, "rcs_recorder", None)
        segments = list(getattr(recorder, "segments", []) or [])
        if not segments:
            return []

        per_run_binned: List[Tuple[np.ndarray, np.ndarray]] = []
        for seg in segments[:30]:
            xb, yb = MainWindow._bin_rcs_curve_points_for_plot(list(seg))
            yb = self._apply_rcs_plot_calibration_to_y(yb)
            if xb.size == 0 or yb.size == 0:
                continue
            per_run_binned.append((np.asarray(xb, dtype=float), np.asarray(yb, dtype=float)))
        if not per_run_binned:
            return []

        bin_m = 0.1
        thr_db = 10.0
        bin_to_xs: Dict[int, List[float]] = {}
        bin_to_ys: Dict[int, List[float]] = {}
        for xb, yb in per_run_binned:
            n = min(int(xb.size), int(yb.size))
            xb2 = np.asarray(xb[:n], dtype=float)
            yb2 = np.asarray(yb[:n], dtype=float)
            m2 = np.isfinite(xb2) & np.isfinite(yb2)
            xb2 = xb2[m2]
            yb2 = yb2[m2]
            if xb2.size == 0:
                continue
            bids = np.floor(xb2 / float(bin_m)).astype(np.int64)
            for k, xk, yk in zip(bids.tolist(), xb2.tolist(), yb2.tolist()):
                bin_to_xs.setdefault(int(k), []).append(float(xk))
                bin_to_ys.setdefault(int(k), []).append(float(yk))

        xb_all: List[float] = []
        yb_all: List[float] = []
        for bid in sorted(bin_to_ys.keys()):
            ys_list = bin_to_ys.get(int(bid), [])
            xs_list = bin_to_xs.get(int(bid), [])
            if not ys_list or not xs_list:
                continue
            ys_arr = np.asarray(ys_list, dtype=float)
            xs_arr = np.asarray(xs_list, dtype=float)
            med = float(np.median(ys_arr))
            keep = np.abs(ys_arr - med) <= float(thr_db)
            if not np.any(keep):
                continue
            xb_all.append(float(np.mean(xs_arr[keep])))
            yb_all.append(float(np.mean(ys_arr[keep])))
        if not xb_all:
            return []
        x_arr = np.asarray(xb_all, dtype=float)
        y_arr = np.asarray(yb_all, dtype=float)
        order_all = np.argsort(x_arr)
        x_arr = x_arr[order_all]
        y_arr = y_arr[order_all]
        if _peak_smooth_rcs_series is not None:
            y_arr = np.asarray(_peak_smooth_rcs_series(y_arr), dtype=float)
        else:
            y_arr = MainWindow._ema_smooth_rcs_array(y_arr, float(RCS_STRAIGHT_EMA_ALPHA))
        out: List[Tuple[float, float]] = []
        for x_val, y_val in zip(x_arr.tolist(), y_arr.tolist()):
            if math.isfinite(float(x_val)) and math.isfinite(float(y_val)):
                out.append((float(x_val), float(y_val)))
        return out


    def _render_rcs_plot_on_axis(
        self,
        ax,
        plot_mode: str,
        *,
        live: bool = False,
        compact: bool = False,
        export: bool = False,
    ) -> None:
        ax.clear()
        ax.set_facecolor("#FCFCFD" if export else "white")

        if plot_mode == "orbit":
            series = self._get_rcs_orbit_plot_series()
            ax.set_title(
                "圆周 RCS",
                fontsize=14 if export else 10,
                fontweight="semibold" if export else "normal",
                pad=14 if export else 8,
            )
            ax.set_theta_zero_location("N")
            ax.set_theta_direction(-1)
            theta_degs = np.arange(0, 360, 30)
            theta_ticks = np.deg2rad(theta_degs)
            ax.set_xticks(theta_ticks)
            ax.set_xticklabels(
                [f"{int(deg)}°" for deg in theta_degs],
                fontsize=11 if export else 8,
                fontweight="bold",
            )
            ax.grid(
                True,
                color="#CFD8DC" if export else "#D7DEE3",
                linestyle="--",
                linewidth=0.8 if export else 0.6,
                alpha=0.7 if export else 0.55,
            )
            ax.set_axisbelow(True)
            if series is None:
                ax.text(
                    0.5,
                    0.5,
                    "暂无圆周RCS数据\n圆弧段勾选RCS并跑完轨迹后，仅用 Cluster Raw 落盘；\n"
                    "载入同目录 Raw 或切换「圆周-RCS」查看（惯导航向优先，时间兜底）。\n"
                    "历史 __orbit_rcs__*.csv 仍可载入。",
                    transform=ax.transAxes,
                    ha="center",
                    va="center",
                    fontsize=11 if export else 9,
                    color="#546E7A",
                )
                return

            raw_angles = np.asarray(series["angles"], dtype=float)
            raw_values = self._apply_rcs_plot_calibration_to_y(
                np.asarray(series["values"], dtype=float)
            )
            fit_angles = np.asarray(series.get("fit_angles", []), dtype=float)
            fit_values = self._apply_rcs_plot_calibration_to_y(
                np.asarray(series.get("fit_values", []), dtype=float)
            )

            raw_n = min(int(raw_angles.size), int(raw_values.size))
            raw_plot_angles = np.asarray(raw_angles[:raw_n], dtype=float)
            raw_plot_values = np.asarray(raw_values[:raw_n], dtype=float)
            raw_finite = np.isfinite(raw_plot_angles) & np.isfinite(raw_plot_values)
            raw_plot_angles = raw_plot_angles[raw_finite]
            raw_plot_values = raw_plot_values[raw_finite]

            if fit_angles.size >= 2 and fit_values.size == fit_angles.size:
                angles = fit_angles
                values = fit_values
            else:
                angles = raw_angles
                values = raw_values

            n = min(int(angles.size), int(values.size))
            angles = np.asarray(angles[:n], dtype=float)
            values = np.asarray(values[:n], dtype=float)
            finite = np.isfinite(angles) & np.isfinite(values)
            angles = angles[finite]
            values = values[finite]
            if angles.size < 2:
                ax.text(
                    0.5,
                    0.5,
                    "圆周 RCS 有效点不足",
                    transform=ax.transAxes,
                    ha="center",
                    va="center",
                    fontsize=11 if export else 9,
                    color="#546E7A",
                )
                return

            if raw_plot_angles.size:
                ax.scatter(
                    raw_plot_angles,
                    raw_plot_values,
                    s=8 if export else 5,
                    marker="o",
                    color="#6E6E6E",
                    alpha=0.46 if export else 0.38,
                    linewidths=0,
                    zorder=2,
                )

            angles_closed = np.append(angles, angles[0])
            values_closed = np.append(values, values[0])
            ax.plot(
                angles_closed,
                values_closed,
                color="#0039D8",
                linewidth=1.85 if export else 1.35,
                alpha=0.96,
                zorder=4,
            )

            tick_step = max(float(ORBIT_RCS_RADIAL_TICK_STEP_DB), 1e-6)
            r_min = -15.0
            r_max = 15.0
            tick_start = r_min
            tick_end = r_max
            r_ticks = np.arange(tick_start, tick_end + tick_step * 0.1, tick_step)
            ax.set_ylim(r_min, r_max)
            ax.set_yticks(r_ticks)
            tick_labels = [
                f"{int(round(tick))}" if abs(float(tick) - round(float(tick))) < 1e-6 else f"{tick:.1f}"
                for tick in r_ticks
            ]
            ax.set_yticklabels(
                tick_labels,
                fontsize=10 if export else 8,
                fontweight="bold",
            )
            ax.text(
                0.5,
                0.95,
                "dBsm",
                transform=ax.transAxes,
                fontsize=11 if export else 9,
                fontweight="bold",
                ha="center",
                va="center",
            )
            return

        ax.set_title(
            self._get_rcs_plot_display_title(plot_mode, compact=compact),
            fontsize=14 if export else 10,
            fontweight="semibold" if export else "normal",
            pad=12 if export else 6,
        )
        ax.set_xlabel("Front Distance (m)" if export else "Distance (m)")
        ax.set_ylabel("RCS (dBsm)" if export else "RCS")
        ax.set_axisbelow(True)
        ax.grid(False)
        ax.yaxis.grid(
            True,
            color="#DCE3E8" if export else "#E7ECF0",
            linestyle="-",
            linewidth=0.8 if export else 0.6,
        )
        if export:
            ax.xaxis.grid(True, color="#EEF2F5", linestyle="--", linewidth=0.65)
        ax.set_xlim(0, RCS_STRAIGHT_X_MAX_M)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        show_legend = export or len(self._loaded_rcs_curves) > 1
        show_ref_labels = export

        if self._rcs_show_only_fitted and self._rcs_fitted is not None:
            xg, yg = self._clip_curve_by_distance(*self._rcs_fitted)
            yg = self._apply_rcs_plot_calibration_to_y(yg)
            if xg.size:
                ax.plot(
                    xg,
                    yg,
                    linewidth=1.7 if export else RCS_FIT_LINE_WIDTH,
                    color="#1565C0",
                    label=None,
                )
            self._plot_rcs_reference_limits(
                ax,
                show_labels=show_ref_labels,
                linewidth=1.8 if export else 1.3,
                alpha=0.95 if export else 0.72,
            )
            handles, labels = ax.get_legend_handles_labels()
            if show_legend and labels:
                ax.legend(loc="best", frameon=False, **self._rcs_legend_kwargs(export))
            return

        if self._loaded_rcs_curves:
            compare_colors = [
                "#1565C0",
                "#C62828",
                "#2E7D32",
                "#6A1B9A",
                "#EF6C00",
                "#00838F",
                "#5D4037",
                "#283593",
            ]
            for index, curve in enumerate(self._loaded_rcs_curves):
                color = compare_colors[index % len(compare_colors)]
                xg, yg = self._clip_curve_by_distance(*curve.fitted)
                yg = self._apply_rcs_plot_calibration_to_y(yg)
                if xg.size and np.any(np.isfinite(yg)):
                    ax.plot(
                        xg,
                        yg,
                        color=color,
                        linewidth=1.7 if export else RCS_FIT_LINE_WIDTH,
                        label=curve.display_name if show_legend else None,
                    )

            self._plot_rcs_reference_limits(
                ax,
                show_labels=show_ref_labels,
                linewidth=1.9 if export else 1.3,
                alpha=0.95 if export else 0.72,
            )
            # 参考上下限曲线可能覆盖全量 x，绘制后再次强制直线测量的距离门
            ax.set_xlim(0.0, float(RCS_STRAIGHT_X_MAX_M))
            handles, labels = ax.get_legend_handles_labels()
            if show_legend and labels:
                ax.legend(loc="best", frameon=False, **self._rcs_legend_kwargs(export))
            return

        segments = list(self.rcs_recorder.segments or [])
        run_lbl = list(self._rcs_segment_run_labels or [])
        segment_labels = list(getattr(self, "_rcs_segment_display_labels", None) or [])
        if segments:
            # 直线测量：显示从 0 开始，但仍只使用 [RCS_STRAIGHT_X_MIN_M, RCS_STRAIGHT_X_MAX_M] 数据做绘制/融合/拟合
            ax.set_xlim(0.0, float(RCS_STRAIGHT_X_MAX_M))
            run_colors = [
                "#1565C0",
                "#F9A825",
                "#C62828",
                "#6A1B9A",
                "#00838F",
                "#5D4037",
                "#283593",
                "#546E7A",
            ]
            def _bin_points_by_x(points: List[CurvePoint]) -> Tuple[np.ndarray, np.ndarray]:
                # 0.1m 分箱；同一雷达周期内（|Δt| 小）多反射点：RCS 按功率线性叠加
                #   P_total = Σ 10^(RCS/10)，RCS_total = 10×log10(P_total)（与 RCS00+RCS01 一致）；
                # 跨时间样本仍用 dB 算术均值以平滑轨迹。
                if not points:
                    return np.asarray([], dtype=float), np.asarray([], dtype=float)
                xs = np.asarray([float(p.x) for p in points], dtype=float)
                ys = np.asarray([float(p.rcs_filt) for p in points], dtype=float)
                ts = np.asarray([float(p.t) for p in points], dtype=float)
                mask = np.isfinite(xs) & np.isfinite(ys) & np.isfinite(ts)
                mask &= (xs >= float(RCS_STRAIGHT_X_MIN_M)) & (xs <= float(RCS_STRAIGHT_X_MAX_M))
                xs = xs[mask]
                ys = ys[mask]
                ts = ts[mask]
                if xs.size == 0:
                    return np.asarray([], dtype=float), np.asarray([], dtype=float)

                # 剔除 RCS 异常跳变点：按 x 排序后，相邻点 |ΔRCS|>10 dBsm 的点直接丢弃
                order0 = np.argsort(xs)
                xs = xs[order0]
                ys = ys[order0]
                ts = ts[order0]
                if xs.size >= 2:
                    keep = np.ones(xs.size, dtype=bool)
                    last_idx = 0
                    for j in range(1, xs.size):
                        if not keep[last_idx]:
                            last_idx = j
                            continue
                        if abs(float(ys[j]) - float(ys[last_idx])) > 10.0:
                            keep[j] = False
                        else:
                            last_idx = j
                    xs = xs[keep]
                    ys = ys[keep]
                    ts = ts[keep]
                    if xs.size == 0:
                        return np.asarray([], dtype=float), np.asarray([], dtype=float)

                bin_m = 0.1
                bin_ids = np.floor(xs / float(bin_m)).astype(np.int64)
                uniq = np.unique(bin_ids)
                xb: List[float] = []
                yb: List[float] = []
                t_gate = float(RCS_BIN_INCOHERENT_SUM_MAX_TIME_SPAN_S)
                for bid in uniq:
                    m = bin_ids == bid
                    if not np.any(m):
                        continue
                    xb.append(float(np.mean(xs[m])))
                    ys_b = ys[m]
                    ts_b = ts[m]
                    if ys_b.size <= 1:
                        yb.append(float(ys_b[0]))
                    elif float(np.max(ts_b) - np.min(ts_b)) <= t_gate:
                        if combine_rcs_db_incoherent_sum is not None:
                            cr = combine_rcs_db_incoherent_sum(ys_b.tolist())
                            yb.append(float(cr) if cr is not None else float(np.mean(ys_b)))
                        else:
                            yb.append(
                                float(
                                    10.0
                                    * math.log10(float(np.sum(10.0 ** (ys_b.astype(float) / 10.0))))
                                )
                            )
                    else:
                        yb.append(float(np.mean(ys_b)))
                x_out = np.asarray(xb, dtype=float)
                y_out = np.asarray(yb, dtype=float)
                order = np.argsort(x_out)
                x_out = x_out[order]
                y_out = y_out[order]
                if _peak_smooth_rcs_series is not None:
                    y_out = np.asarray(_peak_smooth_rcs_series(y_out), dtype=float)
                else:
                    y_out = MainWindow._ema_smooth_rcs_array(y_out, float(RCS_STRAIGHT_EMA_ALPHA))
                return x_out, y_out

            max_show_segments = 30
            segments_to_show = segments[:max_show_segments]
            segment_sizes: List[int] = []
            # 记录每次分箱后的序列，用于“跨次数异常值剔除”后再融合
            per_run_binned: List[Tuple[np.ndarray, np.ndarray]] = []
            for i, seg in enumerate(segments_to_show):
                xb, yb = _bin_points_by_x(seg)
                yb = self._apply_rcs_plot_calibration_to_y(yb)
                if xb.size == 0 or yb.size == 0:
                    segment_sizes.append(0)
                    per_run_binned.append((np.asarray([], dtype=float), np.asarray([], dtype=float)))
                    continue
                color = run_colors[i % len(run_colors)]
                segment_sizes.append(int(min(int(xb.size), int(yb.size))))
                per_run_binned.append((np.asarray(xb, dtype=float), np.asarray(yb, dtype=float)))
                run_no = (
                    int(run_lbl[i])
                    if i < len(run_lbl) and run_lbl[i] is not None
                    else (i + 1)
                )
                legend_label = (
                    str(segment_labels[i]).strip()
                    if i < len(segment_labels) and str(segment_labels[i]).strip()
                    else f"第{run_no}次测量"
                )
                ax.plot(
                    xb,
                    yb,
                    color=color,
                    linewidth=1.25 if export else 1.2,
                    alpha=0.95 if export else 0.88,
                    label=(legend_label if (export or len(segments) > 1) else None),
                    zorder=2,
                )
            if live and getattr(self.rcs_recorder, "_cur", None):
                xb, yb = _bin_points_by_x(list(self.rcs_recorder._cur or []))
                yb = self._apply_rcs_plot_calibration_to_y(yb)
                if xb.size and yb.size:
                    color = run_colors[len(segments) % len(run_colors)]
                    next_no = (int(run_lbl[-1]) + 1) if run_lbl else (len(segments) + 1)
                    ax.plot(
                        xb,
                        yb,
                        color=color,
                        linewidth=1.35 if export else 1.25,
                        linestyle="--",
                        alpha=0.95 if export else 0.85,
                        label=f"第{next_no}次测量(进行中)",
                        zorder=3,
                    )
            valid_curve_count = sum(
                1 for xb, yb in per_run_binned if xb.size > 0 and yb.size > 0
            )
            if valid_curve_count >= 2:
                xb_all, yb_all = MainWindow._fuse_measurement_curves(per_run_binned)
            else:
                xb_all, yb_all = np.asarray([], dtype=float), np.asarray([], dtype=float)
            if xb_all.size and np.any(np.isfinite(yb_all)):
                ax.plot(
                    xb_all,
                    yb_all,
                    linewidth=2.0 if export else 2.2,
                    color="#2E7D32",
                    alpha=0.98 if export else 0.94,
                    label="融合曲线" if len(segments) > 1 else None,
                    zorder=4,
                )
            # 按需求：直线图不显示“次数/分箱点数”旁注，避免图片过于拥挤

            show_legend = bool(
                export or len(self._loaded_rcs_curves) > 1 or len(segments) > 1 or live
            )
            self._plot_rcs_reference_limits(
                ax,
                show_labels=show_ref_labels,
                linewidth=1.9 if export else 1.3,
                alpha=0.95 if export else 0.72,
            )
            handles, labels = ax.get_legend_handles_labels()
            if show_legend and labels:
                ax.legend(loc="best", frameon=False, **self._rcs_legend_kwargs(export))
            return

        fitted_xy: Optional[Tuple[np.ndarray, np.ndarray]] = None
        if self._rcs_fitted is not None:
            fitted_xy = self._rcs_fitted
        elif self.rcs_recorder.point_count() > 0:
            live_segments = [list(seg) for seg in self.rcs_recorder.segments if seg]
            cur = list(getattr(self.rcs_recorder, "_cur", []) or [])
            if cur:
                live_segments.append(cur)
            fitted_xy = self._fitted_curve_from_segments(live_segments)
        if fitted_xy is not None:
            xg, yg = self._clip_curve_by_distance(*fitted_xy)
            yg = self._apply_rcs_plot_calibration_to_y(yg)
            if xg.size and np.any(np.isfinite(yg)):
                ax.plot(
                    xg,
                    yg,
                    linewidth=1.7 if export else RCS_FIT_LINE_WIDTH,
                    color="#1565C0",
                    label=None,
                )

        self._plot_rcs_reference_limits(
            ax,
            show_labels=show_ref_labels,
            linewidth=1.9 if export else 1.3,
            alpha=0.95 if export else 0.72,
        )
        handles, labels = ax.get_legend_handles_labels()
        if show_legend and labels:
            ax.legend(loc="best", frameon=False, **self._rcs_legend_kwargs(export))

    def _draw_rcs(self, live: bool = False) -> None:
        """刷新 RCS 曲线图"""
        plot_mode = self._get_rcs_plot_mode()
        self._ensure_rcs_axes(plot_mode)
        self._render_rcs_plot_on_axis(
            self.rcs_ax,
            plot_mode,
            live=live,
            compact=True,
            export=False,
        )
        self.rcs_canvas.draw_idle()
        return


    def _get_rcs_title(self) -> str:
        if self._loaded_rcs_curves:
            base = (self._rcs_target_name or f"RCS对比({len(self._loaded_rcs_curves)}组)").strip()
            if self._rcs_ref_angle:
                base = f"{base} {self._rcs_ref_angle}"
            return base or "RCS对比"

        base = (self._rcs_target_name or "RCS").strip() or "RCS"
        if self._rcs_ref_angle:
            base = f"{base} {self._rcs_ref_angle}"
        title = base
        if self.rcs_recorder.oid is not None:
            title += f" (ID={self.rcs_recorder.oid})"
        return title


    def _on_rcs_start(
        self,
        segment_index: Optional[int] = None,
        trajectory_name: Optional[str] = None,
    ) -> bool:
        """开始 Cluster(0x701) RCS CSV 录制。"""
        self._orbit_rcs_active = False
        self._orbit_rcs_rows.clear()
        self._orbit_polar_cached_series = None
        self._orbit_rcs_heading0_rad = None
        self._orbit_rcs_clockwise = None
        save_dir = str(self._resolve_rcs_save_dir(trajectory_name or self._get_current_path_name()))
        if self._radial_measurement_spec is not None:
            stem = self._radial_rcs_segment_file_stem(trajectory_name)
            explicit_filename = f"{stem}.csv"
        else:
            traj_tok = self._safe_filename_token(
                str(trajectory_name or self._get_current_path_name()).strip() or "轨迹"
            )
            seg_part = (
                f"_seg{int(segment_index):02d}"
                if segment_index is not None
                else "_segXX"
            )
            stem = f"{traj_tok}_cluster{seg_part}"
            explicit_filename = None
        if not self.controller.begin_cluster_rcs_capture(
            save_dir,
            stem,
            filename=explicit_filename,
        ):
            QtWidgets.QMessageBox.warning(
                self,
                "Cluster RCS",
                "无法启动 CSV：Cluster 接收未就绪或非 Linux CAN（需 can0 Cluster 模式）。",
            )
            self._log("Cluster RCS CSV 启动失败（cluster_csv_runtime 不可用）")
            return False
        self._straight_rcs_collect_all = self._segment_index_is_forward_straight_segment(segment_index)
        self._straight_rcs_last_by_oid.clear()
        self._straight_rcs_rcs_ema_by_oid.clear()
        self._rcs_segment_display_labels = None
        if not self._straight_rcs_collect_all:
            self._straight_rcs_runs = []
            self._rcs_segment_run_labels = None
        self.rcs_recorder.reset()
        self.rcs_lock.disarm()
        self._rcs_recording = True
        self._rcs_fitted = None
        self._rcs_show_only_fitted = False
        self._rcs_orbit_samples = []
        self._loaded_rcs_curves = []
        self._rcs_curve_csv_source_paths = []
        self._rcs_active_segment_index = segment_index
        self._rcs_active_path_name = str(trajectory_name or self._get_current_path_name()).strip() or "轨迹"
        self._rcs_active_target_name = self._get_live_rcs_target_name()
        self._rcs_snapshot_file_target_name = (
            str(self._rcs_active_target_name or self._get_selected_rcs_target_name()).strip()
            or "未命名目标"
        )
        self._rcs_relock_events = []
        segment_text = f" | 分段={segment_index}" if segment_index is not None else ""
        self.rcs_status_label.setText(f"Cluster RCS CSV: 进行中 | 0x701{segment_text}")
        self._log(
            f"【Cluster 0x701】RCS CSV 采集已开始 | "
            f"轨迹={self._rcs_active_path_name}{segment_text} | 目录={save_dir}"
        )
        self._draw_rcs()
        return True


    def _on_orbit_rcs_start(
        self,
        segment_index: Optional[int] = None,
        trajectory_name: Optional[str] = None,
        clockwise: Optional[bool] = None,
    ) -> None:
        """
        圆周段：落盘主文件与直线段相同，均为 DRI Raw 风格 Cluster CSV
        （Data Type/Run Number/Calibration + Time,R,ViewAng,…,DX00..RCS19）；
        仅横向 ROI 放宽（orbit_roi）；圆周段不再单独落盘极坐标 CSV，仅 Cluster Raw；极坐标由 Raw 重放或实时缓存绘制。
        """
        save_dir = str(self._resolve_rcs_save_dir(trajectory_name or self._get_current_path_name()))
        traj_tok = self._safe_filename_token(
            str(trajectory_name or self._get_current_path_name()).strip() or "轨迹"
        )
        seg_part = (
            f"_seg{int(segment_index):02d}"
            if segment_index is not None
            else "_segXX"
        )
        stem = f"{traj_tok}_orbit_cluster{seg_part}"
        if not self.controller.begin_cluster_rcs_capture(
            save_dir, stem, orbit_roi=True
        ):
            QtWidgets.QMessageBox.warning(
                self,
                "Cluster RCS",
                "无法启动圆周段 CSV：Cluster 接收未就绪或非 Linux CAN。",
            )
            self._log("圆周 Cluster RCS CSV 启动失败")
            return
        self.rcs_recorder.reset()
        self.rcs_recorder._ended = False
        self._rcs_segment_run_labels = None
        self._rcs_segment_display_labels = None
        self.rcs_lock.disarm()
        self._orbit_rcs_active = True
        self._orbit_rcs_rows.clear()
        self._orbit_polar_cached_series = None
        pose0 = get_robot_pose()
        self._orbit_rcs_heading0_rad = (
            float(pose0.yaw)
            if pose0 is not None and math.isfinite(float(pose0.yaw))
            else None
        )
        self._orbit_rcs_clockwise = (
            bool(clockwise) if clockwise is not None else self._planned_orbit_clockwise_for_segment(segment_index)
        )
        self._rcs_recording = True
        self._rcs_fitted = None
        self._rcs_show_only_fitted = False
        self._rcs_orbit_samples = []
        self._loaded_rcs_curves = []
        self._rcs_curve_csv_source_paths = []
        self._rcs_active_segment_index = segment_index
        self._rcs_active_path_name = str(trajectory_name or self._get_current_path_name()).strip() or "轨迹"
        self._rcs_active_target_name = self._get_live_rcs_target_name()
        self._rcs_snapshot_file_target_name = (
            str(self._rcs_active_target_name or self._get_selected_rcs_target_name()).strip()
            or "未命名目标"
        )
        self._rcs_relock_events = []
        segment_text = f" | 分段={segment_index}" if segment_index is not None else ""
        self.rcs_status_label.setText(f"圆周 Cluster RCS CSV: 进行中{segment_text}")
        self._log(
            f"【圆周 Cluster 0x701】RCS CSV 已开始 | 轨迹={self._rcs_active_path_name}{segment_text} | "
            f"极角=惯导航向优先/时间兜底"
        )
        self._select_rcs_plot_mode("orbit")
        self._draw_rcs()


    def _finalize_orbit_rcs_recording(
        self,
        *,
        save_raw: bool,
        save_fit_image: bool,
        trajectory_name: Optional[str],
        segment_index: Optional[int],
        cluster_raw_csv_path: Optional[str] = None,
    ) -> Tuple[int, Optional[str], Optional[str]]:
        # save_raw / save_fit_image：圆周段不再另存极坐标 CSV；Cluster Raw 已在 _finalize_cluster_rcs_csv_only 落盘。
        self._orbit_rcs_active = False
        self._rcs_recording = False
        rows = list(self._orbit_rcs_rows)
        self._orbit_rcs_rows.clear()
        self._orbit_rcs_heading0_rad = None
        self._orbit_rcs_clockwise = None
        self.rcs_lock.disarm()
        self.rcs_recorder.reset()
        self.rcs_recorder._cur = []
        self.rcs_recorder._ended = True
        self.rcs_recorder.oid = None

        resolved_segment_index = (
            segment_index if segment_index is not None else self._rcs_active_segment_index
        )
        resolved_trajectory_name = (
            str(trajectory_name or self._rcs_active_path_name or self._get_current_path_name()).strip()
            or "轨迹"
        )
        resolved_target_name = (
            str(
                self._rcs_snapshot_file_target_name
                or self._rcs_active_target_name
                or self._get_selected_rcs_target_name()
            ).strip()
            or "未命名目标"
        )

        if len(rows) < 2:
            self._orbit_polar_cached_series = None
            self._rcs_active_segment_index = None
            self._rcs_active_path_name = None
            self._rcs_active_target_name = None
            self._rcs_snapshot_file_target_name = None
            self._rcs_relock_events = []
            extra = f" | Cluster Raw: {cluster_raw_csv_path}" if cluster_raw_csv_path else ""
            self.rcs_status_label.setText("圆周RCS: 已结束 | 预览点数不足" + extra)
            self._log(
                f"圆周RCS结束: 轨迹={resolved_trajectory_name} | 目标={resolved_target_name} | 点数不足，无极坐标预览{extra}"
            )
            self._draw_rcs()
            return 0, cluster_raw_csv_path, None

        self._orbit_polar_cached_series = self._build_orbit_polar_series_from_rows(rows)
        angle_source = (
            str(self._orbit_polar_cached_series.get("orbit_angle_source", "time"))
            if self._orbit_polar_cached_series is not None
            else "time"
        )
        self._select_rcs_plot_mode("orbit")
        extra = f" | Cluster Raw: {cluster_raw_csv_path}" if cluster_raw_csv_path else ""
        self.rcs_status_label.setText(
            f"圆周RCS: 已结束 | 预览点数={len(rows)} | 角度分箱均值曲线（仅 Raw 落盘）{extra}"
        )
        log_parts = [
            f"圆周RCS结束: 轨迹={resolved_trajectory_name}",
            f"目标={resolved_target_name}",
            f"预览点数={len(rows)}",
            f"极坐标={'惯导方位角' if angle_source == 'heading' else '时间平铺兜底'}后按{ORBIT_RCS_ANGLE_BIN_DEG:g}°分箱算术平均",
        ]
        if resolved_segment_index is not None:
            log_parts.append(f"分段={resolved_segment_index}")
        if cluster_raw_csv_path:
            log_parts.append(f"ClusterRaw={cluster_raw_csv_path}")
        if self._rcs_relock_events:
            log_parts.append(f"自动重锁={len(self._rcs_relock_events)}次")
        self._log(" | ".join(log_parts))

        self._rcs_active_segment_index = None
        self._rcs_active_path_name = None
        self._rcs_active_target_name = None
        self._rcs_snapshot_file_target_name = None
        self._rcs_relock_events = []
        self._draw_rcs()
        return len(rows), cluster_raw_csv_path, None


    def _finalize_cluster_rcs_csv_only(
        self,
        *,
        save_raw: bool,
        trajectory_name: Optional[str],
        segment_index: Optional[int],
        is_orbit: bool,
    ) -> Tuple[int, Optional[str], Optional[str]]:
        """结束 Cluster(0x701) CSV 采集。"""
        rt = getattr(self.controller, "cluster_csv_runtime", None)
        if rt is None or not rt.is_recording():
            self._rcs_recording = False
            self._orbit_rcs_active = False
            return 0, None, None
        n_frames, n_cl = rt.snapshot_stats()
        csv_path = self.controller.end_cluster_rcs_capture()
        self._rcs_recording = False
        self._orbit_rcs_active = False
        # 圆周段极坐标预览来自内存/Cluster Raw 重放；下次启动直线/圆周采集时在各自入口清空
        self.rcs_recorder.reset()
        self.rcs_recorder._cur = []
        self.rcs_recorder._ended = True
        self.rcs_recorder.oid = None
        self.rcs_lock.disarm()
        self._rcs_fitted = None
        self._loaded_rcs_curves = []
        traj = (
            str(trajectory_name or self._rcs_active_path_name or self._get_current_path_name()).strip()
            or "轨迹"
        )
        seg_txt = f" | 分段={segment_index}" if segment_index is not None else ""
        self.rcs_status_label.setText(
            f"Cluster RCS CSV: 已结束 | 帧≈{n_frames} 簇≈{n_cl}"
            + (f" | {csv_path}" if csv_path else "")
        )
        self._log(
            " | ".join(
                [
                    f"Cluster(0x701) RCS CSV 采集结束: 轨迹={traj}{seg_txt}",
                    f"帧≈{n_frames}",
                    f"簇计数≈{n_cl}",
                    (f"文件={csv_path}" if csv_path else "未生成文件"),
                ]
            )
        )
        self._rcs_active_segment_index = None
        self._rcs_active_path_name = None
        self._rcs_active_target_name = None
        self._rcs_snapshot_file_target_name = None
        self._rcs_relock_events = []
        self._draw_rcs()
        return int(n_frames), csv_path if save_raw else None, None


    def _finalize_rcs_recording(
        self,
        *,
        save_raw: bool = False,
        save_fit_image: bool = False,
        trajectory_name: Optional[str] = None,
        segment_index: Optional[int] = None,
    ) -> Tuple[int, Optional[str], Optional[str]]:
        rt_fin = getattr(self.controller, "cluster_csv_runtime", None)
        if rt_fin is not None and rt_fin.is_recording():
            was_orbit = bool(self._orbit_rcs_active)
            n_frames, csv_path, fit_csv = self._finalize_cluster_rcs_csv_only(
                save_raw=save_raw,
                trajectory_name=trajectory_name,
                segment_index=segment_index,
                is_orbit=was_orbit,
            )
            if was_orbit:
                return self._finalize_orbit_rcs_recording(
                    save_raw=save_raw,
                    save_fit_image=save_fit_image,
                    trajectory_name=trajectory_name,
                    segment_index=segment_index,
                    cluster_raw_csv_path=csv_path,
                )
            return n_frames, csv_path, fit_csv
        if self._orbit_rcs_active:
            return self._finalize_orbit_rcs_recording(
                save_raw=save_raw,
                save_fit_image=save_fit_image,
                trajectory_name=trajectory_name,
                segment_index=segment_index,
                cluster_raw_csv_path=None,
            )
        if self.rcs_recorder.oid is None and self.rcs_recorder.point_count() <= 0:
            self._rcs_recording = False
            self._rcs_active_segment_index = None
            self._rcs_active_path_name = None
            self._rcs_active_target_name = None
            self._rcs_snapshot_file_target_name = None
            return 0, None, None

        self.rcs_recorder.finalize()
        self._rcs_recording = False

        point_count = self.rcs_recorder.point_count()
        finalized_segments = [list(seg) for seg in self.rcs_recorder.segments if seg]
        self._rcs_fitted = self._fitted_curve_from_segments(finalized_segments)

        resolved_segment_index = (
            segment_index if segment_index is not None else self._rcs_active_segment_index
        )
        resolved_trajectory_name = (
            str(trajectory_name or self._rcs_active_path_name or self._get_current_path_name()).strip()
            or "轨迹"
        )
        resolved_target_name = (
            str(
                self._rcs_snapshot_file_target_name
                or self._rcs_active_target_name
                or self._get_selected_rcs_target_name()
            ).strip()
            or "未命名目标"
        )
        clip_x0 = (
            float(RCS_STRAIGHT_X_MIN_M)
            if self._segment_index_is_forward_straight_segment(resolved_segment_index)
            else 0.0
        )
        clip_x1 = (
            float(RCS_STRAIGHT_X_MAX_M)
            if self._segment_index_is_forward_straight_segment(resolved_segment_index)
            else float(RCS_MAX_DISTANCE_M)
        )
        self._rcs_fitted = self._clip_curve_by_distance(*self._rcs_fitted, x_min=clip_x0, x_max=clip_x1)

        # 多次往返：前进直线段（speed_sign>0 且近似直线）统一写到同一聚合文件里，不按目标ID拆分
        if self._segment_index_is_forward_straight_segment(resolved_segment_index):
            resolved_target_name = FORWARD_STRAIGHT_RCS_TARGET_NAME
            # 直线测量：把每一次 finalize 出来的段作为“第N次”叠加缓存
            if finalized_segments:
                self._straight_rcs_runs.append(list(finalized_segments[0]))
                if len(self._straight_rcs_runs) > int(self._straight_rcs_max_runs):
                    self._straight_rcs_runs = self._straight_rcs_runs[-int(self._straight_rcs_max_runs) :]
            # 用 UI 缓存的多次数据回填 recorder，用于绘图“第N次”
            if self._straight_rcs_runs:
                self.rcs_recorder.segments = [list(seg) for seg in self._straight_rcs_runs if seg]
                self.rcs_recorder._cur = []
                self.rcs_recorder._ended = True
                self._rcs_segment_run_labels = list(range(1, len(self.rcs_recorder.segments) + 1))
                self._rcs_segment_display_labels = None
            # 同一目标/同一文件的多次前进直线：先形成每次曲线，再做 SG 融合曲线。
            fused = self._fitted_curve_from_segments(self._straight_rcs_runs or [])
            if fused[0].size and fused[1].size:
                self._rcs_fitted = self._clip_curve_by_distance(
                    *fused,
                    x_min=float(RCS_STRAIGHT_X_MIN_M),
                    x_max=float(RCS_STRAIGHT_X_MAX_M),
                )

        raw_path = None
        fit_csv_path: Optional[str] = None
        if save_raw and point_count > 0:
            if resolved_segment_index is not None and finalized_segments:
                raw_path = self._save_rcs_trajectory_snapshot(
                    trajectory_name=resolved_trajectory_name,
                    target_name=resolved_target_name,
                    segment_points=finalized_segments[0],
                )
            else:
                raw_path = self._save_rcs_raw_snapshot(
                    trajectory_name=resolved_trajectory_name,
                    target_name=resolved_target_name,
                    segment_index=resolved_segment_index,
                )
            if raw_path:
                cand = Path(raw_path).with_name(f"{Path(raw_path).stem}_fitted.csv")
                if cand.is_file():
                    fit_csv_path = str(cand)

        point_count_report = int(point_count)
        if raw_path:
            rp = Path(raw_path)
            if rp.is_file():
                try:
                    merged_segs, merged_run_labels = self._parse_saved_rcs_raw_file(str(rp))
                    if merged_segs:
                        self._rcs_fitted = self._clip_curve_by_distance(
                            *self._fitted_curve_from_segments(merged_segs),
                            x_min=clip_x0,
                            x_max=clip_x1,
                        )
                        # 注意：前进直线段希望保留“第N次”叠加显示，不用保存文件解析结果覆盖 UI 缓存
                        if not self._segment_index_is_forward_straight_segment(resolved_segment_index):
                            self.rcs_recorder.reset()
                            self.rcs_recorder.segments = [list(s) for s in merged_segs]
                            self.rcs_recorder._cur = []
                            self.rcs_recorder._ended = True
                            self._rcs_segment_run_labels = merged_run_labels
                            self._rcs_segment_display_labels = self._rcs_segment_file_labels(
                                str(rp), len(merged_segs)
                            )
                        elif len(merged_run_labels) == len(self.rcs_recorder.segments):
                            # 与落盘 CSV 的 SegIdx 分段对齐（按升序映射为 1..N）
                            self._rcs_segment_run_labels = merged_run_labels
                        point_count_report = sum(len(s) for s in merged_segs)
                except Exception:
                    pass

        fit_path = self._save_rcs_fit_image() if save_fit_image else None

        status_parts = [f"RCS录制: 已结束 | 点数={point_count_report}"]
        if raw_path:
            status_parts.append(f"原始数据已保存 {raw_path}")
        if fit_csv_path:
            status_parts.append(f"拟合曲线数据已保存 {fit_csv_path}")
        if fit_path:
            status_parts.append(f"拟合曲线已保存 {fit_path}")
        self.rcs_status_label.setText(" | ".join(status_parts))

        log_parts = [
            f"RCS录制结束: 轨迹={resolved_trajectory_name}",
            f"目标={resolved_target_name}",
            f"点数={point_count_report}",
        ]
        if resolved_segment_index is not None:
            log_parts.append(f"分段={resolved_segment_index}")
        if raw_path:
            log_parts.append(f"原始数据={raw_path}")
        if fit_csv_path:
            log_parts.append(f"拟合数据={fit_csv_path}")
        if fit_path:
            log_parts.append(f"拟合曲线={fit_path}")
        if self._rcs_relock_events:
            log_parts.append(f"自动重锁={len(self._rcs_relock_events)}次")
        self._log(" | ".join(log_parts))

        self._rcs_active_segment_index = None
        self._rcs_active_path_name = None
        self._rcs_active_target_name = None
        self._rcs_snapshot_file_target_name = None
        self._rcs_relock_events = []
        self._draw_rcs()
        return point_count_report, raw_path, fit_path


    def _has_rcs_plot_content(self) -> bool:
        if self._get_rcs_plot_mode() == "orbit":
            return self._get_rcs_orbit_plot_series() is not None
        if self._loaded_rcs_curves:
            return True
        if self._rcs_fitted is not None:
            xg, yg = self._clip_curve_by_distance(*self._rcs_fitted)
            if xg.size and np.any(np.isfinite(yg)):
                return True
        return self.rcs_recorder.point_count() > 0


    def _default_rcs_image_filename(self) -> str:
        stamp = time.strftime("%Y%m%d_%H%M%S")
        if self._get_rcs_plot_mode() == "orbit":
            label = self._safe_filename_token(self._get_selected_rcs_target_name())
            return f"rcs_orbit_{label}_{stamp}.png"
        if self._loaded_rcs_curves:
            if len(self._loaded_rcs_curves) == 1:
                label = self._safe_filename_token(self._loaded_rcs_curves[0].display_name)
                return f"rcs_plot_{label}_{stamp}.png"
            return f"rcs_compare_{len(self._loaded_rcs_curves)}curves_{stamp}.png"
        if self._rcs_fitted is not None and self.rcs_recorder.oid is not None:
            return f"rcs_fit_id{self.rcs_recorder.oid}_{stamp}.png"
        label = self._safe_filename_token(self._get_selected_rcs_target_name())
        return f"rcs_plot_{label}_{stamp}.png"


    def _current_rcs_curve_csv_source_paths(self) -> List[str]:
        """当前距离-RCS图对应的 Raw CSV；保存图片时同步生成 Filtered/combined。"""
        if self._get_rcs_plot_mode() != "distance":
            return []
        candidates = list(getattr(self, "_rcs_curve_csv_source_paths", []) or [])
        if not candidates:
            if self._loaded_rcs_curves:
                candidates = [str(curve.file_path) for curve in self._loaded_rcs_curves]
            elif self._rcs_base_file_path:
                candidates = [str(self._rcs_base_file_path)]

        out: List[str] = []
        seen: Set[str] = set()
        for raw in candidates:
            path = Path(str(raw))
            key = str(path.resolve()) if path.exists() else str(path)
            if key in seen:
                continue
            seen.add(key)
            if not path.is_file() or path.suffix.lower() != ".csv":
                continue
            if MainWindow._is_orbit_cluster_raw_csv(path):
                continue
            if not MainWindow._cluster_csv_has_dri_cluster_header(path):
                continue
            out.append(str(path))
        return out


    def _on_rcs_save_image(self) -> None:
        if not self._has_rcs_plot_content():
            QtWidgets.QMessageBox.information(self, "无图像", "当前没有可保存的RCS图像。")
            self._log("保存RCS图片失败: 当前没有可导出的图像内容")
            return

        default_path = str(Path(self._rcs_save_dir) / self._default_rcs_image_filename())
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self,
            "保存RCS图片",
            default_path,
            "PNG (*.png);;JPEG (*.jpg *.jpeg);;所有文件(*)",
        )
        if not path:
            return

        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        suffix = output_path.suffix.lower()
        fmt = "jpeg" if suffix in {".jpg", ".jpeg"} else "png"
        if not suffix:
            output_path = output_path.with_suffix(".png")
            fmt = "png"

        export_fig = self._create_rcs_export_figure()
        export_fig.savefig(
            str(output_path),
            format=fmt,
            dpi=RCS_EXPORT_DPI,
            facecolor="white",
        )
        export_fig.clf()
        csv_note = ""
        csv_sources = self._current_rcs_curve_csv_source_paths()
        if csv_sources:
            _filtered_paths, combined_path = self._try_export_rcs_curve_csvs(
                csv_sources,
                self._rcs_plot_calibration_db,
            )
            if combined_path is not None:
                csv_note = f" | CSV={combined_path}"
        self.rcs_status_label.setText(f"RCS图片已保存 {output_path}{csv_note}")
        self._log(f"RCS图片已保存: {output_path}{csv_note}")


    def _save_rcs_fit_image(self) -> Optional[str]:
        if self._rcs_fitted is None or self.rcs_recorder.oid is None:
            return None
        xg, yg = self._clip_curve_by_distance(*self._rcs_fitted)
        yg = self._apply_rcs_plot_calibration_to_y(yg)
        valid = np.isfinite(yg)
        if xg.size == 0 or not np.any(valid):
            return None
        stamp = time.strftime("%Y%m%d_%H%M%S")
        filename = f"rcs_fit_id{self.rcs_recorder.oid}_{stamp}.png"
        path = Path(self._rcs_save_dir) / filename

        aspect = self._rcs_preview_aspect_ratio()
        export_width_in = 7.8
        fig = Figure(figsize=(export_width_in, export_width_in / aspect), dpi=140, facecolor="white")
        ax = fig.add_subplot(111)
        ax.set_facecolor("#FCFCFD")
        ax.set_title(f"RCS Fit (ID={self.rcs_recorder.oid})", fontsize=14, fontweight="semibold")
        ax.set_xlabel("Front Distance (m)")
        ax.set_ylabel("RCS (dBsm)")
        ax.grid(False)
        ax.yaxis.grid(True, linestyle="-", linewidth=0.8, color="#DCE3E8")
        ax.xaxis.grid(True, linestyle="--", linewidth=0.65, color="#EEF2F5")
        ax.set_xlim(0, RCS_STRAIGHT_X_MAX_M)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.plot(xg, yg, linewidth=1.7, color="#1565C0", label=None)
        self._plot_rcs_reference_limits(ax, linewidth=1.6, alpha=0.95)
        handles, labels = ax.get_legend_handles_labels()
        if labels:
            ax.legend(loc="best", frameon=False, **self._rcs_legend_kwargs(True))
        fig.tight_layout(rect=(0.04, 0.04, 0.98, 0.86), pad=1.0)
        fig.savefig(
            str(path),
            format="png",
            dpi=RCS_EXPORT_DPI,
            facecolor="white",
        )
        fig.clf()
        return str(path)


    def _convert_target_to_objmeas(self, t, now_ts: float) -> Optional[ObjMeas]:
        """把雷达目标对象转换为 ObjMeas，缺失字段使用默认值。"""
        try:
            oid = int(getattr(t, "cid", getattr(t, "id", getattr(t, "oid", 0))))
            x = float(getattr(t, "x", 0.0))
            y = float(getattr(t, "y", 0.0))
            vx = float(getattr(t, "vx", getattr(t, "vx_rel", 0.0)))
            vy = float(getattr(t, "vy", getattr(t, "vy_rel", 0.0)))
            dyn = int(getattr(t, "dyn", 0))
            rcs = getattr(t, "rcs_db", None)
            if rcs is None:
                rcs = getattr(t, "rcs_kalman", getattr(t, "rcs", 0.0))
            meas_t = getattr(t, "t", now_ts)
            try:
                meas_t = float(meas_t)
            except (TypeError, ValueError):
                meas_t = float(now_ts)
            xr = getattr(t, "x_raw", float("nan"))
            yr = getattr(t, "y_raw", float("nan"))
            try:
                xr = float(xr)
                yr = float(yr)
            except (TypeError, ValueError):
                xr = float("nan")
                yr = float("nan")
            if not (math.isfinite(xr) and math.isfinite(yr)):
                xr = float(x)
                yr = float(y)
            return ObjMeas(
                oid=oid,
                x=x,
                y=y,
                vx=vx,
                vy=vy,
                dyn=dyn,
                rcs_db=float(rcs),
                t=float(meas_t),
                x_raw=float(xr),
                y_raw=float(yr),
                rcs_kf_db=float(getattr(t, "rcs_kf_db", float("nan"))),
            )
        except Exception:
            return None


    def _find_radar_target_object_by_oid(self, targets: List[Any], oid: int) -> Optional[Any]:
        for t in targets:
            if self._extract_radar_target_oid(t) == int(oid):
                return t
        return None


    def _orbit_rcs_effective_forward_gate_m(self) -> float:
        """圆周 RCS 尚无落盘点时放宽前向距离半宽，便于首次锁定；有采样后恢复窄门。"""
        base = float(ORBIT_RCS_FORWARD_GATE_M)
        if not self._orbit_rcs_active:
            return base
        if len(self._orbit_rcs_rows) > 0:
            return base
        return float(max(base, float(ORBIT_RCS_FORWARD_GATE_RELAXED_M)))


    def _pick_meas_in_forward_gate(
        self,
        candidates: List[ObjMeas],
        prefer_oid: Optional[int],
        gate_m: float,
    ) -> Optional[ObjMeas]:
        nom = float(ORBIT_RCS_NOMINAL_FORWARD_M)
        gate = float(gate_m)
        in_gate = [m for m in candidates if abs(float(m.x) - nom) <= gate]
        if not in_gate:
            return None
        if prefer_oid is not None:
            for m in in_gate:
                if int(m.oid) == int(prefer_oid):
                    return m
        return min(in_gate, key=lambda mm: abs(float(mm.x) - nom))


    def _linear_fit_from_points(self, points: List[CurvePoint]) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        if not points or len(points) < 2:
            return None
        xs = np.asarray([float(p.x) for p in points], dtype=float)
        ys = np.asarray([float(p.rcs_filt) for p in points], dtype=float)
        mask = np.isfinite(xs) & np.isfinite(ys)
        mask &= (xs >= float(RCS_STRAIGHT_X_MIN_M)) & (xs <= float(RCS_STRAIGHT_X_MAX_M))
        xs = xs[mask]
        ys = ys[mask]
        if xs.size < 2:
            return None
        order = np.argsort(xs)
        xs = xs[order]
        ys = ys[order]
        if float(np.max(xs) - np.min(xs)) <= 1e-6:
            return np.asarray([float(xs[0])], dtype=float), np.asarray([float(np.mean(ys))], dtype=float)
        coef = np.polyfit(xs, ys, 1)
        x_fit = np.linspace(float(np.min(xs)), float(np.max(xs)), max(2, int(np.ceil((float(np.max(xs))-float(np.min(xs))) / RCS_FIT_GRID_STEP_M)) + 1))
        y_fit = np.polyval(coef, x_fit).astype(float)
        return x_fit.astype(float), y_fit


    def _merge_cluster_pair_for_orbit_rcs(
        self,
        m_use: ObjMeas,
        gate_meas: List[ObjMeas],
        raw_clusters: Optional[List[Any]],
        nom: float,
        gate: float,
    ) -> ObjMeas:
        """
        圆周采样写入：优先对 CAN 帧内前两簇（对应 RCS00/RCS01 顺序）做功率叠加；
        若原始簇不可用则退化为 gate_meas 中 oid 0 与 1。
        """
        if raw_clusters is not None and len(raw_clusters) >= 2:
            c0 = raw_clusters[0]
            c1 = raw_clusters[1]
            try:
                dx0 = float(c0["DX"])
                dx1 = float(c1["DX"])
                dy0 = float(c0["DY"])
                dy1 = float(c1["DY"])
                r0 = float(c0["RCS"])
                r1 = float(c1["RCS"])
            except (KeyError, TypeError, ValueError):
                r0 = float("nan")
            else:
                if (
                    math.isfinite(r0)
                    and math.isfinite(r1)
                    and abs(dx0 - nom) <= gate
                    and abs(dx1 - nom) <= gate
                ):
                    if combine_rcs_db_incoherent_sum is None:
                        cr = float(
                            10.0
                            * math.log10(
                                10.0 ** (r0 / 10.0) + 10.0 ** (r1 / 10.0)
                            )
                        )
                    else:
                        merged = combine_rcs_db_incoherent_sum([r0, r1])
                        if merged is None:
                            return self._merge_cluster_0_1_for_orbit_rcs_legacy(
                                m_use, gate_meas
                            )
                        cr = float(merged)
                    x_avg = (dx0 + dx1) * 0.5
                    y_avg = (dy0 + dy1) * 0.5
                    return replace(
                        m_use,
                        rcs_db=cr,
                        x=x_avg,
                        y=y_avg,
                        x_raw=x_avg,
                        y_raw=y_avg,
                        rcs_kf_db=float("nan"),
                    )

        return self._merge_cluster_0_1_for_orbit_rcs_legacy(m_use, gate_meas)


    def _merge_cluster_0_1_for_orbit_rcs_legacy(
        self, m_use: ObjMeas, gate_meas: List[ObjMeas]
    ) -> ObjMeas:
        """兼容：关联列表中 oid 为 0、1 的两点（旧帧内序号）。"""
        by_oid = {int(m.oid): m for m in gate_meas}
        if 0 not in by_oid or 1 not in by_oid:
            return m_use
        m0, m1 = by_oid[0], by_oid[1]
        if combine_rcs_db_incoherent_sum is None:
            cr = float(
                10.0
                * math.log10(
                    10.0 ** (float(m0.rcs_db) / 10.0)
                    + 10.0 ** (float(m1.rcs_db) / 10.0)
                )
            )
        else:
            merged = combine_rcs_db_incoherent_sum([m0.rcs_db, m1.rcs_db])
            if merged is None:
                return m_use
            cr = float(merged)
        x_avg = (float(m0.x) + float(m1.x)) * 0.5
        y_avg = (float(m0.y) + float(m1.y)) * 0.5
        xr0, yr0 = m0.xy_raw()
        xr1, yr1 = m1.xy_raw()
        xr = (xr0 + xr1) * 0.5
        yr = (yr0 + yr1) * 0.5
        return replace(
            m_use,
            rcs_db=cr,
            x=x_avg,
            y=y_avg,
            x_raw=xr,
            y_raw=yr,
            rcs_kf_db=float("nan"),
        )


    def _update_orbit_rcs_recording(
        self,
        targets: List[Any],
        raw_clusters: Optional[List[Any]] = None,
    ) -> None:
        """圆周段：只关联前方约 40 m 距离门内的目标并采样 (θ=atan2(y,x), RCS)。"""
        now_ts = time.time()
        candidates: List[ObjMeas] = []
        for t in targets:
            m = self._convert_target_to_objmeas(t, now_ts)
            if m is not None:
                candidates.append(m)
        fresh_candidates = [
            m
            for m in candidates
            if now_ts - float(m.t) <= float(self._radar_target_fresh_s)
        ]

        if not fresh_candidates:
            self.rcs_status_label.setText("圆周RCS: 无新鲜雷达目标")
            self._draw_rcs(live=True)
            return

        nom = float(ORBIT_RCS_NOMINAL_FORWARD_M)
        gate = self._orbit_rcs_effective_forward_gate_m()
        in_gate = [m for m in fresh_candidates if abs(float(m.x) - nom) <= gate]

        if not self.rcs_lock.armed:
            m_use = self._pick_meas_in_forward_gate(
                fresh_candidates, self.tracked_target_id, gate
            )
            if m_use is None:
                self.rcs_status_label.setText(
                    f"圆周RCS: 等待前向≈{ORBIT_RCS_NOMINAL_FORWARD_M:.0f}m目标 (±{gate:.0f}m)"
                )
                self._draw_rcs(live=True)
                return
            self.rcs_lock.arm_from(m_use)
            rt = self._find_radar_target_object_by_oid(targets, int(m_use.oid))
            if rt is not None and self.tracked_target_id != int(m_use.oid):
                self._set_tracked_radar_target(rt, auto=True, reason="圆周RCS距离门内锁定")
        else:
            if not in_gate:
                self.rcs_status_label.setText("圆周RCS: 门内暂无可关联目标")
                self._draw_rcs(live=True)
                return
            prev_oid = self.rcs_lock.last_oid
            m_use = self.rcs_lock.associate(in_gate, now_ts)
            if m_use is None:
                self.rcs_status_label.setText("圆周RCS: 关联保持/搜索门内目标…")
                self._draw_rcs(live=True)
                return
            if prev_oid is not None and int(m_use.oid) != int(prev_oid):
                self._rcs_relock_events.append(f"圆周门内关联切换 {prev_oid}->{m_use.oid} t={now_ts:.2f}")
            if self.tracked_target_id != int(m_use.oid):
                rt = self._find_radar_target_object_by_oid(targets, int(m_use.oid))
                if rt is not None:
                    self._set_tracked_radar_target(rt, auto=True, reason="圆周RCS关联")

        m_use = self._merge_cluster_pair_for_orbit_rcs(
            m_use, in_gate, raw_clusters, nom, gate
        )

        r_raw = float(m_use.rcs_db)
        r_f = float(getattr(m_use, "rcs_kf_db", float("nan")))
        if not math.isfinite(r_f):
            r_f = r_raw
        self.rcs_recorder.oid = int(m_use.oid)
        pose = get_robot_pose()
        row: Dict[str, float] = {"t": float(m_use.t), "rcs_filt": float(r_f)}
        if pose is not None and math.isfinite(float(pose.yaw)):
            row["yaw_rad"] = float(pose.yaw)
            row["heading_deg"] = float(math.degrees(float(pose.yaw)))
        if self._orbit_rcs_heading0_rad is not None:
            row["heading0_rad"] = float(self._orbit_rcs_heading0_rad)
        if self._orbit_rcs_clockwise is not None:
            row["orbit_clockwise"] = 1.0 if bool(self._orbit_rcs_clockwise) else 0.0
        # 仅缓存 RCS + 惯导航向供极坐标预览；与 Cluster Raw 行内功率合并一致，不落盘单独极坐标表。
        self._orbit_rcs_rows.append(row)
        self._orbit_polar_cached_series = self._build_orbit_polar_series_from_rows(self._orbit_rcs_rows)
        self.rcs_status_label.setText(
            f"圆周RCS: 进行中 | id={m_use.oid} x={m_use.x:.1f}m RCS={r_raw:.1f}dBsm 点={len(self._orbit_rcs_rows)}"
        )
        self._draw_rcs(live=True)


    def _update_rcs_recording(
        self,
        targets: List,
        raw_clusters: Optional[List[Any]] = None,
    ) -> None:
        """Cluster CSV 写入进度；圆周段仍用簇目标更新极坐标采样。"""
        if not self._rcs_recording:
            return
        rt = getattr(self.controller, "cluster_csv_runtime", None)
        if rt is not None and rt.is_recording():
            nf, nc = rt.snapshot_stats()
            tag = "圆周" if self._orbit_rcs_active else "距离"
            self.rcs_status_label.setText(f"Cluster RCS CSV ({tag}): 帧≈{nf} 簇≈{nc}")
            # 圆周段与 Cluster CSV 同时录制时也必须更新极坐标采样，否则「圆周-RCS」图始终为空
            if self._orbit_rcs_active:
                self._update_orbit_rcs_recording(targets, raw_clusters)
            else:
                self._draw_rcs(live=True)
            return
        if self._orbit_rcs_active:
            self._update_orbit_rcs_recording(targets, raw_clusters)

