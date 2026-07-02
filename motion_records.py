# renamed: motion_records.py
# role: 运动过程记录、实验日志和运行指标。
# contains: 跟踪 sample 缓存、运动记录 CSV、完整运动会话合并、指标导出、停车原因弹窗、手柄接管截断。
# also contains: 运动异常检测状态的日志侧处理和数据分析窗口刷新入口。
# notes: 车辆控制命令仍在 controller/car_control；本文件只处理 UI 侧记录和反馈。
# -*- coding: utf-8 -*-

from ui_shared import *
from ui_shared import _TRACKING_MOTION_INTERNAL_FIELDS, _TRACKING_MOTION_TRACE_FIELDS


MainWindow = None  # 主入口回填真实 MainWindow，供 mixin 内静态方法引用。


class MainWindowMotionMixin:
    def _handle_car_control_mode_transition(
        self,
        power_snapshot: Dict[str, Dict[str, Any]],
    ) -> None:
        car_info = power_snapshot.get("car", {})
        raw_mode = car_info.get("mode")
        try:
            mode = int(raw_mode) if raw_mode is not None else None
        except (TypeError, ValueError):
            mode = None

        previous_mode = self._last_car_control_mode
        self._last_car_control_mode = mode
        status_text = str(car_info.get("status") or "").strip().lower()
        feedback_age_s = car_info.get("feedback_age_s")
        try:
            feedback_age_s = (
                None if feedback_age_s is None else max(0.0, float(feedback_age_s))
            )
        except (TypeError, ValueError):
            feedback_age_s = None

        # 首次收到模式反馈时只做初始化，避免上电时非 CAN 状态误触发。
        if previous_mode is None:
            return
        if mode == previous_mode:
            return
        if feedback_age_s is not None and feedback_age_s > 1.5:
            return
        if status_text in {"stale", "no_feedback"}:
            return

        switched_out_of_can = previous_mode == 1 and mode != 1
        remote_priority = (
            mode == 3
            or status_text == "remote"
        )
        non_can_control = (
            mode is not None
            and mode != 1
            and status_text in {"remote", "standby", "mode_unknown"}
        )
        if not (switched_out_of_can and (remote_priority or non_can_control)):
            return

        mode_text = str(car_info.get("mode_text") or "").strip()
        if not mode_text and mode is not None:
            mode_text = f"模式{mode}"
        handover_to_remote = bool(remote_priority)
        reason = (
            "检测到底盘切换到手柄控制"
            if handover_to_remote
            else f"检测到底盘切换到{mode_text or '非CAN控制'}"
        )
        self._handle_motion_session_interrupt(
            reason,
            request_stop=True,
            trim_motion_record_to_now=handover_to_remote,
            save_rcs_raw=not handover_to_remote,
            popup_title="\u63a7\u5236\u6a21\u5f0f\u5207\u6362",
            popup_text=(
                f"{reason}\n已自动停止当前底盘运动指令下发。"
                + (
                    "\n已自动保存切换前的运动数据，切换后的手柄运动不记录。"
                    if handover_to_remote
                    else "\n运动数据将自动保存。"
                )
            ),
        )


    def _log(self, message: str) -> None:
        ts = time.strftime("%H:%M:%S")
        line = f"[{ts}] {message}"
        self._runtime_log_lines.append(line)
        if len(self._runtime_log_lines) > 2000:
            self._runtime_log_lines = self._runtime_log_lines[-2000:]
        print(line)


    def _emit_tracking_metrics(self, record: Dict[str, Any]) -> None:
        self.tracking_metrics_reported.emit(dict(record))


    @staticmethod
    def _csv_float_text(value: Any, digits: int = 6) -> str:
        try:
            number = float(value)
        except (TypeError, ValueError):
            return ""
        if not math.isfinite(number):
            return ""
        return f"{number:.{digits}f}"


    @staticmethod
    def _csv_int_text(value: Any) -> str:
        try:
            return str(int(value))
        except (TypeError, ValueError):
            return ""


    def _compact_tracking_sample(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        compact: Dict[str, Any] = {}
        for key in _TRACKING_MOTION_INTERNAL_FIELDS:
            if key in sample:
                compact[key] = sample.get(key)

        compact["run_key"] = str(sample.get("run_key") or "")
        compact["run_label"] = str(sample.get("run_label") or "轨迹")
        compact["tracking_mode"] = str(sample.get("tracking_mode") or "stanley")
        compact["segment_kind"] = str(sample.get("segment_kind") or "")
        compact["segment_trajectory_name"] = str(sample.get("segment_trajectory_name") or "")

        _, motion_direction, _ = self._motion_direction_fields(sample)
        compact["motion_direction"] = motion_direction
        return compact


    @staticmethod
    def _sample_has_valid_yaw_feedback(sample: Dict[str, Any]) -> bool:
        raw_value = sample.get("yaw_rate_feedback_valid")
        if raw_value is None:
            return False
        try:
            return int(raw_value) != 0
        except (TypeError, ValueError):
            return False


    def _append_tracking_sample(self, sample: Dict[str, Any]) -> None:
        run_key = str(sample.get("run_key") or "")
        if not run_key:
            return
        sample_copy = self._compact_tracking_sample(sample)
        with self._tracking_sample_lock:
            self._tracking_sample_buffers.setdefault(run_key, []).append(sample_copy)
            if getattr(self, "_motion_full_session_active", False):
                self._motion_full_session_accumulator.append(dict(sample_copy))


    def _arm_trim_next_motion_record(self, cutoff_ts: float, reason: str) -> None:
        self._trim_next_motion_record = True
        self._trim_next_motion_record_ts = max(0.0, float(cutoff_ts))
        self._trim_next_motion_record_reason = str(reason or "").strip()


    def _clear_trim_next_motion_record(self) -> None:
        self._trim_next_motion_record = False
        self._trim_next_motion_record_ts = 0.0
        self._trim_next_motion_record_reason = ""


    def _maybe_trim_motion_record(self, record: Dict[str, Any]) -> Dict[str, Any]:
        if not bool(getattr(self, "_trim_next_motion_record", False)):
            return record

        exit_reason = str(record.get("exit_reason") or "").strip().lower()
        if exit_reason != "stop_flag":
            return record

        cutoff_ts = float(getattr(self, "_trim_next_motion_record_ts", 0.0) or 0.0)
        if cutoff_ts <= 0.0:
            return record

        samples = list(record.get("motion_samples") or [])
        if not samples:
            return record

        trimmed_samples = []
        for sample in samples:
            raw_ts = sample.get("timestamp")
            try:
                sample_ts = float(raw_ts)
            except (TypeError, ValueError):
                sample_ts = 0.0
            if sample_ts <= cutoff_ts + 1e-3:
                trimmed_samples.append(dict(sample))

        if len(trimmed_samples) >= len(samples):
            return record

        trimmed = dict(record)
        trimmed["motion_samples"] = trimmed_samples
        trimmed["record_trimmed_at_handover"] = 1
        if self._trim_next_motion_record_reason:
            trimmed["stop_reason"] = str(trimmed.get("stop_reason") or self._trim_next_motion_record_reason)
        if not trimmed_samples:
            trimmed["samples"] = 0
            trimmed["feedback_samples"] = 0
            trimmed["duration_s"] = 0.0
            trimmed["peak_abs_lateral_error_m"] = 0.0
            trimmed["peak_abs_heading_error_deg"] = 0.0
            trimmed["peak_abs_yaw_rate_error_radps"] = 0.0
            trimmed["peak_abs_cmd_w_radps"] = 0.0
            trimmed["peak_abs_feedback_w_radps"] = 0.0
            trimmed["motion_distance_total_m"] = 0.0
            trimmed["motion_distance_total_signed_m"] = 0.0
            return trimmed

        last_sample = dict(trimmed_samples[-1])
        distance_m, signed_distance_m, distance_kind, distance_label = self._motion_distance_fields(last_sample)
        trimmed["samples"] = len(trimmed_samples)
        trimmed["feedback_samples"] = sum(
            1
            for sample in trimmed_samples
            if self._sample_has_valid_yaw_feedback(sample)
        )
        try:
            trimmed["duration_s"] = max(0.0, float(last_sample.get("relative_time_s", 0.0)))
        except (TypeError, ValueError):
            trimmed["duration_s"] = float(record.get("duration_s", 0.0) or 0.0)
        trimmed["peak_abs_lateral_error_m"] = max(
            abs(float(sample.get("lateral_error_m", 0.0) or 0.0))
            for sample in trimmed_samples
        )
        trimmed["peak_abs_heading_error_deg"] = max(
            abs(math.degrees(float(sample.get("heading_error_rad", 0.0) or 0.0)))
            for sample in trimmed_samples
        )
        trimmed["peak_abs_yaw_rate_error_radps"] = max(
            abs(float(sample.get("yaw_rate_error_radps", 0.0) or 0.0))
            for sample in trimmed_samples
        )
        trimmed["peak_abs_cmd_w_radps"] = max(
            abs(float(sample.get("cmd_w_radps", 0.0) or 0.0))
            for sample in trimmed_samples
        )
        trimmed["peak_abs_feedback_w_radps"] = max(
            abs(float(sample.get("feedback_w_radps", 0.0) or 0.0))
            for sample in trimmed_samples
        )
        trimmed["motion_distance_total_m"] = distance_m
        trimmed["motion_distance_total_signed_m"] = signed_distance_m
        trimmed["motion_distance_kind"] = distance_kind
        trimmed["motion_distance_label"] = distance_label
        return trimmed


    def _pop_tracking_samples(self, run_key: str) -> List[Dict[str, Any]]:
        if not run_key:
            return []
        with self._tracking_sample_lock:
            samples = self._tracking_sample_buffers.pop(run_key, [])
        return [dict(sample) for sample in samples]


    def _on_tracking_metrics_reported(self, record: Dict[str, Any]) -> None:
        rec = dict(record)
        if not bool(rec.get("completed")):
            exit_reason = str(rec.get("exit_reason") or "").strip().lower()
            if exit_reason == "stop_flag":
                stop_reason = self._get_recent_motion_interrupt_reason()
                if stop_reason:
                    rec["stop_reason"] = stop_reason
        rec["exit_reason_text"] = self._describe_tracking_exit_reason(rec)
        rec["motion_samples"] = self._pop_tracking_samples(str(rec.get("run_key") or ""))
        rec = self._maybe_trim_motion_record(rec)
        if getattr(self, "_motion_full_session_active", False):
            self._log(self._format_tracking_metrics_summary(rec))
            self._clear_trim_next_motion_record()
            er = str(rec.get("exit_reason") or "").strip().lower()
            if er in ("pose_lost", "pose_stale", "stop_flag"):
                self._maybe_popup_tracking_stop_reason(rec)
            return
        rec["record_index"] = len(self._tracking_run_records) + 1
        self._tracking_run_records.append(rec)
        self._clear_trim_next_motion_record()
        if int(rec.get("record_trimmed_at_handover", 0) or 0) != 0:
            self._log("检测到手柄接管，运动数据已按切换时刻截断后保存。")
        self._log(self._format_tracking_metrics_summary(rec))
        self._autosave_motion_record(rec)
        self._maybe_popup_tracking_stop_reason(rec)
        run_label = str(rec.get("run_label") or "").strip()
        if self._motion_active and run_label and not run_label.startswith("分段"):
            self._on_motion_run_finished(run_label)


    def _finalize_full_motion_session_record(self) -> None:
        """将一次规划执行中多段采样合并为一条运动记录（CSV 一条），各行仍保留 segment_index 供分析 UI 二次截取。"""
        samples = [dict(s) for s in self._motion_full_session_accumulator]
        if not samples:
            return
        def _seg_sort_key(samp: Dict[str, Any]) -> Tuple[float, int]:
            try:
                ts = float(samp.get("timestamp") or 0.0)
            except (TypeError, ValueError):
                ts = 0.0
            try:
                si = int(samp.get("segment_index") or 0)
            except (TypeError, ValueError):
                si = 0
            return ts, si

        samples.sort(key=_seg_sort_key)
        try:
            t0 = float(samples[0].get("timestamp") or 0.0)
        except (TypeError, ValueError):
            t0 = 0.0
        session_wall = time.time()
        unified_key = f"{session_wall:.6f}|完整运动"
        for s in samples:
            try:
                ts = float(s.get("timestamp") or t0)
            except (TypeError, ValueError):
                ts = t0
            s["relative_time_s"] = max(0.0, ts - t0)
            s["run_key"] = unified_key
            s["run_label"] = "完整运动"
        last = dict(samples[-1])
        rec: Dict[str, Any] = {
            "timestamp": float(last.get("timestamp") or time.time()),
            "run_key": unified_key,
            "run_label": "完整运动",
            "tracking_mode": "mixed",
            "motion_session_unified": 1,
            "completed": True,
            "exit_reason": "goal_arrived",
            "exit_reason_text": "完整规划运动（多段合并）",
            "motion_samples": samples,
            "samples": len(samples),
            "speed_mps": 0.0,
        }
        for key in ("profile_speed_mps", "desired_v_mps", "cmd_v_mps"):
            try:
                v = abs(float(last.get(key) or 0.0))
                if v > 1e-6:
                    rec["speed_mps"] = v
                    break
            except (TypeError, ValueError):
                continue
        try:
            rec["duration_s"] = max(0.0, float(last.get("relative_time_s") or 0.0))
        except (TypeError, ValueError):
            rec["duration_s"] = 0.0
        rec["feedback_samples"] = sum(
            1 for s in samples if self._sample_has_valid_yaw_feedback(s)
        )
        rec["peak_abs_lateral_error_m"] = max(
            abs(float(s.get("lateral_error_m") or 0.0)) for s in samples
        )
        rec["peak_abs_heading_error_deg"] = max(
            abs(math.degrees(float(s.get("heading_error_rad") or 0.0))) for s in samples
        )
        rec["peak_abs_yaw_rate_error_radps"] = max(
            abs(float(s.get("yaw_rate_error_radps") or 0.0)) for s in samples
        )
        rec["peak_abs_cmd_w_radps"] = max(
            abs(float(s.get("cmd_w_radps") or 0.0)) for s in samples
        )
        rec["peak_abs_feedback_w_radps"] = max(
            abs(float(s.get("feedback_w_radps") or 0.0)) for s in samples
        )
        dist_m, signed_m, kind, dist_label = self._motion_distance_fields(last)
        rec["motion_distance_total_m"] = dist_m
        rec["motion_distance_total_signed_m"] = signed_m
        rec["motion_distance_kind"] = kind
        rec["motion_distance_label"] = dist_label
        rec["exit_reason_text"] = self._describe_tracking_exit_reason(rec)
        rec["record_index"] = len(self._tracking_run_records) + 1
        self._tracking_run_records.append(rec)
        self._log(self._format_tracking_metrics_summary(rec))
        self._autosave_motion_record(rec)


    @staticmethod
    def _tracking_mode_label(mode: Optional[str]) -> str:
        value = str(mode or "stanley").strip().lower()
        if value == "stanley":
            return "Stanley"
        if value == "stanley_pid":
            return "Stanley + PID"
        if value == "circle_orbit":
            return "Circle Orbit"
        if value == "mixed":
            return "多段合并"
        # 兼容历史记录：旧数据可能仍写入 pid
        return "PID"


    @staticmethod
    def _motion_direction_fields(record: Dict[str, Any]) -> Tuple[int, str, str]:
        sign = 1
        raw_sign = record.get("speed_sign")
        if raw_sign is not None:
            try:
                sign = -1 if float(raw_sign) < 0.0 else 1
            except (TypeError, ValueError):
                sign = 1
        else:
            for key in ("speed_mps", "profile_speed_mps", "desired_v_mps", "cmd_v_mps"):
                raw_value = record.get(key)
                if raw_value is None:
                    continue
                try:
                    sign = -1 if float(raw_value) < 0.0 else 1
                    break
                except (TypeError, ValueError):
                    continue
        direction = str(record.get("motion_direction") or "").strip().lower()
        if direction not in ("forward", "reverse"):
            direction = "reverse" if sign < 0 else "forward"
        label = str(record.get("motion_direction_label") or "").strip()
        if not label:
            label = "倒回" if direction == "reverse" else "前进"
        return sign, direction, label


    def _motion_distance_fields(
        self,
        record: Dict[str, Any],
        *,
        total: bool = False,
    ) -> Tuple[float, float, str, str]:
        speed_sign, _, _ = self._motion_direction_fields(record)
        segment_kind = str(record.get("segment_kind") or "").strip().lower()
        tracking_mode = str(record.get("tracking_mode") or "").strip().lower()

        if total:
            raw_distance = record.get("motion_distance_total_m", record.get("motion_distance_m"))
            raw_signed = record.get(
                "motion_distance_total_signed_m",
                record.get("motion_distance_signed_m"),
            )
        else:
            raw_distance = record.get("motion_distance_m", record.get("path_s_m"))
            raw_signed = record.get("motion_distance_signed_m")

        try:
            distance = max(0.0, float(raw_distance))
        except (TypeError, ValueError):
            distance = 0.0

        kind = str(record.get("motion_distance_kind") or "").strip()
        label = str(record.get("motion_distance_label") or "").strip()
        if not kind:
            if tracking_mode == "circle_orbit" or segment_kind == "circle":
                kind = "circle_segment"
            elif segment_kind == "line":
                kind = "line_reverse" if speed_sign < 0 else "line_forward"
            else:
                kind = "reverse_segment" if speed_sign < 0 else "forward_segment"
        if not label:
            if kind == "circle_segment":
                label = "圆周段"
            elif kind == "line_reverse":
                label = "直线段后退向"
            elif kind == "line_forward":
                label = "直线段前进向"
            elif kind == "reverse_segment":
                label = "后退向"
            else:
                label = "前进向"

        try:
            signed_distance = float(raw_signed)
        except (TypeError, ValueError):
            signed_distance = distance if kind == "circle_segment" else float(speed_sign * distance)
        return distance, signed_distance, kind, label


    def _reset_motion_stop_context(self) -> None:
        self._last_motion_interrupt_reason = ""
        self._last_motion_interrupt_ts = 0.0


    def _remember_motion_interrupt_reason(self, reason: str) -> None:
        text = str(reason or "").strip()
        if not text:
            return
        self._last_motion_interrupt_reason = text
        self._last_motion_interrupt_ts = time.time()


    def _get_recent_motion_interrupt_reason(self, max_age_s: float = 15.0) -> str:
        text = str(getattr(self, "_last_motion_interrupt_reason", "") or "").strip()
        if not text:
            return ""
        ts = float(getattr(self, "_last_motion_interrupt_ts", 0.0) or 0.0)
        if ts <= 0.0:
            return ""
        if time.time() - ts > max(0.5, float(max_age_s)):
            return ""
        return text


    def _show_motion_stop_popup(self, title: str, text: str) -> None:
        popup_title = str(title or "\u8fd0\u52a8\u505c\u8f66").strip() or "\u8fd0\u52a8\u505c\u8f66"
        popup_text = str(text or "").strip()
        if not popup_text:
            return
        now = time.time()
        popup_key = f"{popup_title}|{popup_text}"
        if (
            popup_key == self._motion_stop_last_popup_key
            and (now - self._motion_stop_last_popup_ts) < float(self._motion_stop_popup_cooldown_s)
        ):
            return
        self._motion_stop_last_popup_key = popup_key
        self._motion_stop_last_popup_ts = now
        QtWidgets.QMessageBox.warning(self, popup_title, popup_text)


    def _describe_tracking_exit_reason(self, record: Dict[str, Any]) -> str:
        exit_reason = str(record.get("exit_reason") or "").strip().lower()
        stop_reason = str(record.get("stop_reason") or "").strip()
        completed = bool(record.get("completed"))
        if completed or exit_reason == "goal_arrived":
            return "\u6b63\u5e38\u5b8c\u6210\uff1a\u5df2\u5230\u8fbe\u76ee\u6807"
        if exit_reason == "pose_stale":
            return "\u5f02\u5e38\u505c\u8f66\uff1a\u5b9a\u4f4d\u6570\u636e\u8d85\u65f6"
        if exit_reason == "pose_lost":
            return "\u5f02\u5e38\u505c\u8f66\uff1a\u65e0\u6cd5\u83b7\u53d6\u5f53\u524d\u4f4d\u59ff"
        if exit_reason == "stop_flag":
            if stop_reason:
                return f"\u5f02\u5e38\u505c\u8f66\uff1a{stop_reason}"
            return "\u5f02\u5e38\u505c\u8f66\uff1a\u6536\u5230\u505c\u6b62\u6216\u6025\u505c\u6307\u4ee4"
        if exit_reason == "loop_exit":
            return "\u5f02\u5e38\u505c\u8f66\uff1a\u63a7\u5236\u5faa\u73af\u63d0\u524d\u9000\u51fa"
        if exit_reason:
            return f"\u5f02\u5e38\u505c\u8f66\uff1a{exit_reason}"
        return "\u5f02\u5e38\u505c\u8f66\uff1a\u672a\u77e5\u539f\u56e0"


    def _maybe_popup_tracking_stop_reason(self, record: Dict[str, Any]) -> None:
        if bool(record.get("completed")):
            return
        exit_reason = str(record.get("exit_reason") or "").strip().lower()
        if exit_reason == "goal_arrived":
            return
        stop_reason = str(record.get("stop_reason") or "").strip()
        if exit_reason == "stop_flag" and stop_reason:
            return
        label = str(record.get("run_label") or "\u8f68\u8ff9").strip() or "\u8f68\u8ff9"
        mode_label = self._tracking_mode_label(str(record.get("tracking_mode") or "stanley"))
        detail = str(record.get("exit_reason_text") or self._describe_tracking_exit_reason(record))
        popup_text = (
            f"\u8fd0\u52a8\u4efb\u52a1: {label}\n"
            f"\u63a7\u5236\u6a21\u5f0f: {mode_label}\n"
            f"\u505c\u8f66\u539f\u56e0: {detail}"
        )
        self._show_motion_stop_popup("\u8fd0\u52a8\u505c\u8f66\u539f\u56e0", popup_text)


    def _format_tracking_metrics_summary(self, record: Dict[str, Any]) -> str:
        record_index = int(record.get("record_index", len(self._tracking_run_records)))
        label = str(record.get("run_label") or "轨迹")
        mode = str(record.get("tracking_mode") or "stanley")
        mode_label = self._tracking_mode_label(mode)
        _, _, direction_label = self._motion_direction_fields(record)
        started_at = time.strftime(
            "%H:%M:%S", time.localtime(float(record.get("timestamp", time.time())))
        )
        speed_mps = float(record.get("speed_mps", 0.0))
        peak_lat = float(record.get("peak_abs_lateral_error_m", 0.0))
        peak_head_deg = float(record.get("peak_abs_heading_error_deg", 0.0))
        peak_yaw_rate_err = float(record.get("peak_abs_yaw_rate_error_radps", 0.0))
        peak_w = float(record.get("peak_abs_cmd_w_radps", 0.0))
        peak_fb_w = float(record.get("peak_abs_feedback_w_radps", 0.0))
        feedback_samples = int(record.get("feedback_samples", 0))
        duration_s = float(record.get("duration_s", 0.0))
        status = str(
            record.get("exit_reason_text")
            or self._describe_tracking_exit_reason(record)
        )
        feedback_text = (
            f"{peak_fb_w:.3f}rad/s" if feedback_samples > 0 else "N/A"
        )
        return (
            f"实验记录#{record_index} | 开始={started_at} | {label} | 模式={mode_label} | 方向={direction_label} | 速度={speed_mps:.2f}m/s | "
            f"横向误差峰值={peak_lat:.3f}m | 航向误差峰值={peak_head_deg:.2f}deg | "
            f"角速度跟踪误差峰值={peak_yaw_rate_err:.3f}rad/s | "
            f"命令角速度峰值(|w_cmd|)={peak_w:.3f}rad/s | "
            f"反馈角速度峰值(|w_fb|)={feedback_text} | 时长={duration_s:.2f}s | 结束={status}"
        )


    def _tracking_motion_sample_headers(self) -> List[str]:
        return list(_TRACKING_MOTION_TRACE_FIELDS)


    def _tracking_motion_sample_row(
        self,
        record_index: int,
        sample_index: int,
        sample: Dict[str, Any],
    ) -> Dict[str, Any]:
        del record_index, sample_index
        _, motion_direction, _ = self._motion_direction_fields(sample)
        return {
            "timestamp": self._csv_float_text(sample.get("timestamp")),
            "relative_time_s": self._csv_float_text(sample.get("relative_time_s")),
            "run_key": str(sample.get("run_key") or ""),
            "run_label": str(sample.get("run_label") or "轨迹"),
            "tracking_mode": str(sample.get("tracking_mode") or "stanley"),
            "segment_index": self._csv_int_text(sample.get("segment_index")),
            "speed_sign": self._csv_float_text(sample.get("speed_sign")),
            "speed_mps": self._csv_float_text(sample.get("speed_mps")),
            "nominal_speed_abs_mps": self._csv_float_text(sample.get("nominal_speed_abs_mps")),
            "lookahead_base_m": self._csv_float_text(sample.get("lookahead_base_m")),
            "arrival_dist_m": self._csv_float_text(sample.get("arrival_dist_m")),
            "slow_down_dist_m": self._csv_float_text(sample.get("slow_down_dist_m")),
            "stanley_gain": self._csv_float_text(sample.get("stanley_gain")),
            "stanley_softening_distance_m": self._csv_float_text(
                sample.get("stanley_softening_distance_m")
            ),
            "stanley_term_rad": self._csv_float_text(sample.get("stanley_term_rad")),
            "lateral_pid_kp": self._csv_float_text(sample.get("lateral_pid_kp")),
            "lateral_pid_ki": self._csv_float_text(sample.get("lateral_pid_ki")),
            "lateral_pid_kd": self._csv_float_text(sample.get("lateral_pid_kd")),
            "heading_pid_kp": self._csv_float_text(sample.get("heading_pid_kp")),
            "heading_pid_ki": self._csv_float_text(sample.get("heading_pid_ki")),
            "heading_pid_kd": self._csv_float_text(sample.get("heading_pid_kd")),
            "yaw_rate_pid_kp": self._csv_float_text(sample.get("yaw_rate_pid_kp")),
            "yaw_rate_pid_ki": self._csv_float_text(sample.get("yaw_rate_pid_ki")),
            "yaw_rate_pid_kd": self._csv_float_text(sample.get("yaw_rate_pid_kd")),
            "cmd_v_mps": self._csv_float_text(sample.get("cmd_v_mps")),
            "cmd_w_radps": self._csv_float_text(sample.get("cmd_w_radps")),
            "desired_v_mps": self._csv_float_text(sample.get("desired_v_mps")),
            "desired_w_radps": self._csv_float_text(sample.get("desired_w_radps")),
            "feedback_v_mps": self._csv_float_text(sample.get("feedback_v_mps")),
            "feedback_w_radps": self._csv_float_text(sample.get("feedback_w_radps")),
            "lateral_error_m": self._csv_float_text(sample.get("lateral_error_m")),
            "stanley_lateral_pd_output_radps": self._csv_float_text(
                sample.get("stanley_lateral_pd_output_radps")
            ),
            "heading_error_rad": self._csv_float_text(sample.get("heading_error_rad")),
            "yaw_rate_error_radps": self._csv_float_text(sample.get("yaw_rate_error_radps")),
            "lateral_pid_output_radps": self._csv_float_text(
                sample.get("lateral_pid_output_radps")
            ),
            "heading_pid_output_radps": self._csv_float_text(
                sample.get("heading_pid_output_radps")
            ),
            "yaw_rate_pid_output_radps": self._csv_float_text(
                sample.get("yaw_rate_pid_output_radps")
            ),
            "path_curvature_inv_m": self._csv_float_text(
                sample.get("path_curvature_inv_m")
            ),
            "path_ff_w_radps": self._csv_float_text(sample.get("path_ff_w_radps")),
            "profile_speed_mps": self._csv_float_text(sample.get("profile_speed_mps")),
            "path_s_m": self._csv_float_text(sample.get("path_s_m")),
            "dist_to_goal_m": self._csv_float_text(sample.get("dist_to_goal_m")),
            "pose_age_s": self._csv_float_text(sample.get("pose_age_s")),
            "current_x_m": self._csv_float_text(sample.get("current_x_m")),
            "current_y_m": self._csv_float_text(sample.get("current_y_m")),
            "nearest_x_m": self._csv_float_text(sample.get("nearest_x_m")),
            "nearest_y_m": self._csv_float_text(sample.get("nearest_y_m")),
            "lookahead_x_m": self._csv_float_text(sample.get("lookahead_x_m")),
            "lookahead_y_m": self._csv_float_text(sample.get("lookahead_y_m")),
            "segment_kind": str(sample.get("segment_kind") or ""),
            "segment_trajectory_name": str(sample.get("segment_trajectory_name") or ""),
            "motion_direction": motion_direction,
        }


    @staticmethod
    def _safe_filename_token(text: str) -> str:
        cleaned = re.sub(r'[\\/:*?"<>|]+', "_", str(text).strip())
        cleaned = cleaned.replace(" ", "_")
        return cleaned or "run"


    def _ensure_recorded_data_dirs(self) -> None:
        self._data_save_root.mkdir(parents=True, exist_ok=True)
        self._motion_save_dir.mkdir(parents=True, exist_ok=True)
        self._rcs_save_dir.mkdir(parents=True, exist_ok=True)


    @staticmethod
    def _unique_output_path(path: Path) -> Path:
        if not path.exists():
            return path
        stem = path.stem
        suffix = path.suffix
        counter = 2
        while True:
            candidate = path.with_name(f"{stem}_{counter:02d}{suffix}")
            if not candidate.exists():
                return candidate
            counter += 1


    def _move_existing_output_to_dir(
        self,
        src_path: Path,
        target_dir: Path,
    ) -> Optional[Path]:
        if not src_path.exists() or not src_path.is_file():
            return None
        target_dir.mkdir(parents=True, exist_ok=True)
        try:
            if src_path.parent.resolve() == target_dir.resolve():
                return src_path
        except Exception:
            pass

        target_path = target_dir / src_path.name
        if target_path.exists():
            target_path = self._unique_output_path(target_path)
        shutil.move(str(src_path), str(target_path))
        return target_path


    def _migrate_matching_outputs(
        self,
        patterns: List[str],
        target_dir: Path,
    ) -> List[str]:
        moved: List[str] = []
        for pattern in patterns:
            for path in sorted(Path.cwd().glob(pattern)):
                try:
                    migrated = self._move_existing_output_to_dir(path, target_dir)
                except Exception as exc:
                    self._log(f"历史数据迁移失败: {path} -> {target_dir} | {exc}")
                    continue
                if migrated is not None:
                    moved.append(str(migrated))
        return moved


    def _migrate_existing_saved_data(self) -> None:
        moved: List[str] = []
        legacy_rcs_dir = Path.cwd() / "rcs_raw_data"
        try:
            if legacy_rcs_dir.exists() and legacy_rcs_dir.is_dir():
                if legacy_rcs_dir.resolve() != self._rcs_save_dir.resolve():
                    for path in sorted(legacy_rcs_dir.iterdir()):
                        if not path.is_file():
                            continue
                        migrated = self._move_existing_output_to_dir(path, self._rcs_save_dir)
                        if migrated is not None:
                            moved.append(str(migrated))
                    try:
                        legacy_rcs_dir.rmdir()
                    except OSError:
                        pass
        except Exception as exc:
            self._log(f"历史RCS目录迁移失败: {legacy_rcs_dir} | {exc}")

        moved.extend(
            self._migrate_matching_outputs(
                [
                    "*__*__*.txt",
                    "*__orbit_rcs__*.csv",
                    "rcs_fit_id*.png",
                    "fitted_id*.pdf",
                    "rcs_plot_*.png",
                    "rcs_compare_*.png",
                ],
                self._rcs_save_dir,
            )
        )
        moved.extend(
            self._migrate_matching_outputs(
                [
                    "motion_trace_*.csv",
                    "tracking_metrics_*.csv",
                ],
                self._motion_save_dir,
            )
        )
        if moved:
            self._log(
                f"历史运动/RCS数据已归档到 {self._data_save_root} | 数量={len(moved)}"
            )


    def _default_motion_trace_filename(self, record: Dict[str, Any]) -> str:
        timestamp = float(record.get("timestamp", time.time()))
        stamp = time.strftime("%Y%m%d_%H%M%S", time.localtime(timestamp))
        traj_name = str(
            record.get("segment_trajectory_name")
            or record.get("run_label")
            or "轨迹"
        ).strip() or "轨迹"
        traj_token = self._safe_filename_token(traj_name)
        mode_token = self._safe_filename_token(str(record.get("tracking_mode") or "pid"))
        return f"motion_trace_{traj_token}_{stamp}_{mode_token}.csv"


    def _save_motion_record_csv(
        self,
        record: Dict[str, Any],
        target_dir: Path,
        *,
        skip_if_saved: bool = False,
        mark_saved: bool = True,
    ) -> Optional[str]:
        samples = list(record.get("motion_samples") or [])
        if not samples:
            return None

        existing_path_text = str(record.get("motion_saved_path") or "").strip()
        if skip_if_saved and existing_path_text:
            return existing_path_text

        target_dir.mkdir(parents=True, exist_ok=True)
        file_path = target_dir / self._default_motion_trace_filename(record)
        if existing_path_text:
            existing_path = Path(existing_path_text)
            try:
                if existing_path.exists() and existing_path.parent.resolve() == target_dir.resolve():
                    file_path = existing_path
                elif file_path.exists():
                    file_path = self._unique_output_path(file_path)
            except Exception:
                if file_path.exists():
                    file_path = self._unique_output_path(file_path)
        elif file_path.exists():
            file_path = self._unique_output_path(file_path)

        headers = self._tracking_motion_sample_headers()
        record_index = int(record.get("record_index", 0))
        with file_path.open("w", encoding="utf-8-sig", newline="") as fp:
            writer = csv.DictWriter(fp, fieldnames=headers)
            writer.writeheader()
            for sample_index, sample in enumerate(samples, start=1):
                writer.writerow(
                    self._tracking_motion_sample_row(record_index, sample_index, sample)
                )

        if mark_saved:
            record["motion_saved_path"] = str(file_path)
            record["motion_saved_at"] = time.time()
        return str(file_path)


    def _save_motion_records(
        self,
        records: List[Dict[str, Any]],
        target_dir: Path,
        *,
        skip_if_saved: bool = False,
        mark_saved: bool = True,
    ) -> List[str]:
        saved_files: List[str] = []
        for record in records:
            file_path = self._save_motion_record_csv(
                record,
                target_dir,
                skip_if_saved=skip_if_saved,
                mark_saved=mark_saved,
            )
            if file_path:
                saved_files.append(file_path)
        return saved_files


    def _autosave_motion_record(self, record: Dict[str, Any]) -> Optional[str]:
        try:
            file_path = self._save_motion_record_csv(
                record,
                Path(self._motion_save_dir),
                skip_if_saved=True,
                mark_saved=True,
            )
        except Exception as exc:
            self._log(f"运动数据自动保存失败: {exc}")
            return None
        if file_path:
            self._refresh_data_analysis_window(focus_path=Path(file_path))
            self._log(f"运动数据已自动保存: {file_path}")
        return file_path


    def _on_emergency_stop_clicked(self) -> None:
        if self.controller.car is not None:
            try:
                self.controller.car.stop()
            except Exception as exc:
                self._log(f"紧急停止：下发零速失败: {exc}")
        else:
            self._log("紧急停止：CAN 未连接，无法下发底盘停止指令")
        if not self._motion_active and not self._rcs_recording:
            self._log("紧急停止：已请求停止（当前无进行中的轨迹跟踪或 RCS 记录）")
            return
        self._handle_motion_session_interrupt(
            "用户按下紧急停止",
            request_stop=False,
            trim_motion_record_to_now=True,
            popup_title="紧急停止",
            popup_text="已停止当前运动指令下发。",
        )


    def _handle_motion_session_interrupt(
        self,
        reason: str,
        *,
        request_stop: bool = False,
        trim_motion_record_to_now: bool = False,
        save_rcs_raw: bool = True,
        popup_title: Optional[str] = None,
        popup_text: Optional[str] = None,
    ) -> None:
        had_motion = bool(self._motion_active)
        had_rcs = bool(self._rcs_recording)
        if not had_motion and not had_rcs:
            return
        self._remember_motion_interrupt_reason(reason)
        if trim_motion_record_to_now and had_motion:
            self._arm_trim_next_motion_record(time.time(), reason)

        if request_stop and self.controller.car is not None:
            try:
                self.controller.car.stop()
            except Exception as exc:
                self._log(f"{reason} | 停止底盘控制失败: {exc}")

        self._motion_active = False
        self._radar_emergency_active = False
        self._reset_motion_guard_runtime(clear_profile=False)

        raw_path = None
        if self._rcs_recording:
            _, raw_path, _ = self._finalize_rcs_recording(save_raw=save_rcs_raw)

        parts = [reason]
        if had_motion:
            if trim_motion_record_to_now:
                parts.append("切换前的运动数据将自动保存，切换后的手柄运动不记录")
            else:
                parts.append(f"运动数据将自动保存到 {self._motion_save_dir}")
        if raw_path:
            parts.append(f"RCS已保存 {raw_path}")
        elif had_rcs:
            parts.append("RCS数据未保存" if not save_rcs_raw else f"RCS数据目录={self._rcs_save_dir}")
        self._log(" | ".join(parts))
        if popup_title:
            self._show_motion_stop_popup(popup_title, popup_text or reason)


    def _on_motion_run_finished(self, source: str) -> None:
        if not self._motion_active and not self._rcs_recording:
            self._radar_emergency_active = False
            self._reset_motion_guard_runtime(clear_profile=False)
            return

        self._motion_active = False
        self._radar_emergency_active = False
        self._reset_motion_guard_runtime(clear_profile=False)

        raw_path = None
        if self._rcs_recording:
            _, raw_path, _ = self._finalize_rcs_recording(save_raw=True)

        summary = f"运动结束: {str(source or '控制线程退出').strip()}"
        if raw_path:
            summary += f" | RCS已自动保存 {raw_path}"
        self._log(summary)


    def _on_save_motion_data(self) -> None:
        records_with_samples = [
            record for record in self._tracking_run_records
            if record.get("motion_samples")
        ]
        live_snapshot_records = self._build_live_motion_snapshot_records()
        if not records_with_samples and not live_snapshot_records:
            QtWidgets.QMessageBox.information(self, "无运动数据", "当前没有可保存的运动采样数据。")
            return

        target_dir = QtWidgets.QFileDialog.getExistingDirectory(
            self,
            "选择运动数据保存目录",
            str(self._motion_save_dir),
        )
        if not target_dir:
            return

        try:
            all_records = records_with_samples + live_snapshot_records
            saved_files = self._save_motion_records(
                all_records,
                Path(target_dir),
                skip_if_saved=False,
                mark_saved=False,
            )
            finished_count = len(records_with_samples)
            live_count = len(live_snapshot_records)
            if live_count > 0:
                self._log(
                    f"已包含运行中快照保存: 已结束={finished_count} 运行中={live_count}"
                )
            if saved_files:
                self._refresh_data_analysis_window(focus_path=Path(saved_files[-1]))
            self._log(f"运动数据已保存: {len(saved_files)}个文件 -> {target_dir}")
            QtWidgets.QMessageBox.information(
                self,
                "保存成功",
                (
                    f"已保存 {len(saved_files)} 个运动数据文件到:\n{target_dir}\n"
                    f"(已结束={finished_count}, 运行中快照={live_count})"
                ),
            )
        except Exception as exc:
            self._log(f"运动数据保存失败: {exc}")
            QtWidgets.QMessageBox.warning(self, "保存失败", f"运动数据保存失败:\n{exc}")


    def _build_live_motion_snapshot_records(self) -> List[Dict[str, Any]]:
        with self._tracking_sample_lock:
            buffer_items = [
                (str(run_key), [dict(sample) for sample in list(samples)])
                for run_key, samples in self._tracking_sample_buffers.items()
                if samples
            ]
        if not buffer_items:
            return []

        base_index = len(self._tracking_run_records) + 1
        now_ts = time.time()
        snapshot_records: List[Dict[str, Any]] = []
        for offset, (run_key, samples) in enumerate(buffer_items):
            if not samples:
                continue
            first_sample = dict(samples[0])
            last_sample = dict(samples[-1])
            run_label = str(
                last_sample.get("run_label")
                or first_sample.get("run_label")
                or "轨迹"
            ).strip() or "轨迹"
            try:
                timestamp = float(last_sample.get("timestamp", now_ts))
            except (TypeError, ValueError):
                timestamp = float(now_ts)
            try:
                duration_s = max(0.0, float(last_sample.get("relative_time_s", 0.0)))
            except (TypeError, ValueError):
                duration_s = 0.0
            try:
                speed_mps = float(
                    last_sample.get(
                        "profile_speed_mps",
                        last_sample.get("desired_v_mps", last_sample.get("cmd_v_mps", 0.0)),
                    )
                )
            except (TypeError, ValueError):
                speed_mps = 0.0

            distance_m, signed_distance_m, distance_kind, distance_label = (
                self._motion_distance_fields(last_sample)
            )
            snapshot_records.append(
                {
                    "record_index": int(base_index + offset),
                    "timestamp": float(timestamp),
                    "run_key": run_key,
                    "run_label": run_label,
                    "tracking_mode": str(last_sample.get("tracking_mode") or "stanley"),
                    "speed_mps": float(speed_mps),
                    "duration_s": float(duration_s),
                    "samples": int(len(samples)),
                    "motion_distance_total_m": float(distance_m),
                    "motion_distance_total_signed_m": float(signed_distance_m),
                    "motion_distance_kind": distance_kind,
                    "motion_distance_label": distance_label,
                    "motion_samples": samples,
                    "record_snapshot_live": 1,
                }
            )
        return snapshot_records

