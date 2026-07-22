# UWB/GNSS/IMU research guidance

## Technical framing

- Indoor centimeter positioning is primarily an UWB+IMU/INS problem. GNSS provides outdoor global coordinates and supports the transition zone.
- Accuracy depends on the complete chain: anchor survey and geometry, timestamp synchronization, UWB ranging quality, NLOS mitigation, extrinsic/lever-arm calibration, state estimation, and validation.
- Prefer raw UWB range or TDoA factors over a black-box UWB position when tight coupling and anchor-level quality control are possible.

## Method families to compare

- Robust UWB front end: MCC/maximum correntropy, UKF, LS initialization, Taylor or Gauss–Newton refinement.
- Geometry: GDOP/PEB or MSE objectives that include NLOS bias; optimize anchor count and placement before deployment.
- Online fusion: ESKF for deterministic real-time cost and covariance-consistent asynchronous updates.
- High-accuracy backend: sliding-window FGO or iSAM2 with IMU preintegration, raw UWB factors, robust losses, and marginalization.
- NLOS handling: classify LOS/NLOS, predict continuous uncertainty, set dynamic observation covariance, then apply Huber/Cauchy as a second protection layer.
- Outdoor–indoor handover: OUTDOOR/TRANSITION/INDOOR state machine with hysteresis; use GNSS C/N0, DOP, satellite count, RTK status, visible UWB anchors, and NIS.

## Evidence discipline

- Always state whether a result is 2D or 3D, static or dynamic, LOS or NLOS, and measured or simulated.
- Keep paper results in their native metric. Do not convert mean error into RMSE or infer a P95 without data.
- Do not use a seamless-transition paper's decimeter/meter accuracy as evidence of indoor centimeter accuracy. Use it to support coordinate unification, map constraints, estimator comparison, and handover logic.

## Project-oriented architecture

1. Sensor layer: UWB raw range/TDoA, IMU, RTK-GNSS; wheel odometry or non-holonomic constraints when available.
2. Calibration/synchronization: surveyed anchors, sensor extrinsics, lever arms, hardware timestamps, online delay checks.
3. Quality front end: CIR/power/residual features, LOS probability, range bias and variance prediction.
4. Fusion: IMU preintegration plus raw range and GNSS factors, dynamic covariance, innovation gating, robust loss.
5. Monitoring: NIS/NEES, rejection reason, visible-anchor geometry, RMSE/P95/max error, continuity and recovery time.

## Suggested acceptance sequence

- Synchronization and static calibration.
- LOS UWB ranging and geometry validation.
- ESKF baseline and covariance consistency.
- Dynamic covariance and NLOS gating.
- Sliding-window tight coupling and robust loss.
- GNSS↔UWB hysteresis handover.
- Cross-room, cross-day, NLOS, packet-loss, transition, and dual-outage testing.
