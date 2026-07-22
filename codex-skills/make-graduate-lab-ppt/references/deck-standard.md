# Deck standard

## Communication job

By the end, a graduate research-group audience should understand which technical route is supported by evidence, what accuracy was actually achieved, what conditions enabled it, and what the project should implement next.

## Default visual language

- Canvas: 16:9, white background, narrow light-gray top band.
- Header: `章节号  结论式标题`, deep blue `#07569C`, left blue vertical accent, thin blue horizontal rule.
- Palette: navy `#083A6C`, blue `#07569C`, pale blue `#E7F2FA`, gray-blue line `#B8CAD9`, red `#C62828`, green `#16816C`.
- Typography: Microsoft YaHei titles; STKaiti/华文楷体 body when requested. English abbreviations retain their canonical spelling.
- Tables: blue header, alternating white/pale-blue rows, red only for the most decision-relevant metrics.
- Footers: source on the left, page number on the right; never let footers compete with evidence.

## Recommended slide archetypes

1. Technical boundary: target accuracy, sensor roles, and evidence ranges.
2. Error budget: calibration, timing, ranging, geometry, extrinsics, estimator consistency.
3. Method pipeline: raw measurement → quality control → initialization → refinement → fusion.
4. Geometry/layout: anchor placement and NLOS-aware objective.
5. Fusion architecture: IMU preintegration plus raw UWB/GNSS factors in a sliding window.
6. Robustness: NLOS classifier, uncertainty prediction, dynamic covariance, robust loss.
7. Evidence table: methods, scenarios, metrics, and project reuse.
8. Implementation and validation: phased milestones with acceptance metrics.

## Writing constraints

- One dominant claim per slide.
- Use short noun phrases and direct conclusions; put elaboration in notes.
- Show units and metric definitions. Do not mix mean error, RMSE, median, P95, and maximum error as if interchangeable.
- Label proposed targets explicitly as “本项目目标” or “建议验收门槛”.

## QA checklist

- No title wraps unintentionally.
- No body text below 16 pt.
- No connector crosses a label or node.
- No border touches or clips glyphs.
- Every reported number has a source and scenario.
- Every slide is inspected at full size after rendering.
- `slides_test.py`: pass.
- `audit_pptx_quality.py`: high=0, medium=0.
