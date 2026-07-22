---
name: make-graduate-lab-ppt
description: Create or revise Chinese graduate lab-meeting and research-group presentation PPTX decks from scientific paper folders, PDFs, code, notes, an existing PPTX, or a visual reference. Use for 研究生组会PPT、文献汇报、算法综述、室内定位/UWB/GNSS/IMU 技术汇报, especially when the deck must be evidence-led, use a compact blue-and-white Chinese academic-defense style, show methods and measured accuracy, use larger readable fonts, avoid AI-like wording, and pass rendered-slide overlap and overflow QA.
---

# Graduate Lab Meeting PPT

Produce a real `.pptx`, not an outline. Use `nature-paper2ppt` for the scientific story and `Presentations` for artifact-tool authoring and rendered QA. When either skill is unavailable, follow its installed replacement without changing the quality gates.

## Required workflow

1. Inventory every input file. Read all papers in the user-designated folder; inspect an existing deck and any reference image before authoring.
2. Classify algorithm and engineering papers as `methods`. Build a problem-to-solution narrative: target and bottleneck → measurement/front end → geometry → fusion backend → robustness → evidence → project implementation → validation.
3. Create a source ledger before writing claims. For every quantitative result record the paper, scenario, sensor setup, metric, value, and whether it is measured, simulated, or a proposed project target.
4. Separate three claims clearly:
   - paper-reported results;
   - conclusions inferred from several papers;
   - targets proposed for the user's project.
5. Select only evidence that advances the argument. Prefer editable native tables and simple diagrams; use source figures only when their axes, legends, panel labels, and captions remain readable.
6. Build the PPTX with `@oai/artifact-tool`. Use the supplied visual reference first; otherwise use `assets/blue-academic-reference.pptx` as the style reference.
7. Render every slide at full size. Fix all unintended overlaps, clipped text, broken connectors, weak contrast, small evidence, and excessive empty space. Run `slides_test.py` and the `nature-paper2ppt` audit; finish with high=0 and medium=0.
8. Keep only the requested final deliverable in the output folder unless the user asks for QA files.

## Content rules

- Emphasize algorithms, positioning mechanisms, experimental conditions, accuracy, failure cases, and implications. Do not discuss code line counts, interface trivia, or generic platform details unless requested.
- For UWB/GNSS/IMU topics, read [references/uwb-gnss-imu-guidance.md](references/uwb-gnss-imu-guidance.md).
- For slide language, styling, typography, and layout constraints, read [references/deck-standard.md](references/deck-standard.md).
- Preserve full paper titles in a reference slide. Summarize each paper by introduction, technology/method, reported result, and impact on the current research.
- Do not claim that indoor GNSS produces centimeter accuracy unless a source demonstrates it. Treat GNSS primarily as the outdoor/global reference and transition sensor; treat UWB+IMU/INS as the indoor precision chain.
- Prefer conclusion-style Chinese titles. Avoid repeated slogans and phrases such as “一句话总结”, “最有价值的方向”, or “不是……而是……”.

## Visual rules

- Use a white canvas, deep blue numbered headers, a thin blue rule, blue subheads, pale-blue alternating rows, and restrained red highlights for conclusions and key metrics.
- Use `Microsoft YaHei` for titles. Use `STKaiti`/华文楷体 for body text when the user requests the established style.
- Default minimums: title 35 pt, section/header 20–24 pt, body 16–20 pt, source labels 9–11 pt.
- Favor one dominant table, workflow, architecture, or evidence figure per slide. Fill the canvas with meaningful evidence; do not shrink text to manufacture whitespace.
- Create connectors before nodes. Keep lines outside text boxes and never place annotation text over a factor, node, chart, or table cell.

## Default deliverables

- Main technical deck: 8–12 slides unless the user specifies otherwise.
- Optional one-page literature synthesis: columns for paper title/introduction, technology and method, reported result, and impact on the research.
- One final `.pptx` with Chinese file naming and concise handoff notes.
