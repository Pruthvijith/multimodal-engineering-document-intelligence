# Design decision and review guide

## Decision record 001

A fixed-layout fusion model makes every field auditable and runs on CPU. It is a discriminative multimodal reader. Generative VLM adaptation, including LLaVA, remains a separate milestone. The default threshold is a design choice. Calibrate it on actual review costs and a separate validation corpus before operational use.

## Operational response

A missing field, duplicate asset string or confidence below threshold returns review. Preserve the source hash to trace a decision. A systematic increase in review rate should trigger sampling by scanner, layout and lighting condition. Add negative tests before allowing a new layout.

## Interview practice

Use these answers after reproducing the experiment. Describe the work as an AI-assisted independent portfolio project.

### What does 100% tell you?

Only that 32 images from the same generator were read correctly. Asset IDs differ across splits, but fonts, positions and visual conventions are shared. I would hold out a renderer and collect real scans.

### Why keep OCR and field validation separate?

Model confidence says little about a mistyped numeric identifier. Field patterns and range checks provide independent reasons to route the card to review.

### How would you evaluate human review?

Measure accepted-record precision and coverage as the threshold changes. Add reviewer correction time and disagreement, then choose a threshold on a separate validation cohort.

