# Reusable Safety Skills: Current Proof-of-Concept Protocol

**Updated 18 September 2026.** The project now trains a shared language-conditioned flow adapter on top of a frozen pretrained robot policy. Geometric checking/repair is the teacher, comparator, and optional fallback; the primary test measures the learned adapter before repair.

- [Read the complete v2.0 implementation and experimental protocol](2026-09-18-language-conditioned-safety-adapter-poc-protocol.md).
- [Printable v2.0 document](2026-09-18-language-conditioned-safety-adapter-poc-protocol.html).
- [Current configuration specification](2026-09-18-language-conditioned-safety-adapter-poc-config.yaml).
- [Archived training-free v1.1 protocol and teacher/evaluator procedures](2026-09-17-reusable-safety-skills-poc-protocol-v1.1.md).
- [Archived v1.1 configuration](2026-09-17-reusable-safety-skills-poc-config-v1.1.yaml).

The current protocol specifies architecture, grounding, skill format, data collection, training, ablations, held-out tests, raw/shielded comparisons, uncertainty reporting, and PSC cost measurement. No adapter training or robot evaluation has been performed as part of this documentation update.
