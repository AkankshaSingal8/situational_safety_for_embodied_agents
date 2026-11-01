# Safety-skill adapter architecture

**21 September 2026 · Visualization of the proposed v2.0 architecture, not experimental results.**

Open [architecture.html](architecture.html) locally. It is self-contained, uses no network services, and includes three views:

- **Inference:** persistent skills, current grounding, frozen policy, shared flow adapter, raw L0 execution, optional L1 checker, and independent actual-state scoring.
- **Training:** checked demonstrations, held-out data boundary, flow-matching supervision, and gradients only into the adapter.
- **Grounding:** an unchanged requirement becomes episode-specific bindings, geometry and conditioning tokens.

Select components to inspect inputs, outputs and scope. The inference route control highlights raw versus shielded execution; it does not simulate robot performance. Keyboard users can switch tabs with arrow keys and inspect components with Enter/Space. On narrow screens the diagram scrolls horizontally.

Vector exports are editable and suitable for figures/slides:

| View | SVG | PDF | PNG |
|---|---|---|---|
| Inference | [runtime.svg](runtime.svg) | [runtime.pdf](runtime.pdf) | [runtime.png](runtime.png) |
| Training | [training.svg](training.svg) | [training.pdf](training.pdf) | [training.png](training.png) |
| Grounding | [grounding.svg](grounding.svg) | [grounding.pdf](grounding.pdf) | [grounding.png](grounding.png) |

The SVG exports show both inference routes; the HTML can highlight one. Solid arrows carry data/actions; dashed amber arrows mark optional shield routing, dashed purple arrows mark optimizer updates, and dashed blue/green arrows mark generation/observation feedback. Color does not indicate measured safety or performance.

The source of truth is the [v2.0 protocol](../../2026-09-18-language-conditioned-safety-adapter-poc-protocol.md). The [September 19 proof-of-life plan](../../2026-09-19-safety-memory-proof-of-life-protocol.md) is a distinct smaller experiment using historical facts and fixed rules; it is not merged into this adapter architecture.

Regenerate with:

```bash
python docs/design/visualizations/safety-skill-adapter/build.py
```

The renderer requires CairoSVG for PNG/PDF exports and Pillow for text fitting. `build.py` defines the vector layouts and component metadata; `page.template.html` supplies the interactive wrapper. No model code or weights are changed by rendering these figures.
