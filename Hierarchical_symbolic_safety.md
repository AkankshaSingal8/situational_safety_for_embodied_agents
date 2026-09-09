## Feature: Hierarchical symbolic safety logic for Franka manipulation scenes

Analyze the safety taxonomy PDF, then map safety Levels 1–5 to the most appropriate symbolic logic representation for each level. The output should include the selected logic type, the safety information it captures, and a one-line reason explaining why that logic is useful for that level.

**Scope:**
- IN: Robot observation image(s), task instruction, candidate objects, safety taxonomy PDF/scenarios, Franka arm tabletop manipulation setting
- OUT: Table of Level-wise mapping from safety Levels 1–5 to symbolic logic types, with predicates/formulas and one-line justification
- OUT: Logic should be useful for downstream safety reasoning, predicate verification, and possible CBF/STL-based enforcement
- NOT included: Code, Low-level controller implementation, CBF-QP code, robot trajectory generation, or full simulation code


**Constraints:**
- The setting is a Franka arm manipulator interacting in a tabletop environment
- Use safety Levels 1–5 from the given safety taxonomy PDF
- For each level, choose the symbolic logic that is most useful for that level
- Give exactly one line of reasoning per level explaining why that symbolic logic is useful
- Prefer logic forms that can support downstream safety checking or enforcement
- Do not overcomplicate Level 1; it should be grounded in geometry/collision predicates
- Do not output vague safety descriptions without formal symbolic structure
- Do not include Level 6 unless explicitly requested

