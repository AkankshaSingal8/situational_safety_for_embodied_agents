# Author's style rules (extracted from their own draft prose, verbatim source)

1. No em-dash asides. Their draft: zero. Use commas, semicolons, or new sentences.
2. No meta-commentary about the writing itself. Banned: "worth saying out loud",
   "stated plainly", "not throat-clearing", "the diagnosis is instructive",
   "which is awkward for", "we could not construct a sharper illustration".
3. No aphorisms or punchy fragments in body prose. "Knowing is not avoiding" stays
   in the title and the teaser caption only.
4. Sparse italics. Their draft italicizes one thing: a quoted instruction.
   Use \textit{} for quoted instructions and defined terms on first use, not for emphasis.
5. Careful enumeration in complete sentences, serial and parallel.
6. Formal connectives: "In principle", "Moreover", "However", "Such examples motivate",
   "Consider a robot asked to", "Rather than".
7. Third person, measured. Claims are stated and then supported, not asserted with force.
8. Structure markers (\textbf{} leads in Related Work / Experiments) are structure, not
   voice. They stay.

## Verification
- grep -c ' --- ' main.tex   (target: 0)
- numeric diff against main.tex.prerewrite (no number may change)
- zero overfull, zero undefined refs
