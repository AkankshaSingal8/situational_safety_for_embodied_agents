"""Build the rollout-viewer page from render_v2.py's manifest and analyze_v2.py's results.
    python make_artifact_v2.py --art runs_v2/artifact --results runs_v2/results_v2.json
writes <art>/index.html (videos are referenced as clips/*.mp4 next to it)
"""
import argparse
import json
import os

ap = argparse.ArgumentParser()
ap.add_argument("--art", default="runs_v2/artifact")
ap.add_argument("--results", default="runs_v2/results_v2.json")
a = ap.parse_args()

M = json.load(open(os.path.join(a.art, "manifest.json")))
base_paths = {(m["split"], m["scene"]): m["path"] for m in M if m["group"] == "base"}
for m in M:
    m["base_path"] = base_paths.get((m["split"], m["scene"]))
R = json.load(open(a.results)) if os.path.exists(a.results) else {}


def pick(split, *keys):
    x = R.get(split)
    for k in keys:
        if x is None:
            return None
        x = x.get(k)
    return None if x is None else x["k"] / x["n"]


ROWS = [
    ("Base trajectory: passes over the laptop", ("summary", "base", "viol_A")),
    ("Base trajectory: tilts the mug past 15°", ("summary", "base", "viol_B")),
    ("A text: laptop avoided, task done", ("summary", "main|A", "A_ok&succ")),
    ("A text: tilt left alone (still over 15°)", ("summary", "main|A", "viol_B")),
    ("B text: mug upright, task done", ("summary", "main|B", "B_ok&succ")),
    ("B text: path left alone (still over laptop)", ("summary", "main|B", "viol_A")),
    ("No-text model: laptop avoided, task done", ("summary", "notext", "A_ok&succ")),
    ("Unseen laptop paraphrases, set 1 (8)", ("exp2_heldout_A",)),
    ("Unseen laptop paraphrases, set 2 (8)", ("exp2_heldout2_A",)),
    ("Unseen upright paraphrases, set 1 (8)", ("exp2_heldout_B",)),
    ("Unseen upright paraphrases, set 2 (8)", ("exp2_heldout2_B",)),
    ("Combined sentence, no combined training", ("exp3", "main|combined", "AB_ok&succ")),
    ("Combined, no training: clauses in turn A→B", ("exp3", "stack_AB|combined", "AB_ok&succ")),
    ("Combined sentence, with combined training", ("exp3", "main_ab|combined", "AB_ok&succ")),
    ("Far-worded combined sentence, no combined training", ("exp3", "main|far", "AB_ok&succ")),
    ("Far-worded combined, no training: clauses in turn A→B", ("exp3", "stack_AB|far", "AB_ok&succ")),
    ("Far-worded combined sentence, with combined training", ("exp3", "main_ab|far", "AB_ok&succ")),
]
table = [dict(label=l, train=pick("train", *k), test=pick("test", *k)) for l, k in ROWS]

GROUPS = [
    ("base", "Base"), ("A", "Constraint A"), ("B", "Constraint B"),
    ("ab_direct", "A+B, no training"), ("ab_stack", "A+B, clauses in turn"), ("ab_trained", "A+B, trained"),
    ("ab_trained_far", "A+B, trained, far wording"),
]
html = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "artifact_template.html")).read()
html = html.replace("<!--__REPORT__-->", open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "report_section.html")).read())
def md_to_html(md, h_offset=1):
    """Minimal Markdown for the generated result files: headings, pipe tables, bullet lists, paragraphs.
    Confidence intervals like '0.985 [0.957,0.995]' get a muted span; long text cells wrap."""
    import html as H
    import re
    out, lines, i = [], md.splitlines(), 0
    ci = re.compile(r"^(\S+) (\[[^\]]+\])$")

    def cell(c, tag):
        c = c.strip()
        m = ci.match(c)
        num = bool(re.match(r"^-?[\d.]+(e-?\d+)?( \[|$|/| ±|, )", c)) or c in ("—",)
        body = f"{H.escape(m.group(1))} <span class=ci>{H.escape(m.group(2))}</span>" if m else H.escape(c)
        cls = "num" if num else ("text" if len(c) > 40 else "")
        return f"<{tag}{' class=' + cls if cls else ''}>{body}</{tag}>"
    while i < len(lines):
        ln = lines[i]
        if ln.startswith("#"):
            k = len(ln) - len(ln.lstrip("#"))
            out.append(f"<h{min(k + h_offset, 4)}>{H.escape(ln[k:].strip())}</h{min(k + h_offset, 4)}>")
        elif ln.startswith("|"):
            rows = []
            while i < len(lines) and lines[i].startswith("|"):
                rows.append([c for c in lines[i].strip().strip("|").split("|")])
                i += 1
            head, body = rows[0], [r for r in rows[1:] if not set("".join(r)) <= set("-: ")]
            out.append('<div class="tablewrap"><table><thead><tr>' + "".join(cell(c, "th") for c in head) + "</tr></thead><tbody>"
                       + "".join("<tr>" + "".join(cell(c, "td") for c in r) + "</tr>" for r in body) + "</tbody></table></div>")
            continue
        elif ln.startswith("- "):
            items = []
            while i < len(lines) and lines[i].startswith("- "):
                items.append(f"<li>{H.escape(lines[i][2:])}</li>"); i += 1
            out.append("<ul>" + "".join(items) + "</ul>")
            continue
        elif ln.strip():
            out.append(f"<p>{H.escape(ln.strip())}</p>")
        i += 1
    return "\n".join(out)


def md_rich(md, h_offset=0):
    """Markdown for hand-written READMEs: headings, pipe tables (also indented under a bullet), nested
    bullet and numbered lists with continuation lines, paragraphs, and inline `code`, **bold**, *italic*."""
    import html as H
    import re
    LIST = re.compile(r"^(\s*)(- |\d+\. )(.*)$")

    def inline(t):
        t = H.escape(t, quote=False)
        t = re.sub(r"`([^`]+)`", r"<code>\1</code>", t)
        t = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", t)
        t = re.sub(r"(?<![\w*])\*([^*\s][^*]*?)\*(?![\w*])", r"<em>\1</em>", t)
        return t

    def cell(c, tag):
        c = c.strip()
        bare = c.replace("**", "")
        num = bool(re.match(r"^\(?-?[\d.]+(e-?\d+)?\)?( \(|$|/| ±|, )", bare)) or bare in ("—", "n/a")
        return f"<{tag}{' class=num' if num else ''}>{inline(c)}</{tag}>"

    def indent(l):
        return len(l) - len(l.lstrip(" "))

    lines, out, i = md.splitlines(), [], 0
    n = len(lines)
    while i < n:
        ln, st = lines[i], lines[i].strip()
        if not st:
            i += 1
        elif ln.startswith("#"):
            k = len(ln) - len(ln.lstrip("#"))
            lv = min(k + h_offset, 4)
            out.append(f"<h{lv}>{inline(ln[k:].strip())}</h{lv}>")
            i += 1
        elif st.startswith("|"):
            rows = []
            while i < n and lines[i].strip().startswith("|"):
                rows.append(lines[i].strip().strip("|").split("|"))
                i += 1
            head, body = rows[0], [r for r in rows[1:] if not set("".join(r)) <= set("-: ")]
            tds = ["".join(cell(c, "td") for c in r) for r in body]
            numcol = [sum(cell(r[k], "td").startswith("<td class=num") for r in body if k < len(r) and r[k].strip())
                      > 0.5 * max(1, sum(1 for r in body if k < len(r) and r[k].strip())) for k in range(len(head))]
            ths = "".join(cell(c, "th").replace("<th>", "<th class=num>", 1) if numcol[k] else cell(c, "th")
                          for k, c in enumerate(head))
            out.append('<div class="tablewrap"><table><thead><tr>' + ths + "</tr></thead><tbody>"
                       + "".join("<tr>" + t + "</tr>" for t in tds) + "</tbody></table></div>")
        elif LIST.match(ln):
            base = indent(ln)
            ordered = LIST.match(ln).group(2)[0].isdigit()
            items = []
            while i < n:
                m = LIST.match(lines[i])
                if m and indent(lines[i]) == base:
                    text = [m.group(3)]
                    i += 1
                    while (i < n and lines[i].strip() and indent(lines[i]) > base and not LIST.match(lines[i])
                           and not lines[i].strip().startswith("|")):
                        text.append(lines[i].strip()); i += 1
                    body = []
                    while i < n:
                        l2 = lines[i]
                        if l2.strip() and indent(l2) > base:
                            body.append(l2); i += 1
                        elif not l2.strip():
                            j = i
                            while j < n and not lines[j].strip():
                                j += 1
                            if j < n and indent(lines[j]) > base:
                                body.extend(lines[i:j]); i = j
                            else:
                                break
                        else:
                            break
                    d = min((indent(b) for b in body if b.strip()), default=0)
                    inner = md_rich("\n".join(b[d:] for b in body), h_offset) if body else ""
                    items.append(f"<li>{inline(' '.join(text))}{inner}</li>")
                else:
                    j = i
                    while j < n and not lines[j].strip():
                        j += 1
                    if j < n and j > i and LIST.match(lines[j]) and indent(lines[j]) == base:
                        i = j; continue
                    break
            tag = "ol" if ordered else "ul"
            out.append(f"<{tag}>" + "".join(items) + f"</{tag}>")
        else:
            para = []
            while (i < n and lines[i].strip() and not lines[i].startswith("#")
                   and not lines[i].strip().startswith("|") and not LIST.match(lines[i])):
                para.append(lines[i].strip()); i += 1
            out.append(f"<p>{inline(' '.join(para))}</p>")
    return "\n".join(out)


HERE = os.path.dirname(os.path.abspath(__file__))
intro = open(os.path.join(HERE, "all_results_intro.html")).read()
res_md = open(os.path.join(HERE, "runs_v2", "results_v2.md")).read().replace("# Residual-flow POC v2 results", "", 1)
seed_md = open(os.path.join(HERE, "runs_v2", "seeds", "seeds.md")).read()
html = html.replace("<!--__ALLRESULTS__-->", intro + '\n<section id="seeds-all">' + md_to_html(seed_md)
                    + '</section>\n<section id="seed0-all">' + md_to_html("# Seed 0, every condition\n" + res_md) + "</section>")
sl_md = open(os.path.join(HERE, "..", "safelibero_residual", "README.md")).read()
sl_md = sl_md[sl_md.index("## Decisions"):sl_md.index("## Files")]
html = html.replace("<!--__SAFELIBERO__-->", open(os.path.join(HERE, "safelibero_intro.html")).read()
                    + '\n<section id="sl-readme"><h2>Full write-up</h2><p>Generated from <code>experiments/safelibero_residual/README.md</code>: decisions, protocol, pre-registered bars and every result in the order it was produced.</p>\n'
                    + md_rich(sl_md, h_offset=1) + "</section>")
html = html.replace("/*__DATA__*/", "const CLIPS = " + json.dumps(M, separators=(",", ":")) + ";\n"
                    "const TABLE = " + json.dumps(table) + ";\n"
                    "const GROUPS = " + json.dumps(GROUPS) + ";")
open(os.path.join(a.art, "index.html"), "w").write(html)
print("wrote", os.path.join(a.art, "index.html"), len(M), "clips", f"{len(html) / 1e6:.2f} MB")
