#!/usr/bin/env python3
from __future__ import annotations

import argparse
import ast
import csv
import json
import re
from dataclasses import dataclass
from difflib import SequenceMatcher
from html import escape
from pathlib import Path
from typing import Any, Dict, List, Tuple


# -----------------------
# Tokenization + alignment
# -----------------------

def toks(s: str) -> List[str]:
    s = (s or "").strip().lower()
    return re.findall(r"[a-z0-9]+(?:'[a-z0-9]+)?", s)

def align(ref: str, hyp: str) -> List[Tuple[str, str]]:
    """
    Return list of (word, tag) for hypothesis words.
    tag in {"ok","ins","rep"}.
    - ok (green): right word, right place
    - rep (yellow): word appears in reference, but wrong place
    - ins (red): word doesn't appear in reference at all (made up)
    """
    r = toks(ref)
    h = toks(hyp)

    # Build a multiset (counter) of reference words for lookup
    ref_words = {}
    for w in r:
        ref_words[w] = ref_words.get(w, 0) + 1

    tagged: List[Tuple[str, str]] = []

    for i, hyp_word in enumerate(h):
        # Check if word is in correct position
        if i < len(r) and hyp_word == r[i]:
            tagged.append((hyp_word, "ok"))
        # Check if word appears anywhere in reference
        elif hyp_word in ref_words:
            tagged.append((hyp_word, "rep"))
        # Word doesn't appear in reference at all
        else:
            tagged.append((hyp_word, "ins"))

    return tagged


# -----------------------
# HTML rendering
# -----------------------

def hyp_html(tagged: List[Tuple[str, str]]) -> str:
    cls = {"ok": "ok", "ins": "ins", "rep": "rep"}
    parts = []
    for w, t in tagged:
        parts.append(f'<span class="{cls.get(t,"rep")}">{escape(w)}</span>')
    return " ".join(parts)

HTML_HEAD = """<!doctype html>
<html><head><meta charset="utf-8">
<style>
body{font-family:system-ui,Arial; margin:20px}
table{border-collapse:collapse; width:100%}
td,th{border:1px solid #ddd; padding:10px; vertical-align:top}
th{background:#f5f5f5; text-align:left}
.ok{color:#0a7a0a; font-weight:700}
.rep{color:#b58900; font-weight:800} /* yellow-ish */
.ins{color:#b00020; font-weight:800}
small{color:#555}
pre{white-space:pre-wrap; word-break:break-word; margin:0}
</style></head><body>
<h2>ASR Report (REF vs HYP)</h2>
<p><span class="ok">green</span>=match, <span class="rep">yellow</span>=mismatch/shift, <span class="ins">red</span>=inserted/made-up</p>
<table>
<tr><th>ID</th><th>Reference</th><th>Hypothesis (colored)</th></tr>
"""

HTML_TAIL = "</table></body></html>"


# -----------------------
# ANSI rendering (terminal colors)
# -----------------------

ANSI = {
    "reset": "\x1b[0m",
    "ok": "\x1b[32;1m",   # bright green
    "rep": "\x1b[33;1m",  # bright yellow
    "ins": "\x1b[31;1m",  # bright red
    "dim": "\x1b[2m",
}

def hyp_ansi(tagged: List[Tuple[str, str]]) -> str:
    out = []
    for w, t in tagged:
        color = ANSI.get(t, ANSI["rep"])
        out.append(f"{color}{w}{ANSI['reset']}")
    return " ".join(out)

def block_sep() -> str:
    return "-" * 96

def safe_one_line(s: str) -> str:
    s = (s or "").replace("\n", " ").strip()
    s = re.sub(r"\s+", " ", s)
    return s


# -----------------------
# Loading decode outputs
# -----------------------

@dataclass
class Rec:
    utt_id: str
    ref: str
    hypo: str
    instruction: str = ""

def _loads_json_or_py(text: str) -> Any:
    text = (text or "").strip()
    if not text:
        return None
    try:
        return json.loads(text)
    except Exception:
        return ast.literal_eval(text)

def load_records(path: Path) -> List[Rec]:
    """
    Returns list of records:
      Rec(utt_id=..., ref=..., hypo=..., instruction=...)

    Supports:
      - jsonl (dict per line)
      - hypo-*.json columnar dict: {"utt_id":[...], "ref":[...], "hypo":[...], ...}
      - hypo-*.json list of dicts
      - mapping dict: {utt_id -> "hyp"} or {utt_id -> {"ref":..,"hypo":..}}
      - python repr variants (single quotes)
    """
    raw = path.read_text(encoding="utf-8", errors="replace")
    if not raw.strip():
        return []

    lines = [ln.strip() for ln in raw.splitlines() if ln.strip()]

    # JSONL case
    if len(lines) >= 2 and all(ln.startswith("{") for ln in lines[:2]):
        out: List[Rec] = []
        for ln in lines:
            d = _loads_json_or_py(ln)
            if not isinstance(d, dict):
                continue
            uid = d.get("utt_id") or d.get("id") or ""
            if not uid:
                continue
            ref = d.get("ref") or ""
            hyp = d.get("hypo") or d.get("hyp") or d.get("text") or ""
            inst = d.get("instruction") or ""
            out.append(Rec(str(uid), str(ref), str(hyp), str(inst)))
        return out

    obj = _loads_json_or_py(raw)

    # List-of-dicts case
    if isinstance(obj, list):
        out: List[Rec] = []
        for d in obj:
            if not isinstance(d, dict):
                continue
            uid = d.get("utt_id") or d.get("id") or ""
            if not uid:
                continue
            ref = d.get("ref") or ""
            hyp = d.get("hypo") or d.get("hyp") or d.get("text") or ""
            inst = d.get("instruction") or ""
            out.append(Rec(str(uid), str(ref), str(hyp), str(inst)))
        return out

    # Dict case: handle BOTH {uid->...} and columnar {"utt_id":[...],...}
    if isinstance(obj, dict):
        # Columnar dict detection: utt_id is a list
        if isinstance(obj.get("utt_id"), list):
            uids = obj.get("utt_id") or []
            refs = obj.get("ref") or []
            hyps = obj.get("hypo") or obj.get("hyp") or obj.get("text") or []
            inst = obj.get("instruction") or []

            n = len(uids)

            def get(arr, i) -> str:
                if isinstance(arr, list) and i < len(arr):
                    return "" if arr[i] is None else str(arr[i])
                return ""

            out: List[Rec] = []
            for i in range(n):
                uid = get(uids, i).strip()
                if not uid:
                    continue
                out.append(Rec(
                    utt_id=uid,
                    ref=get(refs, i),
                    hypo=get(hyps, i),
                    instruction=get(inst, i),
                ))
            return out

        # Mapping dict: {uid -> ...}
        out: List[Rec] = []
        for k, v in obj.items():
            uid = str(k).strip()
            if not uid:
                continue
            if isinstance(v, dict):
                ref = v.get("ref") or ""
                hyp = v.get("merged_hypo") or v.get("hypo") or v.get("hyp") or v.get("text") or ""
                inst = v.get("instruction") or ""
                out.append(Rec(uid, str(ref), str(hyp), str(inst)))
            else:
                out.append(Rec(uid, "", str(v), ""))
        return out

    return []


# -----------------------
# Main
# -----------------------

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--jsonl", required=True, help="decode outputs (.jsonl OR hypo-*.json)")
    ap.add_argument("--out_dir", required=True)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    recs = load_records(Path(args.jsonl))
    if not recs:
        print(f"[WARN] No records loaded from {args.jsonl}")
        return

    # Build outputs
    rows_csv = []
    html_rows = []
    txt_blocks = []
    ansi_blocks = []

    for r in recs:
        ref = r.ref or ""
        hyp = r.hypo or ""
        tagged = align(ref, hyp)

        rows_csv.append((r.utt_id, ref, hyp, " ".join([f"{w}:{t}" for w, t in tagged])))

        html_rows.append(
            f"<tr><td><b>{escape(r.utt_id)}</b></td>"
            f"<td><pre>{escape(ref)}</pre></td>"
            f"<td><pre>{hyp_html(tagged)}</pre></td></tr>"
        )

        # Plain text block
        txt_blocks.append(
            f"{r.utt_id}\n"
            f"REF: {ref if ref.strip() else '(no ref available)'}\n"
            f"HYP: {hyp if hyp.strip() else '(no hyp available)'}\n"
            f"{block_sep()}"
        )

        # ANSI block (colored hypothesis)
        ansi_blocks.append(
            f"{r.utt_id}\n"
            f"{ANSI['dim']}REF:{ANSI['reset']} {ref if ref.strip() else '(no ref available)'}\n"
            f"{ANSI['dim']}HYP:{ANSI['reset']} {hyp_ansi(tagged) if hyp.strip() else '(no hyp available)'}\n"
            f"{ANSI['dim']}{block_sep()}{ANSI['reset']}"
        )

    # Write CSV
    with open(out_dir / "report.csv", "w", newline="", encoding="utf-8") as cf:
        w = csv.writer(cf)
        w.writerow(["utt_id", "ref", "hyp", "hyp_tagged"])
        w.writerows(rows_csv)

    # Write HTML
    (out_dir / "report.html").write_text(
        HTML_HEAD + "\n".join(html_rows) + HTML_TAIL,
        encoding="utf-8"
    )

    # Write plain txt
    (out_dir / "report.txt").write_text(
        "\n".join(txt_blocks) + "\n(END)\n",
        encoding="utf-8"
    )

    # Write ANSI txt
    (out_dir / "report.ansi.txt").write_text(
        "\n".join(ansi_blocks) + f"\n{ANSI['dim']}(END){ANSI['reset']}\n",
        encoding="utf-8"
    )

    print("Wrote:", out_dir / "report.csv")
    print("Wrote:", out_dir / "report.html")
    print("Wrote:", out_dir / "report.ansi.txt")
    print("Wrote:", out_dir / "report.txt")
    print("Tip: view ANSI with: less -R report.ansi.txt")


if __name__ == "__main__":
    main()