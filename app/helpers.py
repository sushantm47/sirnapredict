"""Pure helpers for the dashboard (no Streamlit, so they are easy to test)."""

import pandas as pd

BASES = set("ACGU")
GUIDE_LENGTH = 19
MAX_UPLOAD_BASES = 20_000


def parse_fasta(text):
    """
    Read FASTA or plain sequence text and return (header, sequence).

    Only the first record is used; headers and blank lines are ignored elsewhere.
    """
    if not text or not text.strip():
        raise ValueError("The file is empty")
    header, lines = None, []
    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            continue
        if line.startswith(">"):
            if lines:          # a second record: stop at the first
                break
            header = line[1:].strip() or None
            continue
        lines.append(line)
    sequence = "".join(lines).upper().replace("T", "U")
    if not sequence:
        raise ValueError("No sequence found in the file")
    bad = sorted(set(sequence) - BASES)
    if bad:
        raise ValueError(f"Sequence contains invalid letters: {', '.join(bad)}")
    if len(sequence) < GUIDE_LENGTH:
        raise ValueError(f"Sequence must be at least {GUIDE_LENGTH} nucleotides long")
    if len(sequence) > MAX_UPLOAD_BASES:
        raise ValueError(f"Sequence must be at most {MAX_UPLOAD_BASES:,} nucleotides long")
    return header, sequence


def parse_guide_list(text):
    """Read guides from pasted text: one per line, commas, or a FASTA-style list."""
    guides, errors = [], []
    for i, raw in enumerate(text.replace(",", "\n").splitlines(), start=1):
        line = raw.strip()
        if not line or line.startswith(">"):
            continue
        seq = line.upper().replace("T", "U")
        if set(seq) - BASES:
            errors.append(f"line {i}: invalid letters")
        elif len(seq) != GUIDE_LENGTH:
            errors.append(f"line {i}: {len(seq)} nt, expected {GUIDE_LENGTH}")
        else:
            guides.append(seq)
    return guides, errors


def candidates_frame(candidates):
    """Turn the API's candidate list into a table for display and download."""
    if not candidates:
        return pd.DataFrame()
    df = pd.DataFrame(candidates)
    df["gc_percent"] = (df.pop("gc_content") * 100).round(1)
    df["predicted_knockdown_percent"] = (df["predicted_knockdown"] * 100).round(1)
    return df[["rank", "position", "guide", "target_site", "predicted_knockdown",
               "predicted_knockdown_percent", "gc_percent", "end_asymmetry", "rules_passed"]]


def predictions_frame(predictions):
    """Flatten /predict/batch output into a table."""
    rows = []
    for p in predictions:
        rows.append({
            "guide": p["guide"],
            "predicted_knockdown": p["predicted_knockdown"],
            "call": p["call"],
            "gc_percent": round(p["properties"]["gc_content"] * 100, 1),
            "end_asymmetry": p["properties"]["end_asymmetry"],
            "dg_total": p["properties"]["dg_total"],
            "rules_passed": p["rule_check"]["passed"],
        })
    return pd.DataFrame(rows)


def rule_label(name):
    """Readable text for a Reynolds rule key."""
    return {
        "gc_30_52": "GC content between 30% and 52%",
        "a_at_sense_19": "A at sense position 19",
        "a_at_sense_3": "A at sense position 3",
        "u_at_sense_10": "U at sense position 10",
        "no_gc_at_sense_19": "No G or C at sense position 19",
        "no_g_at_sense_13": "No G at sense position 13",
        "au_rich_guide_5p": "AU-rich guide 5' end",
    }.get(name, name.replace("_", " "))


def api_error_message(status_code, body):
    """Turn a FastAPI error response into one readable line."""
    detail = body.get("detail") if isinstance(body, dict) else None
    if isinstance(detail, list):
        parts = []
        for d in detail:
            field = d.get("loc", [""])[-1]
            msg = d.get("msg", "invalid value")
            parts.append(f"{field}: {msg}" if field and field != "body" else msg)
        return "; ".join(parts)
    if isinstance(detail, str):
        return detail
    return f"Request failed with status {status_code}"
