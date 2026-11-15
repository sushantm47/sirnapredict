"""
siRNAPredict - main page: rank siRNA candidates against a target transcript.

Run from the project root:  streamlit run app/Dashboard.py
The API must be running (default http://localhost:8000, override with API_URL).
"""

import pandas as pd
import streamlit as st
from streamlit.logger import get_logger

from api_client import API_URL, ApiError, design, design_profile, health
from helpers import candidates_frame, parse_fasta

LOGGER = get_logger(__name__)

EXAMPLE_MRNA = (
    "AUGGUGAGCAAGGGCGAGGAGCUGUUCACCGGGGUGGUGCCCAUCCUGGUCGAGCUGGACGGCGACGUAAACGGC"
    "CACAAGUUCAGCGUGUCCGGCGAGGGCGAGGGCGAUGCCACCUACGGCAAGCUGACCCUGAAGUUCAUCUGCACC"
    "ACCGGCAAGCUGCCCGUGCCCUGGCCCACCCUCGUGACCACCCUGACCUACGGCGUGCAGUGCUUCAGCCGCUAC"
    "CCCGACCACAUGAAGCAGCACGACUUCUUCAAGUCCGCCAUGCCCGAAGGCUACGUCCAGGAGCGCACCAUCUUC"
)

st.set_page_config(page_title="siRNAPredict", page_icon="🧬", layout="wide")


def backend_status():
    status = health()
    if not status["online"]:
        st.error("Backend offline 😱")
        st.caption(f"Start it with `uvicorn src.api:app --reload` ({API_URL})")
    elif status.get("model_loaded"):
        st.success(f"Backend online ✅ · model {status.get('version', '')}")
    else:
        st.warning("Backend online, but no model is loaded. Run `python -m src.train`.")
    return status


# ----------------------------------------------------------------- sidebar

with st.sidebar:
    status = backend_status()

    st.header("1 · Pick a target")
    source = st.radio("Where the sequence comes from",
                      ["Example", "Paste sequence", "Upload FASTA"], key="source")

    mrna, label = None, None
    if source == "Example":
        mrna, label = EXAMPLE_MRNA, "Example transcript (300 nt)"
        st.caption("A short demo transcript, so you can try the tool straight away.")
    elif source == "Upload FASTA":
        uploaded = st.file_uploader("mRNA or CDS file", type=["fa", "fasta", "txt"], key="fasta")
        if uploaded:
            try:
                header, mrna = parse_fasta(uploaded.getvalue().decode("utf-8", errors="replace"))
                label = header or uploaded.name
                st.caption(f"{label} · {len(mrna):,} nt")
            except (ValueError, UnicodeDecodeError) as e:
                st.error(f"Could not read the file: {e}")
    else:
        pasted = st.text_area("Sequence (A/C/G/U, or T for DNA)", height=120, key="pasted",
                              placeholder="AUGGUGAGCAAGGGCGAGGAG...")
        if pasted.strip():
            try:
                _, mrna = parse_fasta(pasted)
                label = "Pasted sequence"
                st.caption(f"{len(mrna):,} nt")
            except ValueError as e:
                st.error(str(e))

    st.header("2 · Run it")
    run_clicked = st.button("Design siRNAs", type="primary", key="design", disabled=mrna is None)
    if mrna is None:
        st.caption("Choose or paste a sequence first.")

    with st.expander("Options (sensible defaults set)"):
        top_n = st.slider("How many candidates to show", 5, 50, 10, 5, key="top_n")
        min_gc, max_gc = st.slider("Allowed GC content (%)", 0, 100, (30, 52), key="gc")
        st.caption("Published rules favour 30–52% GC. Widen this if a GC-rich target "
                   "returns too few candidates.")
        avoid_runs = st.checkbox("Skip sequences with 4+ identical bases in a row",
                                 value=True, key="runs")

# ----------------------------------------------------------------- main

st.write("# siRNAPredict 🧬")
st.write("**Which siRNA should you order to silence a gene?** Give this tool the gene's mRNA "
         "sequence. It checks every possible siRNA against it and ranks them by how much "
         "knockdown the model expects.")

with st.expander("How to use this page", expanded=True):
    st.markdown(
        "1. **Pick a target** in the sidebar — start with **Example** if you just want to see it work.\n"
        "2. Click **Design siRNAs**.\n"
        "3. Read the table: the top row is the model's best guess. Each row is one siRNA you "
        "could order.\n\n"
        "**What the columns mean**\n\n"
        "| Column | Meaning |\n"
        "| --- | --- |\n"
        "| `position` | Where on the gene this siRNA binds (nucleotide number) |\n"
        "| `guide` | The siRNA sequence to order — this is the strand that does the silencing |\n"
        "| `target_site` | The matching stretch of the gene it binds to |\n"
        "| `predicted knockdown` | How much of the gene the model expects to be switched off. "
        "70%+ is usually called a hit |\n"
        "| `GC` | Fraction of G and C bases. Mid-range works best |\n"
        "| `rules passed` | How many of the 7 published design rules it satisfies |\n\n"
        "The **Check siRNA** page scores sequences you already have, and **Model Info** shows "
        "how well the model actually performs.")

if run_clicked and mrna:
    try:
        with st.spinner(f"Checking {len(mrna) - 18:,} possible siRNAs..."):
            st.session_state["design_result"] = design(mrna, top_n, min_gc / 100, max_gc / 100,
                                                       avoid_runs)
            st.session_state["design_profile"] = design_profile(mrna)
            st.session_state["design_label"] = label
    except ApiError as e:
        LOGGER.error(e)
        for key in ("design_result", "design_profile"):
            st.session_state.pop(key, None)
        st.error(str(e))

result = st.session_state.get("design_result")
if result:
    st.divider()
    st.subheader(st.session_state.get("design_label") or "Results")
    c1, c2, c3 = st.columns(3)
    c1.metric("Gene length", f"{result['transcript_length']:,} nt")
    c2.metric("siRNAs checked", f"{result['windows_scanned']:,}")
    c3.metric("Passed your filters", f"{result['windows_passing_filters']:,}")

    if not result["candidates"]:
        st.warning("No siRNA passed the filters. Open **Options** in the sidebar and widen the "
                   "GC range — GC-rich genes often need this.")
    else:
        table = candidates_frame(result["candidates"])
        best = table.iloc[0]
        st.success(f"**Best candidate: position {int(best['position'])}** · order this guide: "
                   f"`{best['guide']}` · {best['predicted_knockdown_percent']}% predicted knockdown "
                   f"· {int(best['rules_passed'])}/7 design rules passed")

        st.dataframe(
            table.drop(columns=["predicted_knockdown"]), hide_index=True,
            column_config={
                "predicted_knockdown_percent": st.column_config.ProgressColumn(
                    "predicted knockdown", min_value=0, max_value=100, format="%.1f%%"),
                "rules_passed": st.column_config.NumberColumn("rules passed", format="%d/7"),
                "gc_percent": st.column_config.NumberColumn("GC", format="%.1f%%"),
            })
        st.download_button("Download this list (CSV)", table.to_csv(index=False),
                           file_name="sirna_candidates.csv", mime="text/csv")

        profile = st.session_state.get("design_profile")
        if profile:
            st.subheader("Where the good targets sit on the gene")
            st.caption("Every possible siRNA, in order along the gene. Peaks are promising "
                       "regions; dips are regions to avoid.")
            chart = pd.DataFrame({"predicted knockdown": profile["predicted_knockdown"]},
                                 index=profile["positions"])
            chart.index.name = "position on the gene (nt)"
            st.line_chart(chart, height=260)
elif status["online"]:
    st.info("👈 Pick **Example** in the sidebar and click **Design siRNAs** to try it.")

st.divider()
st.caption("Research and educational use only. Predictions are a ranking aid learned from published "
           "knockdown experiments — they narrow down what to test, and do not replace testing.")
