"""Score siRNAs you already have, one at a time or as a list."""

import pandas as pd
import streamlit as st

from api_client import ApiError, health, predict, predict_batch
from helpers import parse_guide_list, predictions_frame, rule_label

st.set_page_config(page_title="Check siRNA", page_icon="🔬", layout="wide")

st.write("# Check an siRNA 🔬")
st.caption("Score guides you already have, for example candidates from a supplier or a paper.")

with st.sidebar:
    if health()["online"]:
        st.success("Backend online ✅")
    else:
        st.error("Backend offline 😱")

single, batch = st.tabs(["One guide", "Several guides"])

with single:
    guide = st.text_input("Guide (antisense) strand, 5'->3', 19 nt",
                          value="UUGAUAUGUUGGAUGAUGG", key="guide").strip()
    if st.button("Score it", type="primary", key="score_one", disabled=not guide):
        try:
            st.session_state["one"] = predict(guide)
        except ApiError as e:
            st.session_state.pop("one", None)
            st.error(str(e))

    result = st.session_state.get("one")
    if result:
        left, right = st.columns([1, 1])
        with left:
            st.metric("Predicted knockdown", f"{result['predicted_knockdown']:.1%}",
                      help="Fraction of target mRNA expected to be silenced")
            call = result["call"]
            (st.success if call == "active" else st.warning)(
                f"Called **{call}** at the usual 70% threshold")
            props = result["properties"]
            st.write(pd.DataFrame({
                "value": [f"{props['gc_content']:.1%}", f"{props['dg_total']:.2f} kcal/mol",
                          f"{props['end_asymmetry']:+.2f} kcal/mol"]},
                index=["GC content", "Duplex free energy", "End asymmetry (5' - 3')"]))
            st.caption("Positive end asymmetry means the guide's 5' end is less stable, "
                       "which favours it being loaded into RISC.")
        with right:
            st.write(f"**Design rules: {result['rule_check']['passed']}/"
                     f"{result['rule_check']['total']} passed**")
            for name, ok in result["rule_check"]["rules"].items():
                st.write(("✅ " if ok else "❌ ") + rule_label(name))
            st.caption("Reynolds et al. (2004) checklist.")
            st.code(f"guide (antisense) 5'-{result['guide']}-3'\n"
                    f"sense             5'-{result['sense']}-3'", language=None)

with batch:
    text = st.text_area("One guide per line", height=160, key="guides",
                        placeholder="UUGAUAUGUUGGAUGAUGG\nGCAUCAUCCAACAUAUCAA")
    guides, errors = parse_guide_list(text) if text.strip() else ([], [])
    if errors:
        st.warning("Skipped: " + "; ".join(errors[:5])
                   + (f" (+{len(errors) - 5} more)" if len(errors) > 5 else ""))
    if guides and st.button(f"Score {len(guides)} guides", type="primary", key="score_many"):
        try:
            st.session_state["many"] = predictions_frame(predict_batch(guides))
        except ApiError as e:
            st.session_state.pop("many", None)
            st.error(str(e))

    table = st.session_state.get("many")
    if table is not None and not table.empty:
        ranked = table.sort_values("predicted_knockdown", ascending=False)
        shown = ranked.assign(predicted_knockdown=(ranked["predicted_knockdown"] * 100).round(1))
        st.dataframe(shown, hide_index=True,
                     column_config={"predicted_knockdown": st.column_config.ProgressColumn(
                         "predicted knockdown", min_value=0.0, max_value=100.0, format="%.1f%%")})
        st.download_button("Download (CSV)", ranked.to_csv(index=False),
                           file_name="sirna_scores.csv", mime="text/csv")
