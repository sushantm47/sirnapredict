"""How the model was built, validated and scored."""

import pandas as pd
import streamlit as st

from api_client import ApiError, model_info

st.set_page_config(page_title="Model Info", page_icon="📊", layout="wide")
st.write("# About the model 📊")


@st.cache_data(ttl=60)
def load_info():
    return model_info()


try:
    info = load_info()
except ApiError as e:
    st.error(f"Could not load model info: {e}")
    st.stop()

d = info["data"]
st.caption(f"Best model: **{info['best_model']}** · version {info['version']} · "
           f"trained {info['trained_at']} on {d['sirnas']:,} siRNAs across {d['targets']} genes")

st.info(f"**Validation: {info['validation']}.** siRNAs against the same gene share sequence "
        "context, so a random split would let the model see near-duplicates of its own test "
        "data. Every score below comes from genes the model never saw during training.")

scores = pd.DataFrame(info["cv_scores"]).T
names = {"rules": "Reynolds rules (baseline)", "ridge": "Ridge regression",
         "forest": "Gradient-boosted trees"}
scores.index = [names.get(i, i) for i in scores.index]

best = info["cv_scores"][info["best_model"]]
baseline = info["cv_scores"].get("rules", {})
c1, c2, c3 = st.columns(3)
c1.metric("Pearson r", f"{best['pearson_r']:.3f}",
          delta=(f"{best['pearson_r'] - baseline['pearson_r']:+.3f} vs. rules"
                 if baseline else None))
c2.metric("Mean absolute error", f"{best['mae']:.3f}",
          help="Average error in predicted knockdown (0-1)")
c3.metric("AUC", f"{best['auc']:.3f}" if best.get("auc") else "n/a",
          help=f"Separating active (≥{info['active_threshold']:.0%} knockdown) from inactive")

st.subheader("All models, same folds")
st.dataframe(scores.style.format("{:.3f}", na_rep="n/a"))
st.caption("The rules baseline is the published checklist alone, scored as 0-7. Its correlation "
           "is meaningful; its error is not on the same scale, so ignore its MAE.")

left, right = st.columns(2)
with left:
    st.subheader("What the model pays attention to")
    top = pd.DataFrame(info["top_features"]).set_index("feature")["weight"]
    st.bar_chart(top.head(12), horizontal=True, height=380)
    st.caption("`end_asymmetry` is the difference in stability between the guide's two ends; "
               "`dg_*` are nearest-neighbour free energies; `posN_B` means base B at position N.")
with right:
    st.subheader("Performance per target gene")
    per_target = pd.DataFrame(info["per_target"])
    if per_target.empty:
        st.caption("Not enough siRNAs per gene to report this.")
    else:
        st.dataframe(per_target, hide_index=True, height=380)
        st.caption("Correlation between predicted and measured knockdown, gene by gene. "
                   "Spread here is normal: some targets are much harder than others.")
