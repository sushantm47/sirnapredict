"""Explore the measured knockdown data the model was trained on."""

from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[2]
TRAINED = ROOT / "model" / "training_data.csv"       # whatever the model was last fitted on
SYNTHETIC = ROOT / "data" / "synthetic_demo.csv"      # committed fallback
DATA = TRAINED if TRAINED.exists() else SYNTHETIC

st.set_page_config(page_title="Assay Data", page_icon="📈", layout="wide")
st.write("# Assay data 📈")
if not DATA.exists():
    st.info("No dataset found. Run `python -m scripts.prepare_data data/synthetic_demo.csv` "
            "followed by `python -m src.train`.")
    st.stop()

if DATA == SYNTHETIC:
    st.warning("Showing the **synthetic** demo dataset — these efficacies are simulated from "
               "published design rules, not measured. Train on a real dataset to see real "
               "measurements here (see the README).")
else:
    st.caption("The knockdown data the current model was fitted on.")


@st.cache_data
def load():
    return pd.read_csv(DATA)


df = load()
threshold = st.slider("Active threshold (% knockdown)", 50, 95, 70, 5) / 100
df["active"] = df["efficacy"] >= threshold

c1, c2, c3, c4 = st.columns(4)
c1.metric("siRNAs measured", f"{len(df):,}")
c2.metric("Target genes", df["target"].nunique())
c3.metric("Mean knockdown", f"{df['efficacy'].mean():.1%}")
c4.metric(f"Active at ≥{threshold:.0%}", f"{df['active'].mean():.1%}")

left, right = st.columns(2)
with left:
    st.subheader("Distribution of measured knockdown")
    counts = pd.cut(df["efficacy"], bins=[i / 10 for i in range(11)]).value_counts().sort_index()
    counts.index = [f"{int(i.left * 100)}-{int(i.right * 100)}%" for i in counts.index]
    st.bar_chart(counts.rename("siRNAs"))
    st.caption("Most siRNAs are mediocre: a minority reach the active threshold. "
               "This is why ranking candidates before synthesis is worth doing.")
with right:
    st.subheader("Mean knockdown by target gene")
    by_gene = (df.groupby("target")
               .agg(mean_knockdown=("efficacy", "mean"), siRNAs=("efficacy", "size"))
               .sort_values("mean_knockdown", ascending=False))
    st.bar_chart(by_gene["mean_knockdown"], horizontal=True, height=400)
    st.caption("Genes differ a lot, which is why the model is validated on genes it "
               "has never seen.")

st.subheader("Browse")
genes = st.multiselect("Filter by target gene", sorted(df["target"].unique()))
shown = df[df["target"].isin(genes)] if genes else df
shown = shown.sort_values("efficacy", ascending=False).assign(
    knockdown_percent=lambda d: (d["efficacy"] * 100).round(1)).drop(columns=["efficacy"])
st.dataframe(shown, hide_index=True,
             column_config={"knockdown_percent": st.column_config.ProgressColumn(
                 "measured knockdown", min_value=0.0, max_value=100.0, format="%.1f%%")})
