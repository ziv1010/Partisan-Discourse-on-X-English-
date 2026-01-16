#!/usr/bin/env python3
"""
Re-embed the 500-sample with a multilingual encoder, recompute cosine, and emit UMAP coords.

Outputs:
  - min-check/umap_before_after_coords_multilingual.csv
  - Prints cosine summary stats.

Then run:
  MPLBACKEND=Agg MPLCONFIGDIR=/tmp ./pdxvenv/bin/python analyze_drift.py \
    --coords min-check/umap_before_after_coords_multilingual.csv \
    --thr 0.9 \
    --out-prefix min-check/umap_lowcos_multilingual

  MPLBACKEND=Agg MPLCONFIGDIR=/tmp ./pdxvenv/bin/python plot_umap_drift_with_legend.py \
    --coords min-check/umap_before_after_coords_multilingual.csv \
    --out min-check/umap_drift_arrows_with_legend_multilingual.png
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import umap
from sentence_transformers import SentenceTransformer


def summarize_cos(sims: np.ndarray, label: str) -> None:
    qs = np.quantile(sims, [0, 0.10, 0.25, 0.50, 0.75, 0.90, 1.0])
    print(f"[cosine] {label}: min={qs[0]:.4f} p10={qs[1]:.4f} p25={qs[2]:.4f} median={qs[3]:.4f} p75={qs[4]:.4f} p90={qs[5]:.4f} max={qs[6]:.4f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="min-check/eval_500_en.csv", help="CSV with original and translated text")
    ap.add_argument("--text-col", default="tweet", help="Original text column")
    ap.add_argument("--en-col", default="en_text", help="Translated text column")
    ap.add_argument("--out", default="min-check/umap_before_after_coords_multilingual.csv", help="Output coords CSV")
    ap.add_argument("--model", default="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2", help="SentenceTransformer model id")
    args = ap.parse_args()

    df = pd.read_csv(args.input)
    if args.text_col not in df.columns or args.en_col not in df.columns:
        raise SystemExit(f"Missing required columns. Available: {list(df.columns)}")

    texts = df[args.text_col].astype(str).fillna("").tolist()
    en_texts = df[args.en_col].astype(str).fillna("").tolist()

    model = SentenceTransformer(args.model)
    orig_emb = model.encode(texts, convert_to_numpy=True, normalize_embeddings=True, batch_size=32, show_progress_bar=True)
    en_emb = model.encode(en_texts, convert_to_numpy=True, normalize_embeddings=True, batch_size=32, show_progress_bar=True)

    sims = np.sum(orig_emb * en_emb, axis=1)
    summarize_cos(sims, "overall")
    if "script" in df.columns:
        for script in df["script"].dropna().unique():
            mask = df["script"] == script
            summarize_cos(sims[mask.values], label=f"script={script}")

    # UMAP on concatenated embeddings
    combined = np.vstack([orig_emb, en_emb])
    reducer = umap.UMAP(
        n_neighbors=15,
        min_dist=0.1,
        metric="cosine",
        random_state=42,
    )
    coords = reducer.fit_transform(combined)
    n = len(df)
    before = coords[:n]
    after = coords[n:]

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_df = pd.DataFrame(
        {
            "orig_index": df["orig_index"] if "orig_index" in df.columns else df.index,
            args.text_col: df[args.text_col],
            args.en_col: df[args.en_col],
            "script": df["script"] if "script" in df.columns else "",
            "cosine_before_after": sims,
            "umap_x_before": before[:, 0],
            "umap_y_before": before[:, 1],
            "umap_x_after": after[:, 0],
            "umap_y_after": after[:, 1],
        }
    )
    out_df.to_csv(out_path, index=False)
    print(f"[OK] wrote {out_path}")


if __name__ == "__main__":
    main()
