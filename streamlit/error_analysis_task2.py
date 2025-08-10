import os
import sys
import json
import hashlib
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from PIL import Image
from PIL import ImageDraw

# =========================
# Page Setup
# =========================
st.set_page_config(
    page_title="VLSP 2025 - Task 2 Error Analysis",
    page_icon="🧪",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# Minimal CSS tokens and styles for readable UI
st.markdown(
        """
<style>
:root {
    --color-bg: #0B1220;
    --color-text: #E5E7EB;
    --color-surface: #0F172A;
    --color-surface-subtle: #F8FAFC;
    --color-border: #1F2937;
    --color-border-light: #E5E7EB;
    --color-primary: #3B82F6;
    --color-success: #10B981;
    --color-danger: #EF4444;
    --color-warning: #F59E0B;
    --radius: 10px;
}

/***** Header *****/
.main-header {
    background: linear-gradient(90deg, #2563EB 0%, #7C3AED 100%);
    padding: 1.25rem 1.25rem;
    border-radius: var(--radius);
    margin-bottom: 1rem;
    color: white;
}
.main-header h1 { margin: 0; }
.main-header p { margin: 0.25rem 0 0; opacity: 0.9; }

/***** Cards (light boxes for question/answer) *****/
.card {
    background: #FFFFFF;
    color: #111827;
    border: 1px solid var(--color-border-light);
    border-radius: var(--radius);
    padding: 1rem;
}

/***** Choice styles *****/
.choice-normal { background: #FFFFFF; border: 1px solid var(--color-border-light); border-left: 4px solid #E5E7EB; color: #111827; border-radius: 8px; padding: 10px; }
.choice-predicted { background: #FEF3C7; border: 1px solid var(--color-warning); border-left: 4px solid var(--color-warning); color: #111827; border-radius: 8px; padding: 10px; }
.choice-correct { background: #DCFCE7; border: 1px solid var(--color-success); border-left: 4px solid var(--color-success); color: #052e16; border-radius: 8px; padding: 10px; }
.choice-wrong { background: #FEE2E2; border: 1px solid var(--color-danger); border-left: 4px solid var(--color-danger); color: #7f1d1d; border-radius: 8px; padding: 10px; }

/***** Buttons *****/
.stButton > button {
    background: var(--color-primary) !important;
    color: #fff !important;
    border: 0 !important;
    border-radius: 8px !important;
    font-weight: 600 !important;
    padding: 0.5rem 0.9rem !important;
}
.stButton > button:hover { background: #1D4ED8 !important; }
.stButton > button:active { background: #1E40AF !important; }
.stButton > button:disabled { opacity: .6 !important; }

/* Small margins for annotation radios */
[data-testid="stRadio"] {
    margin-top: 8px !important;
    margin-left: 8px !important;
}

/* Chips */
.chip { display: inline-block; padding: 6px 10px; border-radius: 999px; font-weight: 600; font-size: .9rem; margin-right: 8px; border: 1px solid rgba(0,0,0,.05); }
.chip-pred { background: #FEF3C7; color: #92400E; border-color: #F59E0B; }
.chip-gt { background: #DCFCE7; color: #166534; border-color: #16A34A; }

</style>
""",
        unsafe_allow_html=True,
)

# =========================
# Defaults and Paths
# =========================
PROJECT_ROOT = Path(__file__).resolve().parents[1]
# Ensure repository root is in sys.path so 'src' imports work in Streamlit
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))
DEFAULTS = {
    "test_file": PROJECT_ROOT / "data/public_test/vlsp_2025_public_test_task2.json",
    "image_root": PROJECT_ROOT / "data/public_test/public_test_images",
    "pred_files": [
        PROJECT_ROOT / "task2_test_predictions.json",
        PROJECT_ROOT / "task2_train_predictions.json",
        PROJECT_ROOT / "task2_test_submission.json",
        PROJECT_ROOT / "submission_task2.json",
    ],
    "ground_truth": PROJECT_ROOT / "ground_truth_task2.json",
}

# Directory to store persisted detection results
DETECT_CACHE_DIR = PROJECT_ROOT / "data/processed/detections"


# =========================
# Helpers
# =========================

def load_json_safely(path: Path) -> Optional[Any]:
    try:
        if path.exists():
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
    except Exception as e:
        st.warning(f"Failed to load {path.name}: {e}")
    return None


def try_load_predictions() -> Tuple[List[Dict], str]:
    """Try different known prediction formats and return a unified list of dicts.
    Unified keys: id, predicted_answer
    """
    for p in DEFAULTS["pred_files"]:
        data = load_json_safely(p)
        if data is None:
            continue
        src = p.name
        # Case 1: {"results": [...]} with predicted_answer
        if isinstance(data, dict) and isinstance(data.get("results"), list):
            rows = []
            for item in data["results"]:
                if not isinstance(item, dict):
                    continue
                rid = item.get("id") or item.get("question_id")
                pred = item.get("predicted_answer") or item.get("answer") or item.get("prediction")
                if rid is None:
                    continue
                rows.append({"id": str(rid), "predicted_answer": pred, **item})
            if rows:
                return rows, src
        # Case 2: plain list of dicts
        if isinstance(data, list):
            rows = []
            for item in data:
                if not isinstance(item, dict):
                    continue
                rid = item.get("id") or item.get("question_id")
                pred = item.get("predicted_answer") or item.get("answer") or item.get("prediction")
                if rid is None:
                    continue
                rows.append({"id": str(rid), "predicted_answer": pred, **item})
            if rows:
                return rows, src
    return [], ""


def load_test_data(path: Path) -> List[Dict]:
    data = load_json_safely(path)
    return data if isinstance(data, list) else []


def load_ground_truth(path: Path) -> Dict[str, str]:
    data = load_json_safely(path)
    return data if isinstance(data, dict) else {}


def load_law_db() -> List[Dict]:
    """Load law database from common locations.
    Tries data/law_db/vlsp2025_law.json then data/processed/law_db/vlsp2025_law.json
    """
    candidates = [
        PROJECT_ROOT / "data/law_db/vlsp2025_law.json",
        PROJECT_ROOT / "data/processed/law_db/vlsp2025_law.json",
    ]
    for p in candidates:
        data = load_json_safely(Path(p))
        if isinstance(data, list) and data:
            return data
    return []


def get_article_text(law_db: List[Dict], law_id: str, article_id: str) -> str:
    for law in law_db or []:
        if law.get("id") == law_id:
            for article in law.get("articles", []):
                if article.get("id") == article_id:
                    title = article.get("title", "")
                    text = article.get("text", "")
                    return f"**{title}**\n\n{text}" if title else text
    return "⚠️ Không tìm thấy điều luật."


# ---------- Detection persistent cache helpers ----------
def _detector_env_key() -> str:
    api = os.environ.get("ROBOFLOW_API_KEY", "")
    name = os.environ.get("ROBOFLOW_MODEL_NAME", "")
    ver = os.environ.get("ROBOFLOW_MODEL_VERSION", "")
    return f"{name}:{ver}:{'set' if api else 'unset'}"


def _det_cache_path(img_path: str) -> Path:
    # Key by image path + detector env signature
    sig = f"{img_path}|{_detector_env_key()}|v1"
    safe = hashlib.sha1(sig.encode("utf-8")).hexdigest()  # stable across runs
    # include image basename to help humans
    stem = Path(img_path).stem
    return DETECT_CACHE_DIR / f"{stem}__{safe}.json"


def load_detection_cache(img_path: str) -> Optional[List[Dict]]:
    try:
        p = _det_cache_path(img_path)
        if p.exists():
            with open(p, "r", encoding="utf-8") as f:
                data = json.load(f)
                return data if isinstance(data, list) else []
    except Exception:
        pass
    return None


def save_detection_cache(img_path: str, preds: List[Dict]) -> None:
    try:
        DETECT_CACHE_DIR.mkdir(parents=True, exist_ok=True)
        p = _det_cache_path(img_path)
        with open(p, "w", encoding="utf-8") as f:
            json.dump(preds, f, ensure_ascii=False, indent=2)
    except Exception:
        pass

def save_ground_truth(path: Path, gt: Dict[str, str]) -> None:
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(gt, f, ensure_ascii=False, indent=2)
    except Exception as e:
        st.error(f"Failed to save ground truth: {e}")


def normalize_type(qtype: str) -> str:
    ql = (qtype or "").lower()
    if any(k in ql for k in ["mc", "multiple", "choice"]):
        return "mc"
    return "yn"


def question_length(q: str) -> int:
    return len((q or "").strip())


def compute_metrics(rows: List[Dict], gt: Dict[str, str]) -> Dict[str, Any]:
    total = len(rows)
    annotated = 0
    correct = 0
    mc_total = mc_correct = 0
    yn_total = yn_correct = 0

    for r in rows:
        rid = r.get("id")
        qtype = normalize_type(r.get("question_type") or r.get("type") or "")
        pred = r.get("predicted_answer")
        g = gt.get(str(rid))
        if g is None:
            continue
        annotated += 1
        ok = False
        if qtype == "mc":
            mc_total += 1
            ok = (pred == g)
            mc_correct += 1 if ok else 0
        else:
            yn_total += 1
            ok = (str(pred).strip().lower() == str(g).strip().lower())
            yn_correct += 1 if ok else 0
        correct += 1 if ok else 0

    overall_acc = (correct / annotated * 100.0) if annotated else 0.0
    mc_acc = (mc_correct / mc_total * 100.0) if mc_total else 0.0
    yn_acc = (yn_correct / yn_total * 100.0) if yn_total else 0.0

    return {
        "total": total,
        "annotated": annotated,
        "correct": correct,
        "overall_acc": overall_acc,
        "mc_total": mc_total,
        "mc_correct": mc_correct,
        "mc_acc": mc_acc,
        "yn_total": yn_total,
        "yn_correct": yn_correct,
        "yn_acc": yn_acc,
    }


def build_mc_confusion(rows: List[Dict], gt: Dict[str, str]) -> Tuple[List[List[int]], List[str]]:
    labels = ["A", "B", "C", "D"]
    idx = {l: i for i, l in enumerate(labels)}
    mat = [[0] * 4 for _ in range(4)]
    for r in rows:
        qtype = normalize_type(r.get("question_type") or r.get("type") or "")
        if qtype != "mc":
            continue
        rid = str(r.get("id"))
        pred = r.get("predicted_answer")
        g = gt.get(rid)
        if not g or pred not in idx or g not in idx:
            continue
        mat[idx[g]][idx[pred]] += 1
    return mat, labels


def enrich_rows(base_rows: List[Dict], test_data: List[Dict], image_root: Path) -> List[Dict]:
    # map test by id
    tmap: Dict[str, Dict] = {}
    for t in test_data:
        tid = str(t.get("id"))
        if tid:
            tmap[tid] = t
    out = []
    for r in base_rows:
        rid = str(r.get("id"))
        t = tmap.get(rid, {})
        qtext = t.get("question") or r.get("question") or ""
        qtype_raw = t.get("question_type") or r.get("question_type") or r.get("type") or ""
        qtype = normalize_type(qtype_raw)
        choices = t.get("choices") or r.get("options") or {}
        img_id = t.get("image_id") or r.get("image_id")
        rel_articles = t.get("relevant_articles") or r.get("relevant_articles") or []
        img_path = None
        if img_id:
            # Try common extensions
            for ext in [".jpg", ".png", ".jpeg"]:
                p = image_root / f"{img_id}{ext}"
                if p.exists():
                    img_path = str(p)
                    break
        out.append({
            **r,
            "question": qtext,
            "question_type": qtype,
            "question_type_raw": qtype_raw,
            "choices": choices if isinstance(choices, dict) else {},
            "image_path": img_path,
            "image_id": img_id,
            "relevant_articles": rel_articles if isinstance(rel_articles, list) else [],
        })
    return out


# =========================
# Data Loading (Sidebar)
# =========================
with st.sidebar:
    st.markdown("""
    <div class="main-header" style="padding: .75rem 1rem; border-radius: 8px;">
      <h2 style="font-size: 1.1rem;">🧪 Task 2 Error Analysis</h2>
      <p style="font-size:.9rem;opacity:.9;">Focus on analyzing wrong cases and patterns</p>
    </div>
    """, unsafe_allow_html=True)
    # File uploaders (kept as requested)
    uploaded_pred = st.file_uploader("Predictions JSON", type=["json"], key="pred_up")
    uploaded_gt = st.file_uploader("Ground Truth JSON", type=["json"], key="gt_up")
    uploaded_test = st.file_uploader("Test Data JSON", type=["json"], key="test_up")

# Load base data, preferring uploaded files when available
if uploaded_pred is not None:
    try:
        data = json.load(uploaded_pred)
        if isinstance(data, dict) and isinstance(data.get("results"), list):
            base_rows = [{"id": str(d.get("id")), "predicted_answer": d.get("predicted_answer") or d.get("answer") or d.get("prediction"), **d} for d in data["results"] if isinstance(d, dict)]
        elif isinstance(data, list):
            base_rows = [{"id": str(d.get("id")), "predicted_answer": d.get("predicted_answer") or d.get("answer") or d.get("prediction"), **d} for d in data if isinstance(d, dict)]
        else:
            base_rows, src_pred = try_load_predictions()
            if not base_rows:
                st.error("Unsupported predictions format.")
                st.stop()
    except Exception as e:
        st.error(f"Failed to read uploaded predictions: {e}")
        st.stop()
else:
    base_rows, src_pred = try_load_predictions()

# Test data
if uploaded_test is not None:
    try:
        test_data = json.load(uploaded_test)
        test_data = test_data if isinstance(test_data, list) else []
    except Exception as e:
        st.error(f"Failed to read uploaded test data: {e}")
        st.stop()
else:
    test_data = load_test_data(Path(DEFAULTS["test_file"]))

# Ground truth
if uploaded_gt is not None:
    try:
        gt_data = json.load(uploaded_gt)
        gt_data = gt_data if isinstance(gt_data, dict) else {}
    except Exception as e:
        st.error(f"Failed to read uploaded ground truth: {e}")
        st.stop()
else:
    gt_data = load_ground_truth(Path(DEFAULTS["ground_truth"]))

# Inform user if no predictions found and no upload yet
if not base_rows:
    st.warning("Chưa có dữ liệu dự đoán. Hãy upload file Predictions JSON ở sidebar để tiếp tục.")

# Enrich rows with question text, type, choices, image path
rows = enrich_rows(base_rows, test_data, Path(DEFAULTS["image_root"]))
LAW_DB = load_law_db()

# =========================
# Header + KPIs
# =========================
st.markdown(
    """
<div class="main-header">
  <h1>🧪 Subtask 2 — Error Analysis</h1>
  <p>Analyze wrong predictions, patterns, and confusion across MC and Yes/No questions</p>
</div>
""",
    unsafe_allow_html=True,
)

metrics = compute_metrics(rows, gt_data)

k1, k2, k3, k4 = st.columns(4)
with k1:
    st.metric("Total Questions", metrics["total"])
with k2:
    st.metric("Annotated", metrics["annotated"], delta=f"{(metrics['annotated']/metrics['total']*100 if metrics['total'] else 0):.1f}%")
with k3:
    st.metric("Overall Accuracy", f"{metrics['overall_acc']:.1f}%", delta=f"{metrics['correct']}/{metrics['annotated']} correct")
with k4:
    st.metric("MC/Y-N Acc", f"{metrics['mc_acc']:.1f}% / {metrics['yn_acc']:.1f}%", delta=f"MC:{metrics['mc_total']} • Y/N:{metrics['yn_total']}")

st.markdown("---")

# =========================
# Error Analysis Core
# =========================

# Build filtered DataFrame for table and plots
records = []
for r in rows:
    rid = str(r.get("id"))
    qtext = r.get("question", "")
    qtype = r.get("question_type", "yn")
    pred = r.get("predicted_answer")
    gt = gt_data.get(rid)
    is_annotated = gt is not None
    is_correct = None
    if is_annotated:
        if qtype == "mc":
            is_correct = (pred == gt)
        else:
            is_correct = (str(pred).strip().lower() == str(gt).strip().lower())
    records.append({
        "id": rid,
        "type": qtype,
        "question": qtext,
        "predicted": pred,
        "ground_truth": gt,
        "annotated": is_annotated,
        "correct": is_correct,
        "q_len": question_length(qtext),
        "image_path": r.get("image_path"),
    "image_id": r.get("image_id"),
    "choices": r.get("choices", {}),
    "relevant_articles": r.get("relevant_articles", []),
    "question_type_raw": r.get("question_type_raw"),
    })

df = pd.DataFrame(records)
# Ensure expected columns exist even when empty to avoid KeyErrors
if df.empty:
    expected_cols = [
        "id", "type", "question", "predicted", "ground_truth", "annotated",
        "correct", "q_len", "image_path", "image_id", "choices",
        "relevant_articles", "question_type_raw",
    ]
    for col in expected_cols:
        if col not in df.columns:
            df[col] = pd.Series(dtype=object)

# Error patterns and MC confusion
colA, colB = st.columns([1.2, 1.0])
with colA:
    st.markdown("#### 🔍 Top Error Patterns")
    wrong = df[(df["annotated"]) & (df["correct"] == False)]
    if not wrong.empty:
        wrong["pattern"] = wrong.apply(lambda r: f"{r['predicted']} → {r['ground_truth']}", axis=1)
        top = wrong["pattern"].value_counts().head(10).reset_index()
        top.columns = ["pattern", "count"]
        fig = px.bar(top, x="count", y="pattern", orientation="h", height=380)
        fig.update_layout(template="plotly_white", margin=dict(l=10, r=10, t=10, b=10))
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.success("No wrong cases in current filter.")

with colB:
    st.markdown("#### 📊 MC Confusion Matrix")
    mat, labels = build_mc_confusion(rows, gt_data)
    if np.sum(np.array(mat)) > 0:
        fig = go.Figure(go.Heatmap(
            z=mat,
            x=[f"Pred {x}" for x in labels],
            y=[f"GT {x}" for x in labels],
            colorscale="Reds",
            text=mat,
            texttemplate="%{text}",
        ))
        fig.update_layout(template="plotly_white", height=380, margin=dict(l=10, r=10, t=10, b=10))
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Not enough MC annotations to build confusion matrix.")

st.markdown("---")

# Accuracy by question length buckets
st.markdown("#### 📏 Accuracy vs Question Length")
if metrics["annotated"] > 0:
    # Define length bins
    bins = [0, 50, 100, 150, 200, 300, 500]
    labels_bins = ["0-50", "50-100", "100-150", "150-200", "200-300", "300-500+"]
    ann = df[df["annotated"]].copy()
    ann["len_bin"] = pd.cut(ann["q_len"], bins=bins, labels=labels_bins, right=False, include_lowest=True)
    grp = ann.groupby("len_bin").apply(lambda d: (d["correct"].sum() / len(d)) * 100 if len(d) else 0).reset_index(name="accuracy")
    grp["count"] = ann.groupby("len_bin").size().values
    fig = px.bar(grp, x="len_bin", y="accuracy", text=grp.apply(lambda r: f"{r['accuracy']:.1f}% ({int(r['count'])} q)", axis=1))
    fig.update_traces(textposition="outside")
    fig.update_layout(template="plotly_white", yaxis_range=[0, 100], height=360, margin=dict(l=10, r=10, t=10, b=10))
    st.plotly_chart(fig, use_container_width=True)
else:
    st.info("No annotations yet.")

st.markdown("---")

# =========================
# Detail Review + Annotation
# =========================

st.markdown("### 🔎 Detail Review & Quick Annotation")
if not df.empty:
    ids = df["id"].tolist()
    sel_id = st.selectbox("Select Question ID", ids)
    row = df[df["id"] == sel_id].iloc[0]
    # Compute prediction and GT once
    pred = row["predicted"]
    gt = row["ground_truth"]

    c1, c2 = st.columns([1.1, 1.4])
    with c1:
        st.markdown("#### 📸 Image")
        img_p = row["image_path"]
        if img_p and os.path.exists(img_p):
            try:
                st.image(Image.open(img_p), caption=f"Image for {sel_id}", use_container_width=True)
            except Exception as e:
                st.warning(f"Failed to load image: {e}")
        else:
            st.info("No image available.")

    with c2:
        st.markdown("#### ❓ Question & Options")
        if row["type"] == "mc":
            choices = row["choices"] or {}
            g = gt
            opts = []
            for key, text in choices.items():
                css = "choice-normal"
                if pd.notna(g) and key == g and key == pred:
                    css = "choice-correct"
                elif pd.notna(g) and key == g:
                    css = "choice-correct"
                elif key == pred:
                    css = "choice-predicted"
                opts.append(f"<div class='{css}'><b>{key}.</b> {text}</div>")
            card_html = f"<div class='card'><div style='font-weight:600;margin-bottom:8px;'>{row['question']}</div>{''.join(opts)}</div>"
            st.markdown(card_html, unsafe_allow_html=True)

            # Annotate directly under options
            choice_keys = list(choices.keys())
            cur = gt if gt in choice_keys else None
            sel = st.radio(
                "Chọn đáp án đúng:",
                options=choice_keys + ["Clear"],
                index=(choice_keys.index(cur) if cur in choice_keys else len(choice_keys)),
                horizontal=True,
                key=f"ann_mc_{sel_id}",
            )
            if st.button("💾 Lưu", key=f"save_mc_{sel_id}"):
                if sel == "Clear":
                    if sel_id in gt_data:
                        del gt_data[sel_id]
                else:
                    gt_data[sel_id] = sel
                save_ground_truth(Path(DEFAULTS["ground_truth"]), gt_data)
                st.success("Đã lưu")
                st.experimental_rerun()
        else:
            yn = ["Đúng", "Sai"]
            blocks = []
            for label in yn:
                css = "choice-normal"
                if pd.notna(gt) and label == gt and label == pred:
                    css = "choice-correct"
                elif pd.notna(gt) and label == gt:
                    css = "choice-correct"
                elif label == pred:
                    css = "choice-predicted"
                blocks.append(f"<div class='{css}' style='text-align:center;'><b>{label}</b></div>")
            card_html = f"<div class='card'><div style='font-weight:600;margin-bottom:8px;'>{row['question']}</div><div style='display:grid;grid-template-columns:1fr 1fr;gap:8px;'>{''.join(blocks)}</div></div>"
            st.markdown(card_html, unsafe_allow_html=True)

            # Annotate directly under options
            cur = gt if gt in yn else None
            sel = st.radio(
                "Chọn đáp án đúng:",
                options=yn + ["Clear"],
                index=(yn.index(cur) if cur in yn else len(yn)),
                horizontal=True,
                key=f"ann_yn_{sel_id}",
            )
            if st.button("💾 Lưu", key=f"save_yn_{sel_id}"):
                if sel == "Clear":
                    if sel_id in gt_data:
                        del gt_data[sel_id]
                else:
                    gt_data[sel_id] = sel
                save_ground_truth(Path(DEFAULTS["ground_truth"]), gt_data)
                st.success("Đã lưu")
                st.experimental_rerun()

    # (Annotation controls moved above under options)
    # Export submission (minimal) inside Detail Review section
    st.markdown("")
    st.markdown("#### 📤 Export Submission")
    try:
        # Build expected list format
        def to_qtype_label(row_type: str, raw: str) -> str:
            # Prefer raw label if present, else map normalized type
            if pd.notna(raw) and str(raw).strip():
                return str(raw)
            return "Multiple choice" if row_type == "mc" else "Yes/No"

        sub_list = []
        for r in records:
            sub_list.append({
                "id": r["id"],
                "image_id": r.get("image_id"),
                "question": r.get("question"),
                "relevant_articles": r.get("relevant_articles", []),
                "answer": ("" if pd.isna(r["predicted"]) else r["predicted"]),
                "question_type": to_qtype_label(r.get("type"), r.get("question_type_raw")),
            })

        sub_json = json.dumps(sub_list, ensure_ascii=False, indent=2).encode("utf-8")
        st.download_button(
            label="Download task2 submission (list format)",
            data=sub_json,
            file_name=f"submission_task2_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            mime="application/json",
            use_container_width=True,
        )
    except Exception as e:
        st.error(f"Không thể tạo submission: {e}")

else:
    st.info("No items in current filter.")

st.markdown("---")

# =========================
# Sample Inspector (Components View)
# =========================

st.markdown("## 🔬 Sample Inspector — Component View")
st.caption("Xem ảnh, câu hỏi, dự đoán, điều luật trích dẫn và chạy detection cho từng mẫu")

# Inspector filters
fc1, fc2, fc3 = st.columns([1, 1, 2])
with fc1:
    ins_type = st.selectbox("Loại câu hỏi", ["All", "mc", "yn"], key="ins_type")
with fc2:
    ins_status = st.selectbox("Trạng thái", ["All", "Correct", "Wrong", "Unannotated"], key="ins_status")
with fc3:
    ins_q_search = st.text_input("Tìm trong câu hỏi", key="ins_q_search")

fc4, fc5, fc6 = st.columns([1, 1, 1])
with fc4:
    pred_values = sorted({str(x) for x in df["predicted"].dropna().unique().tolist()})
    ins_pred = st.selectbox("Dự đoán", ["All"] + pred_values, key="ins_pred")
# Collect law/article ids
law_ids = sorted({a.get("law_id") for r in rows for a in (r.get("relevant_articles") or []) if isinstance(a, dict) and a.get("law_id")})
art_ids = sorted({a.get("article_id") for r in rows for a in (r.get("relevant_articles") or []) if isinstance(a, dict) and a.get("article_id")})
with fc5:
    ins_law = st.selectbox("Law ID", ["All"] + law_ids, key="ins_law")
with fc6:
    ins_article = st.selectbox("Article ID", ["All"] + art_ids, key="ins_article")

# Apply filters for inspector
f_ins = df.copy()
if ins_type != "All":
    f_ins = f_ins[f_ins["type"] == ins_type]
if ins_status != "All":
    if ins_status == "Correct":
        f_ins = f_ins[f_ins["annotated"] & f_ins["correct"]]
    elif ins_status == "Wrong":
        f_ins = f_ins[f_ins["annotated"] & (~f_ins["correct"])]
    else:
        f_ins = f_ins[~f_ins["annotated"]]
if ins_q_search:
    f_ins = f_ins[f_ins["question"].astype(str).str.contains(ins_q_search, case=False, na=False)]
if ins_pred != "All":
    f_ins = f_ins[f_ins["predicted"].astype(str) == ins_pred]
if ins_law != "All":
    f_ins = f_ins[f_ins["relevant_articles"].apply(lambda arr: any((isinstance(x, dict) and x.get("law_id") == ins_law) for x in (arr or [])))]
if ins_article != "All":
    f_ins = f_ins[f_ins["relevant_articles"].apply(lambda arr: any((isinstance(x, dict) and x.get("article_id") == ins_article) for x in (arr or [])))]

ins_col1, ins_col2 = st.columns([1.2, 1.8])
with ins_col1:
    ids_all = f_ins["id"].tolist()
    if ids_all:
        sel_ins_id = st.selectbox("Chọn Question ID", ids_all, key="ins_sel_id")
        row_ins = df[df["id"] == sel_ins_id].iloc[0]
        img_p = row_ins.get("image_path")
        st.markdown("#### 📸 Ảnh gốc")
        if img_p and os.path.exists(img_p):
            try:
                st.image(Image.open(img_p), caption=f"Image for {sel_ins_id}", use_container_width=True)
            except Exception as e:
                st.warning(f"Không thể mở ảnh: {e}")
        else:
            st.info("Không có ảnh.")

        # Optional detection overlay
        st.markdown("#### 🧩 Detection")
        det_run = st.button("Chạy detection (Roboflow)", key=f"det_run_{sel_ins_id}")
        if det_run:
            try:
                if not img_p or not os.path.exists(img_p):
                    st.warning("Không tìm thấy ảnh để detect.")
                else:
                    # 1) Try load from disk cache
                    cache_path = _det_cache_path(img_p)
                    preds = load_detection_cache(img_p)
                    used_cache = False
                    if not preds:
                        # 2) If not cached, run detector and save
                        if not (os.environ.get("ROBOFLOW_API_KEY") and os.environ.get("ROBOFLOW_MODEL_NAME") and os.environ.get("ROBOFLOW_MODEL_VERSION")):
                            st.warning("Thiếu ROBOFLOW_API_KEY / MODEL_NAME / MODEL_VERSION trong biến môi trường.")
                        else:
                            from src.services.detection_service.detector_pipeline import DetectorPipeline
                            pipeline = DetectorPipeline()
                            preds = pipeline.detect(img_p)
                            save_detection_cache(img_p, preds or [])
                            st.caption(f"Đã lưu cache: {cache_path}")
                    else:
                        used_cache = True
                        st.caption(f"Đang dùng cache: {cache_path}")

                    # Draw overlay if we have predictions list
                    if isinstance(preds, list):
                        try:
                            im = Image.open(img_p).convert("RGB")
                            draw = ImageDraw.Draw(im)
                            for pred in preds:
                                x, y = pred.get("x", 0), pred.get("y", 0)
                                w, h = pred.get("width", 0), pred.get("height", 0)
                                cls = pred.get("class", "")
                                x0, y0 = x - w/2, y - h/2
                                x1, y1 = x + w/2, y + h/2
                                draw.rectangle([x0, y0, x1, y1], outline="#ef4444", width=3)
                                if cls:
                                    draw.text((x0, max(0, y0-12)), cls, fill="#ef4444")
                            cap_note = " (cache)" if used_cache else ""
                            st.image(im, caption=f"Detection Overlay{cap_note} ({len(preds)} boxes)", use_container_width=True)
                        except Exception as e:
                            st.warning(f"Không thể vẽ bounding boxes: {e}")

                        # Show raw predictions (top 10)
                        try:
                            df_preds = pd.DataFrame(preds)
                            if not df_preds.empty:
                                st.dataframe(df_preds.head(10), use_container_width=True)
                            else:
                                st.info("Không có prediction từ detector.")
                        except Exception:
                            pass
            except Exception as e:
                st.error(f"Detection pipeline lỗi: {e}")
    else:
        st.info("Không có dữ liệu để chọn.")

with ins_col2:
    if ids_all:
        st.markdown("#### ❓ Câu hỏi & Tuỳ chọn")
        row_ins = df[df["id"] == sel_ins_id].iloc[0]
        q = row_ins["question"]
        typ = row_ins["type"]
        pred = row_ins["predicted"]
        gt = row_ins["ground_truth"]
        if typ == "mc":
            choices = row_ins.get("choices", {})
            opts = []
            for key, text in choices.items():
                css = "choice-normal"
                if pd.notna(gt) and key == gt and key == pred:
                    css = "choice-correct"
                elif pd.notna(gt) and key == gt:
                    css = "choice-correct"
                elif key == pred:
                    css = "choice-predicted"
                opts.append(f"<div class='{css}'><b>{key}.</b> {text}</div>")
            st.markdown(f"<div class='card'><div style='font-weight:600;margin-bottom:8px;'>{q}</div>{''.join(opts)}</div>", unsafe_allow_html=True)
        else:
            yn_labels = ["Đúng", "Sai"]
            blocks = []
            for label in yn_labels:
                css = "choice-normal"
                if pd.notna(gt) and label == gt and label == pred:
                    css = "choice-correct"
                elif pd.notna(gt) and label == gt:
                    css = "choice-correct"
                elif label == pred:
                    css = "choice-predicted"
                blocks.append(f"<div class='{css}' style='text-align:center;'><b>{label}</b></div>")
            st.markdown(f"<div class='card'><div style='font-weight:600;margin-bottom:8px;'>{q}</div><div style='display:grid;grid-template-columns:1fr 1fr;gap:8px;'>{''.join(blocks)}</div></div>", unsafe_allow_html=True)

        chip_pred = f"<span class='chip chip-pred'>AI Pred: {pred}</span>" if pd.notna(pred) else ""
        chip_gt = f"<span class='chip chip-gt'>GT: {gt}</span>" if pd.notna(gt) else ""
        st.markdown(f"{chip_pred} {chip_gt}", unsafe_allow_html=True)

        st.markdown("#### 📚 Điều luật trích dẫn")
        # Pull relevant_articles from enriched rows matching id
        rels = next((r.get("relevant_articles", []) for r in rows if str(r.get("id")) == sel_ins_id), [])
        if rels:
            for art in rels:
                if isinstance(art, dict) and "law_id" in art and "article_id" in art:
                    label = f"{art['law_id']} — Điều {art['article_id']}"
                    with st.expander(label, expanded=False):
                        content = get_article_text(LAW_DB, art["law_id"], art["article_id"])
                        st.markdown(content)
        else:
            st.info("Không có điều luật liên quan trong dữ liệu.")
# Removed legacy export buttons per request
