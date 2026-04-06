"""
demo_app.py — Movie Recommendation System Demo Dashboard
用途：專題展示用，可互動選擇使用者、推薦方法、NLP Prompt 即時看結果。
啟動：python -m streamlit run demo_app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# 頁面基本設定
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="🎬 電影推薦系統 Demo",
    layout="wide",
    page_icon="🎬",
)

st.title("🎬 電影推薦系統 Demo Dashboard")
st.caption("MovieLens 100K + TMDB + LightGBM LambdaRank · Week 3~6 整合展示")
st.markdown("---")

# ─────────────────────────────────────────────
# 1. 載入模型與資料（快取，只跑一次）
# ─────────────────────────────────────────────
@st.cache_data(show_spinner="⏳ 模型訓練與特徵工程中，請稍候…")
def load_pipeline():
    from movie_lgb_recommender import run_recommender_pipeline
    test_users, top_10_df, test_ground_truth, movies_df, unseen_candidates = run_recommender_pipeline()
    return sorted([int(u) for u in test_users]), top_10_df, test_ground_truth, movies_df, unseen_candidates

test_users, top_10_df, test_ground_truth, movies_df, unseen_candidates = load_pipeline()

# ─────────────────────────────────────────────
# 2. 側邊欄：輸入控制項
# ─────────────────────────────────────────────
with st.sidebar:
    st.header("🕹️ 控制面板")
    st.markdown("---")

    # User ID
    user_id = st.number_input(
        "👤 User ID",
        min_value=int(min(test_users)),
        max_value=int(max(test_users)),
        value=int(test_users[0]),
        step=1,
    )
    if user_id not in test_users:
        st.warning("此 ID 不在 Test Set 中，請重新輸入。")

    st.markdown("---")

    # 推薦方法
    method = st.selectbox(
        "🔧 推薦方法",
        options=["Baseline", "MMR", "Pareto", "Pareto + NLP"],
        index=0,
    )
    
    # MMR Lambda（只在 MMR 時顯示）
    lambda_val = 0.5
    if method == "MMR":
        lambda_val = st.slider("λ (Relevance ↔ Diversity 權衡)", 0.0, 1.0, 0.5, 0.25)
        st.caption("λ=1.0 → 完全看分數；λ=0.0 → 最大化多樣性")
    
    st.markdown("---")

    # NLP Prompt（只在 Pareto + NLP 時顯示）
    nlp_prompt = ""
    if method == "Pareto + NLP":
        nlp_prompt = st.text_input(
            "💬 NLP Prompt",
            value="推薦冷門且多樣的電影",
            placeholder="例：推薦新電影、推薦冷門高評價電影"
        )
        st.caption("支援關鍵字：冷門、多樣、新、評價")

    st.markdown("---")

    # Top-K
    top_k = st.selectbox("📌 推薦數量 (Top-K)", [10, 15, 20], index=0)

    # 執行按鈕
    run_btn = st.button("🚀 產生推薦", type="primary", use_container_width=True)

# ─────────────────────────────────────────────
# 3. 核心推薦函式
# ─────────────────────────────────────────────
GENRE_COLS = [
    "unknown", "Action", "Adventure", "Animation",
    "Children's", "Comedy", "Crime", "Documentary", "Drama", "Fantasy",
    "Film-Noir", "Horror", "Musical", "Mystery", "Romance", "Sci-Fi",
    "Thriller", "War", "Western"
]

def get_candidates(user_id):
    """取得該 User 的候選清單，並補上 novelty_norm fallback"""
    df = unseen_candidates[unseen_candidates['user_id'] == user_id].copy()
    if 'novelty_norm' not in df.columns:
        df['novelty_norm'] = df['novelty'] if 'novelty' in df.columns else 0.5
    return df

def recommend_baseline(candidates, k):
    return candidates.sort_values('predict_score', ascending=False).head(k)

def recommend_mmr(candidates, k, lv):
    from week4_reranking import mmr_rerank
    return mmr_rerank(candidates, GENRE_COLS, lambda_val=lv, k=k)

def recommend_pareto(candidates, k):
    from week4_reranking import pareto_rerank
    return pareto_rerank(candidates, k=k)

def recommend_nlp_pareto(candidates, k, prompt):
    from week5_nlp_pareto import parse_query, dynamic_pareto_rerank
    objectives = parse_query(prompt)
    return dynamic_pareto_rerank(candidates, GENRE_COLS, objectives, k=k), objectives

# ─────────────────────────────────────────────
# 4. 顯示輔助函式
# ─────────────────────────────────────────────
def format_display(df, method_name):
    """
    整齊地挑選欄位並重命名，避免出現空白或不存在的欄位。
    """
    # 基本欄位
    base_cols = {
        'movie_title': 'Movie Title',
        'predict_score': 'Preference (LGB)',
    }
    # 如果有 TMDB 特徵就加入
    optional_cols = {
        'novelty': 'Novelty',
        'recency': 'Recency',
        'quality': 'Quality',
    }
    # 方法專屬欄位
    method_cols = {}
    if method_name == "Pareto" or method_name == "Pareto + NLP":
        method_cols = {'pareto_rank': 'Pareto Rank', 'diversity': 'Diversity'}
    elif method_name == "MMR":
        method_cols = {'mmr_score': 'MMR Score', 'similarity_penalty': 'Sim Penalty'}

    all_wanted = {**base_cols, **optional_cols, **method_cols}
    existing = {k: v for k, v in all_wanted.items() if k in df.columns}

    display = df[list(existing.keys())].copy().rename(columns=existing)
    display.index = range(1, len(display) + 1)
    display.index.name = "Rank"
    return display.round(4)

# ─────────────────────────────────────────────
# 5. 主要顯示區域
# ─────────────────────────────────────────────
# 上半：顯示目前設定摘要
info_cols = st.columns(4)
info_cols[0].metric("👤 User ID", user_id)
info_cols[1].metric("🔧 Method", method)
info_cols[2].metric("📌 Top-K", top_k)
info_cols[3].metric("💬 Prompt", nlp_prompt if nlp_prompt else "—")

st.markdown("---")

if not run_btn:
    st.info("👈 調整左側設定完成後，點擊「🚀 產生推薦」按鈕查看結果。")
    st.stop()

# ─────────────────────────────────────────────
# 6. 執行推薦並顯示結果
# ─────────────────────────────────────────────
candidates = get_candidates(user_id)

if candidates.empty:
    st.error(f"User {user_id} 沒有可用的候選電影，請換一個 User ID。")
    st.stop()

# 執行選定方法
objectives = []
with st.spinner(f"⚙️ 執行 {method} 推薦中…"):
    if method == "Baseline":
        result_df = recommend_baseline(candidates, top_k)
    elif method == "MMR":
        result_df = recommend_mmr(candidates, top_k, lambda_val)
    elif method == "Pareto":
        result_df = recommend_pareto(candidates, top_k)
    elif method == "Pareto + NLP":
        result_df, objectives = recommend_nlp_pareto(candidates, top_k, nlp_prompt)

# 也同時跑一份 Baseline 對照
baseline_df = recommend_baseline(candidates, top_k)

# ─────────────────────────────────────────────
# 7. 呈現推薦結果
# ─────────────────────────────────────────────

# 若 NLP 有捕捉到意圖就顯示
if objectives:
    st.success(f"🎯 NLP 解析結果：目標維度 → **{', '.join(objectives)}**")
elif method == "Pareto + NLP" and not objectives:
    st.warning("⚠️ 未能解析出明確目標關鍵字，改用純 LightGBM 偏好分數排序。")

# 並排顯示：對照用的 Baseline + 選定方法結果
col_baseline, col_method = st.columns(2)

with col_baseline:
    st.subheader("📋 Baseline (LightGBM)")
    st.dataframe(format_display(baseline_df, "Baseline"), use_container_width=True)

with col_method:
    icon_map = {"Baseline": "📋", "MMR": "🔀", "Pareto": "⚖️", "Pareto + NLP": "✨"}
    st.subheader(f"{icon_map.get(method, '🎬')} {method}" + (f"  (λ={lambda_val})" if method == "MMR" else ""))
    st.dataframe(format_display(result_df, method), use_container_width=True)

st.markdown("---")

# ─────────────────────────────────────────────
# 8. Ground Truth 展示（該 User 真實喜歡的電影）
# ─────────────────────────────────────────────
st.subheader("✅ 該使用者的真實喜歡清單（Test Set Ground Truth）")
actual = test_ground_truth.get(user_id, {})
liked_ids = [mid for mid, r in actual.items() if r >= 3.0]

if liked_ids:
    liked_movies = movies_df[movies_df['movie_id'].isin(liked_ids)][['movie_id', 'movie_title']].copy()
    liked_movies['Rating'] = liked_movies['movie_id'].map(actual)
    liked_movies = liked_movies.sort_values('Rating', ascending=False).reset_index(drop=True)
    liked_movies.index += 1
    liked_movies.index.name = "#"
    with st.expander(f"點擊展開（共 {len(liked_movies)} 部高評價電影）"):
        st.dataframe(liked_movies[['movie_title', 'Rating']], use_container_width=True)
else:
    st.info("這名使用者在 Test Set 中沒有高於 3 分的評分紀錄。")
