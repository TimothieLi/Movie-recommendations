"""
demo_app.py — Movie Recommendation System Demo Dashboard
用途：專題展示用，可互動選擇使用者、推薦方法、NLP Prompt 即時看結果。
啟動：python -m streamlit run demo_app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import warnings
from tmdb_api import TMDBClient
from week6_evaluation import calculate_ild_at_k, calculate_novelty_at_k
from movie_lgb_recommender import ndcg_at_k, recall_at_k
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
    user_id = st.selectbox(
        "👤 選擇 User ID",
        options=test_users,
        index=0,
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
    
    st.header("🌐 外部資料整合")
    tmdb_api_key = st.text_input("TMDB API Key", type="password", help="輸入 API Key 啟用 TMDB 推薦")
    if tmdb_api_key:
        st.success("✅ TMDB 整合已開啟")

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
    df['source'] = 'movielens'
    
    if tmdb_api_key:
        with st.spinner("🌐 正在獲取 TMDB 新電影..."):
            client = TMDBClient(tmdb_api_key)
            tmdb_df = client.get_candidates(user_id, count=50, genre_cols=GENRE_COLS)
            if not tmdb_df.empty:
                df = pd.concat([df, tmdb_df], ignore_index=True)
                
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
def format_display(df, method_name, liked_ids=None):
    """
    整齊地挑選欄位並重命名，避免出現空白或不存在的欄位。
    若提供了 liked_ids，則會在匹配的電影名稱前加上 ⭐。
    """
    df = df.copy()
    if liked_ids and 'movie_id' in df.columns:
        df['movie_title'] = df.apply(
            lambda x: f"⭐ {x['movie_title']}" if x['movie_id'] in liked_ids else x['movie_title'],
            axis=1
        )
    
    # 基本欄位
    base_cols = {
        'movie_title': 'Movie Title',
        'predict_score': 'Preference (LGB)',
    }
    # 如果有 TMDB 特徵就加入
    optional_cols = {
        'source': 'Source',
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
        # 額外顯示 lambda 以便觀察
        df['lambda'] = lambda_val
        method_cols['lambda'] = 'Lambda (λ)'

    all_wanted = {**base_cols, **optional_cols, **method_cols}
    existing = {k: v for k, v in all_wanted.items() if k in df.columns}

    display = df[list(existing.keys())].copy().rename(columns=existing)
    if 'Source' in display.columns:
        display['Source'] = display['Source'].apply(lambda x: "🟣 TMDB" if x == 'tmdb' else "🔵 Local")
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

# ─────────────────────────────────────────────
# 7. 呈現推薦結果 (單一方法模式)
# ─────────────────────────────────────────────

# --- 取得 Ground Truth 用於星星標記 ---
actual = test_ground_truth.get(user_id, {})
liked_ids = [mid for mid, r in actual.items() if r >= 3.0]

# (1) 主推薦結果表格
method_icons = {"Baseline": "📋", "MMR": "🔀", "Pareto": "⚖️", "Pareto + NLP": "✨"}
method_captions = {
    "Baseline": "僅考慮個人偏好預測分數（LightGBM LambdaRank），不進行額外重排序。",
    "MMR": "最大邊際相關性（Maximal Marginal Relevance），平衡「相關性」與「多樣性」，避免推薦內容過於雷同。",
    "Pareto": "多目標 Pareto 重排序，同時優化相關性與新穎度（Novelty），尋找兩者的最佳均衡點。",
    "Pareto + NLP": "結合 LLM 語意解析與 Pareto 演算法，將您的自然語言需求動態轉化為推薦目標權重。"
}

st.subheader(f"{method_icons.get(method, '🎬')} 1. {method} 推薦結果" + (f" (λ={lambda_val})" if method == "MMR" else ""))
st.caption(f"{method_captions.get(method, '')} (⭐ 代表命中使用者真實喜歡的電影)")

# 顯示主要結果表格
st.dataframe(format_display(result_df, method, liked_ids=liked_ids), use_container_width=True)

st.markdown("---")

# (2) 推薦詳情 (Featured Movies)
st.subheader(f"🌟 2. 精選推薦詳情 (Featured Details)")
if method == "Pareto + NLP":
    if objectives:
        st.success(f"🎯 NLP 解析意圖：目標維度 → **{', '.join(objectives)}**")
    else:
        st.warning("⚠️ 未能解析出明確目標，已退回個人偏好排序。")

# 只取前 5 部展示詳細資訊
featured_df = result_df.head(5).copy()
for idx, row in featured_df.iterrows():
    with st.container():
        col1, col2 = st.columns([1, 4])
        poster_url = row.get('poster_path')
        if not poster_url or pd.isna(poster_url):
            poster_url = "https://via.placeholder.com/150x225?text=No+Poster"
        with col1:
            st.image(poster_url, use_container_width=True)
        with col2:
            st.markdown(f"### {idx+1}. {row['movie_title']}")
            source_tag = "🟣 TMDB" if row.get('source') == 'tmdb' else "🔵 Local"
            year = row.get('release_year', 'Unknown')
            st.markdown(f"**來源**: {source_tag} | **年份**: {year}")
            active_genres = [g for g in GENRE_COLS if row.get(g) == 1]
            if active_genres:
                st.markdown(f"**類型**: {' · '.join(active_genres)}")
            overview = row.get('overview', "尚無電影簡介資訊。")
            if pd.isna(overview) or not overview:
                overview = "尚無電影簡介資訊。"
            st.write(overview)
            score_cols = st.columns(3)
            score_cols[0].caption(f"⭐ 預測分數: {row.get('predict_score', 0):.2f}")
            score_cols[1].caption(f"🔍 新穎度: {row.get('novelty', 0):.2f}")
            score_cols[2].caption(f"📅 新舊度: {row.get('recency', 0):.2f}")
        st.markdown("---")

# (3) 效能評估指標
st.subheader("📈 3. 效能評估指標")
preds = result_df['movie_id'].tolist()
m_ndcg = ndcg_at_k(actual, preds, k=top_k)
m_recall = recall_at_k(actual, preds, k=top_k, threshold=3.0)
m_novelty = calculate_novelty_at_k(result_df)
m_ild = calculate_ild_at_k(result_df, GENRE_COLS)

metrics_data = {
    "指標項目": ["NDCG@K (排序品質)", "Recall@K (召回率)", "Avg Novelty (新穎度)", "ILD (多樣性)"],
    "數值": [
        f"{m_ndcg*100:.2f}%", 
        f"{m_recall*100:.2f}%", 
        f"{m_novelty*100:.2f}%", 
        f"{m_ild*100:.2f}%"
    ],
    "說明": [
        "越高代表推薦順序越符合使用者真實喜好",
        "越高代表推薦清單中包含越多使用者喜歡的電影",
        "越高代表推薦了越多冷門/小眾的驚喜電影",
        "越高代表推薦清單中電影類型的差異度越大"
    ]
}
st.table(pd.DataFrame(metrics_data))

st.markdown("---")

# (4) 該使用者真正喜歡清單 (Ground Truth)
st.subheader("✅ 4. 該使用者的真實喜歡清單 (Test Set Ground Truth)")
liked_ids = [mid for mid, r in actual.items() if r >= 3.0]
if liked_ids:
    liked_movies = movies_df[movies_df['movie_id'].isin(liked_ids)][['movie_id', 'movie_title']].copy()
    liked_movies['Rating'] = liked_movies['movie_id'].map(actual)
    liked_movies = liked_movies.sort_values('Rating', ascending=False).reset_index(drop=True)
    liked_movies.index += 1
    liked_movies.index.name = "#"
    with st.expander(f"點擊展開（共 {len(liked_movies)} 部高評價電影紀錄）"):
        st.dataframe(liked_movies[['movie_title', 'Rating']], use_container_width=True)
else:
    st.info("這名使用者在 Test Set 中沒有高於 3 分的評分紀錄。")
