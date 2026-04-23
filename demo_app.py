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
# 2. 側邊欄：模式切換與控制
# ─────────────────────────────────────────────
if 'system_mode' not in st.session_state:
    st.session_state['system_mode'] = '離線評估'

with st.sidebar:
    st.header("🎮 系統模式選擇")
    if st.button("📊 離線評估", use_container_width=True, type="primary" if st.session_state['system_mode'] == '離線評估' else "secondary"):
        st.session_state['system_mode'] = '離線評估'
        st.rerun()
    if st.button("💬 互動式推薦", use_container_width=True, type="primary" if st.session_state['system_mode'] == '互動式推薦' else "secondary"):
        st.session_state['system_mode'] = '互動式推薦'
        st.rerun()
    
    mode = st.session_state['system_mode']
    st.markdown("---")
    
    if mode == "離線評估":
        st.header("📊 評估設定")
        user_id = st.selectbox("👤 選擇測試使用者 (Test Set)", options=test_users, index=0)
        method = st.selectbox("🔧 推薦演算法", options=["Baseline", "MMR", "Pareto"], index=0)
        
        lambda_val = 0.5
        if method == "MMR":
            lambda_val = st.slider("λ (Relevance ↔ Diversity)", 0.0, 1.0, 0.5, 0.25)
        
        top_k = st.selectbox("📌 推薦數量 (Top-K)", [10, 15, 20], index=0)
        # 離線評估不使用 TMDB 與 NLP
        tmdb_api_key = None
        nlp_prompt_sidebar = ""
        use_llm = False
        openai_api_key = None
        run_btn = st.button("🚀 產生推薦", type="primary", use_container_width=True)
        
    else:  # 互動式推薦
        st.header("⚙️ 推薦設定")
        with st.expander("🤖 LLM 解析設定"):
            use_llm = st.checkbox("啟用 LLM 語意解析")
            openai_api_key = st.text_input("OpenAI API Key", type="password")
            
        with st.expander("🌐 外部資料整合"):
            tmdb_api_key = st.text_input("TMDB API Key", type="password", help="輸入 API Key 啟用 TMDB 推薦")
            if tmdb_api_key:
                st.success("✅ TMDB 整合已開啟")
        
        top_k = st.selectbox("📌 推薦數量 (Top-K)", [10, 15, 20], index=0)
        # 互動模式固定使用 NLP-Pareto 邏輯，User 固定用第一個作為基礎特徵參考（但不顯示）
        method = "Pareto + NLP"
        user_id = test_users[0] 
        lambda_val = 0.5
        run_btn = False # 由主頁面按鈕觸發

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

def recommend_nlp_pareto(candidates, k, prompt, use_llm=False, api_key=None):
    from week5_nlp_pareto import parse_query_rule, parse_query_llm, dynamic_pareto_rerank
    if use_llm and api_key:
        parsed = parse_query_llm(prompt, api_key=api_key)
    else:
        parsed = parse_query_rule(prompt)
    return dynamic_pareto_rerank(candidates, GENRE_COLS, parsed["objectives"], k=k, parsed_result=parsed), parsed

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
    }
    # 離線模式才顯示模型預測分數
    if method_name != "Pareto + NLP":
        base_cols['predict_score'] = 'Preference (LGB)'
    # 如果有 TMDB 特徵就加入
    optional_cols = {
        'source': 'Source',
        'novelty': 'Novelty',
        'recency': 'Recency',
        'quality': 'Quality',
    }
    # 方法專屬欄位
    method_cols = {}
    if method_name == "MMR":
        method_cols = {'mmr_score': 'MMR Score', 'similarity_penalty': 'Sim Penalty'}
        df['lambda'] = lambda_val
        method_cols['lambda'] = 'Lambda (λ)'
    elif method_name == "Pareto":
        method_cols = {'pareto_rank': 'Pareto Rank', 'diversity': 'Diversity'}
    elif method_name == "Pareto + NLP":
        # 互動模式移除內部過程欄位，顯示最終分數
        method_cols = {'final_score': 'Final Score'}

    all_wanted = {**base_cols, **optional_cols, **method_cols}
    existing = {k: v for k, v in all_wanted.items() if k in df.columns}

    display = df[list(existing.keys())].copy().rename(columns=existing)
    if 'Source' in display.columns:
        display['Source'] = display['Source'].apply(lambda x: "🟣 TMDB" if x == 'tmdb' else "🔵 Local")
    display.index = range(1, len(display) + 1)
    display.index.name = "Rank"
    return display.round(4)

# 上半：顯示目前標題
if mode == "離線評估":
    st.markdown(f"## 🛸 離線效能評估")
    st.markdown("---")
else:
    # 互動模式移除多餘標題與分隔線，直接進入輸入區
    pass

if mode == "離線評估":
    if not run_btn:
        st.info("👈 調整左側設定完成後，點擊「🚀 產生推薦」按鈕查看結果。")
        st.stop()
    candidates = get_candidates(user_id)
    # --- 執行離線評估 ---
    with st.spinner(f"⚙️ 執行 {method} 離線評估中…"):
        if method == "Baseline":
            result_df = recommend_baseline(candidates, top_k)
        elif method == "MMR":
            result_df = recommend_mmr(candidates, top_k, lambda_val)
        elif method == "Pareto":
            result_df = recommend_pareto(candidates, top_k)

    actual = test_ground_truth.get(user_id, {})
    liked_ids = [mid for mid, r in actual.items() if r >= 3.0]

    st.subheader(f"📋 1. {method} 推薦結果 (Offline)")
    st.dataframe(format_display(result_df, method, liked_ids=liked_ids), use_container_width=True)

    st.markdown("---")
    st.subheader("📈 2. 離線效能指標 (Metrics)")
    preds = result_df['movie_id'].tolist()
    m_ndcg = ndcg_at_k(actual, preds, k=top_k)
    m_recall = recall_at_k(actual, preds, k=top_k, threshold=3.0)
    m_novelty = calculate_novelty_at_k(result_df)
    m_ild = calculate_ild_at_k(result_df, GENRE_COLS)

    metrics_data = {
        "指標項目": ["NDCG@K", "Recall@K", "Avg Novelty", "ILD (Diversity)"],
        "數值": [f"{m_ndcg*100:.2f}%", f"{m_recall*100:.2f}%", f"{m_novelty*100:.2f}%", f"{m_ild*100:.2f}%"],
        "說明": ["排序準確度", "喜好命中率", "驚喜度/冷門度", "類型多樣性"]
    }
    st.table(pd.DataFrame(metrics_data))

    if user_id:
        st.markdown("---")
        st.subheader("✅ 3. 真實喜歡清單 (Ground Truth)")
        with st.expander("點擊展開該使用者的歷史高評分紀錄"):
            liked_movies = movies_df[movies_df['movie_id'].isin(liked_ids)].copy()
            liked_movies['Rating'] = liked_movies['movie_id'].map(actual)
            st.dataframe(liked_movies[['movie_title', 'Rating']].sort_values('Rating', ascending=False), use_container_width=True)

else:  # 互動式推薦
    st.markdown("### 💬 互動式推薦需求輸入")
    st.write("請輸入你想看的電影需求（例：給我一些年代久遠的經典片），系統將根據語意解析動態調整推薦權重。")
    
    col_input, col_run = st.columns([4, 1])
    with col_input:
        nlp_prompt = st.text_input(
            "自然語言需求",
            placeholder="例如：推薦冷門但高評價的電影",
            label_visibility="collapsed"
        )
    with col_run:
        interactive_run = st.button("🚀 產生推薦", type="primary", use_container_width=True)

    if interactive_run:
        if not nlp_prompt:
            st.warning("請輸入需求描述後再執行推薦。")
            st.stop()
            
        candidates = get_candidates(user_id)
        # --- 執行互動式推薦 ---
        with st.spinner("🧠 語意解析與跨域推薦計算中…"):
            result_df, parsed = recommend_nlp_pareto(candidates, top_k, nlp_prompt, use_llm=use_llm, api_key=openai_api_key)

        # 1. 語意解析說明
        st.subheader("💡 1. 語意解析說明 (Explainability)")
        p_type = parsed.get("_parser", "rule_based")
        st.success(f"✅ 解析完成 ({p_type.upper()})")
        
        # 顯示目標維度權重 (正規化至 100% 顯示)
        weights = parsed.get("weights", {})
        if weights:
            st.markdown("**🎯 目標維度佔比 (Relative Importance)**")
            total_w = sum(weights.values())
            if total_w == 0: total_w = 1.0
            
            w_cols = st.columns(5)
            w_cols[0].metric("個人偏好", f"{(weights.get('relevance', 0)/total_w)*100:.0f}%")
            w_cols[1].metric("冷門度", f"{(weights.get('novelty', 0)/total_w)*100:.0f}%")
            w_cols[2].metric("多樣性", f"{(weights.get('diversity', 0)/total_w)*100:.0f}%")
            w_cols[3].metric("新舊度", f"{(weights.get('recency', 0)/total_w)*100:.0f}%")
            w_cols[4].metric("評分品質", f"{(weights.get('quality', 0)/total_w)*100:.0f}%")
            st.caption("註：佔比代表各維度對最終排序分數（Final Score）的相對貢獻權重。")

        st.markdown("---")
        
        # 2. 精選推薦詳情 (Top 5 Featured) - 移到前面
        st.subheader("🌟 2. 精選推薦詳情 (Top 5 Featured)")
        for idx, row in result_df.head(5).iterrows():
            with st.container():
                col1, col2 = st.columns([1, 4])
                poster_url = row.get('poster_path', "https://via.placeholder.com/150x225?text=No+Poster")
                if pd.isna(poster_url): poster_url = "https://via.placeholder.com/150x225?text=No+Poster"
                with col1: st.image(poster_url, use_container_width=True)
                with col2:
                    st.markdown(f"### {idx+1}. {row['movie_title']}")
                    st.markdown(f"**來源**: {'🟣 TMDB' if row.get('source') == 'tmdb' else '🔵 Local'} | **年份**: {row.get('release_year', 'Unknown')}")
                    active_genres = [g for g in GENRE_COLS if row.get(g) == 1]
                    if active_genres: st.markdown(f"**類型**: {' · '.join(active_genres)}")
                    st.write(row.get('overview', "尚無電影簡介資訊。"))
                    
                    # 顯示最終分數與特徵
                    s_cols = st.columns(4)
                    s_cols[0].caption(f"🏆 Final Score: {row.get('final_score', 0):.4f}")
                    s_cols[1].caption(f"🔍 Novelty: {row.get('novelty', 0):.2f}")
                    s_cols[2].caption(f"📅 Recency: {row.get('recency', 0):.2f}")
                    s_cols[3].caption(f"⭐ Quality: {row.get('quality', 0):.2f}")
                st.markdown("---")

        # 3. 推薦結果表格 - 移到後面
        st.subheader("🎬 3. 推薦結果與詳情")
        st.dataframe(format_display(result_df, "Pareto + NLP"), use_container_width=True)
    else:
        st.info("💡 請在上方輸入框描述您的電影需求，然後點擊「產生推薦」。")
