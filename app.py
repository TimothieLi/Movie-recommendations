import streamlit as st
import pandas as pd
from movie_lgb_recommender import run_recommender_pipeline
import warnings

warnings.filterwarnings("ignore")

# 設定頁面資訊
st.set_page_config(page_title="Movie Recommender", layout="wide", page_icon="🎬")

st.title("🎬 專題：電影推薦系統 (LightGBM)")

# --- 1. 快取模型與預測結果 ---
# 使用 st.cache_data 讓我們只在「初次打開網頁」時花時間訓練模型，之後都不會重跑
@st.cache_data(show_spinner="模型訓練與特徵工程進行中，大約需要 5~10 秒，請稍後...")
def load_all_data_and_predict():
    # 執行整個 offline batch pipeline
    test_users, top_10_df, test_ground_truth, movies_df, unseen_candidates = run_recommender_pipeline()
    return list(test_users), top_10_df, test_ground_truth, movies_df, unseen_candidates

test_users, top_10_df, test_ground_truth, movies_df, unseen_candidates = load_all_data_and_predict()

# --- 2. 側邊欄 ---
st.sidebar.header("🕹️ 控制面板")
st.sidebar.write("請選擇要預覽的 Test Set User ID")

# 將 test_users 確保轉回 int 進行排序選擇
test_users_sorted = sorted([int(u) for u in test_users])
selected_user_id = st.sidebar.selectbox("User ID", test_users_sorted)

# 取得該 user 的推薦資料 (各頁面共用)
user_top_movies = top_10_df[top_10_df['user_id'] == selected_user_id].copy()

# --- 3. 頁面導航 ---
if 'page_selection' not in st.session_state:
    st.session_state['page_selection'] = "🏠 首頁 (Top-10 推薦)"

st.sidebar.markdown("### 📍 頁面導航")

if st.sidebar.button("🏠 首頁 (Top-10 推薦)", use_container_width=True):
    st.session_state['page_selection'] = "🏠 首頁 (Top-10 推薦)"
if st.sidebar.button("📊 特徵工程 (Week 3)", use_container_width=True):
    st.session_state['page_selection'] = "📊 特徵工程 (Week 3)"
if st.sidebar.button("🏆 Re-ranking 演算法 (Week 4)", use_container_width=True):
    st.session_state['page_selection'] = "🏆 Re-ranking 演算法 (Week 4)"
if st.sidebar.button("✨ NLP 動態推薦 (Week 5)", use_container_width=True):
    st.session_state['page_selection'] = "✨ NLP 動態推薦 (Week 5)"

page_selection = st.session_state['page_selection']

if page_selection == "🏠 首頁 (Top-10 推薦)":
    st.markdown(f"### 👤 目前為 User `{selected_user_id}` 產生專屬推薦")
    st.write("以下為系統預測該名使用者可能最喜歡的 Top-10 電影。如果電影後面標示有 ⭐，代表這部電影也剛好出現在他在 Test Set 的真實喜歡清單中！")
    
    actual_interactions = test_ground_truth.get(selected_user_id, {})
    actual_liked_ids = [m for m, r in actual_interactions.items() if r >= 3.0]
    
    # 實作一組精美的資料表呈現推薦結果
    if user_top_movies.empty:
        st.warning("查無此使用者的預測資料。")
    else:
        # 增加一個 Hit 欄位
        def is_hit(row):
            m_id = row['movie_id']
            if m_id in actual_liked_ids:
                return f"⭐ 命中! (Test Rating: {actual_interactions[m_id]:.0f})"
            return ""
            
        user_top_movies['Hit Match'] = user_top_movies.apply(is_hit, axis=1)
        
        # 整理呈現欄位
        display_df = user_top_movies[['movie_title', 'predict_score', 'Hit Match']].copy()
        display_df.rename(columns={'movie_title': 'Movie Title', 'predict_score': 'Predict Score'}, inplace=True)
        
        # 整理 index 使其變成 Rank 名次 (1~10)
        display_df.index = range(1, len(display_df) + 1)
        display_df.index.name = "Rank / 名次"
        
        # Streamlit 的 dataframe 元件可漂亮地展示
        st.dataframe(display_df, use_container_width=True)
    
    st.markdown("---")
    st.markdown("#### 📖 此名使用者的真實喜歡清單 (由 Test Set 分析)")
    
    # 利用 movies_df 去撈出名稱
    actual_liked_movies = movies_df[movies_df['movie_id'].isin(actual_liked_ids)][['movie_title']]
    if len(actual_liked_movies) > 0:
        st.write(f"此用戶在測試期間，總共給予了 **{len(actual_liked_movies)}** 部電影高評價 (大於等於 3 分)：")
        
        # 把實際喜歡的片單用 expander 縮起來，以免清單太長版面變雜亂
        with st.expander(f"點擊展開 User {selected_user_id} 的實際喜歡電影"):
            st.dataframe(actual_liked_movies.reset_index(drop=True), use_container_width=True)
    else:
        st.info("這名用戶在 Test Set 中並沒有留下這類正向評價。")

elif page_selection == "📊 特徵工程 (Week 3)":
    # ==========================================
    # 4. Week 3 分析專區：Novelty 與 Diversity
    # ==========================================
    st.markdown("### 📊 特徵工程：Novelty 與 Diversity 分析 (Week 3)")
    
    with st.spinner("計算 Novelty 特徵與相似度中..."):
        from week3_features import get_week3_analysis
        
        movies_feat, top3_highest, top3_lowest, m1_title, m2_title, sim_score, fig = get_week3_analysis()
        
        st.subheader("1. Novelty (新穎度) 分布圖")
        st.write("定義：`Novelty(i) = -log(popularity(i) / max_popularity)`，並經過 Min-Max 正規化至 `[0, 1]` 區間。")
        st.pyplot(fig)
        
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("🥶 最冷門電影 (Novelty $\\approx$ 1.0)")
            st.dataframe(top3_highest, use_container_width=True)
            
        with col2:
            st.subheader("🔥 最熱門電影 (Novelty $\\approx$ 0.0)")
            st.dataframe(top3_lowest, use_container_width=True)
            
        st.markdown("---")
        st.subheader("2. Diversity (相似度) 函數測試")
        st.write("定義：`Sim(i, j) = Genre Multi-hot Vector 內積 (Overlap 數量)`")
        st.info(f"電影 **{m1_title}** 與 **{m2_title}** 的 Genre Overlap 數量為： **{sim_score:.0f}**")

elif page_selection == "🏆 Re-ranking 演算法 (Week 4)":
    # ==========================================
    # 5. Week 4 分析專區：Pareto 與 MMR 重新排序
    # ==========================================
    st.markdown("### 🏆 Re-ranking 演算法：Pareto & MMR (Week 4)")
    
    from week4_reranking import pareto_rerank, mmr_rerank
    
    st.write("我們從系統預測的**完整候選清單**中取得該使用者的資料，並加入 Novelty 特徵後重新排序。")
    user_candidates = unseen_candidates[unseen_candidates['user_id'] == selected_user_id].copy()
    
    # 確保我們先跑過 Week 3 以便拿到 movies_feat 的 novelty_norm
    try:
        movies_feat_for_w4 = movies_feat
    except NameError:
        from week3_features import get_week3_analysis
        movies_feat_for_w4, _, _, _, _, _, _ = get_week3_analysis()
        
    user_candidates = user_candidates.merge(movies_feat_for_w4[['movie_id', 'novelty_norm']], on='movie_id', how='left')
    
    genre_cols = [
        "unknown", "Action", "Adventure", "Animation",
        "Children's", "Comedy", "Crime", "Documentary", "Drama", "Fantasy",
        "Film-Noir", "Horror", "Musical", "Mystery", "Romance", "Sci-Fi",
        "Thriller", "War", "Western"
    ]
    
    col_lgb, col_pareto = st.columns(2)
    
    with col_lgb:
        st.subheader("1️⃣ 原始 LightGBM Top-10")
        lgb_display = user_top_movies[['movie_title', 'predict_score']].copy()
        lgb_display.index = range(1, len(lgb_display) + 1)
        st.dataframe(lgb_display, use_container_width=True)
        
    with col_pareto:
        st.subheader("2️⃣ Pareto Re-ranking Top-10")
        pareto_df = pareto_rerank(user_candidates, k=10)
        pareto_display = pareto_df[['movie_title', 'predict_score', 'novelty_norm']].copy()
        pareto_display.index = range(1, len(pareto_display) + 1)
        st.dataframe(pareto_display, use_container_width=True)
        
    st.markdown("---")
    st.subheader("3️⃣ MMR (Maximal Marginal Relevance) Top-10")
    st.write("比較不同 $\lambda$ 對於 Relevance 與 Diversity 的影響：")
    
    lambda_vals = [0.0, 0.25, 0.5, 0.75, 1.0]
    tabs = st.tabs([f"λ={l}" for l in lambda_vals])
    
    for i, l_val in enumerate(lambda_vals):
        with tabs[i]:
            if l_val == 1.0:
                st.write("**完全只看 Relevance (預測分數)**")
            elif l_val == 0.0:
                st.write("**極度著重於 Diversity (Jaccard 相似度懲罰)**")
            else:
                st.write(f"**平衡 Relevance 與 Diversity**")
                
            mmr_df = mmr_rerank(user_candidates, genre_cols, lambda_val=l_val, k=10)
            
            # 要顯示的新增欄位
            display_cols = ['movie_title', 'predict_score', 'similarity_penalty', 'mmr_score', 'lambda']
            mmr_display = mmr_df[display_cols].copy()
            mmr_display.index = range(1, len(mmr_display) + 1)
            st.dataframe(mmr_display, use_container_width=True)

elif page_selection == "✨ NLP 動態推薦 (Week 5)":
    # ==========================================
    # 6. Week 5: NLP 動態目標推薦
    # ==========================================
    st.markdown("### ✨ Week 5: NLP 動態目標推薦")
    
    from week5_nlp_pareto import parse_query, dynamic_pareto_rerank
    import time
    
    query = st.text_input("💬 想找什麼樣的電影？ (輸入完成請按 Enter)", placeholder="例如：推薦冷門且高評價電影", value="推薦冷門且多樣的電影")
    
    col1, col2 = st.columns(2)
    with col1:
        top_k = st.selectbox("📌 推薦輸出數量", [10, 20, 30, 50], index=0)
    with col2:
        st.write("") # 空間排版用
        st.write("")
        sort_by_year = st.checkbox("📅 結果依照年份排序 (由新到舊)")
    
    user_candidates = unseen_candidates[unseen_candidates['user_id'] == selected_user_id].copy()
    
    genre_cols = [
        "unknown", "Action", "Adventure", "Animation",
        "Children's", "Comedy", "Crime", "Documentary", "Drama", "Fantasy",
        "Film-Noir", "Horror", "Musical", "Mystery", "Romance", "Sci-Fi",
        "Thriller", "War", "Western"
    ]
    
    if query:
        # 動態效果
        with st.spinner("🧠 進行語意解析與推薦計算中..."):
            time.sleep(0.8) # 保留適度的運算停頓感
            
            objectives = parse_query(query)
            pareto_nlp_df = dynamic_pareto_rerank(user_candidates, genre_cols, objectives, k=top_k)
            
        if len(objectives) == 0:
            st.warning("⚠️ 無法從你的輸入解析出具體目標（冷門、多樣、新、評價），系統將預設純使用 LightGBM 分數。")
        else:
            st.success(f"🎯 成功捕捉語意意圖！目標維度設定為: **{', '.join(objectives)}**")
            
        st.toast('推薦計算完成！', icon='🎉')
        
        # 整理年份顯示格式
        if 'release_year' in pareto_nlp_df.columns:
            pareto_nlp_df['release_year'] = pareto_nlp_df['release_year'].fillna(0).astype(int).astype(str)
            pareto_nlp_df.loc[pareto_nlp_df['release_year'] == '0', 'release_year'] = "Unknown"
            
        # 決定要印出來的欄位
        display_cols = ['movie_title']
        if 'release_year' in pareto_nlp_df.columns:
            display_cols.append('release_year')
            
        # 根據要求顯示各樣數值 (若該欄位存在)
        for col in ['pareto_rank', 'predict_score', 'preference', 'recency', 'novelty', 'diversity', 'quality']:
            if col in pareto_nlp_df.columns and col not in display_cols:
                display_cols.append(col)
        
        nlp_display = pareto_nlp_df[display_cols].copy()
        
        # 排序
        if sort_by_year and 'release_year' in nlp_display.columns:
            # temporarily convert back to int for proper sorting if possible
            nlp_display['_sort_year'] = pd.to_numeric(nlp_display['release_year'], errors='coerce').fillna(0)
            nlp_display = nlp_display.sort_values('_sort_year', ascending=False).drop(columns=['_sort_year'])
            
        nlp_display.index = range(1, len(nlp_display) + 1)
        nlp_display.index.name = "Rank"
        nlp_display = nlp_display.round(4)
        
        st.dataframe(nlp_display, use_container_width=True)
