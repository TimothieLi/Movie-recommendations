import streamlit as st
import pandas as pd
from movie_lgb_recommender import run_recommender_pipeline
from tmdb_api import TMDBClient
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
if st.sidebar.button("📈 方法比較與分析 (Week 6)", use_container_width=True):
    st.session_state['page_selection'] = "📈 方法比較與分析 (Week 6)"

st.sidebar.markdown("---")
st.sidebar.header("🌐 外部資料整合 (Week 7)")
tmdb_api_key = st.sidebar.text_input("TMDB API Key", type="password", help="請輸入 TMDB API Key 以啟用新電影推薦功能")
if not tmdb_api_key:
    st.sidebar.info("💡 輸入 API Key 可推薦資料庫外的新電影")
else:
    st.sidebar.success("✅ TMDB 已就緒")

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
        pareto_df = pareto_rerank(user_candidates, k=10,
                                   pool_size=100, tie_break='weighted',
                                   selection_mode='soft')
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
    # 6. Week 5: LLM-Assisted Semantic Parsing + Dynamic Pareto Re-ranking
    # ==========================================
    st.markdown("### ✨ Week 5: LLM-Assisted NLP 動態推薦")
    st.caption("本模組採用「語意解析 → 結構化條件 → Rule-based Ranking」三層架構，將自然語言查詢轉為可解釋的多目標推薦控制訊號。")

    from week5_nlp_pareto import parse_query_rule, parse_query_llm, dynamic_pareto_rerank
    import time

    # ── 使用者輸入 ─────────────────────────────────────────────────
    query = st.text_input(
        "💬 想找什麼樣的電影？ (輸入完成請按 Enter)",
        placeholder="例如：推薦冷門但評價不錯的電影",
        value="推薦稍微冷門但很好看的電影"
    )

    col1, col2, col3 = st.columns(3)
    with col1:
        top_k = st.selectbox("📌 推薦輸出數量", [10, 20, 30, 50], index=0)
    with col2:
        sort_by_year = st.checkbox("📅 結果依年份排序（由新到舊）")
    with col3:
        use_llm = st.checkbox("🤖 使用 LLM 語意解析（需 API Key）", value=False)

    openai_api_key = None
    if use_llm:
        openai_api_key = st.text_input("🔑 OpenAI API Key", type="password",
                                       placeholder="sk-...")

    user_candidates = unseen_candidates[unseen_candidates['user_id'] == selected_user_id].copy()

    genre_cols = [
        "unknown", "Action", "Adventure", "Animation",
        "Children's", "Comedy", "Crime", "Documentary", "Drama", "Fantasy",
        "Film-Noir", "Horror", "Musical", "Mystery", "Romance", "Sci-Fi",
        "Thriller", "War", "Western"
    ]

    if query:
        with st.spinner("🧠 進行語意解析與推薦計算中..."):
            time.sleep(0.6)

            # ── 語意解析（LLM 或 Rule-based）────────────────────────
            if use_llm and openai_api_key:
                parsed = parse_query_llm(query, api_key=openai_api_key)
            else:
                parsed = parse_query_rule(query)

            # ── 整合 TMDB 候選電影 ─────────────────────────────────
            final_candidates = user_candidates.copy()
            final_candidates['source'] = 'movielens'
            
            if tmdb_api_key:
                with st.status("🌐 正在從 TMDB 抓取外部候選電影...", expanded=False):
                    tmdb_client = TMDBClient(tmdb_api_key)
                    tmdb_df = tmdb_client.get_candidates(selected_user_id, count=50, genre_cols=genre_cols)
                    if not tmdb_df.empty:
                        # 確保欄位一致後合併
                        final_candidates = pd.concat([final_candidates, tmdb_df], ignore_index=True)
                        st.write(f"已加入 {len(tmdb_df)} 部 TMDB 新電影至候選池")

            objectives = parsed["objectives"]
            pareto_nlp_df = dynamic_pareto_rerank(
                final_candidates, genre_cols, objectives,
                k=top_k, parsed_result=parsed
            )

        # ── 解析結果展示（可解釋性面板）──────────────────────────────
        parser_label = {"llm": "🤖 LLM", "rule_based": "📐 Rule-based",
                        "rule_based_fallback": "📐 Rule-based (LLM fallback)"
                        }.get(parsed.get("_parser", "rule_based"), "📐 Rule-based")

        st.success(f"✅ 解析完成（{parser_label}）")

        with st.expander("🔍 語意解析結果（可解釋性展示）", expanded=True):
            st.markdown(f"**📝 解析說明：** {parsed['explanation']}")

            st.markdown("**🎛️ 目標維度權重：**")
            weight_cols = st.columns(5)
            dim_labels = {"relevance": "個人偏好", "novelty": "冷門度",
                          "diversity": "多樣性", "recency": "近期新穎", "quality": "評分品質"}
            for idx, (dim, label) in enumerate(dim_labels.items()):
                w = parsed["weights"].get(dim, 0.0)
                weight_cols[idx].metric(label=label, value=f"{w:.0%}")

            # 顯示有效 constraints
            active_constraints = {k: v for k, v in parsed["constraints"].items() if v is not None}
            if active_constraints:
                st.markdown("**🔒 啟用的 Rule-based 約束：**")
                c_labels = {"min_quality": "最低品質門檻", "max_novelty": "最大冷門度上限",
                            "min_novelty": "最小冷門度下限", "min_recency": "最低年份新穎度"}
                for ck, cv in active_constraints.items():
                    st.markdown(f"- `{c_labels.get(ck, ck)}`：≥ {cv:.2f}" if "min" in ck
                                else f"- `{c_labels.get(ck, ck)}`：≤ {cv:.2f}")

            if parsed.get("_llm_error"):
                st.warning(f"⚠️ LLM 解析失敗（已自動 fallback）：{parsed['_llm_error']}")

        if len(objectives) == 0:
            st.info("ℹ️ 未偵測到特定目標維度，以個人偏好（predict_score）為主進行推薦。")

        st.toast('推薦計算完成！', icon='🎉')
        
        # 整理年份顯示格式
        if 'release_year' in pareto_nlp_df.columns:
            pareto_nlp_df['release_year'] = pareto_nlp_df['release_year'].fillna(0).astype(int).astype(str)
            pareto_nlp_df.loc[pareto_nlp_df['release_year'] == '0', 'release_year'] = "Unknown"
            
        # 決定要印出來的欄位
        display_cols = []
        
        # 加上來源圖示
        if 'source' in pareto_nlp_df.columns:
            pareto_nlp_df['來源'] = pareto_nlp_df['source'].apply(lambda x: "🟣 TMDB" if x == 'tmdb' else "🔵 Local")
            display_cols.append('來源')
            
        display_cols.append('movie_title')
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

elif page_selection == "📈 方法比較與分析 (Week 6)":
    # ==========================================
    # 7. Week 6: 方法比較與分析
    # ==========================================
    st.markdown("### 📈 Week 6: 系統推薦方法全面比較")
    
    st.write("在這裡我們將針對全體（或抽樣部分）的 Test Users，評估目前所有實作過的推薦系統演算法（LightGBM, MMR, Pareto, Pareto + NLP）在各項客觀基準的綜合表現。")
    
    sample_size = st.slider("選擇要進行評測的 Test User 樣本數量 (數量越大，結果越精準但運算時間較長)", min_value=10, max_value=len(test_users), value=30, step=10)
    
    if st.button("🚀 執行/重新整理效能評測", type="primary"):
        from week6_evaluation import run_week6_experiments
        import matplotlib.pyplot as plt
        import seaborn as sns
        import random
        RANDOM_SEED = 42
        
        genre_cols = [
            "unknown", "Action", "Adventure", "Animation",
            "Children's", "Comedy", "Crime", "Documentary", "Drama", "Fantasy",
            "Film-Noir", "Horror", "Musical", "Mystery", "Romance", "Sci-Fi",
            "Thriller", "War", "Western"
        ]
        
        # Fixed-seed random sampling：從全體 test users 中隨機抽出 sample_size 位使用者。
        # 若 sample_size >= 全體數量則直接使用所有 test users，不呼叫 random.sample() 避免報錯。
        # sorted() 僅為顯示與除錯穩定性，不代表抽樣方式本身。
        all_test_users = [int(u) for u in test_users]
        if sample_size >= len(all_test_users):
            test_users_subset = sorted(all_test_users)
        else:
            rng = random.Random(RANDOM_SEED)
            test_users_subset = sorted(rng.sample(all_test_users, sample_size))
        total_movies = len(movies_df)

        st.info(
            f"📌 **抽樣說明**：本次評測採用 **Fixed-seed Random Sampling**（seed = {RANDOM_SEED}），"
            f"從全體 {len(all_test_users)} 位 Test Users 中隨機抽取 {len(test_users_subset)} 位進行評估，"
            "而非依 User ID 排序取前 N 名，以確保無 sample bias 且結果可重現。"
        )
        
        progress_bar = st.progress(0.0)
        status_text = st.empty()
        
        def update_progress(val):
            progress_bar.progress(val)
            status_text.write(f"正在運算中... (已完成 {int(val*100)}%)")
            
        with st.spinner("評測進行中，包含大量的 Jaccard 與多目標 Pareto 計算..."):
            summary_df = run_week6_experiments(test_users_subset, unseen_candidates, test_ground_truth, genre_cols, total_movies_count=total_movies, pool_size=50, k=10, progress_callback=update_progress)
            
        progress_bar.empty()
        status_text.empty()
        
        st.success("✅ 評測完成！")
        
        st.markdown("#### 📊 評估結果比較表 (Comparison DataFrame)")
        st.dataframe(summary_df.round(4), use_container_width=True)
        
        st.markdown("#### 📉 Trade-off 分布圖 (NDCG vs Novelty & ILD)")
        
        # 使用 matplotlib 與 seaborn 繪製對比圖
        sns.set_theme(style="whitegrid")
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        
        # 第一張圖：NDCG vs Novelty
        sns.scatterplot(data=summary_df, x="Novelty@10", y="NDCG@10", hue="Method", style="Method", s=150, ax=axes[0])
        axes[0].set_title("Trade-off: NDCG@10 vs Novelty@10", fontsize=14, fontweight='bold')
        
        # 將參數標示在點的旁邊
        for i in range(len(summary_df)):
            axes[0].text(summary_df["Novelty@10"][i], summary_df["NDCG@10"][i] - 0.001, 
                         summary_df["Parameters"][i], fontsize=9, alpha=0.8)
                         
        # 第二張圖：NDCG vs ILD
        sns.scatterplot(data=summary_df, x="ILD@10", y="NDCG@10", hue="Method", style="Method", s=150, ax=axes[1])
        axes[1].set_title("Trade-off: NDCG@10 vs ILD@10 (Diversity)", fontsize=14, fontweight='bold')
        
        for i in range(len(summary_df)):
            axes[1].text(summary_df["ILD@10"][i], summary_df["NDCG@10"][i] - 0.001, 
                         summary_df["Parameters"][i], fontsize=9, alpha=0.8)
                         
        plt.tight_layout()
        st.pyplot(fig)
        
        st.markdown("---")
        st.markdown(f"#### 🔍 Sanity Check: 單一用戶直觀比對 (User **{selected_user_id}**)")
        st.write("以下為系統挑出部分極端條件下的 Top-5 呈現差異（避免紙上談兵，直接透過實體清單觀察特徵改變）：")
        
        from week4_reranking import mmr_rerank, pareto_rerank
        from week5_nlp_pareto import dynamic_pareto_rerank, parse_query
        
        user_cands = unseen_candidates[unseen_candidates['user_id'] == selected_user_id].copy()
        if 'novelty_norm' not in user_cands.columns and 'novelty' in user_cands.columns:
            user_cands['novelty_norm'] = user_cands['novelty']
            
        def get_top5_titles(fn):
            res_df = fn(user_cands).head(5)
            # 組合為字串方便觀看
            return " ⭐ ".join(res_df['movie_title'].tolist())
            
        # 綁定參數
        lgbm_fn = lambda c: c.sort_values('predict_score', ascending=False)
        mmr_fn = lambda c: mmr_rerank(c, genre_cols, lambda_val=0.0, k=10)
        par_fn = lambda c: pareto_rerank(c, k=10, pool_size=100, tie_break='weighted', selection_mode='soft')
        nlp_fn = lambda c: dynamic_pareto_rerank(c, genre_cols, parse_query("冷門 多樣"), k=10)
        
        s_data = [
            {"演算法策略": "LightGBM 原味模型 (全憑偏好)", "前 5 推薦電影陣容": get_top5_titles(lgbm_fn)},
            {"演算法策略": "MMR (λ=0.0 / 完全拋棄偏好分數)", "前 5 推薦電影陣容": get_top5_titles(mmr_fn)},
            {"演算法策略": "原版雙目標 Pareto (偏好+新穎)", "前 5 推薦電影陣容": get_top5_titles(par_fn)},
            {"演算法策略": "NLP 動態 Pareto (指令: 冷門+多樣)", "前 5 推薦電影陣容": get_top5_titles(nlp_fn)}
        ]
        
        st.table(pd.DataFrame(s_data))
