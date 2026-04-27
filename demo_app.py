import streamlit as st
import pandas as pd
import numpy as np
import os
import re
import warnings

# --- Pyparsing 相容性補丁 (解決 Gemini API 報錯) ---
try:
    import pyparsing
    if not hasattr(pyparsing, 'DelimitedList'):
        pyparsing.DelimitedList = pyparsing.delimitedList
except ImportError:
    pass

from movie_lgb_recommender import run_recommender_pipeline, ndcg_at_k, recall_at_k, GENRE_COLS
from tmdb_api import TMDBClient
from week6_evaluation import calculate_ild_at_k, calculate_novelty_at_k

warnings.filterwarnings("ignore")

# --- 引入 Optuna 工具 ---
try:
    from optuna_tuning import run_optuna_weight_search
except ImportError as e:
    if "optuna" in str(e):
        st.error("⚠️ 缺少 `optuna` 套件，請執行 `pip install optuna`。")
    else:
        st.error(f"⚠️ 無法載入 optuna_tuning.py: {e}")
except Exception as e:
    st.error(f"⚠️ 載入 Optuna 模組時出錯: {e}")
warnings.filterwarnings("ignore")
 
# ─────────────────────────────────────────────
# 0. 安全取得 Secrets
# ─────────────────────────────────────────────
def get_secret(key, default=None):
    try:
        return st.secrets.get(key, default)
    except Exception:
        return default

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
@st.cache_resource(show_spinner="⏳ 模型訓練與特徵工程中，請稍候…")
def load_pipeline():
    from movie_lgb_recommender import run_recommender_pipeline
    # 回傳 7 個值：test_users, top_10_df, test_ground_truth, movies_df, unseen_candidates, model, features_to_use
    res = run_recommender_pipeline()
    # 確保 test_users 是經過排序的整數列表，方便選單操作
    test_users_sorted = sorted([int(u) for u in res[0]])
    return (test_users_sorted,) + res[1:]

test_users, top_10_df, test_ground_truth, movies_df, unseen_candidates, lgb_model, lgb_features = load_pipeline()

# ─────────────────────────────────────────────
# 2. 側邊欄：模式切換與控制
# ─────────────────────────────────────────────
if 'system_mode' not in st.session_state:
    st.session_state['system_mode'] = '離線評估'

with st.sidebar:
    st.header("🎮 系統模式選擇")
    if st.button("📊 離線評估", type="primary" if st.session_state['system_mode'] == '離線評估' else "secondary"):
        st.session_state['system_mode'] = '離線評估'
        st.rerun()
    if st.button("💬 互動式推薦", type="primary" if st.session_state['system_mode'] == '互動式推薦' else "secondary"):
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
            lambda_val = st.slider("λ (Diversity ↔ Relevance)", 0.0, 1.0, 0.5, 0.25)
        
        top_k = st.selectbox("📌 推薦數量 (Top-K)", [10, 15, 20], index=0)
        # 離線評估不使用 TMDB 與 NLP
        tmdb_api_key = get_secret("TMDB_API_KEY")
        nlp_prompt_sidebar = ""
        use_llm = False
        api_key = get_secret("OPENAI_API_KEY") or get_secret("GEMINI_API_KEY")
        
        # 監控主要參數（不包含 Lambda，以達成拉動滑桿即時更新的效果）
        current_params = (user_id, method, top_k)
        
        if 'run_offline' not in st.session_state:
            st.session_state['run_offline'] = False
        if 'prev_params' not in st.session_state:
            st.session_state['prev_params'] = current_params
            
        # 僅在使用者、方法或數量改變時重設狀態
        if st.session_state['prev_params'] != current_params:
            st.session_state['run_offline'] = False
            st.session_state['prev_params'] = current_params

        if st.button("🚀 產生推薦", type="primary"):
            st.session_state['run_offline'] = True
        
        run_btn = st.session_state['run_offline']
        
    else:  # 互動式推薦
        st.header("⚙️ 推薦設定")
        
        # 自動從 Secrets 讀取 API Key (不再需要手動輸入)
        api_key = get_secret("OPENAI_API_KEY") or get_secret("GEMINI_API_KEY") or ""
        use_llm = True if api_key else False
        
        tmdb_api_key = get_secret("TMDB_API_KEY", "")
        if tmdb_api_key:
            st.success("✅ TMDB 整合已開啟")
        else:
            st.warning("⚠️ 未偵測到 TMDB API Key，將無法顯示海報牆。")
        
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

@st.cache_data(show_spinner=False)
def get_movie_details_from_tmdb(title, api_key):
    """透過電影名稱去 TMDB 抓取海報與簡介"""
    if not api_key: return None, None
    try:
        # 去除標題中的年份括號，例如 "Toy Story (1995)" -> "Toy Story"
        clean_title = title.split(' (')[0]
        client = TMDBClient(api_key)
        # 使用 search 功能
        search_url = f"https://api.themoviedb.org/3/search/movie?api_key={api_key}&query={clean_title}"
        import requests
        resp = requests.get(search_url).json()
        if resp.get('results'):
            first = resp['results'][0]
            poster = f"https://image.tmdb.org/t/p/w500{first['poster_path']}" if first.get('poster_path') else None
            return poster, first.get('overview')
    except:
        pass
    return None, None

def get_candidates(user_id):
    """取得該 User 的候選清單，並補上 novelty_norm fallback"""
    df = unseen_candidates[unseen_candidates['user_id'] == user_id].copy()
    df['source'] = 'movielens'
    
    # 僅在「互動式推薦」模式下且有 API Key 時才抓取 TMDB
    if st.session_state.get('system_mode') == "互動式推薦" and tmdb_api_key:
        with st.spinner("🌐 正在獲取 TMDB 新電影..."):
            try:
                client = TMDBClient(tmdb_api_key)
                tmdb_df = client.get_candidates(user_id, count=50, genre_cols=GENRE_COLS)
                if not tmdb_df.empty:
                    df = pd.concat([df, tmdb_df], ignore_index=True)
            except Exception as e:
                st.warning(f"TMDB 獲取失敗: {e}")
                
    if 'novelty_norm' not in df.columns:
        df['novelty_norm'] = df['novelty'] if 'novelty' in df.columns else 0.5
    return df

def recommend_baseline(candidates, k):
    return candidates.sort_values('predict_score', ascending=False).head(k)

def recommend_mmr(candidates, k, lv):
    from week4_reranking import mmr_rerank
    return mmr_rerank(candidates, GENRE_COLS, lambda_val=lv, k=k)

def recommend_pareto(candidates, k):
    from week5_nlp_pareto import dynamic_pareto_rerank
    # 離線評估專用：專注於相關性 (LGB)、冷門度、品質這三個核心維度
    return dynamic_pareto_rerank(
        candidates, 
        genre_cols=GENRE_COLS, 
        objectives=['novelty', 'quality'], 
        k=k
    )

# ── 修復：將快取函數移至全域範圍，避免定義在 loop 中導致當機 ────────────────
@st.cache_data(show_spinner=False)
def get_base_pareto_layers(_df):
    """快取基礎分層結果，避免 Optuna 搜尋時重複執行昂貴的 Pareto 計算"""
    _df = _df.copy()
    if 'quality' not in _df.columns: _df['quality'] = _df['predict_score']
    if 'novelty_norm' not in _df.columns: _df['novelty_norm'] = 0.5
    # 這裡呼叫原始的 recommend_pareto 來取得所有分層
    from week4_reranking import pareto_rerank
    return pareto_rerank(_df, k=len(_df))
# ────────────────────────────────────────────────────────────────────────

def get_cold_start_candidates(tmdb_api_key=None):
    """取得全體電影作為冷啟動候選池，並整合 TMDB 資料"""
    # 基礎 Local 資料
    df = movies_df.copy()
    df['source'] = 'movielens'
    
    # Cold-start 無個人化預測分數，設為 0
    df['predict_score'] = 0.0
    
    # 整合 TMDB 熱門片 (若有 Key)
    if tmdb_api_key:
        try:
            from tmdb_api import TMDBClient
            client = TMDBClient(tmdb_api_key)
            # 冷啟動直接抓取「熱門電影」作為候選 pool
            tmdb_df = client.get_popular_movies(count=50, genre_cols=GENRE_COLS)
            if not tmdb_df.empty:
                df = pd.concat([df, tmdb_df], ignore_index=True)
        except Exception as e:
            # st.warning(f"TMDB 獲取失敗: {e}")
            pass

    # 確保關鍵維度存在
    if 'novelty_norm' not in df.columns: df['novelty_norm'] = df.get('novelty', 0.5)
    if 'quality' not in df.columns: df['quality'] = 0.5
    if 'recency' not in df.columns:
        if 'release_year' in df.columns:
            from sklearn.preprocessing import MinMaxScaler
            df['recency'] = MinMaxScaler().fit_transform(df[['release_year']].fillna(df['release_year'].median()))
        else:
            df['recency'] = 0.5
    return df

def recommend_nlp_pareto(candidates, k, prompt, use_llm=False, api_key=None):
    from week5_nlp_pareto import parse_query_rule, parse_query_llm, dynamic_pareto_rerank
    if use_llm and api_key:
        parsed = parse_query_llm(prompt, api_key=api_key)
    else:
        parsed = parse_query_rule(prompt)
    
    # 權重處理與對齊
    w = parsed.get("weights", {})
    new_weights = {
        "novelty": w.get("novelty", 0.0),
        "quality": w.get("quality", 0.0),
        "recency": w.get("recency", 0.0),
        "diversity": w.get("diversity", 0.0)
    }
    
    # 檢查是否為 Fallback (無權重)
    if sum(new_weights.values()) == 0:
        parsed["weights"] = {"novelty": 0.0, "quality": 0.0, "recency": 0.0, "diversity": 0.0}
        parsed["explanation"] = "⚠️ 沒有找到相符合的條件，因此推薦最近期的電影。"
        # 為了讓排序能執行，內部給予微量 recency，但顯示維持 0
        internal_weights = {"recency": 1.0} 
    else:
        parsed["weights"] = new_weights
        internal_weights = new_weights

    # 呼叫 Pareto 排序
    result_df = dynamic_pareto_rerank(
        candidates, 
        genre_cols=GENRE_COLS, 
        objectives=['novelty', 'quality', 'recency', 'diversity'], 
        parsed_result=parsed,
        k=k
    )
    return result_df, parsed

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
    }

    # 方法專屬欄位
    method_cols = {}
    if method_name == "MMR":
        # MMR 順序：Penalty -> Score
        # 根據要求移除 Source, Novelty
        optional_cols = {}
        method_cols = {'similarity_penalty': 'Sim Penalty', 'mmr_score': 'MMR Score'}
    elif method_name == "Pareto":
        # Pareto 離線模式：顯示 Novelty, Quality 並加回 Pareto Rank 與 Weighted Score
        optional_cols = {
            'novelty': 'Novelty',
            'quality': 'Quality'
        }
        method_cols = {'pareto_rank': 'Pareto Rank', 'weighted_score': 'Weighted Score'}
    elif method_name == "Pareto + NLP":
        # 互動模式移除內部過程欄位，顯示最終分數
        optional_cols.update({'recency': 'Recency', 'quality': 'Quality', 'diversity': 'Diversity'})
        method_cols = {'final_score': 'Final Score'}
    elif method_name == "Baseline":
        optional_cols = {}
        # 指定要顯示 SHAP 的 5 個特徵
        shap_feat_cols = ['mf_score', 'user_avg_rating', 'user_genre_avg_score', 'cooc_hit_count', 'user_genre_max_score']
        for col in shap_feat_cols:
            if f'SHAP_{col}' in df.columns:
                method_cols[f'SHAP_{col}'] = col
    
    all_wanted = {**base_cols, **optional_cols, **method_cols}
    
    # 構建最終顯示字典
    all_final = {}
    all_final['movie_title'] = 'Movie Title'
    
    # 情況 A: 非 Baseline 且非互動模式，LGB 分數維持在前面 (第二欄)
    if method_name != "Baseline" and method_name != "Pareto + NLP" and 'predict_score' in df.columns:
        all_final['predict_score'] = 'Preference (LGB)'
    
    # 加入中間欄位 (Source, Novelty, Recency, Quality, Pareto Rank, MMR Score 等)
    for k, v in optional_cols.items(): all_final[k] = v
    for k, v in method_cols.items(): all_final[k] = v
    
    # 情況 B: 僅在 Baseline 模式下，LGB 分數放到最後面
    if method_name == "Baseline" and 'predict_score' in df.columns:
        all_final['predict_score'] = 'Preference (LGB)'

    existing = {k: v for k, v in all_final.items() if k in df.columns}
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
            # --- 為 Baseline 計算指定的 5 個 SHAP 特徵 ---
            try:
                from shap_explainer import get_cached_explainer
                explainer = get_cached_explainer(lgb_model, lgb_features)
                X_explain = result_df[lgb_features]
                shap_values = explainer.get_shap_values(X_explain)
                
                # 指定要顯示的 5 個特徵及其索引
                target_features = ['mf_score', 'user_avg_rating', 'user_genre_avg_score', 'cooc_hit_count', 'user_genre_max_score']
                feat_to_idx = {f: i for i, f in enumerate(lgb_features)}
                
                for feat in target_features:
                    if feat in feat_to_idx:
                        idx = feat_to_idx[feat]
                        vals = shap_values[:, idx]
                        result_df[f'SHAP_{feat}'] = [
                            f"{v:+.4f}" for v in vals
                        ]
            except Exception as e:
                st.error(f"SHAP 計算出錯: {e}")
        elif method == "MMR":
            result_df = recommend_mmr(candidates, top_k, lambda_val)
        elif method == "Pareto":
            # Step 3: Pareto 先產生 layer
            with st.spinner("⏳ 正在計算 Pareto 分層 (4-Objective)..."):
                pareto_candidate_df = get_base_pareto_layers(candidates)
            
            st.subheader("🔍 Optuna Weighted Tie-break")
            
            # 初始化 Optuna 相關的 session state
            if 'optuna_res' not in st.session_state:
                st.session_state['optuna_res'] = None

            # 若使用者或 Top-K 改變，清除舊的 Optuna 結果
            if st.session_state.get('optuna_user_cache') != (user_id, top_k):
                st.session_state['optuna_res'] = None
                st.session_state['optuna_user_cache'] = (user_id, top_k)

            actual_dict = test_ground_truth.get(user_id, {})
            
            if not actual_dict:
                st.warning("⚠️ 此使用者在測試集中沒有 Ground Truth 資料，無法執行 NDCG 優化。")
                final_result = pareto_candidate_df.copy()
                final_result['weighted_score'] = (
                    0.5 * final_result['novelty_norm'] + 0.5 * final_result['quality']
                )
                result_df = final_result.sort_values(['pareto_rank', 'weighted_score'], ascending=[True, False]).head(top_k)
            else:
                # 自動執行 Optuna 搜尋 (若尚未計算過)
                if st.session_state['optuna_res'] is None:
                    with st.spinner(f"🔍 Optuna 正在為 User {user_id} 搜尋最佳權重 (50 trials)..."):
                        try:
                            best_weights, best_score, final_result = run_optuna_weight_search(
                                search_df=pareto_candidate_df,
                                actual_dict=actual_dict,
                                ndcg_func=ndcg_at_k,
                                top_k=top_k,
                                n_trials=50,
                                genre_cols=GENRE_COLS
                            )
                            st.session_state['optuna_res'] = (best_weights, best_score, final_result)
                        except Exception as e:
                            st.error(f"執行出錯: {e}")
                            result_df = pareto_candidate_df.head(top_k) # 出錯時的備案

                # 顯示搜尋結果
                if st.session_state['optuna_res']:
                    best_weights, best_score, final_result = st.session_state['optuna_res']
                    st.success(f"✅ Optuna 自動優化完成！NDCG@{top_k}: {best_score * 100:.2f}%")
                    
                    if best_score > 0:
                        st.markdown("#### 🎯 Optuna 最佳權重配置 (離線優化)")
                        cols = st.columns(3)
                        cols[0].metric("Relevance", f"{best_weights['relevance']:.1%}")
                        cols[1].metric("Novelty", f"{best_weights['novelty']:.1%}")
                        cols[2].metric("Quality", f"{best_weights['quality']:.1%}")
                    
                    result_df = final_result
                else:
                    result_df = pareto_candidate_df.head(top_k)

    actual = test_ground_truth.get(user_id, {})
    liked_ids = [mid for mid, r in actual.items() if r >= 3.0]

    st.subheader(f"📋 1. {method} 推薦結果 (Offline)")
    
    # 最終多樣性補丁：在顯示前重新計算每部片對清單多樣性的貢獻
    if 'diversity' not in result_df.columns or (result_df['diversity'] == 0).all():
        from week6_evaluation import calculate_ild_at_k
        # 這裡我們計算一個簡單的指標：該片與清單內其他片的平均 Jaccard 距離
        result_df = result_df.copy()
        genres = result_df[GENRE_COLS].values
        for i in range(len(result_df)):
            if len(result_df) <= 1: 
                result_df.loc[result_df.index[i], 'diversity'] = 1.0
                continue
            # 計算與其他電影的平均距離
            dist = 0
            for j in range(len(result_df)):
                if i == j: continue
                intersection = np.logical_and(genres[i], genres[j]).sum()
                union = np.logical_or(genres[i], genres[j]).sum()
                dist += (1 - intersection / union) if union > 0 else 1.0
            result_df.loc[result_df.index[i], 'diversity'] = dist / (len(result_df) - 1)
        
    st.dataframe(
        format_display(result_df, method, liked_ids=liked_ids)
    )

    # --- 刪除原本的 SHAP 詳細區塊 ---
    pass

    st.markdown("---")
    st.subheader("📈 2. 離線效能指標 (Metrics)")
    st.caption(f"Evaluation Metrics @ Top-{top_k}")
    preds = result_df['movie_id'].tolist()
    m_ndcg = ndcg_at_k(actual, preds, k=top_k)
    m_recall = recall_at_k(actual, preds, k=top_k, threshold=3.0)
    m_novelty = calculate_novelty_at_k(result_df)
    m_ild = calculate_ild_at_k(result_df, GENRE_COLS)

    metrics_data = {
        "指標項目": [f"NDCG@{top_k}", f"Recall@{top_k}", f"Novelty@{top_k}", f"ILD@{top_k}"],
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
            st.dataframe(liked_movies[['movie_title', 'Rating']].sort_values('Rating', ascending=False))

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
        interactive_run = st.button("🚀 產生推薦", type="primary")

    # 修改：只要 nlp_prompt 有內容且按下 Enter (Streamlit 預設行為) 或點擊認按鈕，即執行推薦
    if interactive_run or (nlp_prompt and nlp_prompt.strip()):
        # 改為冷啟動候選池：整合 TMDB
        candidates = get_cold_start_candidates(tmdb_api_key=tmdb_api_key)
        
        # --- 執行互動式推薦 ---
        with st.spinner("🧠 正在根據您的指令在四維空間搜尋最佳解..."):
            result_df, parsed = recommend_nlp_pareto(candidates, top_k, nlp_prompt, use_llm=use_llm, api_key=api_key)

        # 1. 語意解析說明
        st.subheader("💡 1. 語意解析說明 (Explainability)")
        
        # 若為 Fallback 狀態，顯示警告
        if not parsed.get("objectives"):
            st.warning(parsed["explanation"])
        
        p_type = parsed.get("_parser", "rule_based")
        st.success(f"✅ 解析完成 ({p_type.upper()})")
        
        # 移除 LLM 報錯細節 

        weights = parsed.get("weights", {})
        if weights:
            st.markdown("**🎯 冷啟動維度佔比 (Relative Importance)**")
            total_w = sum(weights.values()) if sum(weights.values()) > 0 else 1.0
            
            w_cols = st.columns(4)
            w_cols[0].metric("冷門度", f"{(weights.get('novelty', 0)/total_w)*100:.0f}%")
            w_cols[1].metric("品質評分", f"{(weights.get('quality', 0)/total_w)*100:.0f}%")
            w_cols[2].metric("新舊程度", f"{(weights.get('recency', 0)/total_w)*100:.0f}%")
            w_cols[3].metric("多樣性", f"{(weights.get('diversity', 0)/total_w)*100:.0f}%")
            # 顯示解析思維
            if "explanation" in parsed:
                p_type = parsed.get("_parser", "rule_based")
                if "fallback" in p_type or p_type == "rule_based":
                    st.warning("**💡 備用解析模式 (AI 忙碌中)**")
                else:
                    st.info(f"**🤖 Gemini 解析思維：**  \n{parsed['explanation']}")

        st.markdown("---")
        
        # 2. 精選推薦詳情 (Top 5 Featured)
        st.subheader("🌟 2. 精選推薦詳情 (Top 5 Featured)")
        for idx, row in result_df.head(5).iterrows():
            with st.container():
                # Step 7 & 8: 顯示圖片與簡介
                col1, col2 = st.columns([1, 4])
                
                # 確保變數在迴圈內被正確初始化
                poster_url = row.get('poster_url') if 'poster_url' in row else None
                description = row.get('overview') if 'overview' in row else ""
                
                # --- 新增：針對 Local 電影自動從 TMDB 補完海報與簡介 ---
                if (pd.notnull(tmdb_api_key)) and (pd.isna(poster_url) or not poster_url):
                    p, d = get_movie_details_from_tmdb(row['movie_title'], tmdb_api_key)
                    if p: poster_url = p
                    if d: description = d
                
                # Fallback 處理：確保 poster_url 最終為字串或 None
                if pd.isna(poster_url) or not poster_url:
                    poster_url = row.get('poster_path')
                    if pd.isna(poster_url) or not poster_url:
                        poster_url = "https://via.placeholder.com/150x225?text=MovieLens"
                
                with col1:
                    # 確保 poster_url 是有效的 URL
                    is_valid_url = pd.notnull(poster_url) and str(poster_url).startswith("http")
                    display_url = poster_url if is_valid_url else "https://via.placeholder.com/150x225?text=No+Poster"
                    
                    st.image(display_url, use_container_width=True)
                    if is_valid_url:
                        st.caption("來源：TMDB")
                
                with col2:
                    st.subheader(f"{row['movie_title']}")
                    st.markdown(f"**年份**: {row.get('release_year', 'Unknown')}")
                    
                    active_genres = [g for g in GENRE_COLS if row.get(g) == 1]
                    if active_genres: st.markdown(f"**類型**: {' · '.join(active_genres)}")
                    
                    if description and pd.notnull(description):
                        st.write(description)
                    else:
                        st.write("此電影來自 MovieLens 資料集，正在尋找詳細簡介...")
                    st.markdown("---")

        # 3. 推薦結果表格 - 移到後面
        st.subheader("🎬 3. 推薦結果與詳情")
        st.dataframe(format_display(result_df, "Pareto + NLP"))
    else:
        st.info("💡 請在上方輸入框描述您的電影需求，然後點擊「產生推薦」。")
