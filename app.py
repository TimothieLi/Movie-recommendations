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
def load_and_predict_all():
    # 執行整個 offline batch pipeline
    test_users, top_10_df, test_ground_truth, movies_df = run_recommender_pipeline()
    return list(test_users), top_10_df, test_ground_truth, movies_df

test_users, top_10_df, test_ground_truth, movies_df = load_and_predict_all()

# --- 2. 側邊欄 ---
st.sidebar.header("🕹️ 控制面板")
st.sidebar.write("請選擇要預覽的 Test Set User ID")

# 將 test_users 確保轉回 int 進行排序選擇
test_users_sorted = sorted([int(u) for u in test_users])
selected_user_id = st.sidebar.selectbox("User ID", test_users_sorted)

# --- 3. 畫面呈現 ---
st.markdown(f"### 👤 目前為 User `{selected_user_id}` 產生專屬推薦")
st.write("以下為系統預測該名使用者可能最喜歡的 Top-10 電影。如果電影後面標示有 ⭐，代表這部電影也剛好出現在他在 Test Set 的真實喜歡清單中！")

# 取得該 user 的推薦與真實喜好
user_top_movies = top_10_df[top_10_df['user_id'] == selected_user_id].copy()
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
