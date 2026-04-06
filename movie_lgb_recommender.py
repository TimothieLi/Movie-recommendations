import os
import pandas as pd
import numpy as np
import lightgbm as lgb
import warnings
from collections import defaultdict

warnings.filterwarnings("ignore")

# ==========================================
# [Evaluation Metrics Functions]
# ==========================================
def recall_at_k(actual_dict, predicted_list, k=10, threshold=3.0):
    """
    計算 Recall@K
    actual_dict: 該 user 在 test set 真實的評分字典, {movie_id: rating}
    predicted_list: 推薦系統預測最高分的前 K 名 movie_id 列表
    threshold: 定義 user 真正「喜歡」的分數門檻 (預設 >= 3.0 分算相關)
    """
    # 找出真實喜歡的電影
    relevant_items = set([m for m, r in actual_dict.items() if r >= threshold])
    if not relevant_items:
        return 0.0
    
    top_k_preds = set(predicted_list[:k])
    hits = len(top_k_preds & relevant_items)
    return hits / len(relevant_items) # 或可以除以 min(len(relevant_items), k) 視具體標準而定，這裡採嚴格定義

def dcg_at_k(actual_dict, predicted_list, k=10):
    """計算單筆 DCG@K"""
    k = min(k, len(predicted_list))
    dcg = 0.0
    for i, p in enumerate(predicted_list[:k]):
        if p in actual_dict:
            rel = actual_dict[p]  # 使用實際 rating 當作 relevance 分數
            dcg += rel / np.log2(i + 2)
    return dcg

def ndcg_at_k(actual_dict, predicted_list, k=10):
    """計算單筆 NDCG@K"""
    dcg = dcg_at_k(actual_dict, predicted_list, k)
    
    # 計算 IDCG (Ideal DCG)：將 user 真實評過分的項目，從高到低完美排序取前 K 個
    ideal_ranked = sorted(actual_dict.values(), reverse=True)
    idcg = 0.0
    for i, rel in enumerate(ideal_ranked[:k]):
        idcg += rel / np.log2(i + 2)
        
    if idcg == 0:
        return 0.0
    return dcg / idcg

def run_recommender_pipeline():
    # ==========================================
    # 1. 資料前處理 (Data Preprocessing)
    # ==========================================
    print("Loading datasets...")
    data_cols = ['user_id', 'movie_id', 'rating', 'timestamp']
    
    u_data_path = os.path.join('MovieLens 100K', 'u.data')
    u_item_path = os.path.join('MovieLens 100K', 'u.item')
    
    # Fallback checking current directory
    if not os.path.exists(u_data_path): u_data_path = 'u.data'
    if not os.path.exists(u_item_path): u_item_path = 'u.item'

    if not os.path.exists(u_data_path) or not os.path.exists(u_item_path):
        print("請確定 u.data 與 u.item (MovieLens 100K) 在同一個資料夾，或 'MovieLens 100K' 目錄下。")
        return
    
    ratings_df = pd.read_csv(u_data_path, sep='\t', names=data_cols)
    
    genre_cols = [
        "unknown", "Action", "Adventure", "Animation",
        "Children's", "Comedy", "Crime", "Documentary", "Drama", "Fantasy",
        "Film-Noir", "Horror", "Musical", "Mystery", "Romance", "Sci-Fi",
        "Thriller", "War", "Western"
    ]
    item_cols = ['movie_id', 'movie_title', 'release_date', 'video_release_date', 'imdb_url'] + genre_cols
    movies_df = pd.read_csv(u_item_path, sep='|', names=item_cols, encoding='latin-1')
    movies_df = movies_df[['movie_id', 'movie_title'] + genre_cols]
    
    # ==========================================
    # 2. 資料切分 (Train/Test Split 80/20)
    # ==========================================
    ratings_df = ratings_df.sort_values('timestamp').reset_index(drop=True)
    split_idx = int(len(ratings_df) * 0.8)
    
    train_df = ratings_df.iloc[:split_idx].copy()
    test_df = ratings_df.iloc[split_idx:].copy()
    
    # 結合 rating 與 movie metadata 用於建立特徵
    train_df = train_df.merge(movies_df, on='movie_id', how='left')
    test_df = test_df.merge(movies_df, on='movie_id', how='left')
    
    # ==========================================
    # 3. 特徵工程 (Feature Engineering)
    # ==========================================
    print("Feature Engineering...")
    # [Item Feature] Popularity：電影在 train set 中的互動/被評分次數
    item_pop = train_df.groupby('movie_id').size().reset_index(name='popularity')
    # [User Feature] 使用者平均評分
    user_avg = train_df.groupby('user_id')['rating'].mean().reset_index(name='user_avg_rating')
    
    # [Interaction Feature 原料] 使用者對各 genre 的偏好輪廓
    user_genre_counts = train_df.groupby('user_id')[genre_cols].sum()
    user_genre_profile = user_genre_counts.div(user_genre_counts.sum(axis=1) + 1e-9, axis=0).reset_index()
    profile_cols = {g: f'user_pref_{g}' for g in genre_cols}
    user_genre_profile = user_genre_profile.rename(columns=profile_cols)
    
    def build_features(df):
        df_feat = df.merge(item_pop, on='movie_id', how='left').fillna({'popularity': 0})
        df_feat = df_feat.merge(user_avg, on='user_id', how='left')
        global_mean = train_df['rating'].mean()
        df_feat['user_avg_rating'] = df_feat['user_avg_rating'].fillna(global_mean) 
        
        df_feat = df_feat.merge(user_genre_profile, on='user_id', how='left')
        for g in genre_cols:
            df_feat[f'user_pref_{g}'] = df_feat[f'user_pref_{g}'].fillna(0)
            
        match_series = pd.Series(np.zeros(len(df_feat)), index=df_feat.index)
        for g in genre_cols:
            match_series += df_feat[g] * df_feat[f'user_pref_{g}']
        df_feat['genre_match'] = match_series
        
        return df_feat

    train_df_feat = build_features(train_df)
    test_df_feat = build_features(test_df)
    features_to_use = ['popularity', 'user_avg_rating', 'genre_match'] + genre_cols
    
    # ==========================================
    # 4. 訓練 LightGBM ranking model 
    # ==========================================
    train_df_feat = train_df_feat.sort_values('user_id').reset_index(drop=True)
    test_df_feat = test_df_feat.sort_values('user_id').reset_index(drop=True)
    
    train_groups = train_df_feat.groupby('user_id').size().values
    test_groups = test_df_feat.groupby('user_id').size().values
    
    X_train, y_train = train_df_feat[features_to_use], train_df_feat['rating']
    X_test,  y_test  = test_df_feat[features_to_use], test_df_feat['rating']
    
    print("Building and Training LightGBM LambdaRank Model...")
    lgb_train = lgb.Dataset(X_train, label=y_train, group=train_groups)
    lgb_eval = lgb.Dataset(X_test, label=y_test, group=test_groups, reference=lgb_train)
    
    params = {
        'objective': 'lambdarank',
        'metric': 'ndcg',
        'ndcg_eval_at': [10],
        'learning_rate': 0.1,
        'num_leaves': 31,
        'min_data_in_leaf': 20,
        'verbose': -1,
        'random_state': 42
    }
    
    try:
        callbacks = [lgb.early_stopping(stopping_rounds=20), lgb.log_evaluation(period=20)]
        model = lgb.train(
            params, lgb_train, num_boost_round=300,
            valid_sets=[lgb_train, lgb_eval], callbacks=callbacks
        )
    except AttributeError: # Fallback for older variations
        model = lgb.train(
            params, lgb_train, num_boost_round=300,
            valid_sets=[lgb_train, lgb_eval],
            early_stopping_rounds=20, verbose_eval=20
        )
    print("\nModel training finished!")

    # ==========================================
    # 5. 週 2 評估任務：對 Test Set 中所有 User 進行評估
    # ==========================================
    print("\nEvaluating all users in test set for Recall@10 and NDCG@10 (Batch Strategy)...")
    test_users = test_df['user_id'].unique()
    
    # 建立每個 user 的 ground truth {user_id: {movie_id: rating}}
    test_ground_truth = defaultdict(dict)
    for _, row in test_df.iterrows():
        test_ground_truth[row['user_id']][row['movie_id']] = row['rating']
    
    # 建立 Batch DataFrame：[Test Users] x [All Candidates]
    user_ids_rep = np.repeat(test_users, len(movies_df))
    movie_ids_rep = np.tile(movies_df['movie_id'].values, len(test_users))
    
    candidates_df = pd.DataFrame({'user_id': user_ids_rep, 'movie_id': movie_ids_rep})
    # 融合 Movie Metadata
    candidates_df = candidates_df.merge(movies_df, on='movie_id', how='left')
    
    # 批次提取特徵 (比一個一個 user 迴圈快上百倍)
    candidates_feat = build_features(candidates_df)
    
    # 一次對多達百萬等級的 rows 進行 Rank Score 預測！
    candidates_feat['predict_score'] = model.predict(candidates_feat[features_to_use])
    
    # 過濾掉該名使用者在 Train Set 中已經看過的電影
    train_seen_df = train_df[['user_id', 'movie_id']].copy()
    train_seen_df['seen'] = 1
    candidates_feat = candidates_feat.merge(train_seen_df, on=['user_id', 'movie_id'], how='left')
    unseen_candidates = candidates_feat[candidates_feat['seen'].isnull()].drop('seen', axis=1)
    
    # 針對各個使用者將分數由高到低排序，並取出 Top-10 電影清單
    unseen_candidates = unseen_candidates.sort_values(['user_id', 'predict_score'], ascending=[True, False])
    top_10_df = unseen_candidates.groupby('user_id').head(10)
    top_10_preds = top_10_df.groupby('user_id')['movie_id'].apply(list).to_dict()
    
    # ==========================================
    # 6. 計算整體平均 Recall@10 與 NDCG@10
    # ==========================================
    recalls = []
    ndcgs = []
    for uid in test_users:
        if uid not in test_ground_truth: continue
        actual = test_ground_truth[uid]
        preds = top_10_preds.get(uid, [])
        
        # 我們設定實際給分大於等於 3 首選才是 relevant items (Recall 門檻)
        recalls.append(recall_at_k(actual, preds, k=10, threshold=3.0))
        ndcgs.append(ndcg_at_k(actual, preds, k=10))
        
    avg_recall = np.mean(recalls) * 100
    avg_ndcg = np.mean(ndcgs) * 100
    
    print("\n==========================================")
    print("Baseline Performance (Offline Evaluation)")
    print("==========================================")
    print(f"Mean Recall@10 : {avg_recall:.2f}% (Threshold: rating >= 3.0)")
    print(f"Mean NDCG@10   : {avg_ndcg:.2f}%")
    print("==========================================\n")
    
    return test_users, top_10_df, test_ground_truth, movies_df, unseen_candidates

def main():
    test_users, top_10_df, test_ground_truth, movies_df, unseen_candidates = run_recommender_pipeline()
    
    # ==========================================
    # 7. 額外印出幾位 user 的推薦結果，幫助檢查結果是否合理
    # ==========================================
    sample_users = test_users[:2] # 挑選前 2 位測試集中的 User 作為範例
    
    for uid in sample_users:
        print(f"\n--- [Top-10 Recommended Movies for User {uid}] ---")
        user_top_movies = top_10_df[top_10_df['user_id'] == uid]
        actual_interactions = test_ground_truth[uid]
        actual_liked = [m for m, r in actual_interactions.items() if r >= 3.0]
        
        for rank, row in enumerate(user_top_movies.itertuples(), 1):
            hit_mark = ""
            if row.movie_id in actual_liked:
                hit_mark = f"⭐ [Hit! Test Rating: {actual_interactions[row.movie_id]:.0f}]"
                
            print(f"Rank {rank}: [Score={row.predict_score:.4f}] {row.movie_title} {hit_mark}")

if __name__ == "__main__":
    main()
