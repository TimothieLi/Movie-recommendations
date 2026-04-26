import os
import pandas as pd
import numpy as np
import lightgbm as lgb
import warnings
from collections import defaultdict
import matplotlib.pyplot as plt
from mf_features import train_mf_model, predict_mf_score

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

def run_recommender_pipeline(alpha=1.0, mf_weight=1.0, return_metrics=False, calc_importance=False):
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
    
    # === TMDB Metadata Integration (Week 5) ===
    tmdb_path = os.path.join('TMDB metadata', 'tmdb_5000_movies.csv')
    if not os.path.exists(tmdb_path): tmdb_path = 'tmdb_5000_movies.csv' # Fallback to current dir

    if os.path.exists(tmdb_path):
        tmdb_df = pd.read_csv(tmdb_path)
        
        # 1. MovieLens Title & Year Parsing
        movies_df['title_clean'] = movies_df['movie_title'].str.extract(r'^(.*?)(?:\s*\(\d{4}\))?$')[0].str.lower().str.strip()
        movies_df['year'] = movies_df['movie_title'].str.extract(r'\((\d{4})\)$')[0]
        
        # 2. TMDB Title & Year Parsing
        tmdb_df['title_clean'] = tmdb_df['title'].astype(str).str.lower().str.strip()
        tmdb_df['year'] = pd.to_datetime(tmdb_df['release_date'], errors='coerce').dt.year.astype('Int64').astype(str)
        
        # 3. Merge
        # rename TMDB popularity to avoid conflict with Item Popularity feature in LightGBM
        tmdb_subset = tmdb_df[['title_clean', 'year', 'popularity', 'vote_average', 'vote_count']].rename(columns={'popularity': 'tmdb_popularity'})
        tmdb_subset = tmdb_subset.drop_duplicates(subset=['title_clean', 'year'])
        
        movies_df = movies_df.merge(tmdb_subset, on=['title_clean', 'year'], how='left')
        
        # 4. Fill missing values with median
        movies_df['tmdb_popularity'] = movies_df['tmdb_popularity'].fillna(movies_df['tmdb_popularity'].median())
        movies_df['vote_average'] = movies_df['vote_average'].fillna(movies_df['vote_average'].median())
        
        movies_df['release_year'] = pd.to_numeric(movies_df['year'], errors='coerce')
        movies_df['release_year'] = movies_df['release_year'].fillna(movies_df['release_year'].median())
        
        # 5. Calculate normalized features for Week 5 NLP mapping
        from sklearn.preprocessing import MinMaxScaler
        scaler = MinMaxScaler()
        movies_df['recency'] = scaler.fit_transform(movies_df[['release_year']])
        movies_df['quality'] = scaler.fit_transform(movies_df[['vote_average']])
        
        max_pop = movies_df['tmdb_popularity'].max()
        if max_pop == 0: max_pop = 1
        raw_novelty = -np.log((movies_df['tmdb_popularity'] / max_pop) + 1e-9)
        movies_df['novelty'] = scaler.fit_transform(raw_novelty.values.reshape(-1, 1))
        
        movies_df = movies_df.drop(columns=['title_clean', 'year'])
    else:
        print("TMDB metadata not found!")
        for col in ['tmdb_popularity', 'vote_average', 'vote_count', 'recency', 'quality', 'novelty', 'release_year']:
            movies_df[col] = 0.5
    
    # ==========================================
    # 2. 資料切分 (Train/Validation/Test Split 80/10/10)
    # ==========================================
    ratings_df = ratings_df.sort_values('timestamp').reset_index(drop=True)
    train_end = int(len(ratings_df) * 0.8)
    valid_end = int(len(ratings_df) * 0.9)
    
    train_df = ratings_df.iloc[:train_end].copy()
    valid_df = ratings_df.iloc[train_end:valid_end].copy()
    test_df = ratings_df.iloc[valid_end:].copy()
    
    # 結合 rating 與 movie metadata 用於建立特徵
    train_df = train_df.merge(movies_df, on='movie_id', how='left')
    valid_df = valid_df.merge(movies_df, on='movie_id', how='left')
    test_df = test_df.merge(movies_df, on='movie_id', how='left')
    
    # ==========================================
    # 3. 特徵工程 (Feature Engineering)
    # ==========================================
    print("Feature Engineering...")
    
    # [新增] 訓練 Matrix Factorization 擷取 Latent Features (僅使用 train_df)
    mf_model = train_mf_model(train_df)
    
    # [Item Feature] Popularity：電影在 train set 中的互動/被評分次數
    item_pop = train_df.groupby('movie_id').size().reset_index(name='popularity')
    # [User Feature] 使用者平均評分
    user_avg = train_df.groupby('user_id')['rating'].mean().reset_index(name='user_avg_rating')
    
    # [新增] Item Co-occurrence 計算基底矩陣 (Train-only)
    U_max = int(ratings_df['user_id'].max()) + 2
    M_max = int(max(ratings_df['movie_id'].max(), movies_df['movie_id'].max())) + 2
    
    user_train_history = train_df.groupby('user_id')['movie_id'].apply(set).to_dict()
    
    cooc_matrix = np.zeros((M_max, M_max), dtype=np.float32)
    for uid, history in user_train_history.items():
        hist_list = list(history)
        if len(hist_list) > 1:
            for i in range(len(hist_list)):
                for j in range(i + 1, len(hist_list)):
                    m1, m2 = hist_list[i], hist_list[j]
                    cooc_matrix[m1, m2] += 1
                    cooc_matrix[m2, m1] += 1
                    
    hist_matrix = np.zeros((U_max, M_max), dtype=np.float32)
    for uid, history in user_train_history.items():
        hist_matrix[uid, list(history)] = 1.0
        
    user_cooc_sum_matrix = np.dot(hist_matrix, cooc_matrix)
    user_hist_sizes = np.sum(hist_matrix, axis=1)
    
    cooc_indicator = (cooc_matrix > 0).astype(np.float32)
    user_cooc_hit_matrix = np.dot(hist_matrix, cooc_indicator)
    
    user_cooc_max_matrix = np.zeros((U_max, M_max), dtype=np.float32)
    for uid, history in user_train_history.items():
        if history:
            user_cooc_max_matrix[uid] = np.max(cooc_matrix[list(history)], axis=0)
    
    # [Interaction Feature 原料] 使用者對各 genre 的偏好輪廓
    user_genre_counts = train_df.groupby('user_id')[genre_cols].sum()
    user_genre_profile = user_genre_counts.div(user_genre_counts.sum(axis=1) + 1e-9, axis=0).reset_index()
    profile_cols = {g: f'user_pref_{g}' for g in genre_cols}
    user_genre_profile = user_genre_profile.rename(columns=profile_cols)
    
    # [新增] User History 特徵基礎值：user-genre 的歷史平均評分與互動次數
    user_genre_avg_dict = {}
    user_genre_cnt_dict = {}
    for g in genre_cols:
        g_mask = train_df[g] == 1
        user_genre_avg_dict[f'ug_avg_{g}'] = train_df[g_mask].groupby('user_id')['rating'].mean()
        user_genre_cnt_dict[f'ug_cnt_{g}'] = train_df[g_mask].groupby('user_id').size()
        
    ug_avg_df = pd.DataFrame(user_genre_avg_dict).reset_index().fillna(0)
    ug_cnt_df = pd.DataFrame(user_genre_cnt_dict).reset_index().fillna(0)
    
    def build_features(df, current_alpha=1.0, current_mf_weight=1.0):
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
        
        # [新增] 整合 User History 特徵
        df_feat = df_feat.merge(ug_avg_df, on='user_id', how='left')
        df_feat = df_feat.merge(ug_cnt_df, on='user_id', how='left')
        
        ug_avg_cols = [f'ug_avg_{g}' for g in genre_cols]
        ug_cnt_cols = [f'ug_cnt_{g}' for g in genre_cols]
        
        # 若 user 在 train 從未出現，補 0
        df_feat[ug_avg_cols] = df_feat[ug_avg_cols].fillna(0)
        df_feat[ug_cnt_cols] = df_feat[ug_cnt_cols].fillna(0)
        
        # 針對 candidate 的 genres 取出對應的歷史特徵矩陣
        item_genres_mat = df_feat[genre_cols].values
        ug_avg_mat = df_feat[ug_avg_cols].values
        ug_cnt_mat = df_feat[ug_cnt_cols].values
        
        # 只保留與該電影對應的 genres 相關的特徵
        active_avg_mat = ug_avg_mat * item_genres_mat
        active_cnt_mat = ug_cnt_mat * item_genres_mat
        
        genre_count_per_movie = item_genres_mat.sum(axis=1) + 1e-9
        
        # 1. 該電影對應 genre 的 user 平均分數
        df_feat['user_genre_avg_score'] = active_avg_mat.sum(axis=1) / genre_count_per_movie
        # 2. 該電影對應 genre 的 user 參與總次數
        df_feat['user_genre_total_count'] = active_cnt_mat.sum(axis=1)
        # 3. 該電影擁有的 genre 中的最大歷史 user 評分
        df_feat['user_genre_max_score'] = np.max(active_avg_mat, axis=1)
        # 4. 該電影擁有的 genre 中的最大歷史參與次數
        df_feat['user_genre_max_count'] = np.max(active_cnt_mat, axis=1)
        
        # 移除中介特徵，節省記憶體空間
        df_feat = df_feat.drop(columns=ug_avg_cols + ug_cnt_cols)
        
        # [新增] 整合 Co-occurrence 特徵
        u_idx = np.clip(df_feat['user_id'].astype(int).values, 0, U_max - 1)
        m_idx = np.clip(df_feat['movie_id'].astype(int).values, 0, M_max - 1)
        
        cooc_sum_arr = user_cooc_sum_matrix[u_idx, m_idx]
        cooc_max_arr = user_cooc_max_matrix[u_idx, m_idx]
        cooc_hit_arr = user_cooc_hit_matrix[u_idx, m_idx]
        
        # 精準計算歷史數目並求平均共現（避免 candidate 自己也在 history 裡灌水）
        user_seen_arr = hist_matrix[u_idx, m_idx]
        active_hist_len = user_hist_sizes[u_idx] - user_seen_arr
        active_hist_len[active_hist_len <= 0] = 1.0
        
        cooc_mean_arr = cooc_sum_arr / active_hist_len
        
        # 任務 1 & 2: Normalize features & 新增 hit count
        df_feat['cooc_sum'] = np.log1p(cooc_sum_arr)
        df_feat['cooc_max'] = np.log1p(cooc_max_arr)
        df_feat['cooc_mean'] = np.log1p(cooc_mean_arr)
        df_feat['cooc_hit_count'] = cooc_hit_arr
        
        # 任務 3: Popularity Penalty 
        # (將名稱設為 pop_novelty，以避免覆蓋原先給 Pareto 和 NLP 模組使用的 'novelty' 欄位)
        df_feat['pop_novelty'] = -current_alpha * np.log1p(df_feat['popularity'])
        
        # [新增] 整合 Matrix Factorization (Latent Features)
        mf_scores = predict_mf_score(mf_model, df_feat['user_id'].values, df_feat['movie_id'].values)
        df_feat['mf_score_raw'] = mf_scores
        df_feat['mf_score'] = current_mf_weight * df_feat['mf_score_raw']
        
        return df_feat

    train_df_feat = build_features(train_df, current_alpha=alpha, current_mf_weight=mf_weight)
    valid_df_feat = build_features(valid_df, current_alpha=alpha, current_mf_weight=mf_weight)
    test_df_feat = build_features(test_df, current_alpha=alpha, current_mf_weight=mf_weight)
    features_to_use = [
        'popularity', 'user_avg_rating', 'genre_match',
        'user_genre_avg_score', 'user_genre_total_count',
        'user_genre_max_score', 'user_genre_max_count',
        'cooc_sum', 'cooc_max', 'cooc_mean',
        'cooc_hit_count', 'pop_novelty', 'mf_score'
    ] + genre_cols
    
    # ==========================================
    # 4. 訓練 LightGBM ranking model 
    # ==========================================
    train_df_feat = train_df_feat.sort_values('user_id').reset_index(drop=True)
    valid_df_feat = valid_df_feat.sort_values('user_id').reset_index(drop=True)
    test_df_feat = test_df_feat.sort_values('user_id').reset_index(drop=True)
    
    train_groups = train_df_feat.groupby('user_id').size().values
    valid_groups = valid_df_feat.groupby('user_id').size().values
    test_groups = test_df_feat.groupby('user_id').size().values
    
    X_train, y_train = train_df_feat[features_to_use], train_df_feat['rating']
    X_valid, y_valid = valid_df_feat[features_to_use], valid_df_feat['rating']
    X_test,  y_test  = test_df_feat[features_to_use], test_df_feat['rating']
    
    print("Building and Training LightGBM LambdaRank Model...")
    lgb_train = lgb.Dataset(X_train, label=y_train, group=train_groups)
    lgb_valid = lgb.Dataset(X_valid, label=y_valid, group=valid_groups, reference=lgb_train)
    
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
            valid_sets=[lgb_train, lgb_valid], callbacks=callbacks
        )
    except AttributeError: # Fallback for older variations
        model = lgb.train(
            params, lgb_train, num_boost_round=300,
            valid_sets=[lgb_train, lgb_valid],
            early_stopping_rounds=20, verbose_eval=20
        )
    print("\nModel training finished!")

    # ==========================================
    # 4.5. Validation 評估：針對 Validation Set 中所有 User 進行評估
    # ==========================================
    print("\nEvaluating all users in validation set for Recall@10 and NDCG@10...")
    valid_users = valid_df['user_id'].unique()
    
    valid_ground_truth = defaultdict(dict)
    for _, row in valid_df.iterrows():
        valid_ground_truth[row['user_id']][row['movie_id']] = row['rating']
        
    valid_user_ids_rep = np.repeat(valid_users, len(movies_df))
    valid_movie_ids_rep = np.tile(movies_df['movie_id'].values, len(valid_users))
    
    valid_candidates_df = pd.DataFrame({'user_id': valid_user_ids_rep, 'movie_id': valid_movie_ids_rep})
    valid_candidates_df = valid_candidates_df.merge(movies_df, on='movie_id', how='left')
    
    valid_candidates_feat = build_features(valid_candidates_df, current_alpha=alpha, current_mf_weight=mf_weight)
    valid_candidates_feat['predict_score'] = model.predict(valid_candidates_feat[features_to_use])
    
    # 針對 validation 評估，只排除 train set 中看過的電影
    train_seen_valid_df = train_df[['user_id', 'movie_id']].copy()
    train_seen_valid_df['seen'] = 1
    valid_candidates_feat = valid_candidates_feat.merge(train_seen_valid_df, on=['user_id', 'movie_id'], how='left')
    valid_unseen_candidates = valid_candidates_feat[valid_candidates_feat['seen'].isnull()].drop('seen', axis=1)
    
    valid_unseen_candidates = valid_unseen_candidates.sort_values(['user_id', 'predict_score'], ascending=[True, False])
    valid_top_10_df = valid_unseen_candidates.groupby('user_id').head(10)
    valid_top_10_preds = valid_top_10_df.groupby('user_id')['movie_id'].apply(list).to_dict()
    
    valid_recalls = []
    valid_ndcgs = []
    for uid in valid_users:
        if uid not in valid_ground_truth: continue
        actual = valid_ground_truth[uid]
        preds = valid_top_10_preds.get(uid, [])
        
        valid_recalls.append(recall_at_k(actual, preds, k=10, threshold=3.0))
        valid_ndcgs.append(ndcg_at_k(actual, preds, k=10))
        
    avg_valid_recall = np.mean(valid_recalls) * 100
    avg_valid_ndcg = np.mean(valid_ndcgs) * 100
    
    print("\n==========================================")
    print("Validation Performance")
    print("==========================================")
    print(f"Mean Recall@10 : {avg_valid_recall:.2f}%")
    print(f"Mean NDCG@10   : {avg_valid_ndcg:.2f}%")

    # ==========================================
    # 4.6. 特徵重要性分析 (Feature Importance)
    # ==========================================
    if calc_importance:
        importances = model.feature_importance()
        feature_names = model.feature_name()
    
        feat_imp_df = pd.DataFrame({
            'feature': feature_names,
            'importance': importances
        }).sort_values(by='importance', ascending=False)
    
        print("\nTop Feature Importance")
        print("---------------------")
        for _, row in feat_imp_df.head(15).iterrows():
            print(f"{row['feature']} : {row['importance']}")
    
        plt.figure(figsize=(10, 8))
        top_20_imp = feat_imp_df.head(20).sort_values(by='importance', ascending=True)
        plt.barh(top_20_imp['feature'], top_20_imp['importance'])
        plt.title('LightGBM Feature Importance (Split)')
        plt.xlabel('Importance')
        plt.ylabel('Feature Name')
        plt.tight_layout()
        plt.savefig('feature_importance_lgb.png')
        plt.close()
    
        # ==========================================
        # 4.7. Permutation Importance 分析 (Validation)
        # ==========================================
        baseline_ndcg = avg_valid_ndcg / 100.0
        perm_results = []
        
        eval_df = valid_unseen_candidates.copy()
        
        for feat in features_to_use:
            orig_col = eval_df[feat].values.copy()
            
            shuffled_col = orig_col.copy()
            np.random.shuffle(shuffled_col)
            eval_df[feat] = shuffled_col
            
            # 重新 predict
            eval_df['predict_score'] = model.predict(eval_df[features_to_use])
            
            # 重新排序挑 Top 10
            eval_sorted = eval_df.sort_values(['user_id', 'predict_score'], ascending=[True, False])
            eval_top_10 = eval_sorted.groupby('user_id').head(10)
            eval_preds = eval_top_10.groupby('user_id')['movie_id'].apply(list).to_dict()
            
            # 計算 shuffled NDCG
            pf_ndcgs = []
            for uid in valid_users:
                if uid not in valid_ground_truth: continue
                actual = valid_ground_truth[uid]
                preds = eval_preds.get(uid, [])
                pf_ndcgs.append(ndcg_at_k(actual, preds, k=10))
                
            shuffled_ndcg = np.mean(pf_ndcgs)
            drop_ndcg = baseline_ndcg - shuffled_ndcg
            
            perm_results.append({
                'feature': feat,
                'importance': drop_ndcg
            })
            
            # 還原欄位
            eval_df[feat] = orig_col
            
        perm_df = pd.DataFrame(perm_results).sort_values(by='importance', ascending=False)
        
        print("\nPermutation Feature Importance")
        print("-----------------------------")
        for _, row in perm_df.head(20).iterrows():
            print(f"{row['feature']} : {row['importance']:.6f}")
            
        plt.figure(figsize=(10, 8))
        top_20_perm = perm_df.head(20).sort_values(by='importance', ascending=True)
        plt.barh(top_20_perm['feature'], top_20_perm['importance'])
        plt.title('Permutation Importance (Drop in NDCG@10)')
        plt.xlabel('Importance (Decrease in NDCG)')
        plt.ylabel('Feature Name')
        plt.tight_layout()
        plt.savefig('feature_importance_permutation.png')
        plt.close()

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
    candidates_feat = build_features(candidates_df, current_alpha=alpha, current_mf_weight=mf_weight)
    
    # 一次對多達百萬等級的 rows 進行 Rank Score 預測！
    candidates_feat['predict_score'] = model.predict(candidates_feat[features_to_use])
    
    # 過濾掉該名使用者在 Train Set 與 Validation Set 中已經看過的電影
    seen_df = pd.concat([
        train_df[['user_id', 'movie_id']],
        valid_df[['user_id', 'movie_id']]
    ]).drop_duplicates()
    seen_df['seen'] = 1
    candidates_feat = candidates_feat.merge(seen_df, on=['user_id', 'movie_id'], how='left')
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
    
    if return_metrics:
        return test_users, top_10_df, test_ground_truth, movies_df, unseen_candidates, avg_valid_recall, avg_valid_ndcg, avg_recall, avg_ndcg, model, features_to_use
    return test_users, top_10_df, test_ground_truth, movies_df, unseen_candidates, model, features_to_use

def main():
    # 固定 alpha 為 0.5 (假設這是之前跑出相對穩定的值)
    fixed_alpha = 0.5
    mf_weight_list = [0.3, 0.5, 0.7, 1.0]
    results = []
    
    print("\n==========================================")
    print("Starting MF Weight Sweep Experiment")
    print("==========================================")
    
    final_test_users, final_top_10_df, final_test_ground_truth = None, None, None
    
    for i, w in enumerate(mf_weight_list):
        print(f"\n>>> Running pipeline with mf_weight = {w} (alpha = {fixed_alpha})")
        # 只在最後一次跑 permutation importance, 避免過度花費時間
        run_importance = (i == len(mf_weight_list) - 1)
        ret = run_recommender_pipeline(alpha=fixed_alpha, mf_weight=w, return_metrics=True, calc_importance=run_importance)
        
        # Unpack
        t_users, t_top10, t_gt, t_mdf, t_unseen = ret[:5]
        v_rec, v_ndcg, t_rec, t_ndcg = ret[5:]
        
        results.append({
            'mf_weight': w,
            'val_recall': v_rec,
            'val_ndcg': v_ndcg,
            'test_recall': t_rec,
            'test_ndcg': t_ndcg
        })
        
        final_test_users = t_users
        final_top_10_df = t_top10
        final_test_ground_truth = t_gt
        
    results_df = pd.DataFrame(results)
    
    print("\nMF Weight Sweep Results")
    print("-----------------------")
    for _, row in results_df.iterrows():
        w = row['mf_weight']
        print(f"mf_weight={w} -> Recall={row['val_recall']:.2f}%, NDCG={row['val_ndcg']:.2f}% (Val) | Recall={row['test_recall']:.2f}%, NDCG={row['test_ndcg']:.2f}% (Test)")
        
    # Plotting
    plt.figure(figsize=(12, 5))
    
    # Plot 1: Recall vs MF Weight
    plt.subplot(1, 2, 1)
    plt.plot(results_df['mf_weight'], results_df['val_recall'], marker='o', label='Validation Recall@10')
    plt.plot(results_df['mf_weight'], results_df['test_recall'], marker='s', label='Test Recall@10')
    plt.title('Recall vs MF Weight')
    plt.xlabel('MF Weight')
    plt.ylabel('Recall@10 (%)')
    plt.grid(True)
    plt.legend()
    
    # Plot 2: NDCG vs MF Weight
    plt.subplot(1, 2, 2)
    plt.plot(results_df['mf_weight'], results_df['val_ndcg'], marker='o', label='Validation NDCG@10')
    plt.plot(results_df['mf_weight'], results_df['test_ndcg'], marker='s', label='Test NDCG@10')
    plt.title('NDCG vs MF Weight')
    plt.xlabel('MF Weight')
    plt.ylabel('NDCG@10 (%)')
    plt.grid(True)
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('mf_weight_sweep_results.png')
    print("\nSweep results plot saved to 'mf_weight_sweep_results.png'.")
    
    # ==========================================
    # 7. 額外印出幾位 user 的推薦結果，幫助檢查結果是否合理 (採用最後一個 weight)
    # ==========================================
    if final_test_users is not None:
        sample_users = final_test_users[:2] 
        for uid in sample_users:
            print(f"\n--- [Top-10 Recommended Movies for User {uid}] (mf_weight={mf_weight_list[-1]}) ---")
            user_top_movies = final_top_10_df[final_top_10_df['user_id'] == uid]
            actual_interactions = final_test_ground_truth[uid]
            actual_liked = [m for m, r in actual_interactions.items() if r >= 3.0]
            
            for rank, row in enumerate(user_top_movies.itertuples(), 1):
                hit_mark = ""
                if row.movie_id in actual_liked:
                    hit_mark = f"⭐ [Hit! Test Rating: {actual_interactions[row.movie_id]:.0f}]"
                print(f"Rank {rank}: [Score={row.predict_score:.4f}] {row.movie_title} {hit_mark}")

if __name__ == "__main__":
    main()
