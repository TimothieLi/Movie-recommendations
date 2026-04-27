import pandas as pd
import numpy as np
from movie_lgb_recommender import run_recommender_pipeline, ndcg_at_k, GENRE_COLS
from demo_app import recommend_pareto
from optuna_tuning import run_optuna_weight_search
import warnings

warnings.filterwarnings("ignore")

# 1. 載入資料
test_users, top_10_df, test_ground_truth, movies_df, unseen_candidates, model, features = run_recommender_pipeline()

# 2. 鎖定 User 4
user_id = 4
actual = test_ground_truth.get(user_id, {})
candidates = unseen_candidates[unseen_candidates['user_id'] == user_id].copy()

# 3. 執行 Pareto 候選與 Optuna 優化
pareto_pool = recommend_pareto(candidates, k=100)
best_weights, best_score, final_ranked = run_optuna_weight_search(
    search_df=pareto_pool,
    actual_dict=actual,
    ndcg_func=ndcg_at_k,
    top_k=10,
    n_trials=30,
    genre_cols=GENRE_COLS
)

# 4. 抓取第一名
top_1 = final_ranked.iloc[0]

print(f"\n--- User {user_id} 排序分析 ---")
print(f"最佳權重: {best_weights}")
print(f"\n第一名電影: {top_1['movie_title']}")
print(f"Pareto Rank: {top_1['pareto_rank']}")
print(f"各項數值 (Normalized):")
print(f"- Relevance (LGB): {top_1['predict_score']:.4f}")
print(f"- Novelty: {top_1['novelty_norm']:.4f}")
print(f"- Quality: {top_1['quality']:.4f}")
print(f"- Recency: {top_1['recency']:.4f}")
print(f"- Diversity: {top_1['diversity']:.4f}")
print(f"最終加權得分 (weighted_score): {top_1['weighted_score']:.4f}")
