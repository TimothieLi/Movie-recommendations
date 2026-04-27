import pandas as pd
import numpy as np
from movie_lgb_recommender import run_recommender_pipeline, ndcg_at_k, GENRE_COLS
from week5_nlp_pareto import dynamic_pareto_rerank
from optuna_tuning import run_optuna_weight_search
import warnings
from sklearn.preprocessing import MinMaxScaler

warnings.filterwarnings("ignore")

# 1. 載入資料
test_users, top_10_df, test_ground_truth, movies_df, unseen_candidates, model, features = run_recommender_pipeline()

# 2. 鎖定 User 4
user_id = 4
actual = test_ground_truth.get(user_id, {})
candidates = unseen_candidates[unseen_candidates['user_id'] == user_id].copy()

# 3. 執行 Pareto 候選 (k=100)
pareto_pool = dynamic_pareto_rerank(
    candidates, 
    genre_cols=GENRE_COLS, 
    objectives=['novelty', 'quality', 'recency', 'diversity'], 
    k=100
)

# 4. 執行 Optuna 搜尋
best_weights, best_score, final_ranked = run_optuna_weight_search(
    search_df=pareto_pool,
    actual_dict=actual,
    ndcg_func=ndcg_at_k,
    top_k=10,
    n_trials=30,
    genre_cols=GENRE_COLS
)

# 5. 抓取第一名
top_1 = final_ranked.iloc[0]

# 手動模擬計算過程
scaler = MinMaxScaler()
# 注意：Optuna 內部會對 predict_score 進行一次 MinMaxScaler
rel_norm = scaler.fit_transform(pareto_pool[['predict_score']])[final_ranked.index[0]][0]

print(f"\n--- User {user_id} 排序分析報告 ---")
print(f"最佳權重配置 (W):")
for k, v in best_weights.items():
    print(f"  - {k:10}: {v:.4f} (即 {v:.2%})")

print(f"\n第一名電影: {top_1['movie_title']}")
print(f"Pareto Rank: {top_1['pareto_rank']}")

# 詳細指標
v_rel = rel_norm
v_nov = top_1['novelty_norm']
v_qua = top_1['quality']
v_rec = top_1['recency']
v_div = top_1['diversity']

w_rel = best_weights['relevance']
w_nov = best_weights['novelty']
w_qua = best_weights['quality']
w_rec = best_weights['recency']
w_div = best_weights['diversity']

# 計算驗證
calculated_score = (
    w_rel * v_rel +
    w_nov * v_nov +
    w_qua * v_qua +
    w_rec * v_rec +
    w_div * v_div
)

print(f"\n數值計算過程 (公式: Final Score = Σ W * V):")
print(f"  {w_rel:.4f} * {v_rel:.4f} (Rel)")
print(f"+ {w_nov:.4f} * {v_nov:.4f} (Nov)")
print(f"+ {w_qua:.4f} * {v_qua:.4f} (Qua)")
print(f"+ {w_rec:.4f} * {v_rec:.4f} (Rec)")
print(f"+ {w_div:.4f} * {v_div:.4f} (Div)")
print(f"------------------------------------")
print(f"= 最終得分: {calculated_score:.4f}")
print(f"  (系統存檔得分: {top_1['weighted_score']:.4f})")
