import optuna
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

def run_optuna_weight_search(
    search_df,
    actual_dict,
    ndcg_func,
    top_k=10,
    n_trials=50, 
    genre_cols=None
):
    """
    使用 Optuna 搜尋 3 維最佳權重 (離線評估專用：Relevance, Novelty, Quality)
    """
    if len(actual_dict) == 0:
        default_weights = {"relevance": 0.34, "novelty": 0.33, "quality": 0.33}
        return default_weights, 0.0, search_df.head(top_k)

    df = search_df.copy()
    if 'novelty_norm' not in df.columns: df['novelty_norm'] = df['novelty']

    def objective(trial):
        # 搜尋 3 個維度的權重
        w_rel_raw = trial.suggest_float("w_rel", 0.1, 1.0)
        w_nov_raw = trial.suggest_float("w_nov", 0.0, 0.5)
        w_qua_raw = trial.suggest_float("w_qua", 0.0, 0.5)

        # 權重歸一化
        total = w_rel_raw + w_nov_raw + w_qua_raw
        w_rel = w_rel_raw / total
        w_nov = w_nov_raw / total
        w_qua = w_qua_raw / total

        # 計算綜合得分
        df['temp_score'] = (
            w_rel * df['predict_score'] +
            w_nov * df['novelty_norm'] +
            w_qua * df['quality']
        )
        
        # 取得排序結果
        ranked = df.sort_values(['pareto_rank', 'temp_score'], ascending=[True, False]).head(top_k)
        
        # 計算 NDCG
        preds = ranked['movie_id'].tolist()
        ndcg = ndcg_func(actual_dict, preds, k=top_k)

        # 實作：最平均優先 (Tie-break by balance)
        # 當 NDCG 相同時，我們希望權重越接近 (1/3, 1/3, 1/3) 越好
        balance_penalty = (w_rel - 1/3)**2 + (w_nov - 1/3)**2 + (w_qua - 1/3)**2
        
        # 回傳綜合分數 (懲罰項極小，不影響 NDCG 的主導地位)
        return ndcg - 1e-6 * balance_penalty

    # 使用固定隨機種子，確保結果可重複
    sampler = optuna.samplers.TPESampler(seed=42)
    study = optuna.create_study(direction="maximize", sampler=sampler)
    study.optimize(objective, n_trials=n_trials)

    # 取得最佳權重
    bp = study.best_params
    total_best = bp["w_rel"] + bp["w_nov"] + bp["w_qua"]
    best_weights = {
        "relevance": bp["w_rel"] / total_best,
        "novelty":   bp["w_nov"] / total_best,
        "quality":   bp["w_qua"] / total_best
    }

    # 產生最終排名結果
    df["weighted_score"] = (
        best_weights["relevance"] * df["predict_score"] +
        best_weights["novelty"] * df["novelty_norm"] +
        best_weights["quality"] * df["quality"]
    )
    final_ranked = df.sort_values(['pareto_rank', 'weighted_score'], ascending=[True, False]).head(top_k)
    
    final_ndcg = ndcg_func(actual_dict, final_ranked['movie_id'].tolist(), k=top_k)
    
    return best_weights, final_ndcg, final_ranked
