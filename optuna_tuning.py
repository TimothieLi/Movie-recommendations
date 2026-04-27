import optuna
import pandas as pd
import numpy as np

def run_optuna_weight_search(
    search_df,
    actual_dict,
    ndcg_func,
    top_k=10,
    n_trials=30
):
    """
    使用 Optuna 搜尋最佳 weighted tie-break 權重
    回傳：
        best_weights (dict): {relevance, novelty, quality}
        best_score (float): 最佳 NDCG 分數
        final_ranked (pd.DataFrame): 使用最佳權重排名的結果
    """
    
    # 額外加強：避免無意義搜尋
    if len(actual_dict) == 0:
        default_weights = {"relevance": 0.33, "novelty": 0.33, "quality": 0.33}
        return default_weights, 0.0, search_df.head(top_k)

    def objective(trial):
        # 修改 4：限制範圍避免極端解
        w_relevance = trial.suggest_float("w_relevance", 0.05, 0.9)
        w_novelty   = trial.suggest_float("w_novelty",   0.05, 0.9)
        w_quality   = trial.suggest_float("w_quality",   0.05, 0.9)

        total = w_relevance + w_novelty + w_quality
        if total == 0:
            return 0.0

        # 修改 3：確保權重先 normalize
        w_rel = w_relevance / total
        w_nov = w_novelty / total
        w_qua = w_quality / total

        # 使用局部副本進行計算
        df = search_df.copy()
        df["weighted_score"] = (
            w_rel * df["predict_score"]
            + w_nov * df["novelty_norm"]
            + w_qua * df["quality"]
        )

        # 排序：先 Pareto Rank (升序)，再 weighted_score (降序)
        reranked = df.sort_values(
            ["pareto_rank", "weighted_score"],
            ascending=[True, False]
        ).head(top_k)

        preds = reranked["movie_id"].tolist()
        score = ndcg_func(actual_dict, preds, k=top_k)

        # --- Edge Case Handling ---
        # 如果全部 trial 都是 0 -> Optuna 無法學習
        # 給微小懲罰但仍保留探索能力
        if score == 0.0:
            return -0.001

        # 修改 2：在 objective 加入 tie-break 懲罰項
        # When multiple weight combinations yield identical NDCG,
        # this small penalty ensures selecting a stable and balanced solution.
        penalty = (
            abs(w_rel - 1/3)
            + abs(w_nov - 1/3)
            + abs(w_qua - 1/3)
        )
            
        return score - 1e-4 * penalty

    # 修改 1：固定 Optuna 隨機種子 (seed=42)
    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=42)
    )
    study.optimize(objective, n_trials=n_trials)

    # 取得最佳權重
    best_params = study.best_params
    total_best = sum(best_params.values())
    if total_best == 0:
        best_weights = {"relevance": 0.33, "novelty": 0.33, "quality": 0.33}
    else:
        best_weights = {
            "relevance": best_params["w_relevance"] / total_best,
            "novelty":   best_params["w_novelty"]   / total_best,
            "quality":   best_params["w_quality"]   / total_best
        }

    # 重新產生最佳排序結果
    df_final = search_df.copy()
    df_final["weighted_score"] = (
        best_weights["relevance"] * df_final["predict_score"]
        + best_weights["novelty"] * df_final["novelty_norm"]
        + best_weights["quality"] * df_final["quality"]
    )

    final_ranked = df_final.sort_values(
        ["pareto_rank", "weighted_score"],
        ascending=[True, False]
    ).head(top_k)

    # 取得真正不含 penalty 的 NDCG 分數以供 UI 顯示
    preds = final_ranked["movie_id"].tolist()
    true_best_score = ndcg_func(actual_dict, preds, k=top_k)

    return best_weights, true_best_score, final_ranked
