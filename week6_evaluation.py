import numpy as np
import pandas as pd
import time

from movie_lgb_recommender import ndcg_at_k
from week4_reranking import mmr_rerank, pareto_rerank
from week5_nlp_pareto import dynamic_pareto_rerank, parse_query


def calculate_ild_at_k(recommended_df, genre_cols):
    """
    計算 Intra-List Diversity (ILD) at K
    使用 1 - Jaccard Similarity，並取清單中所有成對 pairwise 的平均值。
    """
    if len(recommended_df) <= 1:
        return 0.0
        
    genres_matrix = recommended_df[genre_cols].values
    n = len(genres_matrix)
    
    total_dissim = 0.0
    pairs = 0
    
    for i in range(n):
        for j in range(i + 1, n):
            g_i = genres_matrix[i]
            g_j = genres_matrix[j]
            intersection = np.sum(np.minimum(g_i, g_j))
            union = np.sum(np.maximum(g_i, g_j))
            
            jaccard_sim = intersection / union if union > 0 else 0.0
            dist = 1.0 - jaccard_sim
            total_dissim += dist
            pairs += 1
            
    return total_dissim / pairs if pairs > 0 else 0.0


def calculate_novelty_at_k(recommended_df):
    """
    計算 Novelty at K (平均 novelty 分數)
    """
    if 'novelty' in recommended_df.columns:
        return recommended_df['novelty'].mean()
    elif 'novelty_norm' in recommended_df.columns:
        return recommended_df['novelty_norm'].mean()
    return 0.0


def evaluate_method(method_fn, user_candidates, genre_cols, ground_truth, user_id, k=10):
    """
    對特定推薦演算法產生的推薦清單進行評價
    """
    # 產生推薦清單
    recom_df = method_fn(user_candidates)
    
    # NDCG@K
    actual_dict = ground_truth.get(user_id, {})
    predicted_list = recom_df['movie_id'].tolist()
    ndcg = ndcg_at_k(actual_dict, predicted_list, k=k)
    
    # Novelty@K
    novelty = calculate_novelty_at_k(recom_df)
    
    # ILD@K
    ild = calculate_ild_at_k(recom_df, genre_cols)
    
    recom_movie_ids = set(recom_df['movie_id'].tolist())
    
    return ndcg, novelty, ild, recom_movie_ids


def run_week6_experiments(test_users, unseen_candidates, test_ground_truth, genre_cols, total_movies_count, pool_size=50, k=10, progress_callback=None):
    """
    執行所有的推薦方法比較大迴圈，包含 LightGBM, MMR, Pareto, 與各種 Pareto+NLP
    """
    # 定義要一起做 benchmarking 的設定檔
    methods_config = [
        {"name": "LightGBM (Baseline)", "params": "N/A", "type": "baseline"},
        {"name": "MMR", "params": "λ=0.0", "type": "mmr", "lambda_val": 0.0},
        {"name": "MMR", "params": "λ=0.25", "type": "mmr", "lambda_val": 0.25},
        {"name": "MMR", "params": "λ=0.5", "type": "mmr", "lambda_val": 0.5},
        {"name": "MMR", "params": "λ=0.75", "type": "mmr", "lambda_val": 0.75},
        {"name": "MMR", "params": "λ=1.0", "type": "mmr", "lambda_val": 1.0},
        {"name": "Pareto Re-ranking", "params": "N/A", "type": "pareto_w4"},
        {"name": "Pareto + NLP", "params": "Query: 冷門", "type": "nlp", "query": "冷門"},
        {"name": "Pareto + NLP", "params": "Query: 多樣", "type": "nlp", "query": "多樣"},
        {"name": "Pareto + NLP", "params": "Query: 新", "type": "nlp", "query": "新"},
        {"name": "Pareto + NLP", "params": "Query: 冷門 + 多樣", "type": "nlp", "query": "冷門而且多樣"}
    ]
    
    # 預先解析 NLP 參數
    for method in methods_config:
        if method["type"] == "nlp":
            method["objectives"] = parse_query(method["query"])
            
    # 準備紀錄分數的資料夾
    results = {
        m_idx: {"ndcg": [], "novelty": [], "ild": [], "covered_items": set()}
        for m_idx in range(len(methods_config))
    }
    
    total_users = len(test_users)
    
    # 遍歷所有的使用者
    for step, user_id in enumerate(test_users):
        candidates = unseen_candidates[unseen_candidates['user_id'] == user_id].copy()
        
        # 相容 Week 4 的純 Pareto
        if 'novelty_norm' not in candidates.columns:
            if 'novelty' in candidates.columns:
                candidates['novelty_norm'] = candidates['novelty']
            else:
                candidates['novelty_norm'] = 0.5
            
        for m_idx, config in enumerate(methods_config):
            # 依據類型裝載推薦函式
            if config["type"] == "baseline":
                fn = lambda c: c.sort_values('predict_score', ascending=False).head(k)
            elif config["type"] == "mmr":
                fn = lambda c: mmr_rerank(c, genre_cols, config["lambda_val"], k=k, pool_size=pool_size)
            elif config["type"] == "pareto_w4":
                fn = lambda c: pareto_rerank(c, k=k, pool_size=pool_size)
            elif config["type"] == "nlp":
                fn = lambda c: dynamic_pareto_rerank(c, genre_cols, config["objectives"], k=k, pool_size=pool_size)
                
            ndcg, nov, ild, recom_ids = evaluate_method(fn, candidates, genre_cols, test_ground_truth, user_id, k=k)
            
            results[m_idx]["ndcg"].append(ndcg)
            results[m_idx]["novelty"].append(nov)
            results[m_idx]["ild"].append(ild)
            results[m_idx]["covered_items"].update(recom_ids)
            
        if progress_callback is not None:
            progress_callback( (step + 1) / total_users )
            
    # 統整為 DataFrame 返回
    summary_rows = []
    for m_idx, config in enumerate(methods_config):
        avg_ndcg = np.mean(results[m_idx]["ndcg"])
        avg_nov = np.mean(results[m_idx]["novelty"])
        avg_ild = np.mean(results[m_idx]["ild"])
        cov = len(results[m_idx]["covered_items"]) / total_movies_count if total_movies_count > 0 else 0.0
        
        summary_rows.append({
            "Method": config["name"],
            "Parameters": config["params"],
            "NDCG@10": avg_ndcg,
            "Novelty@10": avg_nov,
            "ILD@10": avg_ild,
            "Coverage": cov
        })
        
    return pd.DataFrame(summary_rows)
