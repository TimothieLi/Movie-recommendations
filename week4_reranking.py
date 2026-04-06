import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

def pareto_rerank(user_candidates, k=10):
    """
    Pareto Dominance Re-ranking
    找出 Pareto Frontier (Layer 1, Layer 2...)
    item A dominates B 條件：
      - predict_score(A) >= predict_score(B)
      - novelty_norm(A) >= novelty_norm(B)
      - 且至少有一項是大於
    """
    df = user_candidates.copy().reset_index(drop=True)
    selected_indices = []
    remaining_indices = list(df.index)
    layer = 1
    
    while len(selected_indices) < k and remaining_indices:
        current_frontier = set()
        
        # 尋找目前的 non-dominated 集合
        for i in remaining_indices:
            dominated = False
            p_i = df.loc[i, 'predict_score']
            n_i = df.loc[i, 'novelty_norm']
            
            for j in remaining_indices:
                if i == j: continue
                
                p_j = df.loc[j, 'predict_score']
                n_j = df.loc[j, 'novelty_norm']
                
                # Check dominate condition
                if (p_j >= p_i and n_j >= n_i) and (p_j > p_i or n_j > n_i):
                    dominated = True
                    break
            
            if not dominated:
                current_frontier.add(i)
                
        # 對這個 Layer 內的電影使用 LightGBM Score 進行排序 (tie-breaking)
        frontier_df = df.loc[list(current_frontier)].sort_values('predict_score', ascending=False)
        
        # 如果把此層加入會超過 K，只需取需要的數量
        items_to_add = list(frontier_df.index)
        
        # 將這些 candidates 加到 selected_indices
        selected_indices.extend(items_to_add)
        
        # 從 remaining 中移除
        for idx in items_to_add:
            remaining_indices.remove(idx)
            
        layer += 1
        
    # 取 Top K
    final_indices = selected_indices[:k]
    final_df = df.loc[final_indices].copy()
    final_df['pareto_rank'] = range(1, len(final_df) + 1)
    
    return final_df

def mmr_rerank(user_candidates, genre_cols, lambda_val=0.5, k=10, pool_size=50):
    """
    Maximal Marginal Relevance (MMR) Re-ranking
    1. 限制 candidate pool 取原本 predict_score 前 50 名
    2. similarity 使用 Jaccard similarity
    """
    # 限制 Top N pool
    df = user_candidates.sort_values('predict_score', ascending=False).head(pool_size).copy().reset_index(drop=True)
    
    # 計算 Relevance: Min-Max Normalization to [0, 1]
    scaler = MinMaxScaler()
    df['relevance'] = scaler.fit_transform(df[['predict_score']])
    
    selected_indices = []
    unselected_indices = list(df.index)
    
    # 紀錄各個挑選回合的資料
    records = []
    
    while len(selected_indices) < k and unselected_indices:
        if len(selected_indices) == 0:
            # 挑第一部電影：完全看 Relevance，無 similarity 懲罰
            best_idx = df.loc[unselected_indices, 'relevance'].idxmax()
            best_sim = 0.0
            best_mmr = df.loc[best_idx, 'relevance'] * lambda_val # 為了顯示統一
        else:
            best_idx = None
            max_mmr = -np.inf
            best_sim = 0.0
            
            # S: 已選電影的 genre vector matrix
            selected_profiles = df.loc[selected_indices, genre_cols].values
            
            for i in unselected_indices:
                rel_i = df.loc[i, 'relevance']
                g_i = df.loc[i, genre_cols].values
                
                # 計算 Jaccard Similarity: intersection / union
                # selected_profiles (len(S), num_genres), g_i (num_genres,)
                intersections = np.sum(np.minimum(selected_profiles, g_i), axis=1)
                unions = np.sum(np.maximum(selected_profiles, g_i), axis=1)
                
                # 避免分母為 0
                jaccard_sims = np.zeros_like(unions, dtype=float)
                nonzero_mask = unions > 0
                jaccard_sims[nonzero_mask] = intersections[nonzero_mask] / unions[nonzero_mask]
                
                max_sim = np.max(jaccard_sims)
                
                # 計算 MMR 分數
                mmr_score = lambda_val * rel_i - (1 - lambda_val) * max_sim
                
                if mmr_score > max_mmr:
                    max_mmr = mmr_score
                    best_idx = i
                    best_sim = max_sim
            
            best_mmr = max_mmr
                    
        selected_indices.append(best_idx)
        unselected_indices.remove(best_idx)
        
        # 紀錄
        records.append({
            'idx': best_idx,
            'similarity_penalty': best_sim,
            'mmr_score': best_mmr,
            'lambda': lambda_val
        })
        
    # 組合 DataFrame
    final_rows = []
    for rank, rec in enumerate(records, 1):
        idx = rec['idx']
        row_data = df.loc[idx].to_dict()
        row_data['similarity_penalty'] = rec['similarity_penalty']
        row_data['mmr_score'] = rec['mmr_score']
        row_data['lambda'] = rec['lambda']
        row_data['mmr_rank'] = rank
        final_rows.append(row_data)
        
    final_df = pd.DataFrame(final_rows)
    return final_df
