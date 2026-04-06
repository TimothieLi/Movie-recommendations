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

def mmr_rerank(user_candidates, genre_cols, lambda_val=0.5, k=10):
    """
    Maximal Marginal Relevance (MMR) Re-ranking
    Score(i) = lambda * relevance(i) - (1 - lambda) * max_similarity(i, S)
    
    relevance(i) = normalized predict_score to [0,1]
    similarity(i, j) = genre multi-hot overlap (正規化至 [0,1] 以防止尺度過度影響 lambda 權重)
    """
    df = user_candidates.copy().reset_index(drop=True)
    
    # 1. 處理 Relevance: Min-Max Normalization to [0, 1]
    scaler = MinMaxScaler()
    df['relevance'] = scaler.fit_transform(df[['predict_score']])
    
    selected_indices = []
    unselected_indices = list(df.index)
    
    while len(selected_indices) < k and unselected_indices:
        if len(selected_indices) == 0:
            # 挑第一部電影：完全看 Relevance
            best_idx = df.loc[unselected_indices, 'relevance'].idxmax()
        else:
            best_idx = None
            max_mmr = -np.inf
            
            # S: 已選電影的 genre vector matrix
            # shape: (len(S), num_genres)
            selected_profiles = df.loc[selected_indices, genre_cols].values
            
            for i in unselected_indices:
                rel_i = df.loc[i, 'relevance']
                g_i = df.loc[i, genre_cols].values
                
                # 2. 處理 Similarity: max overlap(i, S)
                # overlaps: array of shape (len(S), )
                overlaps = np.dot(selected_profiles, g_i)
                max_sim = np.max(overlaps)
                
                # 為了讓尺度與 relevance 相近 (0~1)，我們將相似度稍微正規化
                # 常見電影類型數量約3~5種，我們用一個簡單的常數5.0來收斂至0~1 (也可以使用 cosine similarity)
                # 這裡就單純使用題意要求的 overlap，並做簡單數值壓縮避免 dominating
                norm_max_sim = max_sim / 5.0 
                
                # 計算 MMR 分數
                mmr_score = lambda_val * rel_i - (1 - lambda_val) * norm_max_sim
                
                if mmr_score > max_mmr:
                    max_mmr = mmr_score
                    best_idx = i
                    
        selected_indices.append(best_idx)
        unselected_indices.remove(best_idx)
        
    final_df = df.loc[selected_indices].copy()
    final_df['mmr_rank'] = range(1, len(final_df) + 1)
    return final_df
