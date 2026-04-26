import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler


def pareto_rerank(user_candidates, k=10, pool_size=100, tie_break='weighted',
                  relevance_weight=0.85, novelty_weight=0.15, epsilon=0.1,
                  selection_mode='soft'):
    """
    Improved Pareto Dominance Re-ranking（雙階段設計 + Epsilon 支配 + Soft 模式）

    ════════════════════════════════════════════════════════════
    【第一階段】Pareto Selection（篩選與分層）
    ────────────────────────────────────────────────────────────
      - 支援 Epsilon Dominance：只有當優勢顯著超過 epsilon 時才構成嚴格支配，
        這能避免因極小分差（如 0.0001）而踢掉高相關性項目的問題，有助於穩住 NDCG。
      - 模式選擇 (selection_mode)：
          * 'hard'：傳統 Pareto 分層，依序取 Layer 1, Layer 2 ... 直到滿足 k 個。
          * 'soft'：不強行按層切斷，而是結合 (1/layer) 與加權分數進行全局排序。
    
    【第二階段】Tie-break Sorting（全局重排序）
    ────────────────────────────────────────────────────────────
      - 在選出的候選集內進行最終排序，確保 Top-1 是當前策略下的最佳選擇。
      - 支援自定義權重 (relevance_weight, novelty_weight)。
      - 預設權重 0.85 / 0.15 為保守策略，平衡 NDCG 與 Novelty。
    ════════════════════════════════════════════════════════════

    Parameters
    ----------
    user_candidates : pd.DataFrame
        候選電影 DataFrame。
    k : int
        最終推薦數量。
    pool_size : int
        候選池大小。
    tie_break : str
        'relevance' 或 'weighted'。
    relevance_weight : float
        相關性權重，預設 0.85。
    novelty_weight : float
        新穎度權重，預設 0.15。
    epsilon : float
        支配邊際 (dominance margin)，預設 0.01。
    selection_mode : str
        'hard' (嚴格分層) 或 'soft' (加權融合層級分數)。
    """
    # 1. 準備資料與安全處理
    actual_pool_size = min(len(user_candidates), pool_size)
    df = user_candidates.sort_values('predict_score', ascending=False).head(actual_pool_size).copy().reset_index(drop=True)

    if df.empty:
        return df

    # 確保必要欄位存在與 fallback
    if 'novelty_norm' not in df.columns:
        df['novelty_norm'] = df['novelty'] if 'novelty' in df.columns else 0.5
    if 'quality' not in df.columns:
        df['quality'] = 0.5
    if 'recency' not in df.columns:
        df['recency'] = 0.5

    # 2. Min-Max Normalization (確保四個維度在同一量級 [0, 1])
    scaler = MinMaxScaler()
    df['_norm_score'] = scaler.fit_transform(df[['predict_score']])
    df['_norm_novelty'] = scaler.fit_transform(df[['novelty_norm']])
    df['_norm_quality'] = scaler.fit_transform(df[['quality']])
    df['_norm_recency'] = scaler.fit_transform(df[['recency']])

    # 3. 多目標 Pareto 分層計算 (4-Objective: Score, Novelty, Quality, Recency)
    remaining_indices = list(df.index)
    df['pareto_layer'] = 0
    current_layer = 1
    
    # 定義要比較的維度
    compare_cols = ['_norm_score', '_norm_novelty', '_norm_quality', '_norm_recency']
    
    while remaining_indices:
        layer_indices = []
        for i in remaining_indices:
            is_dominated = False
            vals_i = df.loc[i, compare_cols].values
            
            for j in remaining_indices:
                if i == j: continue
                vals_j = df.loc[j, compare_cols].values
                
                # 多維 Epsilon Dominance:
                # j 支配 i 的條件：j 在所有維度都不輸 i - eps，且至少有一項勝過 i + eps
                if np.all(vals_j >= vals_i - epsilon) and np.any(vals_j > vals_i + epsilon):
                    is_dominated = True
                    break
            
            if not is_dominated:
                layer_indices.append(i)
        
        df.loc[layer_indices, 'pareto_layer'] = current_layer
        for idx in layer_indices:
            remaining_indices.remove(idx)
        current_layer += 1

    # [Debug] 顯示原始池子分層分佈
    print("\n--- [Debug] Full Pool Pareto Rank Distribution ---")
    print(df["pareto_layer"].value_counts().sort_index())

    # 4. 執行篩選邏輯
    if selection_mode == 'soft':
        # Soft Pareto: 結合層級權益與加權指標分數 (Tie-break 權重僅包含 score 與 novelty)
        df['_final_selection_score'] = (1.0 / df['pareto_layer']) + \
                                      (relevance_weight * df['_norm_score'] + 
                                       novelty_weight * df['_norm_novelty'])
        final_df = df.sort_values('_final_selection_score', ascending=False).head(k).copy()
    else:
        # Hard Pareto
        selected_indices = []
        l = 1
        while len(selected_indices) < k and l < current_layer:
            layer_indices_in_l = df[df['pareto_layer'] == l].index.tolist()
            selected_indices.extend(layer_indices_in_l)
            l += 1
        candidate_indices = selected_indices[:k]
        final_df = df.loc[candidate_indices].copy()

    # 5. 第二階段：Tie-break Sorting
    if tie_break == 'weighted':
        final_df['_tiebreak_score'] = (
            relevance_weight * final_df['_norm_score'] + 
            novelty_weight * final_df['_norm_novelty']
        )
        final_df = final_df.sort_values(['pareto_layer', '_tiebreak_score'], ascending=[True, False])
    else:
        final_df = final_df.sort_values(['pareto_layer', 'predict_score'], ascending=[True, False])

    # 6. 整理輸出
    final_df = final_df.reset_index(drop=True)
    final_df['pareto_rank'] = final_df['pareto_layer']
    
    # [Debug] 輸出前 10 名的多維度狀態
    print("\n--- [Debug] Pareto 4-Objective Top 10 Status ---")
    print(final_df[[
        "movie_title", "_norm_score", "_norm_novelty", 
        "_norm_quality", "_norm_recency", "pareto_rank"
    ]].head(10))

    # 移除內部運算暫存欄位
    tmp_cols = ['_norm_score', '_norm_novelty', '_norm_quality', '_norm_recency', 
                '_final_selection_score', '_tiebreak_score', 'pareto_layer']
    final_df.drop(columns=[c for c in tmp_cols if c in final_df.columns], inplace=True)
    
    return final_df


def mmr_rerank(user_candidates, genre_cols, lambda_val=0.5, k=10, pool_size=50):
    """
    Maximal Marginal Relevance (MMR) Re-ranking
    1. 限制 candidate pool 取原本 predict_score 前 50 名
    2. similarity 使用 Jaccard similarity
    """
    # 限制 Top N pool
    df = user_candidates.sort_values('predict_score', ascending=False).head(pool_size).copy().reset_index(drop=True)
    
    selected_indices = []
    unselected_indices = list(df.index)
    
    # 紀錄各個挑選回合的資料
    records = []
    
    while len(selected_indices) < k and unselected_indices:
        if len(selected_indices) == 0:
            # 挑第一部電影：完全看 Relevance，無 similarity 懲罰
            best_idx = df.loc[unselected_indices, 'predict_score'].idxmax()
            best_sim = 0.0
            # 原始 MMR 公式：λ * predict_score - (1-λ) * max_sim (此處 max_sim=0)
            best_mmr = df.loc[best_idx, 'predict_score'] * lambda_val 
        else:
            best_idx = None
            max_mmr = -np.inf
            best_sim = 0.0
            
            # S: 已選電影的 genre vector matrix
            selected_profiles = df.loc[selected_indices, genre_cols].values
            
            for i in unselected_indices:
                score_i = df.loc[i, 'predict_score']
                g_i = df.loc[i, genre_cols].values
                
                # 計算 Jaccard Similarity: intersection / union
                intersections = np.sum(np.minimum(selected_profiles, g_i), axis=1)
                unions = np.sum(np.maximum(selected_profiles, g_i), axis=1)
                
                jaccard_sims = np.zeros_like(unions, dtype=float)
                nonzero_mask = unions > 0
                jaccard_sims[nonzero_mask] = intersections[nonzero_mask] / unions[nonzero_mask]
                
                max_sim = np.max(jaccard_sims)
                
                # 使用原始分數計算 MMR (不使用 normalized relevance)
                mmr_score = lambda_val * score_i - (1 - lambda_val) * max_sim
                
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
