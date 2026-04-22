import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler


def pareto_rerank(user_candidates, k=10, pool_size=50, tie_break='relevance'):
    """
    Pareto Dominance Re-ranking（雙階段設計）

    ════════════════════════════════════════════════════════════
    【第一階段】Pareto Filtering（Pareto 的角色：篩選）
    ────────────────────────────────────────────────────────────
      - 採用 Pareto Dominance 方法，從 candidate pool 中逐層篩選
      - 找出 Non-dominated 集合（Layer 1, Layer 2 ... 直到收集夠 k 個項目）
      - Item A dominates B 的條件：
          * predict_score(A) >= predict_score(B)
          * novelty_norm(A)  >= novelty_norm(B)
          * 且至少有一項嚴格大於（>）
      - Pareto 只決定「哪些 item 被選中」，不決定最終排序！

    【第二階段】Tie-break Sorting（Tie-break 的角色：排序）
    ────────────────────────────────────────────────────────────
      - 在 Pareto front 收集完畢後，對已選集合進行全局重排序
      - 解決 Pareto 本身不定義最終排序的問題，提升 NDCG 表現
      - 支援兩種策略（透過 tie_break 參數選擇）：
          * 'relevance'：以 predict_score 由高到低排序（預設）
          * 'weighted' ：加權排序，
                         final_score = 0.7 * predict_score_norm
                                     + 0.3 * novelty_norm
                         兼顧相關性（relevance）與新穎度（novelty）
    ════════════════════════════════════════════════════════════

    Parameters
    ----------
    user_candidates : pd.DataFrame
        候選電影 DataFrame，需包含 predict_score 欄位。
    k : int
        最終推薦數量（Top-K）。
    pool_size : int
        Pareto 篩選的候選池大小（從 predict_score 排名前 pool_size 中挑選）。
    tie_break : str
        Tie-break 排序策略。
        - 'relevance'（預設）：使用 predict_score 由高到低排序
        - 'weighted'          ：使用加權分數進行排序

    Returns
    -------
    pd.DataFrame
        經 Pareto 篩選與 Tie-break 排序後的 Top-K 推薦結果，
        含 pareto_rank 欄位。
    """
    df = user_candidates.sort_values('predict_score', ascending=False).head(pool_size).copy().reset_index(drop=True)

    # 確保 novelty_norm 存在（相容 Week 5 TMDB 整合版）
    if 'novelty_norm' not in df.columns:
        if 'novelty' in df.columns:
            df['novelty_norm'] = df['novelty']
        else:
            df['novelty_norm'] = 0.5

    # ──────────────────────────────────────────────
    # 【第一階段】Pareto Filtering
    # 逐層找出 Non-dominated 集合，累積至 k 個候選
    # ──────────────────────────────────────────────
    selected_indices = []
    remaining_indices = list(df.index)
    layer = 1

    while len(selected_indices) < k and remaining_indices:
        current_frontier = set()

        # 尋找目前的 non-dominated 集合（當前 layer 的 Pareto front）
        for i in remaining_indices:
            dominated = False
            p_i = df.loc[i, 'predict_score']
            n_i = df.loc[i, 'novelty_norm']

            for j in remaining_indices:
                if i == j:
                    continue
                p_j = df.loc[j, 'predict_score']
                n_j = df.loc[j, 'novelty_norm']

                # 判斷 i 是否被 j 支配（dominated）
                if (p_j >= p_i and n_j >= n_i) and (p_j > p_i or n_j > n_i):
                    dominated = True
                    break

            if not dominated:
                current_frontier.add(i)

        # 將此 layer 的所有 item 加入已選集合（layer 內以 predict_score 做初步排序）
        frontier_df = df.loc[list(current_frontier)].sort_values('predict_score', ascending=False)
        items_to_add = list(frontier_df.index)
        selected_indices.extend(items_to_add)

        # 從候選池移除已選項目，準備下一個 layer
        for idx in items_to_add:
            remaining_indices.remove(idx)

        layer += 1

    # 取出前 k 個（跨多層 Pareto 篩選後的候選集合）
    candidate_indices = selected_indices[:k]
    final_df = df.loc[candidate_indices].copy()

    # ──────────────────────────────────────────────
    # 【第二階段】Tie-break Sorting（新增）
    # 對 Pareto front 進行全局重排序，提升 NDCG
    # ──────────────────────────────────────────────
    if tie_break == 'weighted':
        # 加權策略：normalize predict_score 後加權合併 novelty_norm
        scaler = MinMaxScaler()
        final_df = final_df.copy()
        final_df['_predict_score_norm'] = scaler.fit_transform(final_df[['predict_score']])
        final_df['_tiebreak_score'] = (
            0.7 * final_df['_predict_score_norm'] +
            0.3 * final_df['novelty_norm']
        )
        final_df = final_df.sort_values('_tiebreak_score', ascending=False)
        # 移除暫時欄位
        final_df.drop(columns=['_predict_score_norm', '_tiebreak_score'], inplace=True)
    else:
        # 預設：relevance 策略，直接以 predict_score 由高到低排序
        final_df = final_df.sort_values('predict_score', ascending=False)

    # 重新指派最終推薦排名
    final_df = final_df.reset_index(drop=True)
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
