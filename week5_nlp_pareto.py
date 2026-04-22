import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import re

def parse_query(query: str) -> list:
    """
    Rule-based NLP Query Parser
    Returns a list of objectives strictly matched.
    """
    objectives = []
    query = query.lower()
    
    if any(keyword in query for keyword in ['冷門', '小眾', '不常見']):
        objectives.append('novelty')
    if any(keyword in query for keyword in ['多樣', '不同', '各種', '豐富']):
        objectives.append('diversity')
    if any(keyword in query for keyword in ['新', '最新', '近期', '最近', '剛']):
        objectives.append('recency')
    if any(keyword in query for keyword in ['評價', '高分', '好評', '優質', '好片']):
        objectives.append('quality')
        
    return objectives

def dynamic_pareto_rerank(user_candidates, genre_cols, objectives, k=10, pool_size=None, tie_break='relevance'):
    """
    Dynamic Pareto Re-ranking（雙階段設計，支援 NLP 動態目標）

    ════════════════════════════════════════════════════════════
    【第一階段】Pareto Filtering（Pareto 的角色：篩選）
    ────────────────────────────────────────────────────────────
      - 支援動態多目標（novelty, diversity, recency, quality）
      - preference（正規化後的 predict_score）永遠作為基準目標之一
      - diversity 在每輪迭代中動態計算（基於已選集合的 Jaccard 距離）
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
    ════════════════════════════════════════════════════════════

    Parameters
    ----------
    user_candidates : pd.DataFrame
        候選電影 DataFrame。
    genre_cols : list
        電影類型欄位名稱列表，用於計算 diversity。
    objectives : list
        動態目標列表，由 parse_query() 生成（如 ['novelty', 'diversity']）。
    k : int
        最終推薦數量（Top-K）。
    pool_size : int or None
        Pareto 篩選候選池大小，預設為 max(50, k*2)。
    tie_break : str
        Tie-break 排序策略，可選 'relevance'（預設）或 'weighted'。
    """
    if pool_size is None:
        pool_size = max(50, k * 2)

    df = user_candidates.sort_values('predict_score', ascending=False).head(pool_size).copy().reset_index(drop=True)

    # 正規化 Preference（作為 Pareto 的基準目標之一）
    scaler = MinMaxScaler()
    df['preference'] = scaler.fit_transform(df[['predict_score']])

    # 確保各目標欄位存在，缺失時補 0.0
    for obj in ['novelty', 'recency', 'quality']:
        if obj not in df.columns:
            df[obj] = 0.0

    # 確認 recency 正確存在
    if 'recency' not in df.columns or df['recency'].sum() == 0:
        print("Warning: recency feature is missing or all zeros!")

    # ──────────────────────────────────────────────
    # 【第一階段】Pareto Filtering - 單一靜態目標捷徑
    # 當只追求單一靜態目標（非 diversity）時，直接排序，無需 Pareto 迭代
    # ──────────────────────────────────────────────
    if len(objectives) == 1 and objectives[0] in ['recency', 'novelty', 'quality']:
        target_obj = objectives[0]
        final_df = df.sort_values([target_obj, 'preference'], ascending=[False, False]).head(k).copy()
        final_df['pareto_rank'] = range(1, len(final_df) + 1)
        if 'diversity' not in final_df.columns:
            final_df['diversity'] = 0.0
        return final_df

    selected_indices = []
    unselected_indices = list(df.index)
    
    while len(selected_indices) < k and unselected_indices:
        frontier_df = df.loc[unselected_indices].copy()
        
        # 動態計算 diversity
        if 'diversity' in objectives:
            if len(selected_indices) > 0:
                selected_profiles = df.loc[selected_indices, genre_cols].values
                diversities = []
                for i in unselected_indices:
                    g_i = df.loc[i, genre_cols].values
                    intersections = np.sum(np.minimum(selected_profiles, g_i), axis=1)
                    unions = np.sum(np.maximum(selected_profiles, g_i), axis=1)
                    
                    jaccard_sims = np.zeros_like(unions, dtype=float)
                    nonzero_mask = unions > 0
                    jaccard_sims[nonzero_mask] = intersections[nonzero_mask] / unions[nonzero_mask]
                    
                    max_jaccard = np.max(jaccard_sims)
                    # diversity = 1 - 最大的 Jaccard 相似度
                    diversities.append(1.0 - max_jaccard)
                frontier_df['diversity'] = diversities
            else:
                # 第一個推薦還沒有可比較的，diversity 視為最高分
                frontier_df['diversity'] = 1.0
                
        # "preference" 永遠作為比較基準的其中一個目標
        compare_cols = ['preference'] + objectives
        
        # Pareto Non-dominated 篩選
        current_frontier = set()
        frontier_indices = list(frontier_df.index)
        
        for i in frontier_indices:
            dominated = False
            vals_i = frontier_df.loc[i, compare_cols].values
            
            for j in frontier_indices:
                if i == j: continue
                vals_j = frontier_df.loc[j, compare_cols].values
                
                # Check Domination
                # J 對 I 來說，每個指標都大於等於，而且至少一個嚴格大於
                if np.all(vals_j >= vals_i) and np.any(vals_j > vals_i):
                    dominated = True
                    break
                    
            if not dominated:
                current_frontier.add(i)
                
        # 【1】修改 Pareto tie-break 機制
        frontier_set_df = frontier_df.loc[list(current_frontier)]
        if len(objectives) == 1:
            # e.g., ['diversity'] 或是原本沒走捷徑的狀況
            best_idx = frontier_set_df[objectives[0]].idxmax()
        else:
            best_idx = frontier_set_df['preference'].idxmax()
        
        # 將最終決定的 diversity 給存回主表，方便事後呈現
        if 'diversity' in objectives:
            df.loc[best_idx, 'diversity'] = frontier_df.loc[best_idx, 'diversity']
            
        selected_indices.append(best_idx)
        unselected_indices.remove(best_idx)
        
    # ──────────────────────────────────────────────
    # 【第二階段】Tie-break Sorting（新增）
    # 對 Pareto front 進行全局重排序，提升 NDCG
    # ──────────────────────────────────────────────
    final_df = df.loc[selected_indices].copy()

    if tie_break == 'weighted':
        # 加權策略：normalize predict_score 後加權合併 novelty_norm（若存在）
        novelty_col = 'novelty_norm' if 'novelty_norm' in final_df.columns else 'novelty'
        if novelty_col not in final_df.columns:
            final_df[novelty_col] = 0.0
        tb_scaler = MinMaxScaler()
        final_df = final_df.copy()
        final_df['_predict_score_norm'] = tb_scaler.fit_transform(final_df[['predict_score']])
        final_df['_tiebreak_score'] = (
            0.7 * final_df['_predict_score_norm'] +
            0.3 * final_df[novelty_col]
        )
        final_df = final_df.sort_values('_tiebreak_score', ascending=False)
        final_df.drop(columns=['_predict_score_norm', '_tiebreak_score'], inplace=True)
    else:
        # 預設：relevance 策略，直接以 predict_score 由高到低排序
        final_df = final_df.sort_values('predict_score', ascending=False)

    # 重新指派最終推薦排名
    final_df = final_df.reset_index(drop=True)
    final_df['pareto_rank'] = range(1, len(final_df) + 1)

    if 'diversity' not in final_df.columns:
        final_df['diversity'] = 0.0

    return final_df
