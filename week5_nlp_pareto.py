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

def dynamic_pareto_rerank(user_candidates, genre_cols, objectives, k=10, pool_size=None):
    """
    Dynamic Pareto Re-ranking 
    - 支援動態更新 diversity
    - preference 為 baseline 指標
    """
    if pool_size is None:
        pool_size = max(50, k * 2)
        
    df = user_candidates.sort_values('predict_score', ascending=False).head(pool_size).copy().reset_index(drop=True)
    
    # 正規化 Preference
    scaler = MinMaxScaler()
    df['preference'] = scaler.fit_transform(df[['predict_score']])
    
    # 如果 user_candidates 裡面有些特徵是空的(剛好缺失)，在這裡預防性做個 0.0
    for obj in ['novelty', 'recency', 'quality']:
        if obj not in df.columns:
            df[obj] = 0.0
            
    # 【3】確認 recency 正確存在
    if 'recency' not in df.columns or df['recency'].sum() == 0:
        print("Warning: recency feature is missing or all zeros!")
        
    # 【2】新增「單一目標優先排序」機制
    # 當只追求單一靜態目標 (非 diversity) 時，直接依目標排序，再以 preference 當次排序
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
        
    final_df = df.loc[selected_indices].copy()
    final_df['pareto_rank'] = range(1, len(final_df) + 1)
    
    if 'diversity' not in final_df.columns:
        final_df['diversity'] = 0.0
        
    return final_df
