import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import re
import json

# --- Pyparsing 相容性補丁 ---
try:
    import pyparsing
    if not hasattr(pyparsing, 'DelimitedList'):
        pyparsing.DelimitedList = pyparsing.delimitedList
except ImportError:
    pass


# ══════════════════════════════════════════════════════════════════════
# Structured Query Schema（結構化查詢格式）
# ══════════════════════════════════════════════════════════════════════
# parse_query_llm() 與 parse_query_rule() 皆回傳此格式：
#
# {
#   "weights": {
#       "relevance": float,   # 與使用者偏好的吻合程度（LightGBM score）
#       "novelty":   float,   # 電影冷門度（越高 = 越小眾）
#       "diversity": float,   # 清單多樣性（越高 = 類型越分散）
#       "recency":   float,   # 電影新穎度（越高 = 越近期）
#       "quality":   float    # 電影評價品質（越高 = 評分越高）
#   },
#   "constraints": {
#       "min_quality":  float | None,  # 最低品質門檻
#       "max_novelty":  float | None,  # 最大冷門度上限
#       "min_novelty":  float | None,  # 最小冷門度下限
#       "min_recency":  float | None   # 最低年份新穎度下限
#   },
#   "objectives": list[str],   # 用於 Pareto filtering（向下相容）
#   "explanation": str         # 人類可讀的解析說明（用於 Streamlit 展示）
# }


def _default_parsed_result() -> dict:
    """回傳一個結構化查詢結果的空白預設值（全部均等權重）"""
    return {
        "weights": {
            "relevance": 1.0,
            "novelty":   0.0,
            "diversity": 0.0,
            "recency":   0.0,
            "quality":   0.0,
        },
        "constraints": {
            "min_quality": None,
            "max_novelty": None,
            "min_novelty": None,
            "min_recency": None,
        },
        "objectives":  [],
        "explanation": "未解析到具體意圖，以純個人偏好（LightGBM）為主。"
    }


# ══════════════════════════════════════════════════════════════════════
# Rule-Based Semantic Parser（規則式語意解析器）
# ══════════════════════════════════════════════════════════════════════

# ── 關鍵字詞典：支援強度分級 ──────────────────────────────────────────
_NOVELTY_KEYWORDS = {
    "high":   ['超冷門', '非常冷門', '極度冷門', '極冷門', '沒人知道', '超小眾', '罕見'],
    "medium": ['冷門', '小眾', '不常見', '少人知道', '比較少人'],
    "low":    ['稍微冷門', '有點冷門', '略冷門', '微冷門'],
}
_DIVERSITY_KEYWORDS = {
    "high":   ['非常多樣', '超多樣', '各種各樣', '豐富多元'],
    "medium": ['多樣', '不同類型', '各種', '豐富', '多元'],
    "low":    ['稍微多樣', '有點多樣', '略多樣'],
}
_RECENCY_KEYWORDS = {
    "high":   ['最新', '最近上映', '剛出', '今年'],
    "medium": ['新', '近期', '最近', '近年'],
    "low":    ['有點新', '稍微新'],
}
_QUALITY_KEYWORDS = {
    "high":   ['非常高評價', '超高評分', '神作', '必看', '經典'],
    "medium": ['高評價', '好評', '高分', '評價好', '優質', '好片', '品質好'],
    "low":    ['還不錯', '評價不差', '有點評價'],
}

# 否定詞：用於翻轉語意（例：「不要太主流」→ novelty 提升）
_NEGATION_TOKENS = ['不要', '不能', '避免', '不太', '不是太', '別太']
_MAINSTREAM_KEYWORDS = ['主流', '熱門', '大眾', '票房', '商業']
_LOW_QUALITY_KEYWORDS = ['爛片', '低分', '差評', '評價差']


def _detect_intensity(query: str, keyword_dict: dict) -> float | None:
    """
    依據關鍵字字典偵測意圖強度，回傳加權值或 None（未偵測到）。
    high → 0.9, medium → 0.7, low → 0.45
    """
    if any(kw in query for kw in keyword_dict.get("high", [])):
        return 0.9
    if any(kw in query for kw in keyword_dict.get("medium", [])):
        return 0.7
    if any(kw in query for kw in keyword_dict.get("low", [])):
        return 0.45
    return None


def parse_query_rule(query: str) -> dict:
    """
    Rule-based Semantic Parser（規則式語意解析器）

    ════════════════════════════════════════════════════════════
    本函式是系統的核心語意解析元件，採用「規則式 + 強度分級」設計。
    相較於單純 keyword on/off，此版本支援：
      - 強度分級（稍微冷門 / 冷門 / 超冷門）
      - 否定語意翻轉（「不要太主流」→ novelty 提升）
      - 複合條件（同時解析多個目標維度）
      - Rule-based 約束（最低品質門檻等）
      - 可解釋輸出（explanation 欄位）
    ════════════════════════════════════════════════════════════

    Parameters
    ----------
    query : str
        使用者自然語言輸入。

    Returns
    -------
    dict
        結構化查詢結果（符合 Structured Query Schema）。
    """
    result = _default_parsed_result()
    q = query.lower().strip()

    # ── 1. 偵測各目標維度的強度 ─────────────────────────────────────
    novelty_score   = _detect_intensity(q, _NOVELTY_KEYWORDS)
    diversity_score = _detect_intensity(q, _DIVERSITY_KEYWORDS)
    recency_score   = _detect_intensity(q, _RECENCY_KEYWORDS)
    quality_score   = _detect_intensity(q, _QUALITY_KEYWORDS)

    # ── 2. 否定語意翻轉 ─────────────────────────────────────────────
    # 「不要太主流」→ novelty 提升
    if any(neg in q for neg in _NEGATION_TOKENS):
        if any(kw in q for kw in _MAINSTREAM_KEYWORDS):
            novelty_score = max(novelty_score or 0.0, 0.7)
        if any(kw in q for kw in _LOW_QUALITY_KEYWORDS):
            quality_score = max(quality_score or 0.0, 0.6)

    # ── 3. 填入加權結果 ─────────────────────────────────────────────
    objectives = []
    explanation_parts = []

    if novelty_score is not None:
        result["weights"]["novelty"] = novelty_score
        result["weights"]["relevance"] = max(0.3, 1.0 - novelty_score * 0.5)
        objectives.append("novelty")
        explanation_parts.append(f"冷門度 {novelty_score:.0%}")

    if diversity_score is not None:
        result["weights"]["diversity"] = diversity_score
        objectives.append("diversity")
        explanation_parts.append(f"多樣性 {diversity_score:.0%}")

    if recency_score is not None:
        result["weights"]["recency"] = recency_score
        objectives.append("recency")
        explanation_parts.append(f"近期新穎度 {recency_score:.0%}")

    if quality_score is not None:
        result["weights"]["quality"] = quality_score
        objectives.append("quality")
        explanation_parts.append(f"評分品質 {quality_score:.0%}")

    # ── 4. Rule-based Constraints ────────────────────────────────────
    # 若使用者提到「評價不錯」/「好看」，加入最低品質約束
    if any(kw in q for kw in ['好看', '不錯', '好評', '值得看', '高分', '優質', '品質']):
        result["constraints"]["min_quality"] = 0.4

    # 若強調「超冷門」，加入最低冷門度下限
    if novelty_score is not None and novelty_score >= 0.8:
        result["constraints"]["min_novelty"] = 0.5

    # 若強調「新」，加入最低年份新穎度下限
    if recency_score is not None and recency_score >= 0.7:
        result["constraints"]["min_recency"] = 0.5

    # ── 5. 組合解析說明（用於 Streamlit 展示）────────────────────────
    result["objectives"] = objectives
    if explanation_parts:
        result["explanation"] = "解析到的偏好維度：" + "、".join(explanation_parts) + "。"
    else:
        result["explanation"] = "未偵測到特定目標，以個人偏好（predict_score）為主。"

    return result


# ══════════════════════════════════════════════════════════════════════
# LLM-Assisted Semantic Parser（LLM 語意解析器）
# ══════════════════════════════════════════════════════════════════════

def parse_query_llm(query: str, api_key: str | None = None) -> dict:
    """
    LLM-Assisted Semantic Parser (支援 OpenAI 與 Google Gemini)
    """
    if not api_key:
        result = parse_query_rule(query)
        result["_parser"] = "rule_based"
        return result

    # 判斷 Provider
    is_gemini = api_key.startswith("AIza")
    
    system_prompt = """You are a semantic parser for a movie recommendation system.
Given a user's natural language query, extract their movie preferences as a structured JSON object.

Output ONLY valid JSON with this exact schema:
{
  "weights": {
    "relevance": <float 0-1, how much user cares about personal preference>,
    "novelty":   <float 0-1, preference for obscure/less popular movies>,
    "diversity": <float 0-1, preference for genre variety>,
    "recency":   <float 0-1, preference for newer movies>,
    "quality":   <float 0-1, preference for highly-rated movies>
  },
  "constraints": {
    "min_quality": <float 0-1 or null>,
    "max_novelty": <float 0-1 or null>,
    "min_novelty": <float 0-1 or null>,
    "min_recency": <float 0-1 or null>
  },
  "objectives": [<list of active objective names, e.g. "novelty", "diversity">],
  "explanation": "<one English sentence summarizing the user's intent>"
}

Rules:
- relevance default is 0.85 unless user explicitly wants to ignore personal taste
- Use 0.9 for "very", 0.7 for normal, 0.45 for "slightly"
- Negation (e.g. "不要太主流") -> increase novelty
- If user says "good quality" -> set min_quality to 0.4"""

    user_prompt = f"Query: {query}"

    try:
        if is_gemini:
            # --- Google Gemini REST API 邏輯 (指定您帳號中可用的型號) ---
            import requests
            url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent?key={api_key}"
            headers = {'Content-Type': 'application/json'}
            payload = {
                "contents": [{
                    "parts": [{"text": f"{system_prompt}\n\n{user_prompt}"}]
                }],
                "generationConfig": {
                    "temperature": 0.0
                }
            }
            response = requests.post(url, headers=headers, json=payload, timeout=10)
            res_json = response.json()
            
            if "candidates" in res_json:
                raw = res_json['candidates'][0]['content']['parts'][0]['text']
            else:
                raise Exception(f"API Error: {res_json.get('error', {}).get('message', 'Unknown Error')}")
        else:
            # --- OpenAI 邏輯 ---
            import openai
            client = openai.OpenAI(api_key=api_key)
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user",   "content": user_prompt},
                ],
                temperature=0.0,
            )
            raw = response.choices[0].message.content.strip()

        # 解析 JSON
        raw = re.sub(r"^```(?:json)?\n?", "", raw, flags=re.MULTILINE)
        raw = re.sub(r"\n?```$", "", raw, flags=re.MULTILINE)
        parsed = json.loads(raw)

        result = _default_parsed_result()
        result["weights"].update(parsed.get("weights", {}))
        result["constraints"].update(parsed.get("constraints", {}))
        result["objectives"]  = parsed.get("objectives", [])
        result["explanation"] = parsed.get("explanation", "")
        result["_parser"]     = f"llm ({'Gemini' if is_gemini else 'OpenAI'})"
        return result

    except Exception as e:
        result = parse_query_rule(query)
        result["_parser"]       = "rule_based_fallback"
        result["_llm_error"]    = str(e)
        return result


def parse_query(query: str) -> list:
    """
    向下相容介面：回傳 objectives list（供舊版 dynamic_pareto_rerank 使用）。
    """
    result = parse_query_rule(query)
    return result["objectives"]


# ══════════════════════════════════════════════════════════════════════
# Dynamic Pareto Re-ranking（支援結構化查詢權重 + Constraints）
# ══════════════════════════════════════════════════════════════════════

def dynamic_pareto_rerank(user_candidates, genre_cols, objectives,
                          k=10, pool_size=None, tie_break='relevance',
                          parsed_result: dict | None = None):
    """
    Dynamic Pareto Re-ranking（雙階段設計，支援 NLP 動態目標）

    ════════════════════════════════════════════════════════════
    【第一階段】Pareto Filtering（Pareto 的角色：篩選）
    ────────────────────────────────────────────────────────────
      - 支援動態多目標（novelty, diversity, recency, quality）
      - preference（正規化後的 predict_score）永遠作為基準目標之一
      - diversity 在每輪迭代中動態計算（基於已選集合的 Jaccard 距離）
      - 支援 parsed_result 中的 constraints：可在 Pareto 前
        預先過濾不符合條件的候選（例如品質不達門檻）

    【第二階段】Tie-break Sorting（Tie-break 的角色：排序）
    ────────────────────────────────────────────────────────────
      - 若傳入 parsed_result，tie-break 加權依 weights 動態調整
      - 否則沿用原本 relevance / weighted 兩種策略
    ════════════════════════════════════════════════════════════

    Parameters
    ----------
    user_candidates : pd.DataFrame
        候選電影 DataFrame。
    genre_cols : list
        電影類型欄位名稱列表。
    objectives : list
        動態目標列表（由 parse_query 或 parsed_result["objectives"] 傳入）。
    k : int
        最終推薦數量（Top-K）。
    pool_size : int or None
        Pareto 篩選候選池大小，預設為 max(50, k*2)。
    tie_break : str
        'relevance' 或 'weighted'（當 parsed_result 傳入時由 weights 自動決定）。
    parsed_result : dict or None
        parse_query_llm() / parse_query_rule() 的完整結構化輸出。
        若傳入，會啟用 constraints 過濾與動態 tie-break 加權。
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

    if 'recency' not in df.columns or df['recency'].sum() == 0:
        print("Warning: recency feature is missing or all zeros!")

    # ── Constraint Filtering（預先過濾不符條件的候選）────────────────
    if parsed_result is not None:
        constraints = parsed_result.get("constraints", {})
        if constraints.get("min_quality") is not None and "quality" in df.columns:
            df = df[df["quality"] >= constraints["min_quality"]].copy()
        if constraints.get("min_novelty") is not None and "novelty" in df.columns:
            df = df[df["novelty"] >= constraints["min_novelty"]].copy()
        if constraints.get("max_novelty") is not None and "novelty" in df.columns:
            df = df[df["novelty"] <= constraints["max_novelty"]].copy()
        if constraints.get("min_recency") is not None and "recency" in df.columns:
            df = df[df["recency"] >= constraints["min_recency"]].copy()
        # 若過濾後候選不足，回退到完整候選池
        if len(df) < k:
            df = user_candidates.sort_values('predict_score', ascending=False).head(pool_size).copy().reset_index(drop=True)
            df['preference'] = scaler.fit_transform(df[['predict_score']])
            for obj in ['novelty', 'recency', 'quality']:
                if obj not in df.columns:
                    df[obj] = 0.0

    df = df.reset_index(drop=True)

    # ──────────────────────────────────────────────
    # 【第一階段】Pareto Filtering - 單一靜態目標捷徑
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
                    diversities.append(1.0 - np.max(jaccard_sims))
                frontier_df['diversity'] = diversities
            else:
                frontier_df['diversity'] = 1.0

        # "preference" 永遠作為比較基準
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
                if np.all(vals_j >= vals_i) and np.any(vals_j > vals_i):
                    dominated = True
                    break
            if not dominated:
                current_frontier.add(i)

        frontier_set_df = frontier_df.loc[list(current_frontier)]
        if len(objectives) == 1:
            best_idx = frontier_set_df[objectives[0]].idxmax()
        else:
            best_idx = frontier_set_df['preference'].idxmax()

        if 'diversity' in objectives:
            df.loc[best_idx, 'diversity'] = frontier_df.loc[best_idx, 'diversity']

        selected_indices.append(best_idx)
        unselected_indices.remove(best_idx)

    # ──────────────────────────────────────────────
    # 【第二階段】Tie-break Sorting（支援動態加權）
    # ──────────────────────────────────────────────
    final_df = df.loc[selected_indices].copy()
    tb_scaler = MinMaxScaler()

    if parsed_result is not None and len(final_df) > 1:
        # 動態加權 tie-break：使用 parsed_result 的 weights
        w = parsed_result.get("weights", {})
        final_df = final_df.copy()
        final_df['_pref_norm'] = tb_scaler.fit_transform(final_df[['predict_score']])

        score = w.get("relevance", 0.85) * final_df['_pref_norm']
        for dim in ['novelty', 'diversity', 'recency', 'quality']:
            if dim in final_df.columns and w.get(dim, 0) > 0:
                score += w.get(dim, 0) * final_df[dim]

        final_df['final_score'] = score
        final_df = final_df.sort_values('final_score', ascending=False)
        final_df.drop(columns=[c for c in ['_pref_norm', '_tiebreak_score'] if c in final_df.columns], inplace=True)

    elif tie_break == 'weighted':
        # 舊版 weighted 策略（向下相容）
        novelty_col = 'novelty_norm' if 'novelty_norm' in final_df.columns else 'novelty'
        if novelty_col not in final_df.columns:
            final_df[novelty_col] = 0.0
        final_df = final_df.copy()
        final_df['_predict_score_norm'] = tb_scaler.fit_transform(final_df[['predict_score']])
        final_df['final_score'] = (
            0.7 * final_df['_predict_score_norm'] +
            0.3 * final_df[novelty_col]
        )
        final_df = final_df.sort_values('final_score', ascending=False)
        final_df.drop(columns=['_predict_score_norm', '_tiebreak_score'], inplace=True, errors='ignore')
    else:
        final_df = final_df.sort_values('predict_score', ascending=False)

    final_df = final_df.reset_index(drop=True)
    final_df['pareto_rank'] = range(1, len(final_df) + 1)

    if 'diversity' not in final_df.columns:
        final_df['diversity'] = 0.0

    return final_df
