# Movie Recommendation System 🎬
### 多目標 LightGBM 排序 · Pareto 重新排序 · 基於規則的 NLP 偏好控制

![Python](https://img.shields.io/badge/python-3.10%2B-green)
![License](https://img.shields.io/badge/license-MIT-lightgrey)
![LightGBM](https://img.shields.io/badge/model-LightGBM%20LambdaRank-orange)
![Streamlit](https://img.shields.io/badge/demo-Streamlit-red)
 
> **推薦系統能否同時兼顧準確性與多樣性？使用者是否能用簡單的語言決定兩者之間的權衡？**
> 本專案結合了 LightGBM LambdaRank 基準模型、基於 Pareto 的重新排序策略以及基於規則的 NLP 條件映射器，並在 MovieLens 100K 數據集上進行了嚴謹的訓練、驗證與測試集切分評估。

## 系統核心架構

本專案旨在將機器學習模型的實驗評估與前端展示作深度整合。系統設計包含資料前處理、特徵工程、多目標重新排序（Re-ranking）及系統層級的評估。

主要核心檔案劃分如下：
1. **`movie_lgb_recommender.py`**:
   - 負責資料載入與 **Train / Validation / Test** 切分。確保 Test Set 僅用於最終無偏差的評估。
   - 進行混合特徵工程，包含使用者的歷史偏好與協同過濾訊號。
2. **`demo_app.py`**:
   - 基於 [Streamlit](https://streamlit.io/) 構建的前端互動介面，支援冷啟動與個人化雙模式。

## 🌐 線上展示 (Live Demo)

👉 **立即在線嘗試系統**:  
[https://movie-recommendations-timothie.streamlit.app](https://movie-recommendations-timothie.streamlit.app)

## 方法演進 (Weekly Progress Mapping)

- **Week 1–4: 混合推薦基準與 Pareto 重新排序**
  - 建構 LightGBM LambdaRank 基準模型，並引入 Pareto Dominance 策略平衡相關性與新穎度。
- **Week 5–7: LLM 語意解析與 TMDB 整合**
  - 導入 Gemini API 作為語意解析器，將自然語言轉為結構化權重。
  - 整合 TMDB API 抓取即時電影資訊。
- **Week 8: 系統穩定性優化與 UI/UX 精煉**
  - **冷啟動模式優化**：重構 `dynamic_pareto_rerank`，解決 Vintage 模式與年份約束的邏輯衝突。
  - **互動介面精煉**：移除冗餘的 Source 標籤與詳細報錯，優化海報牆顯示邏輯與過時的 Streamlit 語法。
  - **穩健性增強**：實作語意解析的「優雅降級」機制，當 API 忙碌時自動切換至 Rule-based 模式。
  - **效能修補**：修正了 `tb_scaler` 未定義等關鍵 Bug，確保 Tie-break 排序的穩定性。

## 📊 實驗結果 (測試集)

| 評測方法 | Recall@10 | NDCG@10 | Novelty@10 | ILD@10 (多樣性) |
| :--- | :---: | :---: | :---: | :---: |
| **基準模型 (LGB + MF)** | 3.32% | 0.1312 | 0.2546 | 0.8536 |
| **Pareto (Optuna 3D 最佳解)** ✅ | 2.63% | **0.1885** | **0.3112** | 0.7406 |

*(數據來自測試集, seed=42, k=10)*

## 規格化開發 (Spec-Driven Development)

👉 [spec/movie_recommender.yaml](spec/movie_recommender.yaml)

## 專案結構 (Project Structure)

```text
大三專題/
│── app.py                    # 完整前端入口
│── demo_app.py               # 互動推薦展示入口
│── movie_lgb_recommender.py  # 模型訓練核心
│── week5_nlp_pareto.py       # 語意解析與冷啟動核心排序器
│── tmdb_api.py               # TMDB API 整合
│── optuna_tuning.py          # 權重自動優化
└── README.md                 # 專案文件
```

## 🚀 快速啟動

```bash
pip install -r requirements.txt
python -m streamlit run demo_app.py
```

> **私密資訊管理**：請在 `.streamlit/secrets.toml` 中加入 `OPENAI_API_KEY` 與 `TMDB_API_KEY`。

## 未來展望
- **Deep Learning Ranking Models**
- **Real-time Feedback Loop**
