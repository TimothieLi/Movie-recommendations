# Movie Recommendation System 🎬
### 多目標 LightGBM 排序 · Pareto 重新排序 · 基於規則的 NLP 偏好控制

![Python](https://img.shields.io/badge/python-3.10%2B-green)
![License](https://img.shields.io/badge/license-MIT-lightgrey)
![LightGBM](https://img.shields.io/badge/model-LightGBM%20LambdaRank-orange)
![Streamlit](https://img.shields.io/badge/demo-Streamlit-red)
 
🌐 **線上展示**: [movie-recommendations-timothie.streamlit.app](https://movie-recommendations-timothie.streamlit.app)

> **推薦系統能否同時兼顧準確性與多樣性？使用者是否能用簡單的語言決定兩者之間的權衡？**
> 本專案結合了 LightGBM LambdaRank 基準模型、基於 Pareto 的重新排序策略以及基於規則的 NLP 條件映射器，並在 MovieLens 100K 數據集上進行了嚴謹的訓練、驗證與測試集切分評估。

本專案實作一個基於 **LightGBM (LambdaRank)** 為核心的混合式推薦系統（Hybrid Recommendation System），屬於大三專題之研究型專案。本系統利用 [MovieLens 100K Dataset](https://grouplens.org/datasets/movielens/100k/) 與 TMDB 資料集，探討從初始的 Learning-to-Rank 模型到結合多目標最佳化（Multi-Objective Optimization）與自然語言映射（Natural-Language Condition Mapping）的方法演進。

## 系統核心架構

本專案旨在將機器學習模型的實驗評估與前端展示作深度整合。系統設計包含資料前處理、特徵工程、多目標重新排序（Re-ranking）及系統層級的評估。

主要核心檔案劃分如下：
1. **`movie_lgb_recommender.py`**:
   - 負責資料載入與 **Train / Validation / Test** 切分。其中，Validation Set 專供模型的 Early Stopping 與超參數挑選（Model Selection）使用，確保 Test Set 僅用於最終無偏差的評估。
   - 進行混合特徵工程，包含使用者的歷史偏好（User History）、共現性特徵（Co-occurrence）以及基於矩陣分解的潛在偏好訊號（MF / Latent Preference）等協同過濾訊號（Collaborative Signals）。
   - 訓練基於 LightGBM 的 LambdaRank 排序模型。
2. **`app.py` 與 `demo_app.py`**:
   - 基於 [Streamlit](https://streamlit.io/) 構建的前端互動介面。
   - 提供百萬級候選清單的批次推論（Batch Evaluation）與快取機制，達成 0 毫秒延遲的即時預測與多方法比較展示。

## 🌐 線上展示 (Live Demo)

👉 **立即在線嘗試系統**:  
[https://movie-recommendations-timothie.streamlit.app](https://movie-recommendations-timothie.streamlit.app)

這是一個部署在 Streamlit Cloud 的網頁應用程式，展示了：
- **LightGBM LambdaRank**: 核心推薦模型的排序表現。
- **基於 Pareto 的重新排序**: 即時多目標最佳化（相關性 vs. 新穎度）。
- **NLP 驅動控制**: 使用自然語言查詢動態調整優化目標。

*無需本地設定 — 只需打開連結即可與系統互動。*

## 方法演進 (Weekly Progress Mapping)

為有系統地推進研究，本專案依階段迭代不同的演算法機制，對應之檔案與研究流程如下：

- **Week 1–2: LightGBM Baseline (`movie_lgb_recommender.py`, `mf_features.py`)**
  - 建構以 LightGBM ranking 為核心的混和式基準模型（Hybrid Baseline）。
  - 將 Matrix Factorization (MF) 的 latent preference signal 以及 collaborative signals 融合至排序特徵中，大幅提升系統的 Recall 表現。
- **Week 3: Novelty & Diversity Features (`week3_features.py`)**
  - 實作多維度特徵：包含電影新穎度（Novelty）懲罰及基於 Multi-hot Genre 的多樣性（Diversity）計算，為後續多目標最佳化奠定基礎。
- **Week 4: Re-ranking Strategies (`week4_reranking.py`)**
  - 導入重新排序策略（Re-ranking）以平衡推薦準確度與多元性，本專案最終採用**改良版 Pareto-based Re-ranking**。
  - 亦實作 Maximal Marginal Relevance (MMR) 演算法作為對照基準（透過 $\lambda$ 參數動態調配 Relevance 與 Diversity 之平衡）。

  **改良版 Pareto Re-ranking（最終採用方法）**

  本方法以 Pareto Dominance 作為多目標篩選核心，同時引入多項改良機制以提升排序品質：

  - **雙目標 Pareto 分層（Non-dominated Sorting）**：對候選集合依 `relevance`（`predict_score`）與 `novelty` 進行非支配排序，Pareto Layer 1 為當前最優前沿，Layer 2、3 依序向後延伸。Pareto 負責的問題是「哪些 item 應被納入候選」。
  - **Epsilon-Dominance 支配邊際**：傳統 Pareto 支配對極小分差（如 0.0001）過度敏感，可能錯誤淘汰高相關性項目。本方法加入 epsilon 門檻（`ε = 0.01`），只有當一方優勢**顯著超過** ε 時才構成嚴格支配，有效穩住 NDCG。
  - **Soft Selection（軟性選取）**：不以硬性 Pareto 層切斷候選集，而是將各 item 所在的層級（Layer Benefit = `1 / layer`）與加權分數（`relevance_weight × score + novelty_weight × novelty`）融合為統一排序依據，允許層級較深但分數優異的 item 重回推薦名單，顯著改善 NDCG。
  - **Weighted Tie-break 排序（第二階段）**：在選出候選集後，以可調整比例（預設 `0.85 × relevance + 0.15 × novelty`）進行最終全局重排序。Tie-break 負責的問題是「已選 item 的呈現順序」，與 Pareto 的篩選角色明確分離。
  - **較大候選池（`pool_size = 100`）**：擴大初始候選範圍，讓 Pareto 有更充足的空間識別具潛力的冷門 / 高品質電影。
- **Week 5: LLM-Assisted Semantic Parsing (`week5_nlp_pareto.py`)**
  - 本模組定位為「LLM-assisted semantic parsing for rule-based recommendation control」，採用三層架構：
    1. **語意解析層**：支援兩種模式。預設為 `parse_query_rule()`（強度分級規則式解析，含否定語意翻轉）；可選啟用 `parse_query_llm()`，以 OpenAI `gpt-4o-mini` 作為 semantic parser 將自然語言轉為結構化條件，若 API 呼叫失敗則自動 fallback。
    2. **結構化條件層**：輸出統一 Schema（`weights` / `constraints` / `objectives` / `explanation`），支援目標權重化（非 on/off）、強度分級（稍微冷門 / 冷門 / 超冷門）與 rule-based 約束（如 `min_quality`、`min_novelty`）。
    3. **Ranking 控制層**：`dynamic_pareto_rerank()` 依解析結果動態調整 Pareto 篩選目標與 tie-break 加權比例，後端全程維持 rule-based / score-based，保持可解釋性。
- **Week 6: System-Level Evaluation (`week6_evaluation.py`)**
  - 建構大規模的 Batch 評測環境。繪製 NDCG vs. Novelty 及 NDCG vs. ILD 的 Trade-off 曲線，量化並比較各階段演算法的實質效益。
- **Week 7: External Data Integration (`tmdb_api.py`)**
  - **TMDB API 整合**：導入資料庫外（Out-of-database）推薦功能。透過 TMDB Discover API 即時抓取最新、熱門或高評價電影，並與本地 MovieLens 候選集動態融合。
  - **跨域特徵對齊**：自動將 TMDB 元數據轉化為系統可理解的 `Novelty`、`Recency` 與 `Quality` 特徵，使外部電影能無縫參與 Pareto 重排序。
  - **展示中心升級**：`demo_app.py` 進化為「研究成果展示板」，支援單一方法深度檢視、電影海報牆（Featured Posters）、劇情大綱顯示及百分比標準化評估指標（NDCG/Recall/Novelty/ILD）。
- **Week 8: UI/UX Refactoring & Dual-Mode System (`demo_app.py`)**
  - **雙模式架構（Mode Isolation）**：重構展示介面，完全隔離「離線評估（Offline Evaluation）」與「互動式推薦（Interactive Recommendation）」。
  - **按鈕式持久化導覽**：採用 `st.session_state` 實作按鈕式切換，解決 Streamlit 頁面重整導致的選擇遺失問題，並提供更流暢的視覺回饋。
  - **解析度可視化優化**：在互動模式下，將 NLP 解析權重正規化為 100% 佔比顯示（Relative Importance），並以卡片式 Scorecard 呈現，讓使用者直觀理解各維度對 Final Score 的影響力。
  - **冷啟動體驗優化**：在互動模式中移除 User-specific 的 LightGBM 偏好欄位與 Raw JSON debug 資訊，將介面焦點完全鎖定在 Query-based 的推薦結果與可解釋性說明。

## 📊 實驗結果 (測試集)

本專案採用嚴謹的離線評估流程，並以 **Recall@10** 與 **NDCG@10** 作為衡量推薦準確度與排序品質之主要指標。多目標方法亦額外評估 Novelty@10 與 ILD@10（清單內成對多樣性）。

> 所有重新排序方法（MMR、Pareto、基於 NLP 的方法）均套用於由基準模型產生的**相同候選池**上，以確保公平比較。

| 評測方法 | Recall@10 | NDCG@10 | Novelty@10 | ILD@10 (多樣性) |
| :--- | :---: | :---: | :---: | :---: |
| **基準模型 (LGB + MF)** | **2.90%** | **0.1359** | 0.2889 | 0.7457 |
| **MMR (λ=0.25)** | - | 0.1290 | 0.3143 | 0.9131 |
| **MMR (λ=0.5)** | - | 0.1387 | 0.2984 | 0.8798 |
| **MMR (λ=0.75)** | - | 0.1384 | 0.2869 | 0.8058 |
| **Pareto 重新排序 (Soft)** ✅ | 2.73% | 0.1222 | 0.3507 | 0.7463 |
| **NLP + Pareto (查詢: 冷門)** | - | 0.0849 | **0.3754** | 0.7495 |
| **NLP + Pareto (查詢: 多樣)** | - | 0.1307 | 0.3155 | **0.9156** |
| **NLP + Pareto (查詢: 新)** | - | **0.1817** | 0.3115 | 0.7484 |
| **NLP + Pareto (查詢: 冷門+多樣)** | - | 0.1359 | 0.2889 | 0.7457 |


*(Recall@10 僅在 Baseline 全測試集上計算；重新排序方法以相同候選池評估 NDCG/Novelty/ILD。Pareto 方法採 pool_size=100；數據來自測試集, seed=42, k=10)*

### ⚙️ 評估協議 (Evaluation Protocol)

- **資料集**: MovieLens 100K
- **資料切分**: 基於時間的 訓練 / 驗證 / 測試 (80 / 10 / 10)
- **指標**: Recall@10, NDCG@10, Novelty@10, ILD@10 (Intra-List Diversity)
  - **候選池大小**: 每位使用者 100 個 (Pareto); 50 個 (MMR / NLP)
- **重新排序**: 套用於由基準模型產生的相同候選集
- **隨機種子**: 42 (確保可重現性)
- **相關性閾值**: 評分 Rating ≥ 3.0

### 💡 核心洞察 (Key Insights)

- **MMR (λ=0.5)** 在重新排序方法中取得了最佳的 NDCG@10 (0.1387)，略微**優於基準模型**，同時也提升了多樣性 (ILD: 0.8798 vs 0.7457)。
- **Pareto 重新排序 (Soft)** ✅ 是**最終採用的多目標重新排序方法**。它結合了非支配排序與 Epsilon-Dominance (ε=0.01)、基於層級的軟性選取，以及在 100 個候選項目上的加權 Tie-break (0.85 × 相關性 + 0.15 × 新穎度)。此設計達到了 NDCG@10 = 0.1222 與 Novelty@10 = 0.3507 —— 與硬性 Pareto 相比恢復了約 17% 的 NDCG 差距，同時保留了新穎度優勢。核心設計原則是明確的角色分離：**Pareto 負責多目標過濾**（哪些項目被選中），而 **Tie-break 負責最終排序**。
- **NLP + Pareto (查詢: 新)** 在所有方法中取得了最高的 NDCG@10 (0.1817)，這表明偏向「新進電影」的查詢與該資料集中的使用者偏好高度吻合。
- **NLP + Pareto (查詢: 冷門)** 最大化了 Novelty@10 (0.3754)，但代價是較低的 NDCG (0.0849)，展示了透過自然語言條件實現的可控權衡。
- MMR 中的 λ 參數提供了一個平滑的旋鈕：如預期，較低的 λ 會增加多樣性 (ILD ↑) 但會犧牲相關性 (NDCG ↓)。

*(實驗基準以真實評分 Rating $\ge 3.0$ 作為相關性判斷閥值；數據由 `run_readme_experiment.py` 產出，seed=42, pool_size=50, k=10)*

## 規格化開發 (Spec-Driven Development)

本專案採用 **Spec-driven development** 的精神，將系統設計與架構顯性化。我們將系統的目標、資料處理原則、基準模型、擴充模組及評估限制等，定義為一份清楚的規格文件，不僅有助於未來的開發維護，也方便協作者或 AI 快速理解專案全貌。

您可以查看位於 `spec/` 目錄下的規格檔案：
👉 [spec/movie_recommender.yaml](spec/movie_recommender.yaml)

## 專案結構 (Project Structure)

```text
大三專題/
│── spec/                     # OpenSpec 系統規格文件目錄
│   └── movie_recommender.yaml
│── app.py                    # 完整前端互動介面入口
│── demo_app.py               # 精簡展示版前端入口
│── requirements.txt          # 環境依賴套件清單
│── movie_lgb_recommender.py  # 核心資料處理、特徵工程與 LightGBM LambdaRank 訓練
│── mf_features.py            # 純 Numpy 實作的 Latent Preference (Matrix Factorization)
│── week3_features.py         # 多目標特徵計算 (Novelty, Diversity)
│── week4_reranking.py        # 重新排序策略實作 (MMR, Pareto)
│── week5_nlp_pareto.py       # Rule-based NLP 動態權重映射
│── week6_evaluation.py       # 系統層級綜合評估與 Trade-off 視覺化
│── tmdb_api.py               # TMDB API 整合與跨域資料抓取
│── optuna_tuning.py          # 基於 Optuna 的多目標權重自動優化
└── README.md                 # 專題文件 (本檔案)
```

## 資料集準備 (Dataset Setup)

本專案依賴兩個開源資料集。為確保模型成功訓練與特徵擴充正確，請依照以下結構放置檔案：

### 1. MovieLens 100K Dataset
* **來源**：[GroupLens 官方下載點](https://grouplens.org/datasets/movielens/100k/) (下載 `ml-100k.zip` 並解壓縮)
* **必需檔案**：`u.data` (使用者歷史評分) 與 `u.item` (電影屬性資訊)
* **預期路徑**：請在專案根目錄建立 `MovieLens 100K` 資料夾並放置檔案。
  ```text
  大三專題/
  └── MovieLens 100K/
      ├── u.data
      └── u.item
  ```

### 2. TMDB 5000 Movies Metadata (進階多目標特徵需要)
* **來源**：[Kaggle TMDB 5000 Movie Dataset](https://www.kaggle.com/datasets/tmdb/tmdb-movie-metadata)
* **必需檔案**：`tmdb_5000_movies.csv`
* **預期路徑**：請在專案根目錄建立 `TMDB metadata` 資料夾並放置。
  ```text
  大三專題/
  └── TMDB metadata/
      └── TMDB_5000_movies.csv
  ```

## 環境設定與啟動 (Setup & Installation)

本專案建議使用 **Python 3.10+** 並透過 `venv` 建立獨立虛擬環境來執行，以確保實驗環境之可重現性 (Reproducibility)。

```bash
# 1. 進入專案資料夾
cd 大三專題

# 2. 建立虛擬環境
python -m venv venv

# 3. 啟動虛擬環境
# macOS / Linux:
source venv/bin/activate
# Windows:
# venv\Scripts\activate

# 4. 安裝依賴套件
pip install -r requirements.txt

# 5. 啟動展示應用程式
# 完整功能版（含所有頁籤）
python -m streamlit run app.py
# 專題 Demo 展示特化版
python -m streamlit run demo_app.py
```
> **Week 5 LLM 模式（選用）**：若要啟用 `gpt-4o-mini` 語意解析，請在 Streamlit 介面中勾選「🤖 使用 LLM 語意解析」並輸入 OpenAI API Key。
>
> **Week 7 TMDB 模式（選用）**：若要推薦資料庫外的最新電影，請在側邊欄輸入你的 TMDB API Key。系統會自動將外部新電影標記為 🟣 **TMDB** 並進行多目標排序。

## 🚀 部署到 Streamlit Cloud

本專案已針對 **Streamlit Cloud** 進行優化，您可以輕鬆將其部署為線上 Web App。

### 部署步驟

1. **登入 Streamlit Cloud**：造訪 [Streamlit Cloud](https://streamlit.io/cloud) 並使用 GitHub 帳號登入。
2. **點選 New app**：在控制面板點擊右上角的 "Create app"。
3. **選擇 GitHub repo**：選擇本專案所在的儲存庫與分支。
4. **設定主程式路徑**：將 Main file path 設為 `demo_app.py`。
5. **點擊 Deploy**：系統將自動偵測 `requirements.txt` 並安裝環境。

### 注意事項
- **Python 版本**：建議在 Advanced Settings 中選擇 **Python 3.11** 以獲得最佳相容性。
- **LightGBM 問題**：若部署時遇到 LightGBM 編譯錯誤，建議在 `requirements.txt` 中嘗試固定版本（如 `lightgbm==3.3.5`）。
- **Secrets 管理**：若希望預設啟用 LLM 或 TMDB 功能，可在 Streamlit Cloud 的 **Secrets** 設定中加入 `OPENAI_API_KEY` 與 `TMDB_API_KEY`。

部署完成後，您的應用程式將可透過公開 URL 存取（例如 `https://your-app.streamlit.app`）。

## 未來展望
- **Deep Learning Ranking Models**：嘗試將現行 LambdaRank 樹狀架構替換或增強為神經網路排序架構。
- **Online Learning & Feedback Loop**：整合即時互動回饋機制，模擬生產環境中動態調整模型權重的特性。
