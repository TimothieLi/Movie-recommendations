# Movie Recommendation System 🎬

這是一個基於 **LightGBM (LambdaRank)** 所建構的電影推薦系統，屬於大三專題專案的一部分。系統採用 [MovieLens 100K Dataset](https://grouplens.org/datasets/movielens/100k/) 作為訓練與測試資料集。

## 網站展示 (Streamlit Frontend)

為了將訓練出的離線預測結果具象化，我們實作了一個互動式的 Web App 前端：
- **快速切換**：可從側邊欄自由選擇 Test Set 中的 User ID。
- **極速預測**：採用批次 (Batch Evaluation) 結合記憶體快取 (`@st.cache_data`)，百萬條電影候選清單可在幾秒內全數預測完畢，切換使用者延遲為 0 毫秒！
- **推薦命中指示**：系統精選前 10 名推薦電影。如果該部電影剛好出現在該用戶的「測試期間真實好評名單」中，則會標亮為 `⭐ 命中!`。

## 系統核心架構

本專案將機器學習與 UI 流暢結合在一起。主要包含兩支程式檔案：

1. **`movie_lgb_recommender.py`**:
   - 負責資料載入與 Train / Test 拆分。
   - 進行特徵工程 (User Avg Rating, Movie Popularity, Genre Matches)。
   - 訓練 LightGBM (LambdaRank) 排序模型。
   - 內建 Week 2 評核機制 (`Recall@10` 與 `NDCG@10`)。
2. **`app.py`**:
   - 使用 [Streamlit](https://streamlit.io/) 建立的前端互動網頁。
   - 負責串接 `run_recommender_pipeline()` 並利用 Dataframe 完美呈現推薦結果與評分。

## Offline 表現 (Baseline)

系統的初版表現為：
- **Mean Recall@10** : ~4.74% (以真實 Rating >= 3.0 當作相關基準)
- **Mean NDCG@10** : ~14.10%
*(由於特徵簡潔，這是一個理想的 Baseline)*

## 執行與安裝說明 (Installation)

請先確保您具有 Python 3.8+ 且有基本的 `venv` 環境。安裝步驟如下：

```bash
# 1. 進入專案資料夾
cd 大三專題

# 2. 啟動虛擬環境 (如果沒有請先 python -m venv venv)
source venv/bin/activate

# 3. 安裝依賴 (含 Streamlit)
pip install pandas numpy lightgbm streamlit

# 4. 啟動前端程式
streamlit run app.py
```
> 註：執行前請確保此資料夾下有 `MovieLens 100K` 資料集 (包含 `u.data` 及 `u.item`)。

## 最新進度：Week 3 & Week 4 演算法升級

除了基本的 LightGBM 模型外，系統現已整合多目標策略，以提升推薦滿意度：

- **👉 Week 3: 多目標特徵工程 (Multi-objective Feature Engineering)**
  - 實作了 **Novelty (新穎度)** 計算：`Novelty(i) = -log(popularity(i) / max_popularity)`
  - 實作了 **Diversity (多樣性)** 相似度基礎：基於 Genre 的 Multi-hot 內積算法計算。
  - 獨立腳本：`week3_features.py`，支援繪製電影分布。

- **👉 Week 4: 雙重重新排序策略 (Re-ranking)**
  - **Pareto Dominance**：尋找「不可支配」的電影集合 (Pareto Frontier) 進行重排，在不妥協分數的情況下保證最佳新穎度。
  - **Maximal Marginal Relevance (MMR)**：實作 Greedy 貪婪演算法，透過動態切換 $\lambda$ (0.0 ~ 1.0) 自由調配 Relevance 與 Diversity 的拉扯權衡。
  - 獨立腳本：`week4_reranking.py`。
  
- **👉 Week 5: TMDB 資料串接與 NLP 動態多目標推薦**
  - **資料擴充**：將 MovieLens 與 TMDB (tmdb_5000_movies) 資料庫進行清洗配對，萃取 `popularity`, `vote_average` 及 `release_year`，擴充為進階評估特徵 (`novelty`, `quality` & `recency`)。
  - **自然語言解析**：實作了 Rule-based NLP 分析器，支援直覺的語意關鍵字組合（如：「推薦近期上映而且多樣化的好片」）。
  - **動態 Pareto Re-ranking**：打破傳統單純排序，實作高階動態 Pareto 機制。當指令包含單一目標時採用高速排序；當面臨 `diversity` (多樣性) 目標時，則採用 Greedy 邊界尋優，每一輪皆會根據「當前已選片單」即時翻新候選池的 Jaccard 相似度！
  - 獨立腳本：`week5_nlp_pareto.py`。
  
這三個禮拜的進階演算法成果都已完整串接並整合於目前所見的 Streamlit 介面中，並透過側邊欄 (Sidebar) 獨立頁面呈現，隨時可以進行網頁互動展示！

## 未來展望
- 嘗試將現行 LambdaRank 架構神經網路化 (Deep Learning Ranking Models)。
- 支援真實使用者的歷史即時操作回饋迴圈 (Online Learning)。
