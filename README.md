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

## 專案結構 (Project Structure)

```text
大三專題/
│── app.py                    # 完整前端互動介面入口
│── demo_app.py               # 精簡展示版前端入口
│── requirements.txt          # 環境依賴套件清單
│── movie_lgb_recommender.py  # 核心資料載入、特徵工程與 LightGBM 訓練
│── week3_features.py         # 多目標特徵工程 (新穎度、多樣性)
│── week4_reranking.py        # 重新排序策略 (Pareto, MMR)
│── week5_nlp_pareto.py       # TMDB 資料串接、自然語言解析與動態 Pareto
│── week6_evaluation.py       # 系統層級評估與圖表繪製
└── README.md                 # 專案說明文件
```

## 資料集準備 (Dataset Setup)

本專案依賴兩個開源資料集。為確保程式能正確讀取資料，請依照以下說明下載並放置檔案：

### 1. MovieLens 100K
* **來源**：[GroupLens 官方下載點](https://grouplens.org/datasets/movielens/100k/) (下載 `ml-100k.zip` 並解壓縮)
* **必需檔案**：`u.data` (使用者評分) 與 `u.item` (電影資訊)
* **預期路徑**：建議在專案根目錄建立 `MovieLens 100K` 資料夾，並將檔案放入其內。
* **結構範例**：
  ```text
  大三專題/
  └── MovieLens 100K/
      ├── u.data
      └── u.item
  ```
*(程式兼容於根目錄下作為替代位置)*

### 2. TMDB 5000 Movies Metadata (Week 5 之後需要)
* **來源**：[Kaggle TMDB 5000 Movie Dataset](https://www.kaggle.com/datasets/tmdb/tmdb-movie-metadata)
* **必需檔案**：`tmdb_5000_movies.csv`
* **預期路徑**：請在專案根目錄建立 `TMDB metadata` 資料夾，並將 CSV 檔案放入其內。
* **結構範例**：
  ```text
  大三專題/
  └── TMDB metadata/
      └── tmdb_5000_movies.csv
  ```

## 環境設定與安裝 (Setup & Installation)

本專案建議使用 **Python 3.10+** 並透過 `venv` 建立獨立虛擬環境來執行。

```bash
# 1. 進入專案資料夾
cd 大三專題

# 2. 建立並啟動虛擬環境
python -m venv venv

# macOS / Linux 系統請執行：
source venv/bin/activate
# Windows 系統請執行：
# venv\Scripts\activate

# 3. 安裝所有依賴套件
pip install -r requirements.txt

# 4. 啟動 Streamlit 應用程式
python -m streamlit run app.py
```
> 註：執行前請確保此資料夾下有 `MovieLens 100K` 資料集 (包含 `u.data` 及 `u.item`)，以及相關 TMDB 資料集。

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
  
- **👉 Week 6: 方法比較與系統評估分析 (System-Level Evaluation)**
  - **多維評估指標**：實作了 `NDCG@K`（排序準確度）、`Novelty@K`（新穎度平均）、`ILD@K（Intra-List Diversity）`（清單內成對相似度）、`Coverage`（涵蓋率）共四大量化指標。
  - **大規模批次比較**：在相同候選池（前 50 名）與 Top-10 條件下，對 LightGBM Baseline、MMR（五種 λ）、Pareto 及四種 NLP 目標組合進行全面 Batch 評測。
  - **Trade-off 視覺化**：以 Seaborn 散佈圖繪製 `NDCG vs Novelty` 及 `NDCG vs ILD (Diversity)` 對決，清楚展現不同方法在準確度與多元性之間的取捨。
  - 評估腳本：`week6_evaluation.py`。

這五個階段的成果完整地串接並整合於 Streamlit 互動介面中，隨時可以進行網頁互動展示！

## 啟動方式 (How to Run)

本專案提供兩個 Streamlit 入口：

```bash
# 完整功能版（含 Week 3~6 所有頁籤）
python -m streamlit run app.py

# 精簡展示版（專題 Demo 用）
python -m streamlit run demo_app.py
```

> 若同時啟動兩個版本，請指定不同 Port：
> ```bash
> python -m streamlit run demo_app.py --server.port 8502
> ```

## Demo 操作說明

`demo_app.py` 是一個專為專題展示設計的輕量級互動面板：

1. 在側邊欄輸入 **User ID**（Test Set 範圍內）
2. 從下拉選單選擇 **推薦方法**（Baseline / MMR / Pareto / Pareto+NLP）
3. 若選擇 MMR，可用 Slider 即時調整 **λ 值**
4. 若選擇 Pareto + NLP，可輸入 **自然語言 Prompt**（例如：推薦冷門且多樣的電影）
5. 點擊「🚀 產生推薦」查看推薦結果
6. 畫面同時呈現 **Baseline 對照** 與 **所選方法的 Top-K 結果**

## Offline 表現 (Baseline)

系統的初版表現為：
- **Mean Recall@10** : ~4.74% (以真實 Rating >= 3.0 當作相關基準)
- **Mean NDCG@10** : ~14.10%
*(由於特徵簡潔，這是一個理想的 Baseline)*

## 未來展望
- 嘗試將現行 LambdaRank 架構神經網路化 (Deep Learning Ranking Models)。
- 支援真實使用者歷史即時操作的回饋迴圈 (Online Learning)。

## 可重現性聲明 (Reproducibility)

本專案致力於維持公開運行的可重現性：
1. 專案的執行依賴已統一列於 `requirements.txt` 中。
2. 開發與驗證均建議於隔離的虛擬環境 (venv) 下進行。
3. 本專案已於開發環境進行基本功能測試；為確保推薦系統模型能正確運行，使用者需嚴格依照上方【資料集準備】的說明配置相關資料檔案。

### 建議驗證步驟

為確認您的環境是否建置完備，建議新使用者可透過以下流程自行驗證：
1. **建立乾淨環境**：開啓終端機並建立全新的虛擬環境 (`python -m venv venv`)。
2. **安裝相依套件**：啟動環境後執行 `pip install -r requirements.txt`。
3. **啟動前端應用**：執行 `python -m streamlit run app.py`；若順利進入網頁且無出現模組遺漏錯誤或資料路徑報錯，即視為驗證成功。
