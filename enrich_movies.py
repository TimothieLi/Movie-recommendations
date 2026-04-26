import pandas as pd
import numpy as np
import os

# 1. 載入資料
u_item_path = os.path.join('MovieLens 100K', 'u.item')
genre_cols = [
    "unknown", "Action", "Adventure", "Animation",
    "Children's", "Comedy", "Crime", "Documentary", "Drama", "Fantasy",
    "Film-Noir", "Horror", "Musical", "Mystery", "Romance", "Sci-Fi",
    "Thriller", "War", "Western"
]
item_cols = ['movie_id', 'movie_title', 'release_date', 'video_release_date', 'imdb_url'] + genre_cols
movies_df = pd.read_csv(u_item_path, sep='|', names=item_cols, encoding='latin-1')
movies_df = movies_df[['movie_id', 'movie_title'] + genre_cols]

tmdb_path = os.path.join('TMDB metadata', 'tmdb_5000_movies.csv')
tmdb_df = pd.read_csv(tmdb_path)

# Step 1: 清理 MovieLens 標題
movies_df["title_clean"] = (
    movies_df["movie_title"]
    .str.replace(r"\(\d{4}\)", "", regex=True)
    .str.strip()
    .str.lower()
)

# Step 2: 處理 TMDB dataset
tmdb_df["title_clean"] = tmdb_df["title"].astype(str).str.lower().str.strip()

# Step 3: Merge TMDB metadata (包含 overview, poster_path 如果有的話)
# 先去重，避免 merge 後資料量膨脹
tmdb_cols = ["title_clean", "overview", "vote_average", "popularity", "release_date"]
if "poster_path" in tmdb_df.columns:
    tmdb_cols.append("poster_path")

tmdb_subset = tmdb_df[tmdb_cols].drop_duplicates(subset=["title_clean"])

merged_df = movies_df.merge(
    tmdb_subset,
    on="title_clean",
    how="left"
)

# Step 4: 建立圖片 URL
if "poster_path" in merged_df.columns:
    merged_df["poster_url"] = merged_df["poster_path"].apply(
        lambda x: f"https://image.tmdb.org/t/p/w500{x}" if pd.notnull(x) else None
    )
else:
    merged_df["poster_url"] = None

# 補足 recency, quality, novelty 等欄位 (供 demo_app 使用)
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()

# Quality (用 vote_average)
merged_df['vote_average'] = merged_df['vote_average'].fillna(merged_df['vote_average'].median())
merged_df['quality'] = scaler.fit_transform(merged_df[['vote_average']])

# Recency (用 release_date)
merged_df['release_year_val'] = pd.to_datetime(merged_df['release_date'], errors='coerce').dt.year
merged_df['release_year_val'] = merged_df['release_year_val'].fillna(merged_df['release_year_val'].median())
merged_df['recency'] = scaler.fit_transform(merged_df[['release_year_val']])

# Novelty (用 popularity)
merged_df['popularity'] = merged_df['popularity'].fillna(merged_df['popularity'].median())
max_pop = merged_df['popularity'].max()
if max_pop == 0: max_pop = 1
raw_novelty = -np.log((merged_df['popularity'] / max_pop) + 1e-9)
merged_df['novelty'] = scaler.fit_transform(raw_novelty.values.reshape(-1, 1))

# Step 5: 儲存新 dataset
merged_df.to_csv("movies_with_metadata.csv", index=False)

print(f"✅ 成功儲存 movies_with_metadata.csv, 總數: {len(merged_df)}")
match_rate = merged_df['overview'].notnull().sum() / len(merged_df)
print(f"📈 TMDB 對齊比例 (以 Overview 為準): {match_rate*100:.2f}%")
