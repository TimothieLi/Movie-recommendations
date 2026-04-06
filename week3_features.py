import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def get_week3_analysis():
    # ==========================================
    # 0. 讀取資料 (與 baseline 相同)
    # ==========================================
    data_cols = ['user_id', 'movie_id', 'rating', 'timestamp']
    u_data_path = os.path.join('MovieLens 100K', 'u.data')
    if not os.path.exists(u_data_path): u_data_path = 'u.data'
    ratings_df = pd.read_csv(u_data_path, sep='\t', names=data_cols)
    
    genre_cols = [
        "unknown", "Action", "Adventure", "Animation",
        "Children's", "Comedy", "Crime", "Documentary", "Drama", "Fantasy",
        "Film-Noir", "Horror", "Musical", "Mystery", "Romance", "Sci-Fi",
        "Thriller", "War", "Western"
    ]
    item_cols = ['movie_id', 'movie_title', 'release_date', 'video_release_date', 'imdb_url'] + genre_cols
    u_item_path = os.path.join('MovieLens 100K', 'u.item')
    if not os.path.exists(u_item_path): u_item_path = 'u.item'
    movies_df = pd.read_csv(u_item_path, sep='|', names=item_cols, encoding='latin-1')
    movies_df = movies_df[['movie_id', 'movie_title'] + genre_cols]

    # ==========================================
    # 1. 計算 popularity
    # ==========================================
    popularity_df = ratings_df.groupby('movie_id').size().reset_index(name='popularity')
    movies_feat = movies_df.merge(popularity_df, on='movie_id', how='left').fillna({'popularity': 0})
    
    # ==========================================
    # 2. 定義 novelty
    # ==========================================
    max_popularity = movies_feat['popularity'].max()
    movies_feat['novelty_raw'] = -np.log((movies_feat['popularity'] + 1) / (max_popularity + 1))
    
    # ==========================================
    # 3. 對 novelty 做 Min-Max Normalization
    # ==========================================
    min_nov = movies_feat['novelty_raw'].min()
    max_nov = movies_feat['novelty_raw'].max()
    
    if max_nov > min_nov:
        movies_feat['novelty_norm'] = (movies_feat['novelty_raw'] - min_nov) / (max_nov - min_nov)
    else:
        movies_feat['novelty_norm'] = 0.0

    # 取得極端值
    top3_highest_novelty = movies_feat.nlargest(3, 'novelty_norm')[['movie_title', 'popularity', 'novelty_norm']]
    top3_lowest_novelty = movies_feat.nsmallest(3, 'novelty_norm')[['movie_title', 'popularity', 'novelty_norm']]

    # ==========================================
    # 4. 定義電影相似度函數
    # ==========================================
    def compute_movie_similarity(movie_id_1, movie_id_2):
        m1 = movies_feat[movies_feat['movie_id'] == movie_id_1]
        m2 = movies_feat[movies_feat['movie_id'] == movie_id_2]
        
        if m1.empty or m2.empty:
            return 0.0
            
        m1_genres = m1[genre_cols].values[0]
        m2_genres = m2[genre_cols].values[0]
        return np.dot(m1_genres, m2_genres)

    # 測試第一部電影和第二部電影的相似度
    test_m1 = movies_feat.iloc[0]['movie_id']
    test_m2 = movies_feat.iloc[1]['movie_id']
    sim_score = compute_movie_similarity(test_m1, test_m2)
    m1_title = movies_feat.iloc[0]['movie_title']
    m2_title = movies_feat.iloc[1]['movie_title']

    # ==========================================
    # 5. 繪製 novelty 分布圖
    # ==========================================
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(movies_feat['novelty_norm'], bins=50, color='skyblue', edgecolor='black')
    ax.set_title('Novelty Distribution [Min-Max Normalized]')
    ax.set_xlabel('Novelty (0 = most popular, 1 = least popular)')
    ax.set_ylabel('Number of Movies')
    ax.grid(axis='y', alpha=0.75)
    
    return movies_feat, top3_highest_novelty, top3_lowest_novelty, m1_title, m2_title, sim_score, fig

if __name__ == "__main__":
    get_week3_analysis()
