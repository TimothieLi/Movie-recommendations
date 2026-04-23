import requests
import pandas as pd
import numpy as np
from datetime import datetime

class TMDBClient:
    """
    TMDB API Client for fetching external movie candidates.
    """
    def __init__(self, api_key):
        self.api_key = api_key
        self.base_url = "https://api.themoviedb.org/3"
        self.img_base_url = "https://image.tmdb.org/t/p/w500"
        
        # TMDB Genre ID to MovieLens Genre Name Mapping
        self.genre_mapping = {
            28: "Action",
            12: "Adventure",
            16: "Animation",
            35: "Comedy",
            80: "Crime",
            99: "Documentary",
            18: "Drama",
            10751: "Children's",
            14: "Fantasy",
            27: "Horror",
            10402: "Musical",
            9648: "Mystery",
            10749: "Romance",
            878: "Sci-Fi",
            53: "Thriller",
            10752: "War",
            37: "Western"
        }
        
    def fetch_discover_movies(self, sort_by="popularity.desc", min_vote=6.0, page=1):
        """
        Fetch movies from TMDB discover endpoint.
        """
        endpoint = f"{self.base_url}/discover/movie"
        params = {
            "api_key": self.api_key,
            "language": "zh-TW",
            "sort_by": sort_by,
            "vote_average.gte": min_vote,
            "vote_count.gte": 100,
            "page": page
        }
        
        try:
            response = requests.get(endpoint, params=params)
            response.raise_for_status()
            data = response.json()
            return data.get("results", [])
        except Exception as e:
            print(f"Error fetching TMDB movies: {e}")
            return []

    def get_candidates(self, user_id, count=50, genre_cols=None):
        """
        Fetch and format TMDB movies as system candidates for a specific user.
        """
        raw_movies = self.fetch_discover_movies()
        if not raw_movies:
            return pd.DataFrame()
            
        formatted_list = []
        
        # Calculate max popularity for normalization
        max_pop = max([m.get("popularity", 1) for m in raw_movies]) if raw_movies else 1
        
        for m in raw_movies[:count]:
            # Basic info
            release_date = m.get("release_date", "")
            release_year = 0
            if release_date:
                try:
                    release_year = datetime.strptime(release_date, "%Y-%m-%d").year
                except:
                    pass
            
            # Feature mapping
            # novelty = inverse popularity (higher novelty means lower popularity)
            pop = m.get("popularity", 0)
            norm_pop = pop / max_pop if max_pop > 0 else 0
            novelty = 1.0 - norm_pop
            
            # quality = vote_average / 10
            quality = m.get("vote_average", 0) / 10.0
            
            # recency = (year - 1920) / (2024 - 1920)
            recency = max(0, (release_year - 1920) / (2024 - 1920)) if release_year > 0 else 0
            
            # predict_score: default for external movies
            predict_score = 0.5 
            
            movie_item = {
                "user_id": user_id,
                "movie_id": f"tmdb_{m['id']}",
                "movie_title": m.get("title", m.get("original_title", "Unknown")),
                "release_year": release_year,
                "predict_score": predict_score,
                "novelty": novelty,
                "quality": quality,
                "recency": recency,
                "source": "tmdb",
                "overview": m.get("overview", ""),
                "poster_path": f"{self.img_base_url}{m.get('poster_path', '')}" if m.get("poster_path") else None
            }
            
            # Genre mapping (Multi-hot)
            if genre_cols:
                for col in genre_cols:
                    movie_item[col] = 0
                
                tmdb_genre_ids = m.get("genre_ids", [])
                for gid in tmdb_genre_ids:
                    if gid in self.genre_mapping:
                        ml_genre = self.genre_mapping[gid]
                        if ml_genre in genre_cols:
                            movie_item[ml_genre] = 1
            
            formatted_list.append(movie_item)
            
        return pd.DataFrame(formatted_list)
