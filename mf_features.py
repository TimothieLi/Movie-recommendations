import numpy as np

class PureNumpyMF:
    """
    純 Numpy 實作的 Matrix Factorization (SGD)。
    優點：0 依賴風險，支援任何作業系統環境，不用額外編譯 C++ 擴展，完美融入目前專案。
    此特徵負責從 Train_df 中抽取 User-Item 的隱含偏好 (Latent Preference Signal)，
    並提供給下游 LightGBM 進行更細緻的排序決策。
    """
    def __init__(self, n_factors=20, n_epochs=15, lr=0.01, reg=0.05):
        self.n_factors = n_factors
        self.n_epochs = n_epochs
        self.lr = lr
        self.reg = reg
        
    def fit(self, df):
        print("Training Pure Numpy Matrix Factorization (Latent Preferences)...")
        users = df['user_id'].unique()
        items = df['movie_id'].unique()
        
        self.user2idx = {u: i for i, u in enumerate(users)}
        self.item2idx = {item: i for i, item in enumerate(items)}
        
        self.global_mean = df['rating'].mean()
        
        n_users = len(users)
        n_items = len(items)
        
        # 參數初始化
        self.P = np.random.normal(0, 0.1, (n_users, self.n_factors))
        self.Q = np.random.normal(0, 0.1, (n_items, self.n_factors))
        self.b_u = np.zeros(n_users)
        self.b_i = np.zeros(n_items)
        
        u_idx = np.array([self.user2idx[u] for u in df['user_id']])
        i_idx = np.array([self.item2idx[i] for i in df['movie_id']])
        ratings = df['rating'].values - self.global_mean
        
        for epoch in range(self.n_epochs):
            # for 迴圈逐筆更新 SGD
            for u, i, r in zip(u_idx, i_idx, ratings):
                err = r - (self.b_u[u] + self.b_i[i] + np.dot(self.P[u], self.Q[i]))
                
                self.b_u[u] += self.lr * (err - self.reg * self.b_u[u])
                self.b_i[i] += self.lr * (err - self.reg * self.b_i[i])
                
                Pu = self.P[u]
                Qi = self.Q[i]
                
                self.P[u] += self.lr * (err * Qi - self.reg * Pu)
                self.Q[i] += self.lr * (err * Pu - self.reg * Qi)
                
        print("MF Training Completed!")
        
    def predict(self, user_id, movie_id):
        # 推論，若找不到則退回 global mean
        u = self.user2idx.get(user_id, -1)
        i = self.item2idx.get(movie_id, -1)
        
        pred = self.global_mean
        if u != -1:
            pred += self.b_u[u]
        if i != -1:
            pred += self.b_i[i]
        if u != -1 and i != -1:
            pred += np.dot(self.P[u], self.Q[i])
            
        return pred
    
    def predict_batch(self, user_ids, movie_ids):
        # 優化預測，避免 Python 外部 for 迴圈拖慢大量推論
        preds = []
        for u_id, i_id in zip(user_ids, movie_ids):
            preds.append(self.predict(u_id, i_id))
        return np.array(preds)


def train_mf_model(train_df):
    model = PureNumpyMF(n_factors=15, n_epochs=15, lr=0.01, reg=0.05)
    model.fit(train_df)
    return model

def predict_mf_score(model, user_ids, movie_ids):
    return model.predict_batch(user_ids, movie_ids)
