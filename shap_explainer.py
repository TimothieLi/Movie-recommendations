import shap
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st

class MovieShapExplainer:
    def __init__(self, model, feature_names):
        """
        初始化 SHAP 解釋器
        model: 訓練好的 LightGBM 模型
        feature_names: 模型使用的特徵名稱列表
        """
        self.model = model
        self.feature_names = feature_names
        # 使用 TreeExplainer，這對樹模型（LightGBM）非常快
        self.explainer = shap.TreeExplainer(model)

    def get_shap_values(self, X):
        """
        計算給定樣本的 SHAP values
        X: 特徵矩陣 (pd.DataFrame or np.array)
        """
        # 如果是 DataFrame，確保特徵順序正確
        if isinstance(X, pd.DataFrame):
            X = X[self.feature_names]
        
        shap_values = self.explainer.shap_values(X)
        
        # 對於 LambdaRank 或二元分類，shap_values 可能是一個列表
        # 通常我們取 index 1 或直接使用（視版本而定）
        if isinstance(shap_values, list):
            return shap_values[0] # LambdaRank 通常只有一個 output
        return shap_values

    def get_top_features(self, single_shap_values, top_n=5):
        """
        取得單一樣本最重要的前 N 個特徵及其影響力
        """
        feature_impact = []
        for i, val in enumerate(single_shap_values):
            feature_impact.append({
                'feature': self.feature_names[i],
                'shap_value': val
            })
        
        # 按絕對值大小排序
        feature_impact.sort(key=lambda x: abs(x['shap_value']), reverse=True)
        return feature_impact[:top_n]

    def plot_shap_bar(self, single_shap_values, title="SHAP Explanation"):
        """
        使用 matplotlib 繪製簡單的水平柱狀圖
        """
        top_features = self.get_top_features(single_shap_values, top_n=10)
        top_features.reverse() # 讓大的在上面
        
        names = [f['feature'] for f in top_features]
        values = [f['shap_value'] for f in top_features]
        colors = ['#ff0051' if v > 0 else '#008bfb' for v in values] # 正紅負藍
        
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.barh(names, values, color=colors)
        ax.set_title(title)
        ax.set_xlabel("SHAP Value (Impact on Predict Score)")
        plt.tight_layout()
        return fig

@st.cache_resource
def get_cached_explainer(_model, _feature_names):
    return MovieShapExplainer(_model, _feature_names)
