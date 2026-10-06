"""Exploratory clustering and PCA; not predictive evidence."""
from typing import Any

def analyze(draws: list[dict[str, Any]], max_number: int) -> dict[str, Any]:
    try:
        import numpy as np
        from sklearn.cluster import KMeans
        from sklearn.decomposition import PCA
        matrix = np.array([[int(n in d["numbers"]) for n in range(1, max_number + 1)] for d in draws])
        if len(matrix) < 3:
            raise ValueError("insufficient history")
        k = min(3, len(matrix))
        labels = KMeans(n_clusters=k, random_state=42, n_init=10).fit_predict(matrix)
        pca = PCA(n_components=min(2, matrix.shape[1])).fit(matrix)
        return {"clustering": {"algorithm": "kmeans", "label_counts": {str(i): int(sum(labels == i)) for i in range(k)}},
                "pca": {"components": int(pca.n_components_), "explained_variance_ratio": [float(x) for x in pca.explained_variance_ratio_]},
                "warning": "Phân cụm/PCA chỉ mang tính thăm dò."}
    except (ImportError, ValueError):
        return {"clustering": None, "pca": None, "warning": "Không đủ dữ liệu hoặc thiếu thư viện ML."}
