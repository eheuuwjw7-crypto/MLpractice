import os

os.environ["OMP_NUM_THREADS"] = "1"

from scipy.cluster.vq import kmeans
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.metrics import calinski_harabasz_score,silhouette_score
import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"
ORI_DIR = DATA_DIR / "customers.csv"

def dm01_user_cluster():
    dataset = pd.read_csv(ORI_DIR)
    # dataset.info()
    # print(dataset)

    x = dataset.iloc[:,[3,4]]
    # print(x)

    sse_list = []
    sc_list = []

    for i in range(2,11):
        kmeans = KMeans(n_clusters=i, random_state=42, n_init='auto')
        kmeans.fit(x)
        sse_list.append(kmeans.inertia_)
        y_pred = kmeans.predict(x)
        sc_list.append(silhouette_score(x,y_pred))

    plt.plot(range(2,11),sse_list)
    plt.title("elbow method")
    plt.xlabel("K")
    plt.ylabel("SSE")
    plt.grid()
    plt.show()

    plt.plot(range(2, 11), sc_list)
    plt.title("sc")
    plt.xlabel("K")
    plt.ylabel("sc")
    plt.grid()
    plt.show()

def dm02_user_cluster_analyse():
    dataset = pd.read_csv(ORI_DIR)
    x = dataset.iloc[:, [3, 4]]
    kmeans = KMeans(n_clusters=5, random_state=42, n_init='auto',max_iter=100)
    kmeans.fit(x)
    y_pred = kmeans.predict(x)

    cluster_centers = kmeans.cluster_centers_
    cluster_names = {}

    # KMeans 给出的类别编号是随机编号，不一定 0 就代表某一类。
    # 这里根据聚类中心的位置，给每个簇起一个更容易看懂的名字。
    for cluster_id, center in enumerate(cluster_centers):
        income, spending = center
        if income < 45 and spending >= 60:
            cluster_names[cluster_id] = "Youth"
        elif income < 45 and spending < 60:
            cluster_names[cluster_id] = "TA"
        elif income >= 70 and spending >= 60:
            cluster_names[cluster_id] = "Traditional"
        elif income >= 70 and spending < 60:
            cluster_names[cluster_id] = "Normal"
        else:
            cluster_names[cluster_id] = "Standard"

    cluster_colors = {
        "Standard": "red",
        "Traditional": "blue",
        "Normal": "green",
        "Youth": "cyan",
        "TA": "magenta",
    }

    plt.figure(figsize=(10, 6), dpi=100)

    for cluster_id, cluster_name in cluster_names.items():
        cluster_data = x[y_pred == cluster_id]
        plt.scatter(
            cluster_data.iloc[:, 0],
            cluster_data.iloc[:, 1],
            c=cluster_colors[cluster_name],
            label=cluster_name
        )

    plt.scatter(
        cluster_centers[:, 0],
        cluster_centers[:, 1],
        c='black',
        s=160,
        label='Centroids'
    )

    plt.title("Clusters of customers")
    plt.xlabel("Annual Income (k$)")
    plt.ylabel("Spending Score (1-100)")
    plt.legend()
    plt.show()

if __name__ == '__main__':
    # dm01_user_cluster()
    dm02_user_cluster_analyse()
