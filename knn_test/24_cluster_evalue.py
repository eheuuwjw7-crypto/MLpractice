import os

# Windows + MKL 环境下运行 KMeans 可能会反复提示内存泄漏 warning。
# sklearn 官方提示可以限制 OMP 线程数来避免这个 warning。
os.environ["OMP_NUM_THREADS"] = "4"

from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.metrics import calinski_harabasz_score,silhouette_score

def dm01_sse():
    sse_list = []

    x,y = make_blobs(n_samples=1000, n_features=2, centers=[[-1,-1],[0,0],[1,1],[2,2]], cluster_std=[0.4,0.2,0.2,0.2], random_state=42)
    for clu_num in range(1,100):
        kmeans = KMeans(n_clusters=clu_num, max_iter=100, random_state=42, n_init='auto')
        kmeans.fit(x)
        sse_list.append(kmeans.inertia_)

    plt.figure(figsize=(18,8),dpi=100)
    plt.xticks(range(1,100,3))
    plt.xlabel("K")
    plt.ylabel("SSE")
    plt.grid()
    plt.title("variance")
    plt.plot(range(1,100),sse_list,'or-')
    plt.show()
    
def dm02_sc():
    sc_list = []

    x,y = make_blobs(n_samples=1000, n_features=2, centers=[[-1,-1],[0,0],[1,1],[2,2]], cluster_std=[0.4,0.2,0.2,0.2], random_state=42)
    for clu_num in range(2,100):
        kmeans = KMeans(n_clusters=clu_num, max_iter=100, random_state=42, n_init='auto')
        kmeans.fit(x)
        y_pred = kmeans.predict(x)
        sc_value = silhouette_score(x,y_pred)
        sc_list.append(sc_value)

    plt.figure(figsize=(18,8),dpi=100)
    plt.xticks(range(1,100,3))
    plt.xlabel("K")
    plt.ylabel("sc")
    plt.grid()
    plt.title("variance")
    plt.plot(range(2,100),sc_list,'or-')
    plt.show()
    
def dm03_ch():
    ch_list = []

    x,y = make_blobs(n_samples=1000, n_features=2, centers=[[-1,-1],[0,0],[1,1],[2,2]], cluster_std=[0.4,0.2,0.2,0.2], random_state=42)
    for clu_num in range(2,100):
        kmeans = KMeans(n_clusters=clu_num, max_iter=100, random_state=42, n_init='auto')
        kmeans.fit(x)
        y_pred = kmeans.predict(x)
        ch_value = calinski_harabasz_score(x,y_pred)
        ch_list.append(ch_value)

    plt.figure(figsize=(18,8),dpi=100)
    plt.xticks(range(1,100,3))
    plt.xlabel("K")
    plt.ylabel("ch")
    plt.grid()
    plt.title("variance")
    plt.plot(range(2,100),ch_list,'or-')
    plt.show()

if __name__ == '__main__':
    # dm01_sse()
    dm02_sc()