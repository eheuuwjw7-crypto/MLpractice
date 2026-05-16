from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.metrics import calinski_harabasz_score

x,y = make_blobs(n_samples=1000, n_features=2, centers=[[-1,-1],[0,0],[1,1],[2,2]], cluster_std=[0.4,0.2,0.2,0.2], random_state=42)
plt.figure()
plt.scatter(x[:,0],x[:,1],marker='o')
# plt.show()

y_pred = KMeans(n_clusters=4,random_state=22).fit_predict(x)
plt.scatter(x[:,0],x[:,1],c=y_pred)
plt.show()
print(calinski_harabasz_score(x,y_pred))