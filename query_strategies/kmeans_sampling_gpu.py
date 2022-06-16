import numpy as np
from .strategy import Strategy
from sklearn.cluster import KMeans
import faiss

class KMeansSamplingGPU(Strategy):
    def __init__(self, dataset, net, args_input, args_task):
        super(KMeansSamplingGPU, self).__init__(dataset, net, args_input, args_task)

    def query(self, n):
        unlabeled_idxs, unlabeled_data = self.dataset.get_unlabeled_data()
        embeddings = self.get_embeddings(unlabeled_data).numpy()
        cluster_learner = FaissKmeans(n_clusters = n, gpu = True)
        cluster_learner.fit(embeddings)
        dis, q_idxs = cluster_learner.predict(embeddings)
        q_idxs = q_idxs.T[0]
        
        return unlabeled_idxs[q_idxs]


class FaissKmeans:
    def __init__(self, n_clusters=8, gpu=True, n_init=10, max_iter=300):
        self.n_clusters = n_clusters
        self.n_init = n_init
        self.max_iter = max_iter
        self.kmeans = None
        self.cluster_centers_ = None
        self.inertia_ = None
        self.gpu = gpu

    def fit(self, X):
        self.kmeans = faiss.Kmeans(d=X.shape[1],
                                   k=self.n_clusters,
                                   niter=self.max_iter,
                                   nredo=self.n_init,
                                   gpu = self.gpu)
        self.kmeans.train(X.astype(np.float32))
        self.cluster_centers_ = self.kmeans.centroids
        self.inertia_ = self.kmeans.obj[-1]

    def predict(self, X):
        D, I = self.kmeans.index.search(X.astype(np.float32), 1)
        return D, I