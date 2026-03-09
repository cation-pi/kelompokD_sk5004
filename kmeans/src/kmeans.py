import numpy as np
import logging
from tqdm import tqdm
from src.distance import squared_euclidean_distance

# 1.classic: randomly pick k centroids from the examples as initial cluster centers
def initialize_random(x, k):
    n_samples = x.shape[0]
    random_indices = np.random.choice(n_samples, k, replace=False)
    return x[random_indices]

# 2: assign each example to the nearest centroid
def assign_clusters(x, centroids):
    # menghitung jarak dari semua titik data x ke setiap centroid (vektorisasi)
    distances = np.array([squared_euclidean_distance(x, c) for c in centroids])
    # mencari index centroid dengan jarak minimum (terdekat)
    return np.argmin(distances, axis=0)

# 3: move the centroids to the center of the examples that were assigned to it
def update_centroids(x, labels, k):
    n_features = x.shape[1]
    new_centroids = np.zeros((k, n_features))
    
    for i in range(k):
        cluster_points = x[labels == i]
        
        if len(cluster_points) > 0:
            # c_i = center of mass (rata-rata) dari titik2 di cluster
            new_centroids[i] = np.mean(cluster_points, axis=0)
        else:
            # handling jika ada cluster yang kosong selama iterasi
            new_centroids[i] = x[np.random.choice(x.shape[0])]
            
    return new_centroids

# 1.++: k-means++ initialization menggunakan d^2 weighting
def initialize_kmeans_pp(x, k):
    n_samples = x.shape[0]
    centroids = []
    
    # 1a: pilih center awal seragam secara acak dari data
    first_idx = np.random.choice(n_samples)
    centroids.append(x[first_idx])
    
    # 1c: ulangi hingga k center terpilih
    for _ in range(1, k):
        # hitung d(x)^2: jarak terpendek dari setiap data ke center yang sudah ada
        d_sq = np.min([squared_euclidean_distance(x, c) for c in centroids], axis=0)
        
        # 1b: gunakan distribusi probabilitas berdasarkan jarak kuadrat
        probs = d_sq / np.sum(d_sq)
        
        # pilih center berikutnya secara acak dengan bobot probabilitas d^2
        next_idx = np.random.choice(n_samples, p=probs)
        centroids.append(x[next_idx])
        
    return np.array(centroids)

# implementasi classic k-means
def kmeans(x, k, max_iters=100, tol=1e-4):
    logging.info(f"running classic k-means dengan k={k}")
    centroids = initialize_random(x, k)
    
    for i in tqdm(range(max_iters), desc="k-means iterations"):
        old_centroids = centroids.copy()
        
        labels = assign_clusters(x, centroids)
        centroids = update_centroids(x, labels, k)
        
        # periksa konvergensi: perubahan centroid di bawah nilai toleransi
        shift = np.sum((centroids - old_centroids) ** 2)
        if shift < tol:
            logging.info(f"k-means konvergen pada iterasi ke-{i+1}")
            break
    else:
        logging.warning("k-means mencapai batas iterasi maksimum tanpa konvergen penuh.")
        
    return labels, centroids

# implementasi k-means++
def kmeans_pp(x, k, max_iters=100, tol=1e-4):
    logging.info(f"running k-means++ dengan k={k}")
    centroids = initialize_kmeans_pp(x, k)
    
    # lanjut dengan prosedur k-means standar
    for i in tqdm(range(max_iters), desc="k-means++ iterations"):
        old_centroids = centroids.copy()
        
        labels = assign_clusters(x, centroids)
        centroids = update_centroids(x, labels, k)
        
        shift = np.sum((centroids - old_centroids) ** 2)
        if shift < tol:
            logging.info(f"k-means++ konvergen pada iterasi ke-{i+1}")
            break
    else:
        logging.warning("k-means++ mencapai batas iterasi maksimum tanpa konvergen penuh.")
        
    return labels, centroids