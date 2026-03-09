import numpy as np
import matplotlib.pyplot as plt
import logging
from tqdm import tqdm
from src.metrics import calculate_sse, calculate_silhouette_score

# mencari titik siku (elbow) dengan pendekatan geometris
def find_elbow_point(k_values, sse_values):
    p1 = np.array([k_values[0], sse_values[0]])
    p2 = np.array([k_values[-1], sse_values[-1]])
    
    distances = []
    for i in range(len(k_values)):
        p3 = np.array([k_values[i], sse_values[i]])
        # menghitung jarak dari titik p3 ke garis lurus p1-p2
        dist = np.abs(np.cross(p2 - p1, p1 - p3)) / np.linalg.norm(p2 - p1)
        distances.append(dist)
        
    best_k_idx = np.argmax(distances)
    return k_values[best_k_idx]

# mencari k terbaik (elbow method)
def choose_k_elbow(x, clustering_algo, max_k=10, plot=True, **kwargs):
    logging.info("running metode elbow...")
    k_values = list(range(1, max_k + 1))
    sse_values = []
    
    # run k-means berkali2 dan hitung sse
    for k in tqdm(k_values, desc="elbow evaluation"):
        labels, centroids = clustering_algo(x, k, **kwargs)
        sse = calculate_sse(x, labels, centroids)
        sse_values.append(sse)
        
    best_k = find_elbow_point(k_values, sse_values)
    logging.info(f"metode elbow menyarankan k={best_k}")
    
    # visualisasi k (sumbu x) vs sse (sumbu y)
    if plot:
        plt.figure(figsize=(8, 5))
        plt.plot(k_values, sse_values, marker='o', linestyle='-')
        plt.axvline(x=best_k, color='red', linestyle='--', label=f'suggested k={best_k}')
        plt.title('elbow method')
        plt.xlabel('number of clusters')
        plt.ylabel('distortion (sse)')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.show()
        
    return best_k

# mencari nilai k terbaik (silhouette analysis)
def choose_k_silhouette(x, clustering_algo, max_k=10, plot=True, **kwargs):
    logging.info("running metode silhouette...")
    # k=1 tidak valid untuk silhouette, jadi mulai dari 2
    k_values = list(range(2, max_k + 1)) 
    sil_scores = []
    
    # run k-means berkali2 dan hitung silhouette score
    for k in tqdm(k_values, desc="silhouette evaluation"):
        labels, centroids = clustering_algo(x, k, **kwargs)
        score = calculate_silhouette_score(x, labels)
        sil_scores.append(score)
        
    # pilih k dengan silhouette score terbesar
    best_k_idx = np.argmax(sil_scores)
    best_k = k_values[best_k_idx]
    
    logging.info(f"metode silhouette menyarankan k={best_k} (score maksimum: {sil_scores[best_k_idx]:.4f})")
    
    if plot:
        plt.figure(figsize=(8, 5))
        plt.plot(k_values, sil_scores, marker='s', linestyle='-', color='orange')
        plt.axvline(x=best_k, color='red', linestyle='--', label=f'best k={best_k}')
        plt.title('silhouette method evaluation')
        plt.xlabel('number of clusters')
        plt.ylabel('average silhouette score')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.show()
        
    return best_k