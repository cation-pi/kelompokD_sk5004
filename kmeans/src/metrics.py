import numpy as np

# menghitung sse
def calculate_sse(x, labels, centroids):
    sse = 0.0
    
    # iterasi untuk setiap cluster j
    for j, centroid in enumerate(centroids):
        # implementasi w^(i,j) = 1, ambil hanya titik2 yang masuk di cluster j
        cluster_points = x[labels == j]
        
        if len(cluster_points) > 0:
            # hitung sse
            distances = np.sum((cluster_points - centroid) ** 2, axis=-1)
            sse += np.sum(distances)
            
    return sse

# menghitung koefisien silhouette untuk setiap titik data (individual)
def calculate_silhouette_samples(x, labels):
    n_samples = len(x)
    s_i = np.zeros(n_samples)
    unique_labels = np.unique(labels)

    for i in range(n_samples):
        current_label = labels[i]
        
        # cluster cohesion (a_i)
        ## cari semua titik di cluster yang sama, kecuali titik i itu sendiri
        same_cluster_mask = (labels == current_label)
        same_cluster_mask[i] = False 
        
        ## jika cluster hanya memiliki 1 titik, nilai silhouette = 0
        if np.sum(same_cluster_mask) == 0:
            s_i[i] = 0.0
            continue
            
        ## rata-rata euclidean distance dari titik i ke titik lain di cluster yang sama
        a_i = np.mean(np.linalg.norm(x[same_cluster_mask] - x[i], axis=-1))
        
        # cluster separation (b_i)
        b_i = np.inf
        for label in unique_labels:
            if label != current_label:
                other_cluster_mask = (labels == label)
                
                if np.sum(other_cluster_mask) > 0:
                    ## rata-rata euclidean distance dari titik i ke cluster tetangga
                    dist_to_other = np.mean(np.linalg.norm(x[other_cluster_mask] - x[i], axis=-1))
                    
                    ## cari jarak terdekat ke cluster tetangga
                    if dist_to_other < b_i:
                        b_i = dist_to_other
                        
        # koefisien silhouette
        s_i[i] = (b_i - a_i) / max(a_i, b_i)
        
    return s_i

# menghitung nilai keseluruhan (mean) koefisien silhouette untuk dataset
def calculate_silhouette_score(x, labels):
    s_i = calculate_silhouette_samples(x, labels)
    return np.mean(s_i)