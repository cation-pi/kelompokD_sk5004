# metrics

## SSE (sum of squared errors)
SSE adalah sum distance^2 antara data point dan centroid cluster-nya.

$SSE = \sum_{i=1}^n \sum_{j=1}^k w^{(i,j)} \|x^{(i)} - \mu^{(j)}\|_2^2$

$\mu^{(j)}$ is the representative point (centroid) for cluster j. $w^{(i,j)}$ = 1 if the example, $x^{(i)}$, is in cluster j, or 0 otherwise.

$w^{(i,j)} = 
\begin{cases} 
1, & \text{if } x^{(i)} \in j \\
0, & \text{otherwise}
\end{cases}$

SSE ini akan dipakai oleh elbow method dan untuk debugging k-means. 

## silhouette score
$s^{(i)}$ is the difference between cluster cohesion ($a^{(i)}$) and cluster separation ($b^{(i)}$) divided by the greater of the two. $b^{(i)}$ quantifies how dissimilar an example is from other clusters, $a^{(i)}$ tells us how similar it is to the other examples in its own cluster. 

Steps:
1. calculate the cluster cohesion (rata-rata jarak titik x ke semua titik lain di cluster yang sama)
2. calculate the cluster separation from the next closest cluster as the average distance between the example and all examples in the nearest cluster.
3. calculate the silhouette

$s^{(i)} = \frac{b^{(i)} - a^{(i)}}{max\{b^{(i)}, a^{(i)}\}}$

* The silhouette coefficient is bounded in the range –1 to 1
* The silhouette coefficient is 0 if $b^{(i)} = a^{(i)}$
* $b^{(i)} >> a^{(i)}$ we get close to an ideal silhouette coefficient

## metrics.py documentation
* untuk perhitungan silhouette, digunakan euclidean distance biasa (bukan squared)
* 