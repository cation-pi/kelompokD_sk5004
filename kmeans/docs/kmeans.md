# k-means algorithm

## urutan fungsi
initialize_random()
assign_clusters()
update_centroids()
kmeans()
initialize_kmeans_pp()
kmeans_pp()

## classic k-means
### Python Machine Learning 3rd Edition by Sebastian Raschka & Vahid Mirjalili
1. Randomly pick k centroids from the examples as initial cluster centers.
2. Assign each example to the nearest centroid, $\mu^{(j)}, j \in {1, \dots, k}$
3. Move the centroids to the center of the examples that were assigned to it.
4. Repeat steps 2 and 3 until the cluster assignments do not change or a user-defined tolerance or maximum number of iterations is reached.

### Arthur & Vassilvitskii (2007)
1. Arbitrarily choose $k$ initial centers $C = \{c_1, \ldots, c_k\}$.
2. For each $i \in \{1, \ldots, k\}$, set the cluster $C_i$ to be the set of points in $X$ that are closer to $c_i$ than they are to $c_j$ for all $j \neq i$.
3. For each $i \in \{1, \ldots, k\}$, set $c_i$ to be the center of mass of all points in $C_i$: $c_i := \frac{1}{|C_i|} \sum_{x \in C_i} x.$
4. Repeat Steps 2 and 3 until $C$ no longer changes.

## k-means++
### Python Machine Learning 3rd Edition by Sebastian Raschka & Vahid Mirjalili
1. Initialize an empty set, **M**, to store the $k$ centroids being selected.
2. Randomly choose the first centroid, $\mu^{(j)}$, from the input examples and assign it to **M**.
3. For each example, $x^{(i)}$, that is not in **M**, find the minimum squared distance,  
   $d(x^{(i)}, M)^2$, to any of the centroids in **M**.
4. To randomly select the next centroid, $\mu^{(p)}$, use a weighted probability distribution equal to  
   $\frac{d(\mu^{(p)}, M)^2}{\sum_i d(x^{(i)}, M)^2}$.
5. Repeat steps 2 and 3 until $k$ centroids are chosen.
6. Proceed with the classic k-means algorithm.

### Arthur & Vassilvitskii (2007)
At any given time, let $D(x)$ denote the shortest distance from a data point x to the closest center we have already chosen. Then, we deﬁne the following algorithm.
1a. Choose an initial center $c_1$ uniformly at random from $\mathcal{X}$.
1b. Choose the next center $c_i$, selecting $c_i = x' \in \mathcal{X}$ with probability  
$\frac{D(x')^2}{\sum_{x \in \mathcal{X}} D(x)^2}.$
1c. Repeat Step 1b until we have chosen a total of $k$ centers.
2-4. Proceed as with the standard $k$-means algorithm.

We call the weighting used in Step 1b simply "$D^2$ weighting".