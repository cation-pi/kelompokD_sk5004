# distance
## squared euclidean distance
$d(x, y)^2 = \sum_{j=1}^m (x_j - y_j)^2 = \|x - y\|_2^2$

## minkowski distance
$d(x, y) = \left( \sum_{i=1}^n |x_i - y_i|^p \right)^{1/p}$

## cosine distance
$d(x, y) = 1 - \frac{x \cdot y}{\|x\| \|y\|}$

## chebyshev distance
$d(x, y) = \max_i |x_i - y_i|$

## manhattan distance
$d(x, y) = \sum_{i=1}^n |x_i - y_i|$


## documentation untuk distance.py
* euclidean pakai yg squared untuk meminimalkan SSE dan menghilangkan operasi akar kuadrat. 
* parameter `axis=-1` digunakan untuk mendukung broadcasting. ini membantu saat menghitung jarak antara banyak titik data dengan beberapa centroid sekaligus tanpa perlu looping manual. broadcasting di NumPy: dengan menggunakan axis=-1, jika x adalah kumpulan data berukuran (N, D) dan y adalah sebuah centroid (D,), numpy akan otomatis menghitung jarak untuk seluruh N baris sekaligus. 
* epsilon kecil ($10^{-10}$) pada pembagi di cosine_distance itu sebagai trik keamanan (NumPy akan memberikan peringatan jika terjadi pembagian dengan nol saat norm salah satu vektor adalah 0).
* 