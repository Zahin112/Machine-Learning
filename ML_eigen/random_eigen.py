
import numpy as np

#n = 3 
n = int(input("n = "))
# m = np.random.randint(25,size = (n, n))
# print(m)
# mx = np.sum(np.abs(m), axis=1)
# np.fill_diagonal(m, mx)
# print(m)
# print(np.linalg.det(m))
while True:
    m = np.random.randint(25, size = (n, n))
    if np.linalg.det(m) != 0:
        break
    # if np.linalg.matrix_rank(m) == n: 
    #     break
# print(np.linalg.det(m))

eigen_values, eigen_vectors = np.linalg.eig(m)

d = np.diag(eigen_values)
# print(d)

inv = np.linalg.inv(eigen_vectors)
vec = np.dot(eigen_vectors, np.dot(d, inv))

# print(m)
# print("final:\n", vec)

print(np.allclose(m, vec))
