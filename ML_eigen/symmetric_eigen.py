
import numpy as np

# n = 5 
n = int(input("n = "))
# According to https://en.wikipedia.org/wiki/Positive-definite_matrix, for any square matrix A,
# A' * A is positive semi-definite, and rank(A' * A) is equal to rank(A) . Matrices are invertible
# if they have full rank. So all we have to do is generate an initial random matrix with full rank and
# we can then easily find a positive semi-definite matrix derived from it.
while True:
    m = np.random.randint(25, size=(n, n))
    if np.linalg.matrix_rank(m) == n: 
        break
# print(m)
m_symm = (m + m.T)
# print(m_symm)
# print(np.linalg.det(m_symm))

eigen_values, eigen_vectors = np.linalg.eig(m_symm)

d = np.diag(eigen_values)
# print(d)

inv = np.linalg.inv(eigen_vectors)
vec = np.dot(eigen_vectors, np.dot(d, inv))
# print("final:\n", vec)

print(np.allclose(m_symm, vec))
