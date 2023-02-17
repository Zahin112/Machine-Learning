
import numpy as np
from numpy.linalg import svd


# n = 3 
# m = 2
n = int(input("n = "))
m = int(input("m = "))
# while True:
#     m = np.random.randint(25,size = (n, n))
#     if np.linalg.det(m) != 0:
#         break
    # if np.linalg.matrix_rank(m) == n: 
    #     break

mat = np.random.randint(25, size = (n, m))
# print(mat)
# print(np.linalg.det(m))

builtin = np.linalg.pinv(mat)
# print(builtin)

u, s, v = svd(mat)

D = 1.0/s
d = np.zeros((n, m))
d[:m, :m] = np.diag(D)
# print(d)
vec = v.T.dot(d.T).dot(u.T)
# print(vec)

print(np.allclose(builtin, vec))
