import numpy as np


def normalize(v):
    return v / np.sqrt(v.dot(v))


def gs(u):
    v = [normalize(u[0])]
    for i in range(1, len(u)):
        s = u[i]
        for j in range(i):
            s = s - np.dot(u[i], v[j]) * v[j]
        v.append(normalize(s))
    return v


coeffs_1 = np.array([1, 1, 0, 0])
coeffs_2 = np.array([0, 1, 1, 0])
coeffs_3 = np.array([0, 0, 1, 1])
#coeffs_4=np.array([0,0,0,1])

F = gs([coeffs_1, coeffs_2, coeffs_3])
for v in F:
    print(v)
