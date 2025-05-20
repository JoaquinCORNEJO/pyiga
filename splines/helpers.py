import numpy as np


def find_span_linear(degree, knotvector, nbctrlpts, knot):
    span = degree + 1
    while (span < nbctrlpts) and (knotvector[span] <= knot):
        span += 1
    return span - 1


def basis_function_ders(degree, knotvector, span, knot, order):
    left = np.ones(degree + 1)
    right = np.ones_like(left)
    ndu = np.ones((degree + 1, degree + 1))
    for j in range(1, degree + 1):
        left[j] = knot - knotvector[span + 1 - j]
        right[j] = knotvector[span + j] - knot
        saved = 0.0
        for r in range(0, j):
            ndu[j][r] = right[r + 1] + left[j - r]
            temp = ndu[r][j - 1] / ndu[j][r]
            ndu[r][j] = saved + (right[r + 1] * temp)
            saved = left[j - r] * temp
        ndu[j][j] = saved

    nders = min(order, degree)
    ders = np.zeros((nders + 1, degree + 1))
    for j in range(0, degree + 1):
        ders[0][j] = ndu[j][degree]

    aa = np.ones((2, degree + 1))
    for r in range(0, degree + 1):
        s1, s2 = 0, 1
        aa[0][0] = 1.0
        for k in range(1, nders + 1):
            d = 0.0
            rk = r - k
            pk = degree - k
            if r >= k:
                aa[s2][0] = aa[s1][0] / ndu[pk + 1][rk]
                d = aa[s2][0] * ndu[rk, pk]
            j1 = 1 if rk >= -1 else -rk
            j2 = k - 1 if (r - 1) <= pk else degree - r
            for j in range(j1, j2 + 1):
                aa[s2][j] = (aa[s1][j] - aa[s1][j - 1]) / ndu[pk + 1][rk + j]
                d += aa[s2][j] * ndu[rk + j][pk]
            if r <= pk:
                aa[s2][k] = -aa[s1][k - 1] / ndu[pk + 1][r]
                d += aa[s2][k] * ndu[r][pk]
            ders[k][r] = d
            j = s1
            s1 = s2
            s2 = j

    r = float(degree)
    for k in range(1, nders + 1):
        for j in range(0, degree + 1):
            ders[k][j] *= r
        r *= degree - k

    return [ders[ii, :] for ii in range(nders + 1)]
