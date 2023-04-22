import numpy as np

# 다항식 회귀 분석을 수행하는 함수, 입력 data: x, 출력 data: y, degree: 함수의 차수
def polyfitT(x, y, degree):
    # 다항식 회귀 분석 수행
    coeffs = np.polyfit(x, y, degree)
    p = np.poly1d(coeffs)
    yhat = p(x)
    # 결정 계수 계산
    ybar = np.sum(y) / len(y)
    ssreg = np.sum((yhat - ybar) ** 2)
    sstot = np.sum((y - ybar) ** 2)
    results = ssreg / sstot
    return results

# IV fitting 함수 x: 입력 데이터 x, q: Iph 값, w: Rs 값
# alp: A 값, xi: 보간에 사용될 입력 데이터 x ,yi: 보간에 사용될 출력 데이터 y
def IVfittting(x, q, w, alp, xi = [], yi = []):
    polyfiti = np.polyfit(xi, yi, 12)       # 12차 다항식 보간 수행
    fiti = np.poly1d(polyfiti)
    return abs(q * (np.exp(x / w) - 1)) + alp * fiti(x)     # IV curve fitting 계산