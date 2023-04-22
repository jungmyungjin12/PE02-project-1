import xml.etree.ElementTree as elemTree
import matplotlib.pyplot as plt
import numpy as np

def R_square(X,Y,Y_reg): # R square 값을 계산하는 함수
    Y_mean=sum(Y)/Y.size # 전류의 평균값
    SST=sum((Y-Y_mean)**2) # 전체 데이터와 평균값 간 차이 제곱의 합
    SSE=sum((Y_reg-Y_mean)**2) # 추정값과 평균값 간 차이 제곱의 합
    return SSE/SST # R square 값을 반환

def Best_fit_R(X,Y): # 가장 적합한 차수를 결정하는 함수
    Rs = []
    for i in range(1,9):
        coef = np.polyfit(X,Y,i)
        func = np.poly1d(coef)
        fitted_data = func(X)
        Rs.append(R_square(X,Y,fitted_data))
    max_degree = Rs.index(max(Rs))+1
    return max(Rs)

def fit_IV_R(X,Y): # 전류와 전압 값을 이용하여 R square 값을 계산하는 함수
    coef = np.polyfit(X,Y,12)
    func = np.poly1d(coef)
    result = R_square(X, Y, func(X))
    return f'{result:.6f}'