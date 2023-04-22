import os
import xml.etree.ElementTree as elemTree
import matplotlib.pyplot as plt
import numpy as np

def R_square(X,Y,Y_reg): # R_square 값을 반환하는 함수 정의
    Y_mean=sum(Y)/Y.size # 측정 데이터 Y에 대한 평균값을 가지는 변수
    SST=sum((Y-Y_mean)**2) # 측정 데이터와 평균값 차 제곱의 합
    SSE=sum((Y_reg-Y_mean)**2) # 근사 데이터와 측정 데이터 평균값 차 제곱의 합
    SSR=sum((Y-Y_reg)**2)
    return 1-SSR/SST # R_square 값 반환
def Best_fit_R(X,Y):
    Rs = []
    for i in range(1,9)
        coef = np.polyfit(X,Y)
        func = np.poly1d(coef)
        fitted_data = func(X)
        Rs.append(X,Y,fitted_data)
    max_degree = Rs.index(max(Rs))