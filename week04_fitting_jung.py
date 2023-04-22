import xml.etree.ElementTree as elemTree
import matplotlib.pyplot as plt
import numpy as np

def R_square(X,Y,Y_reg): # obtain R square
    Y_mean=sum(Y)/Y.size # mean of Current
    SST=sum((Y-Y_mean)**2) # total sum of square ( sum of square of difference between data and mean )
    SSE=sum((Y_reg-Y_mean)**2) # Residual sum of square ( sum of square of difference between )
    return SSE/SST # return R sqaure
def Best_fit_R(X,Y):
    Rs = []
    for i in range(1,9):
        coef = np.polyfit(X,Y,i)
        func = np.poly1d(coef)
        fitted_data = func(X)
        Rs.append(R_square(X,Y,fitted_data))
    max_degree = Rs.index(max(Rs))+1
    return max(Rs)
def fit_IV_R(X,Y):
    coef = np.polyfit(X,Y,12)
    func = np.poly1d(coef)
    result = R_square(X, Y, func(X))
    return f'{result:.6f}'