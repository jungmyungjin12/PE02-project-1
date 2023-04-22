from lmfit import Model
import xml.etree.ElementTree as ET
import matplotlib.pyplot as plt
from fitting import *
from filter import *


def iv(x):
    tree = ET.parse(x)              # XML 파일 파싱
    # XML 파일에서 Voltage, Current 데이터 가져오기
    b = tree.find(".ElectroOpticalMeasurements/ModulatorSite/Modulator/PortCombo/IVMeasurement/Voltage")
    c = tree.find(".ElectroOpticalMeasurements/ModulatorSite/Modulator/PortCombo/IVMeasurement/Current")
    # Voltage, Current 데이터 전처리
    x_2 = b.text.split(",")
    y_2 = c.text.split(",")
    x_list = list(map(float, x_2))
    y_list = list(map(float, y_2))
    # Current 데이터 절댓값으로 변환
    y_list_1 = []
    for i in range(len(y_list)):
        g = abs(y_list[i])
        y_list_1.append(g)

    plt.plot(x_list, y_list_1, "ro", label='initial fit')       # Initial plot 그리기
    # Y 스케일을 로그 스케일로 변환
    plt.yscale("log")
    plt.title("IV-analysis")
    plt.xlabel('Voltage [V]')
    plt.ylabel('Current [A]')

    gmodel = Model(IVfittting)      # IV fitting 모델 생성

    result = gmodel.fit(y_list_1, x=x_list, q=1, w=1, alp=1, xi=x_list, yi=y_list_1)        # IV fitting 수행
    # R-squared 값 계산
    yhat = result.best_fit
    ybar = np.sum(y_list_1) / len(y_list_1)
    ssreg = np.sum((yhat - ybar) ** 2)
    sstot = np.sum((y_list_1 - ybar) ** 2)
    results = ssreg / sstot
    # Best fit plot 그리기
    plt.plot(x_list, result.best_fit, 'b-', label='best fit R^2={}'.format(results))
    # Maximum Current와 Minimum Current에 해당하는 지점에 녹색 숫자로 라벨링
    plt.text(-1, result.best_fit[4], str(result.best_fit[4]), color='g', horizontalalignment='center',
             verticalalignment='bottom')
    plt.text(1, result.best_fit[12], str(result.best_fit[12]), color='g', horizontalalignment='center',
             verticalalignment='bottom')
    plt.title('IV-fitting')     # 그래프 제목 설정
    plt.legend()                # 범례 표시