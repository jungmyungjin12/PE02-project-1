import xml.etree.ElementTree as ET
import numpy as np
from numpy import exp
from lmfit import Model

# rsq_ref 계산을 위한 polyfitT 함수 정의
def polyfitT(x, y, degree):
    # x, y 데이터에 대해 degree 차원 다항식으로 fitting 후, fitting 계수를 coeffs에 저장
    coeffs = np.polyfit(x, y, degree)
    # 다항식으로 fitting한 모델 p에 대해 R-squared 값을 계산하여 반환
    # R-squared: fitting한 모델이 데이터를 얼마나 잘 설명하는지에 대한 지표 (1에 가까울수록 설명력이 좋다고 판단됨)
    p = np.poly1d(coeffs)
    yhat = p(x)  # fitting된 모델 p에 x 값을 입력하여 예측한 y 값 (fitting된 curve 상의 값)을 yhat에 저장
    ybar = np.sum(y) / len(y)  # y 값 (실제 데이터)의 평균값을 ybar에 저장

    # R-squared 계산에 필요한 변수 ssreg, sstot 계산
    ssreg = np.sum((yhat - ybar) ** 2)  # or sum([ (yihat - ybar)**2 for yihat in yhat])
    sstot = np.sum((y - ybar) ** 2)     # or sum([ (yi - ybar)**2 for yi in y])
    # R-squared 값을 계산하여 반환
    results = ssreg / sstot
    return results

# 참조 측정값을 이용하여 R-squared 계산하는 Rsq_Ref 함수 정의
def Rsq_Ref(x):
    tree = ET.parse(x)      # xml 파일을 parsing하여 tree 객체 생성

    # xml 파일에서 필요한 정보인 wavelength와 transmission 값을 찾아서 L7, IL7 변수에 저장
    L7 = tree.find(".ElectroOpticalMeasurements/ModulatorSite/Modulator[2]/PortCombo/WavelengthSweep/L")
    IL7 = tree.find(".ElectroOpticalMeasurements/ModulatorSite/Modulator[2]/PortCombo/WavelengthSweep/IL")

    # L7, IL7 변수에 저장된 값들을 ',' 기준으로 split하여 리스트로 변환 후, 각 값을 float 형식으로 변환하여 L7, IL7 변수에 저장
    L7 = L7.text.split(",")
    IL7 = IL7.text.split(",")
    L7 = list(map(float, L7))
    IL7 = list(map(float, IL7))
    # 참조 측정값에 대해 polyfitT 함수를 사용하여 R-squared 값을 계산하여 Rsq_Ref에 저장 후 반환
    Rsq_Ref = polyfitT(L7, IL7, 6)
    return Rsq_Ref
# --------------------------------------------------------------------------------------------------------------------#

# rsq_fit
def Rsq_fit(x):.
    tree = ET.parse(x)          # xml parsing
    # 전류-전압 데이터를 찾습니다.
    b = tree.find(".ElectroOpticalMeasurements/ModulatorSite/Modulator/PortCombo/IVMeasurement/Voltage")
    c = tree.find(".ElectroOpticalMeasurements/ModulatorSite/Modulator/PortCombo/IVMeasurement/Current")
    # 데이터를 리스트로 변환합니다.
    x_2 = b.text.split(",")
    y_2 = c.text.split(",")
    x_list = list(map(float, x_2))
    y_list = list(map(float, y_2))
    # 전류 값의 절대값을 구합니다.
    y_list_1 = []
    for i in range(len(y_list)):
        g = abs(y_list[i])
        y_list_1.append(g)

    # 12차 다항식 적합을 수행합니다.
    polyfiti = np.polyfit(x_list, y_list_1, 12)
    fiti = np.poly1d(polyfiti)
    # 가우시안 함수와 폴리노미얼 함수를 합친 함수를 정의합니다.
    def gaussian(x, q, w, alp):
        return abs(q * (exp(x / w) - 1)) + alp * fiti(x)
    # lmfit 라이브러리를 사용하여 데이터에 적합합니다.
    gmodel = Model(gaussian)
    result = gmodel.fit(y_list_1, x=x_list, q=1, w=1, alp=1)
    # R-제곱 값을 계산합니다.
    yhat = result.best_fit
    ybar = np.sum(y_list_1) / len(y_list_1)
    ssreg = np.sum((yhat - ybar) ** 2)
    sstot = np.sum((y_list_1 - ybar) ** 2)
    results = ssreg / sstot
    return results