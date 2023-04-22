import xml.etree.ElementTree as ET
import numpy as np

# x, y 데이터와 차수(degree)를 입력 받아 다항식 회귀 분석을 수행하는 polyfitT() 함수를 정의
def polyfitT(x, y, degree):
    coeffs = np.polyfit(x, y, degree)   # 다항식 최소 자승법을 이용하여 회귀분석 수행
    # r-squared
    p = np.poly1d(coeffs)   # 회귀식 계수를 이용하여 다항식 객체 생성
    # fit values, and mean
    yhat = p(x)                         # or [p(z) for z in x]  # 회귀식을 이용하여 예측값 계산
    ybar = np.sum(y) / len(y)           # or sum(y)/len(y)  # 실제값의 평균 계산
    ssreg = np.sum((yhat - ybar) ** 2)  # or sum([ (yihat - ybar)**2 for yihat in yhat])  # 회귀선과 실제값의 차이 제곱합 계산
    sstot = np.sum((y - ybar) ** 2)     # or sum([ (yi - ybar)**2 for yi in y])  # 실제값과 평균값의 차이 제곱합 계산
    results = ssreg / sstot             # 결정 계수 계산
    return results

# x라는 xml 파일 경로를 입력받아 해당 xml 파일을 분석하는 Errorcheck() 함수를 정의,Rsq_Ref 값과 x값(해당 xml 파일의 특정 데이터)을 사용하여 데이터 품질을 판단
# 입력한 XML 파일에서 데이터를 추출하여 오차를 검사하는 함수
def Errorcheck(x):
    tree = ET.parse(x)   # XML 파일을 파싱하여 ElementTree 객체 생성
    L7 = tree.find(".ElectroOpticalMeasurements/ModulatorSite/Modulator[2]/PortCombo/WavelengthSweep/L")
    # XML 내의 특정 경로를 찾아 해당 Element 객체 반환
    IL7 = tree.find(".ElectroOpticalMeasurements/ModulatorSite/Modulator[2]/PortCombo/WavelengthSweep/IL")
    L7 = L7.text.split(",")   # 텍스트 데이터를 쉼표로 구분하여 리스트로 변환
    IL7 = IL7.text.split(",")
    L7 = list(map(float, L7))   # 문자열을 실수형으로 변환
    IL7 = list(map(float, IL7))
    Rsq_Ref = polyfitT(L7, IL7, 6)   # 다항식 최소 자승법을 이용하여 회귀분석 수행

    c = tree.find(".ElectroOpticalMeasurements/ModulatorSite/Modulator/PortCombo/IVMeasurement/Current")
    y_2 = c.text.split(",")
    y_list = list(map(float, y_2))
    y_list_1 = []
    for i in range(len(y_list)):
        g = abs(y_list[i])
        y_list_1.append(g)
    x = y_list_1[12]
    x = float(x)
    if Rsq_Ref >= 0.996 and x >= (10**-7):
        return "No Error"
    elif Rsq_Ref <= 0.996 and x >= (10**-7):
        return "Rsq_Ref Error"
    elif Rsq_Ref >= 0.996 and x <= (10**-7):
        return "IV-fitting"
    else:
        return "Rsq_Ref Error and IV-fitting"

# 데이터 추출 후, 다항식 최소 자승법을 이용하여 회구분석을 진행하고 오류 유형을 결정하는 함수
def ErrorFlag(x):
    tree = ET.parse(x)          # XML 파일 파싱하여 ElementTree 객체 생성
    # XML 내의 특정 경로를 찾아 해당 Element 객체 반환
    L7 = tree.find(".ElectroOpticalMeasurements/ModulatorSite/Modulator[2]/PortCombo/WavelengthSweep/L")
    IL7 = tree.find(".ElectroOpticalMeasurements/ModulatorSite/Modulator[2]/PortCombo/WavelengthSweep/IL")

    # 쉼표로 구분된 텍스트 데이터를 리스트로 변환하고 문자열을 실수형으로 변환
    L7 = list(map(float, L7.text.split(",")))
    IL7 = list(map(float, IL7.text.split(",")))

    Rsq_Ref = polyfitT(L7, IL7, 6)                  # 다항식 최소 자승법을 이용하여 회귀분석 수행

    # XML 내의 특정 경로를 찾아 해당 Element 객체 반환
    c = tree.find(".ElectroOpticalMeasurements/ModulatorSite/Modulator/PortCombo/IVMeasurement/Current")

    y_list = list(map(float, c.text.split(",")))    # 쉼표로 구분된 텍스트 데이터를 리스트로 변환하고 문자열을 실수형으로 변환
    y_list_1 = [abs(y) for y in y_list]             # 리스트 y_list의 절대값을 구하여 y_list_1 리스트에 저장
    x = float(y_list_1[12])                         # y_list_1 리스트의 13번째 원소를 가져와서 실수형으로 변환

    # 조건문을 이용하여 오류 유형 결정
    if Rsq_Ref >= 0.996 and x >= (10 ** -7):
        return 0
    elif Rsq_Ref <= 0.996 and x >= (10 ** -7):
        return 1
    elif Rsq_Ref >= 0.996 and x <= (10 ** -7):
        return 2
    else:
        return 3