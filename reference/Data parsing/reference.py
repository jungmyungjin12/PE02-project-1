import xml.etree.ElementTree as ET   # xml 데이터를 파싱하기 위한 모듈
import matplotlib.pyplot as plt      # 데이터 시각화를 위한 모듈
from fitting import *                # fitting 모듈에서 사용되는 함수들을 import


def reference(x):
    tree = ET.parse(x)      # xml 파일을 파싱하여 tree 객체로 변환
    # tree 객체에서 L과 IL element를 찾음
    L = tree.find(".ElectroOpticalMeasurements/ModulatorSite/Modulator[2]/PortCombo/WavelengthSweep/L")
    IL = tree.find(".ElectroOpticalMeasurements/ModulatorSite/Modulator[2]/PortCombo/WavelengthSweep/IL")
    # L, IL 값을 리스트로 변환
    L_7 = L.text.split(",")
    IL_7 = IL.text.split(",")
    # 리스트를 float 형태로 변환
    L_list_7 = list(map(float, L_7))
    IL_list_7 = list(map(float, IL_7))
    # 산점도를 그리고 투명도(alpha), 마커 크기(s) 등의 속성을 설정
    plt.scatter(L_list_7, IL_list_7, s=15, label="reference", alpha=0.01, facecolor='none', edgecolor='r')
    # 6차 다항식으로 fitting하고 fitting 결과를 plot
    polyfiti = np.polyfit(L_list_7, IL_list_7, 6)
    fiti = np.poly1d(polyfiti)
    x = polyfitT(L_list_7, IL_list_7, 6)  # R^2 값 계산
    plt.plot(L_list_7, fiti(L_list_7), label="{}th R^2 = {}".format(6, '%0.5f' % x))  # R^2 값을 label로 포함하여 plot
    # 범례, 타이틀, x축 라벨, y축 라벨 등 시각화 요소를 설정
    plt.legend(loc="best")
    plt.title("Reference fitting")
    plt.xlabel('Wavelength [nm]')
    plt.ylabel('Measured transmission [dB]')

