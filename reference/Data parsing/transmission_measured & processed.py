import xml.etree.ElementTree as ET
import matplotlib.pyplot as plt
from filter import *

# transmission measured 함수 정의
def measured(x):
    tree = ET.parse(x)                  # XML 파일을 parsing 합니다
    for i in range(1, 7):               # DBias에 대한 loop
        # L과 IL 값을 parsing 합니다.
        L = tree.find(".ElectroOpticalMeasurements/ModulatorSite/Modulator/PortCombo/WavelengthSweep[{}]/L".format(i))
        IL = tree.find(".ElectroOpticalMeasurements/ModulatorSite/Modulator/PortCombo/WavelengthSweep[{}]/IL".format(i))
        L_i = L.text.split(",")
        IL_i = IL.text.split(",")
        L_list_i = list(map(float, L_i))
        IL_list_i = list(map(float, IL_i))
        # DCBias 값을 가져옵니다.
        DBias = tree.find(".ElectroOpticalMeasurements/ModulatorSite/Modulator/PortCombo/WavelengthSweep[{}]".format(i))
        # 그래프를 그립니다.
        plt.plot(L_list_i, IL_list_i, ".", label=DBias.get("DCBias"))

    # reference 값을 가져와서 그래프를 그립니다.
    L = tree.find(".ElectroOpticalMeasurements/ModulatorSite/Modulator[2]/PortCombo/WavelengthSweep/L")
    IL = tree.find(".ElectroOpticalMeasurements/ModulatorSite/Modulator[2]/PortCombo/WavelengthSweep/IL")
    L_7 = L.text.split(",")
    IL_7 = IL.text.split(",")
    L_list_7 = list(map(float, L_7))
    IL_list_7 = list(map(float, IL_7))
    DBias = tree.find(".ElectroOpticalMeasurements/ModulatorSite/Modulator[2]/PortCombo/WavelengthSweep")
    plt.plot(L_list_7, IL_list_7, ".", label="reference")

    # 그래프에 대한 label, title, xlabel, ylabel 설정
    plt.legend(loc=(0, 0))
    plt.title("Transmission spectra - as measured")
    plt.xlabel('Wavelength [nm]')
    plt.ylabel('Measured transmission [dB]')

# transmission processed

def processed(x):
    tree = ET.parse(x)      # xml parsing
    # 레퍼런스(2번 모듈레이터)에 대한 데이터를 불러옵니다.
    L = tree.find(".ElectroOpticalMeasurements/ModulatorSite/Modulator[2]/PortCombo/WavelengthSweep/L")
    IL = tree.find(".ElectroOpticalMeasurements/ModulatorSite/Modulator[2]/PortCombo/WavelengthSweep/IL")
    L_7 = L.text.split(",")
    IL_7 = IL.text.split(",")
    L_list_7 = list(map(float, L_7))
    IL_list_7 = list(map(float, IL_7))
    # 6차 다항식으로 피팅합니다.
    polyfit6 = np.polyfit(L_list_7, IL_list_7, 6)
    fit6 = np.poly1d(polyfit6)

    # 나머지 모듈레이터에 대한 데이터를 불러옵니다.
    for i in range(1, 7):
        L = tree.find(".ElectroOpticalMeasurements/ModulatorSite/Modulator/PortCombo/WavelengthSweep[{}]/L".format(i))
        IL = tree.find(".ElectroOpticalMeasurements/ModulatorSite/Modulator/PortCombo/WavelengthSweep[{}]/IL".format(i))
        L_i = L.text.split(",")
        IL_i = IL.text.split(",")
        L_list_i = list(map(float, L_i))
        IL_list_i = list(map(float, IL_i))
        DBias = tree.find(".ElectroOpticalMeasurements/ModulatorSite/Modulator/PortCombo/WavelengthSweep[{}]".format(i))
        # 6차 다항식으로 피팅된 값을 빼고 그래프를 그립니다.
        plt.plot(L_list_i, IL_list_i - fit6(L_list_i), ".", label=DBias.get("DCBias"))

    # 레퍼런스 데이터도 같은 방식으로 처리합니다.
    plt.plot(L_list_7, IL_list_7 - fit6(L_list_7), ".", label=DBias.get("DCBias"))
    # 그래프에 라벨을 추가하고 그래프를 그립니다.
    plt.legend(loc=(0, 0))
    plt.title("Transmission spectra - as processed")
    plt.xlabel('Wavelength [nm]')
    plt.ylabel('flat Measured transmission [dB]')
