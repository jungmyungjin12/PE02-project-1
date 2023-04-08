import xml.etree.ElementTree as ET  # xml 모듈과 matplotlib 모듈을 import합니다.
import matplotlib.pyplot as plt

root = ET.parse('HY202103_D07_(0,0)_LION1_DCM_LMZC.xml').getroot()  # xml 파일을 파싱합니다.

v = []  # 빈 리스트를 만듭니다.
for waveLengthSweep in root.findall('.//WavelengthSweep'):  # WavelengthSweep 태그를 찾습니다.
    waveValues = []  # 빈 리스트를 만듭니다.
    for child in waveLengthSweep:  # WavelengthSweep의 자식 태그들을 찾습니다.
        waveValues.append(list(map(float, child.text.split(','))))  # 자식 태그의 텍스트를 ,로 split해서 리스트로 변환하고, 모든 요소를 float로 변환합니다.
    waveValues.append(waveLengthSweep.attrib['DCBias'])  # DCBias를 waveValues 리스트의 마지막에 추가합니다.
    v.append(waveValues)  # waveValues 리스트를 v 리스트에 추가합니다.

# Spectrum graph of raw data
plots = []  # 빈 리스트를 만듭니다.
for i in range(len(v) - 1):  # v 리스트의 마지막 요소는 REF로 제외하고 반복합니다.
    line, = plt.plot(v[i][0], v[i][1], label="DCBias=\"" + str(v[i][2]) + "\"")  # plot을 그리고, 레이블을 설정합니다.
    plots.append(line)  # plot을 plots 리스트에 추가합니다.

line, = plt.plot(v[6][0], v[6][1], color='gray', label="REF")  # REF plot을 그립니다.

plt.gca().add_artist(plt.legend(handles=[line], loc='upper right'))  # REF 레이블을 추가합니다.
plt.legend(handles=plots, ncol=2, loc="lower center")  # 나머지 레이블을 추가합니다.
plt.title("Transmission spectra - as measured", fontsize=12)  # 그래프 제목을 설정합니다.
plt.xlabel('Wavelength [nm]', fontsize=12)  # x축 레이블을 설정합니다.
plt.ylabel('Measured transmission [dB]', fontsize=12)  # y축 레이블을 설정합니다.
plt.show()  # 그래프를 출력합니다.