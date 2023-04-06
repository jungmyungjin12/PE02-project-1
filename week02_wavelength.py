import xml.etree.ElementTree as ET
import matplotlib.pyplot as plt

# xml 파일을 읽어들임
tree = ET.parse("HY202103_D07_(0,0)_LION1_DCM_LMZC.xml")
root = tree.getroot()

# dcbias 값과 wavelength, IL 값을 저장할 리스트 생성
dcbias_values = []
wavelength_values = []
IL_values = []

# 각 wavelength 블록에서 dcbias, wavelength, IL 값을 추출하여 리스트에 저장
for wavelength in root.iter("WavelengthSweep"):
    dcbias_values.append(float(wavelength.get("DCBias")))
    wavelength_values.append([float(x) for x in wavelength.find('L').text.split(',')])
    IL_values.append([float(x) for x in wavelength.find('IL').text.split(',')])

# 각 dcbias 값별로 그래프를 생성하여 한번에 표시
for i in range(len(dcbias_values)):
    if i == len(dcbias_values) - 1:
        plt.plot(wavelength_values[i], IL_values[i], label='ref_0V')
    else:
        plt.plot(wavelength_values[i], IL_values[i], label=str(dcbias_values[i]))

plt.xlabel('Wavelength [nm]')
plt.ylabel('Measured transmission [dB]')
plt.legend(loc='lower center', ncol = 4)
plt.show()