# 필요한 라이브러리 가져오기
import xml.etree.ElementTree as elemTree
import matplotlib.pyplot as plt
import numpy as np

# XML 파일 파싱
tree = elemTree.parse("HY202103_D07_(0,0)_LION1_DCM_LMZC.xml")
root = tree.getroot()

# 'Current' 태그에서 전류 값을 추출하여 리스트로 변환
for i in root.iter('Current'):
    a = i.text
I = np.array(list(map(float, a.split(','))))

# 'Voltage' 태그에서 전압 값을 추출하여 리스트로 변환
for i in root.iter('Voltage'):
    b = i.text
V = np.array(list(map(float,b.split(','))))

# 전류 값의 절대값을 계산
abs_I = np.abs(I)

# 다항식으로 근사
z = np.polyfit(V, abs_I, 15)
p = np.poly1d(z)

# 다항 회귀 모델의 적합도를 나타내는 지표인 R-squared 값을 계산
# 모델의 예측 값과 실제 값의 차이를 나타내는 오차 제곱합(Residual Sum of Squares)을 계산
SSE = np.sum((p(V) - np.mean(abs_I)) ** 2)
# 종속 변수의 전체 변동을 나타내는 총 오차 제곱합(Total Sum of Squares)을 계산
SST = np.sum((abs_I - np.mean(abs_I)) ** 2)
# R-squared 값을 계산
r_squared = (SSE / SST)

# 그래프 생성
plt.subplot(1,2,1)  # 1행 2열중 첫번째에 그래프 생성
plt.plot(V, abs_I, 'ro', V, abs(p(V)), '-') # 전압 값과 전류 값의 절대값을 포함하는 그래프 생성
plt.title('IV - analysis', fontdict={'weight': 'bold', 'size':10}) # 그래프 제목 설정
plt.xlabel('Voltage[V]', labelpad=8 , fontdict={'weight': 'bold', 'size':8}) # x 축 레이블 설정
plt.ylabel('Current[A]', labelpad=8 , fontdict={'weight': 'bold', 'size':8}) # y 축 레이블 설정
plt.yscale('logit') # y 축 로그 스케일로 설정
plt.grid(True) # 그리드 추가
plt.text(0.02, 0.8, f"R_squared = {r_squared:.20f}", fontsize=10, transform=plt.gca().transAxes) # R_squared 값을 그래프에 표시
plt.text(-2,p(-2), p(-2), fontsize = 8) # 그래프에 (-2, p(-2)) 위치에 p(-2) 값을 텍스트로 표시
plt.text(-1,p(-1), p(-1), fontsize = 8) # 그래프에 (-1, p(-1)) 위치에 p(-1) 값을 텍스트로 표시
plt.text(0.5,p(1), p(1), fontsize = 8) # 그래프에 (0.5, p(1)) 위치에 p(1) 값을 텍스트로 표시

# dcbias 값과 wavelength, IL 값을 저장할 리스트 생성
dcbias_values = []
wavelength_values = []
IL_values = []

# 각 wavelength 블록에서 dcbias, wavelength, IL 값을 추출하여 리스트에 저장
for wavelength in root.iter("WavelengthSweep"):
    dcbias_values.append(float(wavelength.get("DCBias")))
    wavelength_values.append([float(x) for x in wavelength.find('L').text.split(',')])
    IL_values.append([float(x) for x in wavelength.find('IL').text.split(',')])

plt.subplot(1,2,2)  # 1행 2열중 2번째에 그래프 생성
# 각 dcbias 값별로 그래프를 생성하여 한번에 표시
for i in range(len(dcbias_values)):
    if i == len(dcbias_values) - 1:
        plt.plot(wavelength_values[i], IL_values[i], label='ref_0V')
    else:
        plt.plot(wavelength_values[i], IL_values[i], label=str(dcbias_values[i])+'V')
plt.title('Transmission spectrum - as measured', fontdict={'weight': 'bold', 'size':10})
plt.xlabel('Wavelength [nm]', labelpad=8 , fontdict={'weight': 'bold', 'size':8})   # x 축 레이블 설정
plt.ylabel('Measured transmission [dB]', labelpad=8 , fontdict={'weight': 'bold', 'size':8})    # y 축 레이블 설정
plt.legend(loc='lower center', ncol=4)    # 범례 생성 및 위치 지정
plt.show()