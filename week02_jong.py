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
I = np.array(list(map(float,a.split(','))))

# 'Voltage' 태그에서 전압 값을 추출하여 리스트로 변환
for i in root.iter('Voltage'):
    b = i.text
V = np.array(list(map(float,b.split(','))))


# 전류 값의 절대값을 계산
abs_I = np.abs(I)

# 그래프 생성
plt.subplot(1,2,1)
plt.plot(V, abs_I, 'ko',label= 'data') # 전압 값과 전류 값의 절대값을 포함하는 그래프 생성
plt.title('IV analysis', fontdict = {'weight': 'bold', 'size':15}) # 그래프 제목 설정
plt.xlabel('Voltage[V]', labelpad=8 , fontdict={'weight': 'bold', 'size':12}) # x 축 레이블 설정
plt.ylabel('Current[A]', labelpad=8 , fontdict={'weight': 'bold', 'size':12}) # y 축 레이블 설정
plt.yscale('logit') # y 축 로그 스케일로 설정
plt.legend()

# 해당위치의 전류 값 찾아내서 응용과제의 화면처럼 text 로 보여줌
plt.text(-2.1, 10**-7, '-1V ={}'.format(abs_I[np.where(V == -1)[0][0]]), fontdict={'size':8})
plt.text(-2.1, 10**-7.3, '1V ={}'.format(abs_I[np.where(V == 1)[0][0]]), fontdict={'size':8})


fit2 = np.polyfit(V, abs_I, 1000)
fit2_function = np.poly1d(fit2)

SST = sum((abs_I - abs_I.mean())**2)
SSE = sum((fit2_function(V)-abs_I.mean())**2)
R_Squared = SSE/SST
plt.text(-2, 10**-6.7, 'R-squared ={}'.format(R_Squared), fontdict={'size':8})
plt.subplot(1,2,1)


plt.plot(V, fit2_function(V), 'r-', label= 'best-fit')
plt.legend(loc = 'upper left', fontsize = 7)

plt.grid(True) # 그리드 추가
#----------------------------------------------------------------------------------------------------------------------

# XML 파일 파싱
tree = elemTree.parse("HY202103_D07_(0,0)_LION1_DCM_LMZC.xml")
root = tree.getroot()
WLL = []
TRL = []

for wave in root.iter('L'):
    WL = wave.text
    WLL.append([float(i) for i in WL.split(',')])

for trans in root.iter('IL'):
    Tr = trans.text
    TRL.append([float(i) for i in Tr.split(',')])

Bias_list = [i.get('DCBias') for i in root.iter('WavelengthSweep')]

for i in range(0,len(WLL)):
  if i == len(WLL) - 1:
    plt.subplot(1,2,2)
    plt.plot(WLL[i], TRL[i])
  else:
    plt.subplot(1,2,2)
    plt.plot(WLL[i], TRL[i],label = '{}V'.format(Bias_list[i]))

plt.legend(loc='best', ncol=3,fontsize = 7)
plt.title('Transmission Spectra - as measured', fontsize = 10) # 그래프 제목 설정
plt.xlabel('Wavelength[nm]', fontsize = 10) # x 축 레이블 설정
plt.ylabel('Measured transmission[dB]', fontsize = 10) # y 축 레이블 설정


plt.show()

