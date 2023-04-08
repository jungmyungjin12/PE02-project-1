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


fit2 = np.polyfit(V, abs_I, 1000) # polyfit을 이용하여 V와 절대값 I와의 그래프를 1000차로 근사하여 각 계수들을 fit2에 지정.
fit2_function = np.poly1d(fit2)  # poly1d를 이용하여 그 계수들을 객체로, 연산가능한 다항식 형태로 변경시킴.

SST = sum((abs_I - abs_I.mean())**2)
SSE = sum((fit2_function(V)-abs_I.mean())**2)
R_Squared = SSE/SST

plt.text(-2, 10**-6.7, 'R-squared ={}'.format(R_Squared), fontdict={'size':8})

plt.subplot(1,2,1) # 그래프를 1행 2열 디자인으로 구성 하고 이 plot 을 첫 번째 자리에 올림
plt.plot(V, fit2_function(V), 'r-', label= 'best-fit') # V와 다항식에(V)를 대입해 도출한 값을 x y 좌표로 plot함.
plt.legend(loc = 'upper left', fontsize = 7)

plt.grid(True) # 그리드 추가
#----------------------------------------------------------------------------------------------------------------------

# XML 파일 파싱
tree = elemTree.parse("HY202103_D07_(0,0)_LION1_DCM_LMZC.xml")
root = tree.getroot()

WLL = []      # wavelength 값이 들어갈 WLL 과 transmission 손실 dB값이 들어갈 TRL을 지정.
TRL = []
for wave in root.iter('L'):   # L[nm]의 텍스트 값을 순회하며 웨이브의 텍스트 값들을 WLL에 append
    WL = wave.text
    WLL.append([float(i) for i in WL.split(',')])

for trans in root.iter('IL'):    # IL[dB]의 텍스트 값을 순회하며 웨이브의 텍스트 값들을 TRL에 append
    Tr = trans.text
    TRL.append([float(i) for i in Tr.split(',')])

Bias_list = [i.get('DCBias') for i in root.iter('WavelengthSweep')] # 범례를 붙이기 위해 모든 Wave length 딕셔너리에
for i in range(0,len(WLL)):   # DCBias key 값에 대응되는 요소들을 list 형태로 DC_list에 저장,
  if i == len(WLL) - 1:        # 참조값의 범례 제외해 응용과제 결과물과 일치키기 위하여 if 로 경우를 정해줌
    plt.subplot(1,2,2)            # 1행 2열의 디자인 형태중 2번째 자리에 plot함
    plt.plot(WLL[i], TRL[i])
  else:
    plt.subplot(1,2,2)            # 참조값의 범례가 아닌 경우에는 DCbias[V]로 범례를 표시해야 하기 때문에 라벨링을 함.
    plt.plot(WLL[i], TRL[i],label = '{}V'.format(Bias_list[i]))

plt.legend(loc='best', ncol=3,fontsize = 7)
plt.title('Transmission Spectra - as measured', fontsize = 10) # 그래프 제목 설정
plt.xlabel('Wavelength[nm]', fontsize = 10) # x 축 레이블 설정
plt.ylabel('Measured transmission[dB]', fontsize = 10) # y 축 레이블 설정


plt.show()