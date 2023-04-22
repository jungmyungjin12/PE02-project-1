# 필요한 라이브러리 가져오기
import xml.etree.ElementTree as elemTree
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# XML 파일 파싱
tree = elemTree.parse("HY202103_D07_(0,0)_LION1_DCM_LMZC.xml")
root = tree.getroot()
plt.subplots(constrained_layout=True)

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

plt.subplot(2,3,4)


plt.plot(V, abs_I, 'ko',label= 'data') # 전압 값과 전류 값의 절대값을 포함하는 그래프 생성
plt.title('IV analysis', fontdict = {'weight': 'bold', 'size':10}) # 그래프 제목 설정
plt.xlabel('Voltage[V]', labelpad=8 , fontdict={'weight': 'bold', 'size':7}) # x 축 레이블 설정
plt.ylabel('Current[A]', labelpad=8 , fontdict={'weight': 'bold', 'size':7}) # y 축 레이블 설정
plt.yscale('logit') # y 축 로그 스케일로 설정
plt.legend()

# 해당위치의 전류 값 찾아내서 응용과제의 화면처럼 text 로 보여줌
plt.text(-1.95, 10**-7, '-1V ={}'.format(abs_I[np.where(V == -1)[0][0]]), fontdict={'size':7})
plt.text(-1.95, 10**-7.5, '1V ={}'.format(abs_I[np.where(V == 1)[0][0]]), fontdict={'size':7})


fit2 = np.polyfit(V, abs_I, 1000) # polyfit을 이용하여 V와 절대값 I와의 그래프를 1000차로 근사하여 각 계수들을 fit2에 지정.
fit2_function = np.poly1d(fit2)  # poly1d를 이용하여 그 계수들을 객체로, 연산가능한 다항식 형태로 변경시킴.

SST = sum((abs_I - abs_I.mean())**2)
SSE = sum((fit2_function(V)-abs_I.mean())**2)
R_Squared = SSE/SST

plt.text(-2, 10**-6.5, 'R-squared ={}'.format(R_Squared), fontdict={'size':7})

plt.subplot(2,3,4) # 그래프를 1행 2열 디자인으로 구성 하고 이 plot 을 첫 번째 자리에 올림
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
    plt.subplot(2,3,1)            # 1행 2열의 디자인 형태중 2번째 자리에 plot함
    plt.plot(WLL[i], TRL[i])
  else:
    plt.subplot(2,3,1)            # 참조값의 범례가 아닌 경우에는 DCbias[V]로 범례를 표시해야 하기 때문에 라벨링을 함.
    plt.plot(WLL[i], TRL[i],label = '{}V'.format(Bias_list[i]))

plt.legend(loc='best', ncol=3,fontsize = 7)
plt.title('Transmission Spectra - as measured', fontsize = 10) # 그래프 제목 설정
plt.xlabel('Wavelength[nm]', fontsize = 10) # x 축 레이블 설정
plt.ylabel('Measured transmission[dB]', fontsize = 10) # y 축 레이블 설정

plt.subplot(2,3,2)
plt.plot(WLL[-1],TRL[-1])

square_list=[]
for i in range(1,11):
  fit3 = np.polyfit(WLL[-1], TRL[-1], i)  # polyfit을 이용하여 V와 절대값 I와의 그래프를 i차로 근사하여 각 계수들을 fit2에 지정.
  fit3_function = np.poly1d(fit3)
  plt.plot(WLL[-1], fit3_function(WLL[-1]), label = '{}th'.format(i))
  SST = sum((np.array(TRL[-1]) - np.array(TRL[-1]).mean()) ** 2)
  SSE = sum((fit3_function(np.array(WLL[-1])) - np.array(TRL[-1]).mean()) ** 2)
  square_list.append(SSE / SST),

plt.text(1550, -12, 'R-squared ={}'.format(square_list[square_list.index(max(square_list))-1]), fontdict={'size':6})
plt.text(1550, -13, 'R-squared ={}'.format(max(square_list)), fontdict={'color':  'red','size':6})
plt.title('Transmission Spectra - as measured', fontsize = 10) # 그래프 제목 설정
plt.xlabel('Wavelength[nm]', fontsize = 10) # x 축 레이블 설정
plt.ylabel('Fitting reference data [dB]', fontsize = 10) # y 축 레이블 설정
plt.legend(loc='best', ncol=3,fontsize = 6)
#============================================================================================================

fit3 = np.polyfit(WLL[-1], TRL[-1], square_list.index(max(square_list))+ 1)  # polyfit을 이용하여 V와 절대값 I와의 그래프
fit3_function = np.poly1d(fit3)
Bias_list = [i.get('DCBias') for i in root.iter('WavelengthSweep')] # 범례를 붙이기 위해 모든 Wave length 딕셔너리에

for i in range(0,len(WLL)):   # DCBias key 값에 대응되는 요소들을 list 형태로 DC_list에 저장,
  if i == len(WLL) - 1:        # 참조값의 범례 제외해 응용과제 결과물과 일치키기 위하여 if 로 경우를 정해줌
    plt.subplot(2,3,3)            # 1행 2열의 디자인 형태중 2번째 자리에 plot함
    plt.plot(WLL[i], np.array(TRL[i])-fit3_function(WLL[-1]))
  else:
    plt.subplot(2,3,3)            # 참조값의 범례가 아닌 경우에는 DCbias[V]로 범례를 표시해야 하기 때문에 라벨링을 함.
    plt.plot(WLL[i], np.array(TRL[i])-fit3_function(WLL[-1]),label = '{}V'.format(Bias_list[i]))

plt.legend(loc='best', ncol=3,fontsize = 7)
plt.title('Flat Transmission Spectra - as measured', fontsize = 10) # 그래프 제목 설정
plt.xlabel('Wavelength[nm]', fontsize = 10) # x 축 레이블 설정
plt.ylabel('Measured transmission[dB]', fontsize = 10) # y 축 레이블 설정
#plt.show()

Wafer = root.find('TestSiteInfo').get('Wafer')
Lot = root.find('TestSiteInfo').get('Batch')
Mask = root.find('TestSiteInfo').get('Maskset')
TestSite = root.find('TestSiteInfo').get('TestSite')
Date = root.get('CreationDate')
from datetime import datetime

date_string = Date
dt = datetime.strptime(date_string, '%a %b %d %H:%M:%S %Y')
Date = dt.strftime('%Y%m%d_%H%M%S')

Operator = root.get('Operator')
Name=''

for i in root.iter('Modulator'):
  Name = i.get('Name')
if 'LMZ'in Name:
  Script_ID = 'Process LMZ'
else:
  Script_ID = 'Process ???'

Row = root.find('TestSiteInfo').get('DieRow')
Column = root.find('TestSiteInfo').get('DieColumn')


Analysis_Wavelength = ''
for i in root.iter("DesignParameter"):
  if i.get('Name') == 'Design wavelength':
    Analysis_Wavelength = i.text
Script_Version = 0.1
Script_Owner = 'B2'
if max(square_list) <= 0.6:
  Error_description = 'Ref. spec. Error'
  ErrorFlag = 1
else:
  Error_description = 'No error'
  ErrorFlag = 0
Rsq_of_Ref =max(square_list)
Max_transmission_of_Ref = max(fit3_function(WLL[-1]))
Rsq_of_IV=R_Squared
I_at_plusone=abs_I[np.where(V == -1)[0][0]]
I_at_minusone=abs_I[np.where(V == -1)[0][0]]


values=[]
values.append([Lot,Wafer,Mask,TestSite,Name,Date,Script_ID,Script_Version,Script_Owner,Operator,Row,Column,ErrorFlag,Error_description,
               Analysis_Wavelength,Rsq_of_Ref,Max_transmission_of_Ref,Rsq_of_IV,I_at_minusone,I_at_plusone])

    # 데이터 담을 변수 선언

df = pd.DataFrame(values, index=None, columns= ['Lot',	'Wafer',	'Mask',	'TestSite',	'Name',	'Date',	'Script ID',	'Script Version',	'Script Owner',	'Operator',	'Row',	'Column',	'ErrorFlag',	'Error description',	'Analysis Wavelength [nm]',	'Rsq of Ref. spectrum (Nth)',	'Max transmission of Ref. spec. (dB)',	'Rsq of IV',	'I at -1V [A]',	'I at 1V [A]'])

df = df.reset_index()  # 인덱스를 열로 변환
df = df.drop('index', axis=1)  # 첫 번째 열 삭제

df.to_csv('AnalysisResult_A2.csv', index=False)  # csv 파일로 저장 (인덱스 제외)

