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
I = (list(map(float,a.split(','))))

# 'Voltage' 태그에서 전압 값을 추출하여 리스트로 변환
for i in root.iter('Voltage'):
    b = i.text
V = (list(map(float,b.split(','))))

# 전류 값의 절대값을 계산
abs_I = np.abs(I)

# 그래프 생성
plt.plot(V, abs_I, 'bo--') # 전압 값과 전류 값의 절대값을 포함하는 그래프 생성
plt.title('IV analysis', fontdict = {'weight': 'bold', 'size':15}) # 그래프 제목 설정
plt.xlabel('Voltage[V]', labelpad=8 , fontdict={'weight': 'bold', 'size':12}) # x 축 레이블 설정
plt.ylabel('Current[A]', labelpad=8 , fontdict={'weight': 'bold', 'size':12}) # y 축 레이블 설정
plt.yscale('logit') # y 축 로그 스케일로 설정
plt.grid(True) # 그리드 추가
plt.show() # 그래프 표시
