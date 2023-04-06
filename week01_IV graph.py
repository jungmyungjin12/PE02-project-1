# 필요한 라이브러리 가져오기
import xml.etree.ElementTree as ET
import matplotlib.pyplot as plt
import numpy as np

# XML 파일 파싱
tree = ET.parse("HY202103_D07_(0,0)_LION1_DCM_LMZC.xml")
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

# 3차 다항식으로 근사
z = np.polyfit(V, abs_I, 15)
p = np.poly1d(z)

SSE = np.sum((p(V)-np.mean(abs_I))**2)
SST = np.sum((abs_I - np.mean(abs_I))**2)
r_squared = (SSE / SST)

# 그래프 생성
plt.plot(V, abs_I, 'ro', V, abs(p(V)), '-') # 전압 값과 전류 값의 절대값을 포함하는 그래프 생성
plt.title('IV analysis', fontdict = {'weight': 'bold', 'size':15}) # 그래프 제목 설정
plt.xlabel('Voltage[V]', labelpad=8 , fontdict={'weight': 'bold', 'size':12}) # x 축 레이블 설정
plt.ylabel('Current[A]', labelpad=8 , fontdict={'weight': 'bold', 'size':12}) # y 축 레이블 설정
plt.yscale('logit') # y 축 로그 스케일로 설정
plt.grid(True) # 그리드 추가
plt.text(0.02, 0.8, f"r_squared = {r_squared:.20f}", fontsize=12, transform=plt.gca().transAxes)
plt.text(-2,p(-2), p(-2), fontsize = 8)
plt.text(-1,p(-1), p(-1), fontsize = 8)
plt.text(0.5,p(1), p(1), fontsize = 8)
plt.show() # 그래프 표시