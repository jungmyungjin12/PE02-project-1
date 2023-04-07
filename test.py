# import library
import xml.etree.ElementTree as ET
import numpy as np
import matplotlib.pyplot as plt
from lmfit import Model
def shockely_diode(voltage,rev_sat_I,temp_voltage):
    return rev_set_I*(np.exp(voltage/temp_voltage-1)

tree = ET.parse("HY202103_D07_(0,0)_LION1_DCM_LMZC.xml")
root = tree.getroot() # get root from tree

for i in root.iter('Current'): # parsing I, V data
    I=np.array(list(map(float,i.text.split(','))))
    # I=abs(I) # absolute value of Current data
for i in root.iter('Voltage'):
    V=np.array(list(map(float,i.text.split(','))))

# 모델 인스턴스 생성
model = Model(shockely_diode)

# 초기 매개 변수 설정
params = model.make_params(
    reverse_sat_I=1e-6,
    temp_voltage=300.0,
)

# 모델 피팅
result = model.fit(I, params, voltage=V)

# 결과 출력
print(result.fit_report())

# 그래프 그리기
plt.scatter(V, I, label='Experimental Data')
plt.plot(V, result.best_fit, label='Fitted Curve')
plt.xlabel('Voltage (V)')
plt.ylabel('Current (A)')
plt.legend()
plt.show()
