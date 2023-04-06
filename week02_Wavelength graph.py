#  import library needed
import xml.etree.ElementTree as elemTree
import matplotlib.pyplot as plt
import numpy as np
# XML file parsing
tree = elemTree.parse("HY202103_D07_(0,0)_LION1_DCM_LMZC.xml")
root = tree.getroot()
color=['b','g','r','c','m','y','k'] # 색을 담아놓은 변수
temp=0 # 색을 바꾸기 위한 임시 변수

for i in root.iter('WavelengthSweep'): # iterator를 이용한 parsing
    wavelength=np.array(list(map(float,i.find('L').text.split(','))))
    gain=np.array(list(map(float,i.find('IL').text.split(','))))
    bias=i.attrib['DCBias']
    if temp==6:
        plt.plot(wavelength,gain,label='reference(0V)',color=color[temp])
        continue
    plt.plot(wavelength,gain,label=bias+'V',color=color[temp])
    temp+=1 # color의 다음 index 색으로 바꾸기 위해 변수에 +1

plt.xlabel('Wavelength[nm]', labelpad=8 , fontdict={'weight': 'bold', 'size':8})
plt.ylabel('Gain[dB]', labelpad=8 , fontdict={'weight': 'bold', 'size':8})
plt.title('Transmission graph', fontdict = {'weight': 'bold', 'size':10})
plt.legend(ncol=4,loc='lower center',fontsize=5) # 범례 표시
plt.xticks(fontsize=6) # 축 눈금 레이블 fontsize 설정
plt.yticks(fontsize=6)
plt.show()
# for i in data:
#     plt.plot(data[i][0],data[i][1])

# data={}

# for i in range(10):
#     if i==6:
#         color.append('k')
#     else:
#         color.append('C{}'.format(i))

# data[bias]=np.append([wavelength],[gain],axis=0)
