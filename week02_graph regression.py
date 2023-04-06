import xml.etree.ElementTree as elemTree
import matplotlib.pyplot as plt
import numpy as np
def data_fitting(X,Y,N):
    cof=np.polyfit(X,Y,N)
    fit_data=np.zeros([X.size])
    for i in range(N+1):
        fit_data+=(X**(N-i)*cof[i])
    return abs(fit_data)
def R_square(X,Y,Y_reg):
    Y_mean=sum(Y)/Y.size
    SST=sum((Y-Y_mean)**2)
    SSE=sum((Y_reg-Y_mean)**2)
    return SSE/SST

tree = elemTree.parse("HY202103_D07_(0,0)_LION1_DCM_LMZC.xml")
root = tree.getroot()

for i in root.iter('Current'):
    I=np.array(list(map(float,i.text.split(','))))
    I=abs(I)
for i in root.iter('Voltage'):
    V=np.array(list(map(float,i.text.split(','))))

data_fitted=data_fitting(V,I,16)
plt.plot(V,data_fitted,'k--',label='best-fit')
plt.plot(V,I,'ro',label='data')
plt.yscale('logit')

# print('{:.20f}'.format(R_square(V,I,data_fitting(V,I,16))))

plt.xlabel('Voltage[V]', labelpad=8 , fontdict={'weight': 'bold', 'size':8})
plt.ylabel('Current[A]', labelpad=8 , fontdict={'weight': 'bold', 'size':8})
plt.title('IV analysis', fontdict = {'weight': 'bold', 'size':10})
plt.legend(loc='upper left',fontsize=7) # 범례 표시
plt.xticks(fontsize=6) # 축 눈금 레이블 fontsize 설정
plt.yticks(fontsize=6)
plt.text(0.02,0.8,'R_square = {:.15f}'.format(R_square(V,I,data_fitted)),fontsize=8,transform=plt.gca().transAxes)
plt.text(-2,data_fitted[0]*1.5,'{:.11f}'.format(data_fitted[0]),fontsize=6)
plt.text(-1,data_fitted[4]*1.5,'{:.11f}'.format(data_fitted[4]),fontsize=6)
plt.text(1,data_fitted[12]*1.5,'{:.11f}'.format(data_fitted[12]),fontsize=6)
plt.show()
