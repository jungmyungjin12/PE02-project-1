# import necessary library
import xml.etree.ElementTree as elemTree
import matplotlib.pyplot as plt
import numpy as np

def data_fitting(X,Y,N): # Approximation function definition
    cof=np.polyfit(X,Y,N) # regression (result -> list of coefficient)
    fit_data=np.zeros([X.size])  # define default matrix
    for i in range(N+1): # repeat as degree of polynomial
        fit_data+=(X**(N-i)*cof[i]) # data approximated
    return abs(fit_data) # return absolute value of data
def R_square(X,Y,Y_reg): # obtain R square
    Y_mean=sum(Y)/Y.size # mean of Current
    SST=sum((Y-Y_mean)**2) # total sum of square ( sum of square of difference between data and mean )
    SSE=sum((Y_reg-Y_mean)**2) # Residual sum of square ( sum of square of difference between )
    return SSE/SST # return R sqaure

tree = elemTree.parse("HY202103_D07_(0,0)_LION1_DCM_LMZC.xml") # objectify XML file
root = tree.getroot() # get root from tree

for i in root.iter('Current'): # parsing I, V data
    I=np.array(list(map(float,i.text.split(','))))
    I=abs(I) # absolute value of Current data
for i in root.iter('Voltage'):
    V=np.array(list(map(float,i.text.split(','))))

plt.subplot(1,2,1) # plot on one GUI
data_fitted=data_fitting(V,I,16) # fitted data using defined function
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

# ---------------------------------------------------------------------------------------------------------------------
plt.subplot(1,2,2)

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