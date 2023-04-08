# import necessary library
import xml.etree.ElementTree as elemTree
import matplotlib.pyplot as plt
import numpy as np

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

z = np.polyfit(V, I, 15)
p = np.poly1d(z)

plt.subplot(1,2,1) # plot on one GUI (regression graph)
plt.plot(V,p(V),'k--',label='best-fit') # plot approximated graph as a dotted line
plt.plot(V,I,'ro',label='data') # plot I-V graph as points
plt.yscale('logit') # set up y axis scale as log

# print('{:.20f}'.format(R_square(V,I,data_fitting(V,I,16))))

# set up background of graph
plt.xlabel('Voltage[V]', labelpad=8 , fontdict={'weight': 'bold', 'size':8})
plt.ylabel('Current[A]', labelpad=8 , fontdict={'weight': 'bold', 'size':8})
plt.title('IV analysis', fontdict = {'weight': 'bold', 'size':10})
plt.grid(True) # 그리드 추가
plt.legend(loc='upper left',fontsize=7) # show legend
plt.xticks(fontsize=6) # modulate axis label's fontsize
plt.yticks(fontsize=6)
# show particular data using text method in mathplotlib library
plt.text(0.02,0.8,'R_square = {:.15f}'.format(R_square(V,I,abs(p(V)))),fontsize=8,transform=plt.gca().transAxes)
plt.text(0.02,0.75,'-1V = {:.12f}[A]'.format(p(-1)),fontsize=8,transform=plt.gca().transAxes)
plt.text(0.02,0.7,'+1V = {:.12f}[A]'.format(p(1)),fontsize=8,transform=plt.gca().transAxes)
# plt.gca().transAxes -> help set up the position of text(x: 0~1, y:0~1) 0 4 12
plt.text(-2,p(-2)*1.5,'{:.11f}A'.format(p(-2)),fontsize=6)
plt.text(-1,p(-1)*1.5,'{:.11f}[A]'.format(p(-1)),fontsize=6)
plt.text(0.5,p(1)*1.5,'{:.11f}[A]'.format(p(1)),fontsize=6)


v = []  # 빈 리스트를 만듭니다.
for waveLengthSweep in root.findall('.//WavelengthSweep'):  # WavelengthSweep 태그를 찾습니다.
    waveValues = []  # 빈 리스트를 만듭니다.
    for child in waveLengthSweep:  # WavelengthSweep의 자식 태그들을 찾습니다.
        waveValues.append(list(map(float, child.text.split(','))))  # 자식 태그의 텍스트를 ,로 split해서 리스트로 변환하고, 모든 요소를 float로 변환합니다.
    waveValues.append(waveLengthSweep.attrib['DCBias'])  # DCBias를 waveValues 리스트의 마지막에 추가합니다.
    v.append(waveValues)  # waveValues 리스트를 v 리스트에 추가합니다.

plt.subplot(1,2,2)
# Spectrum graph of raw data
plots = []  # 빈 리스트를 만듭니다.
for i in range(len(v) - 1):  # v 리스트의 마지막 요소는 REF로 제외하고 반복합니다.
    line, = plt.plot(v[i][0], v[i][1], label=str(v[i][2]) + 'V')  # plot을 그리고, 레이블을 설정합니다.
    plots.append(line)  # plot을 plots 리스트에 추가합니다.

line, = plt.plot(v[6][0], v[6][1], color='gray', label="REF")  # REF plot을 그립니다.

plt.gca().add_artist(plt.legend(handles=[line], loc='upper right'))  # REF 레이블을 추가합니다.
plt.legend(handles=plots, ncol=3, loc="lower center", fontsize=5)  # 나머지 레이블을 추가합니다.
plt.title("Transmission spectra - as measured", fontdict = {'weight': 'bold', 'size':10})  # 그래프 제목을 설정합니다.
plt.xlabel('Wavelength [nm]', labelpad=8 , fontdict={'weight': 'bold', 'size':8})  # x축 레이블을 설정합니다.
plt.ylabel('Measured transmission [dB]', labelpad=8 , fontdict={'weight': 'bold', 'size':8})  # y축 레이블을 설정합니다.
plt.show()  # 그래프를 출력합니다.