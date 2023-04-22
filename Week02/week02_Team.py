# 필요한 library 불러오기
import xml.etree.ElementTree as elemTree
import matplotlib.pyplot as plt
import numpy as np

def R_square(X,Y,Y_reg): # R_square 값을 반환하는 함수 정의
    Y_mean=sum(Y)/Y.size # 측정 데이터 Y에 대한 평균값을 가지는 변수 
    SST=sum((Y-Y_mean)**2) # 측정 데이터와 평균값 차 제곱의 합
    SSE=sum((Y_reg-Y_mean)**2) # 근사 데이터와 측정 데이터 평균값 차 제곱의 합
    return SSE/SST # R_square 값 반환

tree = elemTree.parse("HY202103_D07_(0,0)_LION1_DCM_LMZC.xml") # XML 파일 객체
root = tree.getroot() # tree에서 root를 가져옮

for i in root.iter('Current'): # IV 데이터 parsing
    I=np.array(list(map(float,i.text.split(','))))
    I=abs(I) # absolute value of Current data
for i in root.iter('Voltage'):
    V=np.array(list(map(float,i.text.split(','))))

z = np.polyfit(V, I, 15) # 15차 다항식으로 V-I 그래프 근사
p = np.poly1d(z) # z(계수들을 리스트로 가짐)을 통해 다항식 함수 모델을 만듦

plt.subplot(1,2,1) # 한 GUI에 그래프 두 개를 그리기 위해 subplot(전체 행 개수, 열 개수, 위치) 사용
plt.plot(V,p(V),'k--',label='best-fit') # 근사 데이터 그래프 검은색 점선으로 plot
plt.plot(V,I,'ro',label='data') # 측정 데이터 그래프 빨간색 점으로 plot
plt.yscale('logit') # y축 scale logit으로 지정)

# 그래프 label, 디자인 설정
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
plt.text(-2,p(-2)*1.5,'{:.11f}A'.format(p(-2)),fontsize=6) # y좌표에 1.5를 곱해주는 이유 = text가 점과 겹쳐서 보이기 때문에 1.5를 곱해 text 위치를 상향조정
plt.text(-1,p(-1)*1.5,'{:.11f}[A]'.format(p(-1)),fontsize=6)
plt.text(0.5,p(1)*1.5,'{:.11f}[A]'.format(p(1)),fontsize=6)


v = []  # 빈 리스트 생성
for waveLengthSweep in root.findall('.//WavelengthSweep'):  # WavelengthSweep 태그 찾기
    waveValues = []  # 빈 리스트 생성
    for child in waveLengthSweep:  # WavelengthSweep의 자식 태그들을 찾기
        waveValues.append(list(map(float, child.text.split(','))))  # 자식 태그의 텍스트를 ,로 split해서 리스트로 변환하고, 모든 요소를 float으로 변환
    waveValues.append(waveLengthSweep.attrib['DCBias'])  # DCBias를 waveValues 리스트의 마지막에 추가
    v.append(waveValues)  # waveValues 리스트를 v 리스트에 추가

plt.subplot(1,2,2)
# Spectrum graph of raw data
plots = []  # 빈 리스트 생성
for i in range(len(v) - 1):  # v 리스트의 마지막 요소는 REF로 제외하고 반복
    line, = plt.plot(v[i][0], v[i][1], label=str(v[i][2]) + 'V')  # plot을 그리고, 레이블을 설정
    plots.append(line)  # plot을 plots 리스트에 추가

line, = plt.plot(v[6][0], v[6][1], color='gray', label="REF")  # REF data plot

plt.gca().add_artist(plt.legend(handles=[line], loc='upper right',fontsize=5))  # REF 레이블을 추가
plt.legend(handles=plots, ncol=3, loc="lower center", fontsize=5)  # 나머지 레이블을 추가
plt.title("Transmission spectra - as measured", fontdict = {'weight': 'bold', 'size':10})  # 그래프 제목을 설정
plt.xlabel('Wavelength [nm]', labelpad=8 , fontdict={'weight': 'bold', 'size':8})  # x축 레이블을 설정
plt.ylabel('Measured transmission [dB]', labelpad=8 , fontdict={'weight': 'bold', 'size':8})  # y축 레이블을 설정
plt.show()  # 그래프를 출력
