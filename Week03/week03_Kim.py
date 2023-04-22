# 필요한 library 불러오기
import xml.etree.ElementTree as elemTree
import matplotlib.pyplot as plt
import numpy as np

def R_square(X,Y,Y_reg): # R_square 값을 반환하는 함수 정의
    Y_mean=sum(Y)/Y.size # 측정 데이터 Y에 대한 평균값을 가지는 변수
    SST=sum((Y-Y_mean)**2) # 측정 데이터와 평균값 차 제곱의 합
    SSE=sum((Y_reg-Y_mean)**2) # 근사 데이터와 측정 데이터 평균값 차 제곱의 합
    return SSE/SST # R_square 값 반환

tree = elemTree.parse("HY202103_D07_(0,0)_LION1_DCM_LMZC.xml")  # XML 파일 객체
root = tree.getroot()                                           # tree에서 root를 가져옮

for i in root.iter('Current'): # IV 데이터 parsing
    I=np.array(list(map(float,i.text.split(','))))
    I=abs(I) # absolute value of Current data
for i in root.iter('Voltage'):
    V=np.array(list(map(float,i.text.split(','))))

z = np.polyfit(V, I, 12) # 12차 다항식으로 V-I 그래프 근사
p = np.poly1d(z)         # z(계수들을 리스트로 가짐)을 통해 다항식 함수 모델을 만듦

plt.subplot(2,3,4) # 한 GUI에 그래프 두 개를 그리기 위해 subplot(전체 행 개수, 열 개수, 위치) 사용
plt.plot(V,p(V),'k--',label='best-fit') # 근사 데이터 그래프 검은색 점선으로 plot
plt.plot(V,I,'ro',label='data') # 측정 데이터 그래프 빨간색 점으로 plot
plt.yscale('logit') # y축 scale logit으로 지정)

# 그래프 label, 디자인 설정
plt.xlabel('Voltage[V]', labelpad=4 , fontdict={'weight': 'bold', 'size':10})
plt.ylabel('Current[A]', labelpad=4 , fontdict={'weight': 'bold', 'size':10})
plt.title('IV analysis', fontdict = {'weight': 'bold', 'size':10})
plt.grid(True) # 그리드 추가
plt.legend(loc='upper left',fontsize=8) # show legend
plt.xticks(fontsize=8) # modulate axis label's fontsize
plt.yticks(fontsize=8)
# show particular data using text method in mathplotlib library
plt.text(0.02,0.8,'R_square = {:.15f}'.format(R_square(V,I,abs(p(V)))),fontsize=10,transform=plt.gca().transAxes)
plt.text(0.02,0.75,'-1V = {:.12f}[A]'.format(p(-1)),fontsize=10,transform=plt.gca().transAxes)
plt.text(0.02,0.7,'+1V = {:.12f}[A]'.format(p(1)),fontsize=10,transform=plt.gca().transAxes)
# plt.gca().transAxes -> help set up the position of text(x: 0~1, y:0~1) 0 4 12
plt.text(-2,p(-2)*1.5,'{:.11f}A'.format(p(-2)),fontsize=8) # y좌표에 1.5를 곱해주는 이유 = text가 점과 겹쳐서 보이기 때문에 1.5를 곱해 text 위치를 상향조정
plt.text(-1,p(-1)*1.5,'{:.11f}[A]'.format(p(-1)),fontsize=8)
plt.text(0.5,p(1)*1.5,'{:.11f}[A]'.format(p(1)),fontsize=8)


v = []  # 빈 리스트 생성
for waveLengthSweep in root.findall('.//WavelengthSweep'):  # WavelengthSweep 태그 찾기
    waveValues = []  # 빈 리스트 생성
    for child in waveLengthSweep:  # WavelengthSweep의 자식 태그들을 찾기
        waveValues.append(list(map(float, child.text.split(','))))  # 자식 태그의 텍스트를 ,로 split해서 리스트로 변환하고, 모든 요소를 float으로 변환
    waveValues.append(waveLengthSweep.attrib['DCBias'])  # DCBias를 waveValues 리스트의 마지막에 추가
    v.append(waveValues)  # waveValues 리스트를 v 리스트에 추가

plt.subplot(2,3,1)
# Spectrum graph of raw data
plots = []  # 빈 리스트 생성
for i in range(len(v) - 1):  # v 리스트의 마지막 요소는 REF로 제외하고 반복
    line, = plt.plot(v[i][0], v[i][1], label=str(v[i][2]) + 'V')  # plot을 그리고, 레이블을 설정
    plots.append(line)  # plot을 plots 리스트에 추가

line, = plt.plot(v[6][0], v[6][1], color='gray', label="REF")  # REF data plot

plt.gca().add_artist(plt.legend(handles=[line], loc='upper right',fontsize=7))  # REF 레이블을 추가
plt.legend(handles=plots, ncol=3, loc="lower center", fontsize=8)  # 나머지 레이블을 추가
plt.title("Transmission spectra - as measured", fontdict = {'weight': 'bold', 'size':12})  # 그래프 제목을 설정
plt.xlabel('Wavelength [nm]', labelpad=6 , fontdict={'weight': 'bold', 'size':8})  # x축 레이블을 설정
plt.ylabel('Measured transmission [dB]', labelpad=6 , fontdict={'weight': 'bold', 'size':8})  # y축 레이블을 설정


# ----------------------------------------------------------------------------
# font 설정
font_axis = {                       # x, y축 라벨의 폰트 설정
    'family': 'monospace',          # 폰트 스타일
    'weight': 'bold',               # 폰트 두께
    'size': 10                      # 폰트 크기
}

font_title = {                     # 그래프 제목 폰트 설정
    'family': 'monospace',         # 폰트 스타일
    'weight': 'bold',              # 폰트 두께
    'size': 12                     # 폰트 크기
}
# R-squre 함수 정의
def R_square(X,Y,Y_reg):           # R_square 값을 반환하는 함수 정의
    Y_mean=sum(Y)/Y.size           # 측정 데이터 Y에 대한 평균값을 가지는 변수
    SST=sum((Y-Y_mean)**2)         # 측정 데이터와 평균값 차 제곱의 합
    SSE=sum((Y_reg-Y_mean)**2)     # 근사 데이터와 측정 데이터 평균값 차 제곱의 합
    return SSE/SST                 # R_square 값 반환

plt.subplot(2, 3, 2)            # 1행 2열의 첫번째 자리에 그래프를 그릴 공간 생성

wl_R, tm_R = [], []             # wavelength reference, transmission reference 리스트 생성

# XML 파일에서 wavelength, transmission 데이터 추출
for i in root.iter():
    if i.tag == 'Modulator':                            # 태그 이름이 'Modulator'인 경우
        if i.attrib.get('Name') == 'DCM_LMZC_ALIGN':    # 속성 이름이 'DCM_LMZC_ALIGN'인 경우
            # wavelength 값을 추출하여 리스트 형태로 저장
            wl_R = list(map(float, i.find('PortCombo').find('WavelengthSweep').find('L').text.split(',')))
            # transmission 값을 추출하여 리스트 형태로 저장
            tm_R = list(map(float, i.find('PortCombo').find('WavelengthSweep').find('IL').text.split(',')))

# R-square 1차부터 8차 근사 반복문
R_squared = []
for degree in range(1, 9):                                      # 1차부터 8차 다항식으로 근사
    z = np.polyfit(np.array(wl_R), np.array(tm_R), degree)      # degree차 다항식으로 그래프 근사
    p = np.poly1d(z)                                            # z(계수들을 리스트로 가짐)을 통해 다항식 함수 모델을 만듦
    tm_reg = p(wl_R)
    R_sq = R_square(np.array(wl_R), np.array(tm_R), tm_reg)     # 근사값으로 R_square 계산
    R_squared.append(R_sq)
    print(f"{degree}차 함수: R_square = {R_sq:.10f}")
print('R-squared:',R_squared)

# R-square 최대값 구하기
max_R_squared = max(R_squared)     # R_squared 리스트에서 최대값을 구함
print(f"최대 R_square 값: {max_R_squared:.10f}")
max_index = R_squared.index(max_R_squared)    # R_squared 리스트에서 최대값의 인덱스를 구함
plt.text(0.5, 0.1, f"Max R_square = {max_R_squared:.8f}", ha='center', va='center')
plt.annotate(f"Max R_square {max_R_squared:.8f}", xy=(wl_R[0], tm_R[0]), xytext=(wl_R[0]+15, tm_R[0]+2))
# R-square 최대값 이전 차수 구하기
prev_R_squared = R_squared[max_index - 1]  # 최대값의 이전 차수에 대한 R_square 값을 구함
plt.text(0.5, 0.15, f"R_square = ({max_index}): {prev_R_squared:.8f}", ha='center', va='center')
plt.annotate(f"R_square = {prev_R_squared:.8f}", xy=(wl_R[0], tm_R[0]), xytext=(wl_R[0]+15, tm_R[0]+2.5))

# 추세선 그래프 그래프에 출력
fit = np.polyfit(np.array(wl_R), np.array(tm_R), 8)     # 회귀 분석을 수행하여 추세선 계수 구하기
fit_eq = np.poly1d(fit)     # 추세선 방정식 구하기
print(f'Fitting equation : {fit_eq}')
plt.plot(wl_R, fit_eq(wl_R), label='Fitting')     # 추세선 데이터 플롯
# plt.text(0.2, 0.3, f"y:{fit_eq}", fontsize = 6,transform=plt.gca().transAxes)     # 근사한 8차 방정식 출력

plot = []  # 빈 리스트 생성
for i in range(len(R_squared)): # R_squared 리스트의 크기만큼 반복
    degree = i + 1 # 근사 차수
    line, = plt.plot(wl_R, np.poly1d(np.polyfit(wl_R, tm_R, degree))(wl_R), label=str(degree) +'th') # plot을 그리고, 레이블을 설정
    plot.append(line) # plot을 plots 리스트에 추가

line, = plt.plot(wl_R, fit_eq(wl_R), color='gray', label="Fitting") # 회귀 분석 결과 plot
plt.plot(wl_R, tm_R, color='#ED2B2A', linestyle=':', label='Reference') # REF data plot

plt.title('Transmission spectra - as measured', fontdict=font_title)        # Write a label with a setting of font_title
plt.xlabel('Wavelength[nm]', labelpad=8, fontdict=font_axis)                # Write a label with a setting of axis
plt.ylabel('Measured transmission[dB]', labelpad=8, fontdict=font_axis)     # Write a label with a setting of axis
plt.xticks(fontsize=8)                                                      # Set the font size of axis value
plt.yticks(fontsize=8)                                                      # Set the font size of axis value
plt.legend(loc='lower center', ncol=2, fontsize=5)
plt.subplots_adjust(wspace=0.3, hspace=0.3)                                 # 그래프 간격 설정

plt.show()          # 그래프를 출력
