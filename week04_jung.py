# 라이브러리 가져오기
import xml.etree.ElementTree as elemTree
import matplotlib.pyplot as plt
import numpy as np

# R_square 값을 반환해주는 함수 정의
def R_square(X,Y,Y_reg): # R_square 값을 반환하는 함수 정의
    Y_mean=sum(Y)/Y.size # 측정 데이터 Y에 대한 평균값을 가지는 변수
    SST=sum((Y-Y_mean)**2) # 측정 데이터와 평균값 차 제곱의 합
    SSE=sum((Y_reg-Y_mean)**2) # 근사 데이터와 측정 데이터 평균값 차 제곱의 합
    SSR=sum((Y-Y_reg)**2)
    return 1-SSR/SST # R_square 값 반환

# N차 다항식으로 근사했을 때 함수 식 표현
def polyeq(X,Y,N):
    equation=''
    coef=np.polyfit(X,Y,N)
    func=np.poly1d(coef)
    plt.plot(X,func(X))
    for i in range(N+1):
        if i==0:
            equation+=f'{coef[i]:.1f}x^{N-i}'
        else:
            equation+=f'+{coef[i]:.1f}x^{N - i}'
    return equation
def fit_data(X,Y,N):
    coef = np.polyfit(X, Y, N)
    func = np.poly1d(coef)
    fit_data = func(X)
    return fit_data

# SSE/SST
# def R_square(X,Y,Y_reg): # R_square 값을 반환하는 함수 정의
#     Y_mean=sum(Y)/Y.size # 측정 데이터 Y에 대한 평균값을 가지는 변수
#     SST=sum((Y-Y_mean)**2) # 측정 데이터와 평균값 차 제곱의 합
#     SSE=sum((Y_reg-Y_mean)**2) # 근사 데이터와 측정 데이터 평균값 차 제곱의 합
#     return SSE/SST # R_square 값 반환

# XML 파일 객채화를 통한 데이터 파싱 준비
tree = elemTree.parse("data_file/HY202103_D07_(0,0)_LION1_DCM_LMZC.xml") # XML 파일 객체
root = tree.getroot() # tree에서 root를 가져옮
# 사용되는 변수 초기값 설정
temp_1=0 # 반복되는 횟수를 세기 위한 변수
Rs=[] # 각 피팅 degree에 대한 R_square를 담는 변수
plots=[] # label에 대한 문자열들을 담을 변수 -> 추후 handle을 이용한 범례 추가에 쓰임
color=['b','g','r','c','m','y','k'] # 마커의 색상을 for문에서 반복하면서 사용하기 위해 변수 선언
graph_square=10 # 다항함수 피팅에 대한 변수
label_fontsize=7 # label의 폰트 사이즈를 조절하기 위한 변수 초기화(뒤에서 일일이 다 조절할 필요가 없음)
title_fontsize=9 # title의 폰트 사이즈를 조절하기 위한 변수 초기화
legend_fontsize=4.5 # legend의 폰트 사이즈를 조절하기 위한 변수 초기화
labelpad_size=2 # labelpad(간격)을 조절하기 위한 변수 초기화
n=5 # text의 위치를 조절하는 데에 사용하는 변수

# data parsing
for i in root.iter('Current'): # IV 데이터 parsing
    I=np.array(list(map(float,i.text.split(','))))
    I=abs(I) # absolute value of Current data
for i in root.iter('Voltage'):
    V=np.array(list(map(float,i.text.split(','))))

z = np.polyfit(V, I, 15) # 15차 다항식으로 V-I 그래프 근사
p = np.poly1d(z) # z(계수들을 리스트로 가짐)을 통해 다항식 함수 모델을 만듦

for i in root.iter('Modulator'):
    if temp_1 == 1:
        for k in i.iter('WavelengthSweep'):
            wave_len=np.array(list(map(float,k.find('L').text.split(','))))
            trans=np.array(list(map(float,k.find('IL').text.split(','))))
    temp_1+=1

plt.subplot(2,3,4)
plt.plot(V,p(V),'k--',label='best-fit') # 근사 데이터 그래프 검은색 점선으로 plot
plt.plot(V,I,'ro',label='data') # 측정 데이터 그래프 빨간색 점으로 plot
plt.yscale('logit') # y축 scale logit으로 지정)

# 그래프 label, 디자인 설정
plt.xlabel('Voltage[V]', labelpad=labelpad_size , fontdict={'weight': 'bold', 'size':label_fontsize})
plt.ylabel('Current[A]', labelpad=labelpad_size , fontdict={'weight': 'bold', 'size':label_fontsize})
plt.title('IV analysis', fontdict = {'weight': 'bold', 'size':title_fontsize})
plt.grid(True) # 그리드 추가
plt.legend(loc='upper left',fontsize=legend_fontsize) # show legend
plt.xticks(fontsize=6) # modulate axis label's fontsize
plt.yticks(fontsize=6)
# show particular data using text method in mathplotlib library
plt.text(0.02,0.8,'R_square = {:.15f}'.format(R_square(V,I,abs(p(V)))),fontsize=4,transform=plt.gca().transAxes)
plt.text(0.02,0.75,'-1V = {:.12f}[A]'.format(p(-1)),fontsize=4,transform=plt.gca().transAxes)
plt.text(0.02,0.7,'+1V = {:.12f}[A]'.format(p(1)),fontsize=4,transform=plt.gca().transAxes)

# plt.gca().transAxes -> help set up the position of text(x: 0~1, y:0~1) 0 4 12
plt.text(-2,p(-2)*1.5,'{:.11f}A'.format(p(-2)),fontsize=4) # y좌표에 1.5를 곱해주는 이유 = text가 점과 겹쳐서 보이기 때문에 1.5를 곱해 text 위치를 상향조정
plt.text(-1,p(-1)*1.5,'{:.11f}[A]'.format(p(-1)),fontsize=4)
plt.text(0.5,p(1)*1.5,'{:.11f}[A]'.format(p(1)),fontsize=4)
# --------------------------------------------------------------------------------------------
for k in [2,5]: # 피팅 그래프는 plot은 동일하게 진행하기 위해서 for 문을 사용하여 subplot 실행
    plt.subplot(2,3,k)

    line, = plt.plot(wave_len,trans, '.', markersize=0.25, label="raw data")  # REF data plot
    plt.gca().add_artist(plt.legend(handles=[line], loc='upper right',fontsize=5))  # REF 레이블을 추가
    if k==2:
        for i in range(1, graph_square + 1):
            coef = np.polyfit(wave_len, trans, i)
            trans_func = np.poly1d(coef)
            Rs.append(R_square(wave_len, trans, trans_func(wave_len)))
            line, = plt.plot(wave_len, trans_func(wave_len), label=f'{i}th', alpha=0.5)
            plots.append(line)
            plt.title('Transmission spectra - Processed and fitting',fontdict={'weight': 'bold', 'size': title_fontsize})
        max_ind = Rs.index(max(Rs))
        for i in range(-2,1):
            if i==0:
                plt.text(0.3, 0.45 + 0.05 * i, f'R\u00B2({max_ind + i + 1}st)= {Rs[max_ind + i]}', fontsize=5, transform=plt.gca().transAxes,color='red',weight='bold')
                continue
            plt.text(0.3, 0.45+0.05*i, f'R\u00B2({max_ind+i+1}st)= {Rs[max_ind+i]}', fontsize=5, transform=plt.gca().transAxes)
    else:
        plt.title('Processed and fitting of reference', fontdict={'weight': 'bold', 'size': title_fontsize})
        plt.text(0.35-0.055*n, 0.45,f'f({n}st) = {polyeq(wave_len,trans,n)}', fontsize=5,transform=plt.gca().transAxes)
        plt.text(0.28, 0.36,f'wavelength at Max : {wave_len[np.where(trans==max(trans))[0][0]]}[nm]', fontsize=5,transform=plt.gca().transAxes)
        plt.text(0.28, 0.31,f'wavelength at Min : {wave_len[np.where(trans==min(trans))[0][0]]}[nm]', fontsize=5,transform=plt.gca().transAxes)

    plt.xlabel('Wavelength[nm]', labelpad=labelpad_size , fontdict={'weight': 'bold', 'size':7})
    plt.ylabel('Measured transmission[dB]', labelpad=labelpad_size , fontdict={'weight': 'bold', 'size':label_fontsize})
    plt.legend(handles=plots,ncol=4,loc='lower center',fontsize=legend_fontsize) # show legend
    plt.xticks(fontsize=6) # modulate axis label's fontsize
    plt.yticks(fontsize=6)
# --------------------------------------------------------------------------------------------------------------------------
for k in [1, 3]:
    plt.subplot(2,3,k)
    temp_2=0
    plots=[]
    for i in root.iter('WavelengthSweep'): # data parsing using iterator
        wavelength=np.array(list(map(float,i.find('L').text.split(','))))
        gain=np.array(list(map(float,i.find('IL').text.split(','))))
        bias=i.attrib['DCBias'] # legend of each graph
        if temp_2==6: # plot reference graph with another way of naming label
            ref_gain = fit_data(wavelength, gain, max_ind + 1)
            if k == 1:
                line,=plt.plot(wavelength,gain,label='reference(0V)',color=color[temp_2])
                plt.gca().add_artist(plt.legend(handles=[line], loc='upper right', fontsize=5))  # REF 레이블을 추가
                plt.title('Transmission spectra - as measured', fontdict={'weight': 'bold', 'size': title_fontsize})
            else:
                line, = plt.plot(wavelength, gain - ref_gain, label='reference(0V)', color=color[temp_2])
                plt.gca().add_artist(plt.legend(handles=[line], loc='upper right', fontsize=5))  # REF 레이블을 추가
                plt.title('Flat Transmission spectra - as measured', fontdict={'weight': 'bold', 'size': title_fontsize})
            continue
        if k == 1:
            line2,=plt.plot(wavelength,gain,label=bias+'V',color=color[temp_2])
            plots.append(line2)
        else:
            line2, = plt.plot(wavelength, gain - ref_gain, label=bias + 'V', color=color[temp_2])
            plots.append(line2)
        temp_2+=1 # to change color, plus 1 to number of repetition variable
    plt.legend(handles=plots,ncol=4,loc='lower center',fontsize=legend_fontsize) # show legend

# set up the background of graph
plt.xlabel('Wavelength[nm]', labelpad=labelpad_size , fontdict={'weight': 'bold', 'size':label_fontsize})
plt.ylabel('measured transmission[dB]', labelpad=labelpad_size , fontdict={'weight': 'bold', 'size':label_fontsize})
plt.xticks(fontsize=6) # modulate axis label's fontsize
plt.yticks(fontsize=6)

# 그래프 크기 조절하기
plt.gcf().set_size_inches(20,5)
plt.subplots_adjust(wspace=0.3,hspace=0.3)

plt.show()# show graph to user
