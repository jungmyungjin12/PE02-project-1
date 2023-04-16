import xml.etree.ElementTree as etree
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import r2_score
from lmfit import Model

def current(V_D, I_s, n, T):    # 쇼클리 다이오드 방정식 정의
    k = 1.380649 * 10**(-23)
    q = 1.602 * 10**(-19)
    return I_s*(np.exp(q*V_D/(n*k*T))-1)

xml_file = etree.parse('HY202103_D07_(0,0)_LION1_DCM_LMZC.xml')     # XML 파일 불러오기
root = xml_file.getroot()           # get root(element) of file

# 폰트 설정
font_axis = {
    'family': 'monospace',
    'weight': 'bold',
    'size': 18
}
font_title = {                 # font setting for title
    'family': 'monospace',     # font style
    'weight': 'bold',          # font weight
    'size': 25                 # font size
}

# Voltage - Current (Raw data)
vol, cur = [], []               # Voltage, Current값 초기화
plt.subplot(1, 2, 1)            # 1행 2열의 첫번째 자리에 그래프를 그릴 공간 생성

for i in root.iter():           # Return iterator for the root element
    if i.tag == 'Voltage':
        vol = list(map(float, i.text.split(',')))       # tag가 'Voltage'면 데이터를 리스트로 저장
    elif i.tag == 'Current':
        cur = list(map(float, i.text.split(',')))       # tag가 'Current'면 데이터를 리스트로 저장

plt.plot(vol, list(map(abs, cur)), 'co', label='raw_data')   # 전압 - 전류 그래프를 청록색 원형 마크로 그림

# Voltage-Current(Fitting)

# V_D <= 0.25V(threshold voltage)
p_num = 10     # 10차 다항식으로 근사
fit = np.polyfit(np.array(vol)[:10], np.array(list(map(abs, cur)))[:10], p_num)
# 전류 값을 전압 값에 대해 다항식으로 fitting하여 계수를 구함
fit_eq = np.poly1d(fit)         # 계수를 이용해 fitting 방정식 생성
print(f'Fitting equation : {fit_eq}')
print(f'r2 = {r2_score(list(map(abs, cur))[:10], fit_eq(vol[:10]))}\n')         # R square 출력

# V_D > 0.25V(threshold voltage)
Cmodel = Model(current)             # fitting 모델 정의(함수 정의)
params = Cmodel.make_params(I_s=1e-15, n=1, T=300)         # 초기값 세팅(포화 전류)
result = Cmodel.fit(list(map(abs, cur))[10:], params, V_D=vol[10:])         # fitting 수행
print(f'result of best values : {result.best_values}')          # best fitting의 매개변수 출력
print(f'result of best fit : {result.best_fit}')                # fitting 이후 전류 데이터 출력
print(f'r2 = {r2_score(list(map(abs, cur))[10:], result.best_fit)}\n')          # R square 출력

# 0.25보다 작은 값에 대해서 fit_eq 함수를, 0.25보다 큰 값은 정의한 모델에서 값을 가져와 fit_plot리스트에 저장
fit_plot = []
n1, n2 = 0, 0
for v in vol:
    if v <= 0.25:
        fit_plot.append(fit_eq(vol[n1]))
        n1 += 1
    else:
        fit_plot.append(result.best_fit.tolist()[n2])
        n2 += 1
# r2 score를 계산하여 출력
print(f'r2 = {r2_score(list(map(abs, cur)), fit_plot)}\n')
plt.plot(vol, fit_plot, '--', label='Fitting')          # fitting 그래프를 결과로 나타냄

plt.text(0.02,0.8,'R_square = {:.15f}'.format(r2_score(list(map(abs, cur)), fit_plot)),fontsize=8,transform=plt.gca().transAxes)
plt.text(0.02,0.75,'R_square = {:.15f}'.format(r2_score(list(map(abs, cur))[10:], result.best_fit)),fontsize=8,transform=plt.gca().transAxes)
plt.text(-2,fit_eq(-2)*1.5,'{:.11f}'.format(fit_eq[0]),fontsize=6)
plt.text(-1,fit_eq(-1)*1.5,'{:.11f}'.format(fit_eq[4]),fontsize=6)


# Setting of graph
plt.title('IV-analysis', fontdict=font_title)       # 폰트 설정이 적용된 레이블 작성
plt.xlabel('Voltage[V]', labelpad=10, fontdict=font_axis)   # 폰트와 15pt 여백이 적용된 x축 레이블 작성
plt.ylabel('Current[A]', labelpad=10, fontdict=font_axis)   # 폰트와 15pt 여백이 적용된 y축 레이블 작성
plt.xticks(fontsize=14)         # x축 눈금값의 폰트 크기 설정
plt.yticks(fontsize=14)         # y축 눈금값의 폰트 크기 설정
plt.yscale('logit')     # y축 스케일을 로그 형식으로 변경
plt.minorticks_off()
plt.legend(loc='best', ncol=2, fontsize=13)     # 범례 위치, 열의 수, 폰트 크기 설정
plt.grid(True, which='major', alpha=0.5)        # 주 눈금선에 대한 그리드 표시 및 투명도 설정
plt.show()