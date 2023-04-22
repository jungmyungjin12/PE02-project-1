# 필요한 라이브러리를 import
import xml.etree.ElementTree as etree
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score
from lmfit import Model
import time

def current(V_D, I_s, n):                          # V_D, I_s, n을 이용한 전류 계산 함수
    return I_s*(np.exp((V_D/(n*0.026))-1))

def load_data(file_root):                            # xml 파일에서 데이터 추출 함수
    wafer, mask, test, name, date, oper, row, col, analysis_wl = [], [], [], [], [], [], [], [], []
    # 데이터를 저장할 빈 리스트들

    for data in root.iter():
        if data.tag == 'OIOMeasurement':                # 'OIOMeasurement' 태그에서 날짜와 작업자 이름 추출
            date.append(data.get('CreationDate'))
            oper.append(data.get('Operator'))

        elif data.tag == 'TestSiteInfo':                # 'TestSiteInfo' 태그에서 관련 정보 추출
            test.append(data.get('TestSite'))
            wafer.append(data.get('Wafer'))
            mask.append(data.get('Maskset'))
            row.append(data.get('DieRow'))
            col.append(data.get('DieColumn'))

        elif data.tag == 'DesignParameter':             # 'DesignParameter' 태그에서 분석 대상 파장 추출
            if data.attrib.get('Name') == 'Design wavelength':
                analysis_wl.append(data.text)

        elif data.tag == 'ModulatorSite':               # 'ModulatorSite' 태그에서 모듈레이터 이름 추출
            name.append(data.find('Modulator').get('Name'))

    return wafer, mask, test, name, date, oper, row, col, analysis_wl

xml_file = etree.parse("data_file/HY202103_D07_(0,0)_LION1_DCM_LMZC.xml")     # xml 파일 불러오기
root = xml_file.getroot()           # root 엘리먼트 추출

# 폰트 설정
font_title = {                 # 타잍틀 글꼴 설정
    'family': 'monospace',     # 글꼴 스타일 선정
    'weight': 'bold',          # 글꼴 굵기 설정
    'size': 15                 # 글꼴 크기 설정
}

# ==================================================================================================================== #
# 전체 plot 크기 설정 및 제목 설정
plt.figure(figsize=(20, 10))
plt.suptitle('HY202103_D07_(0,0)_LION1_DCM_LMZC', fontsize=20, weight='bold')

# Voltage-Current(Raw data)
vol, cur = [], []           # vol, cur 리스트 생성
plt.subplot(2, 3, 4)        # 그래프 위치 설정

# XML 파일에서 Voltage 태그와 Current 태그의 값을 추출하여 리스트로 저장
for i in root.iter():
    if i.tag == 'Voltage':
        vol = list(map(float, i.text.split(',')))
    elif i.tag == 'Current':
        cur = list(map(float, i.text.split(',')))
# 추출한 데이터를 산점도로 시각화
plt.plot(vol, list(map(abs, cur)), 'co', label='raw_data')

# Voltage-Current(Fitting)
# V_D <= 0.25V(threshold voltage)
p_num = 5
fit = np.polyfit(np.array(vol)[:10], np.array(list(map(abs, cur)))[:10], p_num)
fit_eq = np.poly1d(fit)
print('[Fitting about V_D <= 0.25 (IV analysis)]')
print('[Fitting equation]')
print(fit_eq)
print(f'r² = {r2_score(list(map(abs, cur))[:10], fit_eq(vol[:10]))}\n')

# V_D > 0.25V(threshold voltage)
Cmodel = Model(current)
params = Cmodel.make_params(I_s=5e-12, n=1)
result = Cmodel.fit(list(map(abs, cur))[10:], params, V_D=vol[10:])
print('[Fitting about V_D > 0.25 (IV analysis)]')
print(f'result of best values : {result.best_values}')
print(f'result of best fit : {result.best_fit}')
print(f'r² = {r2_score(list(map(abs, cur))[10:], result.best_fit)}\n')

# fitting 결과를 리스트에 저장
fit_plot = []
n1, n2 = 0, 0
for v in vol:
    if v <= 0.25:
        fit_plot.append(fit_eq(vol[n1]))
        n1 += 1
    else:
        fit_plot.append(result.best_fit.tolist()[n2])
        n2 += 1

# r-squared 값을 계산
rsq_iv = r2_score(list(map(abs, cur)), fit_plot)
print('[Result of fitting (IV analysis)]')
print(f'r² = {rsq_iv}\n')

# fitting 결과를 선으로 시각화
plt.plot(vol, fit_plot, '--', label='Fitting')

# Setting of graph
plt.title('IV-analysis', fontdict=font_title)
plt.xlabel('Voltage[V]', fontsize=10)
plt.ylabel('Current[A]', fontsize=10)
plt.xticks(fontsize=10)
plt.yticks(fontsize=10)
plt.yscale('logit')
plt.minorticks_off()
plt.legend(loc='best', ncol=2, fontsize=10)
plt.grid(True, which='major', alpha=0.5)

# ==================================================================================================================== #

# Wavelength-Transmission(Raw data)

# 빈 리스트와 변수 초기화
wl_list, tm_list = [], []
wl_ref, tm_ref = [], []
arr_tm = []
DC_bias = -2.0
plot_color = ['lightcoral', 'coral', 'gold', 'lightgreen', 'lightskyblue', 'plum']
color_number = 0

plt.subplot(2, 3, 1)        # 2행 3열의 subplot 중 첫 번째 그래프 설정
# xml 파일에서 'WavelengthSweep' tag 탐색
for i in root.iter():
    if i.tag == 'WavelengthSweep':         # 'DCBias' 속성이 입력한 값과 같으면 wavelength과 transmission 값을 리스트에 추가
        if i.attrib.get('DCBias') == str(DC_bias):
            wl = list(map(float, i.find('L').text.split(',')))
            wl_list.append(wl)
            tm = list(map(float, i.find('IL').text.split(',')))
            tm_list.append(tm)
            plt.plot(wl, tm, plot_color[color_number], label=f'DCBias = {DC_bias}V')    # 그래프를 그릴 때 사용할 색상 지정
            DC_bias += 0.5          # DC bias 값을 0.5씩 증가하면서 그래프의 색상 변경
            color_number += 1
        # 그래프의 제목, x축과 y축 레이블, 범례 설정
        plt.title('Transmission spectra - as measured', fontdict=font_title)
        plt.xlabel('Wavelength[nm]', fontsize=10)
        plt.ylabel('Measured transmission[dB]', fontsize=10)
        plt.legend(loc='lower center', ncol=2, fontsize=10)

    # Reference
    # xml 파일에서 'Modulator' tag 탐색
    elif i.tag == 'Modulator':
        # 'Name' 속성이 'DCM_LMZC_ALIGN'과 같으면 reference spectrum의 wavelength과 transmission 값을 리스트에 추가
        if i.attrib.get('Name') == 'DCM_LMZC_ALIGN':
            wl_ref = list(map(float, i.find('PortCombo').find('WavelengthSweep').find('L').text.split(',')))
            tm_ref = list(map(float, i.find('PortCombo').find('WavelengthSweep').find('IL').text.split(',')))
            plt.plot(wl_ref, tm_ref, color='#7f7f7f', linestyle=':', label='Reference')
            plt.subplot(2, 3, 2)        # subplot 중 두 번째 그래프 설정
            plt.plot(wl_ref, tm_ref, color='#7f7f7f', linestyle=':', label='Reference')
            arr_tm = np.array(tm_ref)
            # reference spectrum의 transmission 값 중 최댓값과 최솟값, 해당 wavelength 값 출력
            print(f'Max transmission of Ref. spec : {np.max(arr_tm)}dB at wavelength : {wl_ref[np.argmax(arr_tm)]}nm')
            print(f'Min transmission of Ref. spec : {np.min(arr_tm)}dB at wavelength : {wl_ref[np.argmin(arr_tm)]}nm\n')

# Wavelength-Transmission(Fitting)
rsq_ref = []
for p in range(2, 7):                       # 다항식 차수를 2부터 6까지 변화시켜가며 반복문 실행
    start_time = time.time()                # 시간 측정 시작
    # np.polyfit() 함수를 사용하여 다항식 계수 계산
    fit = np.polyfit(np.array(wl_ref), np.array(tm_ref), p)
    run_time = time.time() - start_time     # 다항식 계수 계산 시간 측정
    fit_eq = np.poly1d(fit)                 # 계산된 다항식 계수를 이용하여 다항식 객체 생성
    print(f'[Fitting equation(ref)-{p}th]')
    print(fit_eq)                           # 계산된 다항식 출력
    print(f'r²={r2_score(tm_ref, fit_eq(wl_list[0]))}')
    print(f'run time : {run_time}s\n')
    rsq_ref.append(r2_score(tm_ref, fit_eq(wl_list[0])))        # 계산된 r² 값을 리스트에 추가
    plt.plot(wl_ref, fit_eq(wl_ref), label=f'{p}th R² : {r2_score(tm_ref, fit_eq(wl_list[0]))}')    # 그래프 출력

plt.title('Transmission spectra - as measured', fontdict=font_title)
plt.xlabel('Wavelength[nm]', fontsize=10)
plt.ylabel('Measured transmission[dB]', fontsize=10)
plt.legend(loc='lower center', fontsize=10)

DC_bias = -2.0
plt.subplot(2, 3, 3)
# DC 바이어스 값이 변화하는 것을 반복문으로 구현하여 그래프 출력
for j in range(6):
    plt.plot(wl_ref, tm_ref - fit_eq(wl_ref))
    plt.plot(wl_list[j], tm_list[j]-fit_eq(wl_list[j]), plot_color[j], label=f'DC_bias={DC_bias}')
    DC_bias += 0.5

plt.title('Flat Transmission spectra - as measured', fontdict=font_title)
plt.xlabel('Wavelength[nm]', fontsize=10)
plt.ylabel('Measured transmission[dB]', fontsize=10)
plt.legend(loc='lower center', ncol=2, fontsize=10)

# plt.savefig('HY202103_D07_(0,0)_LION1_DCM_LMZC.png')

wafer, mask, test, name, date, oper, row, col, analysis_wl = load_data(root)
# 결과 데이터를 DataFrame으로 생성하여 csv 파일로 저장
df = pd.DataFrame({'Wafer': wafer, 'Mask': mask, 'TestSite': test, 'Name': name, 'Date': date,
                   'Operator': oper, 'Row': row, 'Column': col, 'Analysis Wavelength': analysis_wl,
                   'Rsq of Ref. spectrum (Nth)': rsq_ref[2], 'Max transmission of Ref. spec. (dB)': np.max(arr_tm),
                   'Rsq of IV': rsq_iv, 'I at -1V [A]': cur[4],
                   'I at 1V [A]': abs(cur[-1])})

# df.to_csv('HY202103_D07_(0,0)_LION1_DCM_LMZC.csv')
plt.show()