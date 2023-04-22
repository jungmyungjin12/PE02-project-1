import os
import xml.etree.ElementTree as elemTree
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import week04_parsed_data as data
import pickle as pk
import week04_fitting_jung as fit
import os

# 사용할 변수 선언
count = 0.1 # 돌린 횟수를 저장할 변수
R_max_ref = np.array([])
R_IV = np.array([])
Error_flag = np.array([])
Error_dsc = np.array([])
Script_version = np.array([])
Script_owner = np.array([])
Users = {'audwl': 'B1','J Seo' : 'B2,','junjuns' : 'B3','User' : 'B4'}
name = ['Lot','Wafer','Mask','TestSite','Name','Date','Script ID','Scipt Version','Script Owner','Operator','Row','Column'
        ,'ErrorFlag','Error description','Analysis Wavelength','Rsq of Ref.spectrum(Nth)','Max_transmission of Ref.spec.(dB)','Rsq of IV','I at -1V[A]','I at 1V']

username = os.environ['USERNAME']

# 돌린 횟수 데이터를 저장 및 변수에 할당하는 코드
try:
    with open('count.txt','r') as f: # txt 파일에서 숫자 데이터(돌린 횟수) 읽기
        count = float(f.read())
        count += 0.1
except FileNotFoundError: # 처음에 아무 아무 숫자가 없어 생기는 오류 방지
    with open('count.txt','w') as f:
        f.write(str(count))

with open('count.txt','w') as f: # +0.1이 된 횟수를 다시 작성(w는 원래 있던 데이터를 삭제하고 다시 씀)
    f.write(str(count))

# R_max_ref와 R_IV 데이터를 구하는 코드
for i in range(len(data.WL_ref)):
    R_max_ref = np.append(R_max_ref, np.array(fit.Best_fit_R(data.WL_ref[i],data.TR_ref[i])))
for i in range(len(data.I)):
    R_IV = np.append(R_IV, np.array(float(fit.fit_IV_R(data.V[i],abs(data.I[i])))))

# 전체 데이터를 활용(구분)하기 편하도록 reshape
R_max_ref = R_max_ref.reshape(len(data.file_names),1)
R_IV = R_IV.reshape(len(data.file_names),1)
Error_flag = np.array(list( 0 if x >= 0.95 else 1 for x in list(R_max_ref))).reshape(len(data.file_names),1)
Error_dsc = np.array(list( 'No Error' if x == 0 else 'Ref. spec. Error' for x in list(Error_flag))).reshape(len(data.file_names),1)
Script_version = np.full((len(data.file_names),1),count)
Script_owner = np.full((len(data.file_names),1),Users[username])

# Week04_parsed_data에서 parsing된 데이터들 가져와서 DataFrame을 만들고 적절히 변환하여 엑셀 데이터로 변환
df = pd.DataFrame([data.Lot,data.Wafer_name,data.Mask_name,data.TestSite,data.Name,data.Date,data.Script_id,Script_version,Script_owner,data.Operator,data.row
                ,data.column,Error_flag,Error_dsc,data.Analysis_WL,R_max_ref,data.Max_TR_ref,R_IV,data.I_n_1V,data.I_p_1V],index=name)
df = df.transpose()
df.to_excel('PE02_week04_Result.xlsx',index=False) # index=False로 기본 index 설정을 삭제
