import os
import xml.etree.ElementTree as elemTree
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import week04_parsed_data as data
import pickle as pk
import week04_fitting_jung as fit
import os

count = 0.1
R_max_ref = np.array([])
R_IV = np.array([])
Error_flag = np.array([])
Error_dsc = np.array([])
Script_version = np.array([])
Script_owner = np.array([])
Users = {'audwl': 'B1','J Seo' : 'B2,','junjuns' : 'B3','User' : 'B4'}
name = ['Lot','Wafer','Mask','TestSite','Name','Date','Script ID','Scipt Version','Script Owner','Operator','Row','column'
        ,'ErrorFlag','Error description','Analysis','Rsq of Ref.spectrum(Nth)','Max_transmission of Ref.spec.(dB)','Rsq of IV','I at -1V[A]','I at 1V']

username = os.environ['USERNAME']


try:
    with open('count.txt','r') as f:
        count = float(f.read())
        count += 0.1
except FileNotFoundError:
    with open('count.txt','w') as f:
        f.write(str(count))

with open('count.txt','w') as f:
    f.write(str(count))

'''
dir_path='data_file'
file_names = [file for file in os.listdir(dir_path) if file.endswith('.xml')]
for file_name in file_names:
    tree = elemTree.parse(f'data_file/{file_name}')
    root = tree.getroot()
'''

for i in range(len(data.WL_ref)):
    R_max_ref = np.append(R_max_ref, fit.Best_fit_R(data.WL_ref[i],data.TR_ref[i]))
for i in range(len(data.I)):
    R_IV = np.append(R_IV, fit.fit_IV_R(data.I[i],data.V[i]))

R_max_ref = R_max_ref.reshape(len(data.file_names),1)
R_IV = R_IV.reshape(len(data.file_names),1)
Error_flag = np.array(list( 0 if x >= 0.95 else 1 for x in list(R_max_ref))).reshape(len(data.file_names),1)
Error_dsc = np.array(list( 'No Error' if x == 0 else 'Ref. spec. Error' for x in list(Error_flag))).reshape(len(data.file_names),1)
Script_version = np.full((len(data.file_names),1),count)
Script_owner = np.full((len(data.file_names),1),Users[username])

# pd.DataFrame([data.Lot,data.Wafer_name,data.Mask_name,data.TestSite,data.Name,data.Script_id,Script_version,data.Operator,data.Row,data.Column
#               ,Error_flag,data.Analysis_WL,R_max_ref,data.Max_TR_ref,R_IV,data.I_n_1V,data.I_p_1V])

