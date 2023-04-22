import os
import xml.etree.ElementTree as elemTree
import matplotlib.pyplot as plt
import numpy as np

dir_path='data_file'
file_names = [file for file in os.listdir(dir_path) if file.endswith('.xml')]
print(file_names)

I = np.array([])
V = np.array([])
WL = np.array([])
WL_ref = np.array([])
TR = np.array([])
TR_ref = np.array([])
data_length_TR = 0
DC_bias=[]
temp1 = 0
temp2 = 0
Lot = 0
Wafer_name = 0
Mask_name = 0
TestSite = 0
Date = 0


for file_name in file_names:
    tree = elemTree.parse(f'data_file/{file_name}')
    root = tree.getroot()

    for MD in root.iter('Modulator'):
        for current in root.iter('Current'):
            I = np.append(I,abs(np.array(list(map(float,current.text.split(','))))))
        for voltage in root.iter('Voltage'):
            V = np.append(V,list(map(float,voltage.text.split(','))))

        for WL_sweep in MD.iter('WavelengthSweep'):
            if temp1 == 1:
                WL_ref = np.append(WL_ref, np.array(list(map(float, WL_sweep.find('L').text.split(',')))))
                TR_ref = np.append(TR_ref, np.array(list(map(float, WL_sweep.find('IL').text.split(',')))))
                continue
            WL = np.append(WL, np.array(list(map(float, WL_sweep.find('L').text.split(',')))))
            TR = np.append(TR, np.array(list(map(float, WL_sweep.find('IL').text.split(',')))))
            DC_bias.append(WL_sweep.attrib['DCBias'])
        temp1+=1
    WL=WL.reshape(int(WL.size/WL_ref.size), WL_ref.size)
    TR=TR.reshape(int(WL.size/WL_ref.size), WL_ref.size)
