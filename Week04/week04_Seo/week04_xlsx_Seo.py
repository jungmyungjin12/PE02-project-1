import xml.etree.ElementTree as elemTree
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import datetime as dt
import os

# count.txt 파일이 이미 존재한다면, 파일에서 이전에 저장한 숫자 값을 읽어옵니다.
if os.path.isfile('../count.txt'):
    with open('../count.txt', 'r') as f:
        count = int(f.read())
# count.txt 파일이 존재하지 않는다면, 숫자 1을 저장합니다.
else:
    count = 0.1
    with open('../count.txt', 'w') as f:
        f.write(str(count))

# 아래 코드는 실행될 때마다 count를 1씩 증가시키고, 그 값을 출력합니다.
count += 0.1   # count를 1 증가시킵니다.
print(count)   # count를 출력합니다.

# 실행한 숫자 값을 파일에 저장합니다.
with open('../count.txt', 'w') as f:
    f.write(str(count))

username = os.environ.get('username')
if username == 'J Seo':
    user = 'B1'

tree = elemTree.parse("../data_file/HY202103_D07_(0,0)_LION1_DCM_LMZC.xml") # objectify XML file
root = tree.getroot() # get root from tree

def R_square(X,Y,Y_reg): # obtain R square
    Y_mean=sum(Y)/Y.size # mean of Current
    SST=sum((Y-Y_mean)**2) # total sum of square ( sum of square of difference between data and mean )
    SSE=sum((Y_reg-Y_mean)**2) # Residual sum of square ( sum of square of difference between )
    return SSE/SST # return R sqaure

for i in root.iter('Current'): # parsing I, V data
    I=np.array(list(map(float,i.text.split(','))))
    I=abs(I) # absolute value of Current data
for i in root.iter('Voltage'):
    V=np.array(list(map(float,i.text.split(','))))


z = np.polyfit(V, I, 12)
p = np.poly1d(z)
iv_rsquared = R_square(V,I,abs(p(V)))

v = []  # 빈 리스트를 만듭니다.
for waveLengthSweep in root.findall('.//WavelengthSweep'):  # WavelengthSweep 태그를 찾습니다.
    waveValues = []  # 빈 리스트를 만듭니다.
    for child in waveLengthSweep:  # WavelengthSweep의 자식 태그들을 찾습니다.
        waveValues.append(list(map(float, child.text.split(','))))  # 자식 태그의 텍스트를 ,로 split해서 리스트로 변환하고, 모든 요소를 float로 변환합니다.
    waveValues.append(waveLengthSweep.attrib['DCBias'])  # DCBias를 waveValues 리스트의 마지막에 추가합니다.
    v.append(waveValues)  # waveValues 리스트를 v 리스트에 추가합니다.

z1 = np.polyfit(v[6][0], v[6][1], 9)
p1 = np.poly1d(z1)
npw = np.array(v[6][0])
npl = np.array(v[6][1])
spec_rsq = R_square(npw, npl, p1(v[6][0]))

name = []
a = Lot = []
name.append('Lot')
b = Wafer = []
name.append('Wafer')
c = Mask = []
name.append('Mask')
d = Testsite = []
name.append('Testsite')

for info in root.iter("TestSiteInfo"):
    Lot.append(info.get("Batch"))
    Wafer.append(info.get("Wafer"))
    Mask.append(info.get("Maskset"))
    Testsite.append(info.get("TestSite"))

e = Name = []
temp=0
name.append('Name')
for info in root.iter("DeviceInfo"):
    if temp==0:
        Name.append(info.get("Name"))
    temp+=1

f = Date = []
name.append('Date')
for info in root.iter("OIOMeasurement"):
    date = info.get("CreationDate")
time = dt.datetime.strptime(date, "%a %b %d %H:%M:%S %Y")
date = time.strftime("%Y%m%d_%H%M%S")
Date.append(date)

g = Script_ID = []
name.append('Script ID')
for info in root.iter("TestSiteInfo"):
    if info.get("TestSite") == "DCM_LMZC" :
        Script_ID.append("process LMZ")

h = Script_Version = []
name.append('Script Version')
Script_Version.append(count)

i1 = Script_Owner = []
name.append('Script Owner')
Script_Owner.append(user)

j = Operator = []
name.append('Operator')
for info in root.iter("OIOMeasurement"):
    Operator.append(info.get("Operator"))

k = Row = []
name.append('Row')
l = Column = []
name.append('Column')
for info in root.iter("TestSiteInfo"):
    Row.append(info.get("DieRow"))
    Column.append(info.get("DieColumn"))

m = ErrorFlag = []
name.append("ErrorFlag")
n = Error_description = []
name.append("Error description")

if iv_rsquared > 0.95:
    ErrorFlag.append('0')
    Error_description.append('No Error')
else:
    ErrorFlag.append('1')
    Error_description('Ref. spec. Error')

o = Analysis_Wavelength = []
name.append('Analysis Wavelength')
for info in root.iter("DesignParameter"):
    if info.get("Name") == "Design wavelength":
        Analysis_Wavelength.append(info.text)

q = ref_spec_rsq = []
name.append('Rsq of Ref. spectrum (Nth)')
ref_spec_rsq.append(spec_rsq)

r = max_trans = []
name.append('Max transmission of Ref. spec. (dB)')
max_trans.append(round(max(v[6][1]),1))

s = rsq_of_iv = []
name.append('Rsq of IV')
rsq_of_iv.append(iv_rsquared)

t = iatm1 = []
name.append('I at -1V [A]')
iatm1.append(p(-1))
u = iat1 = []
name.append('I at 1V [A]')
iat1.append(p(1))

df = pd.DataFrame([a,b,c,d,e,f,g,h,i1,j,k,l,m,n,o,q,r,s,t,u], index=name)
df = df.transpose()
df.to_excel('PE2_GroupNumber_Result.xlsx', index=True)
