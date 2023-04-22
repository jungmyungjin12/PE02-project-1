import xml.etree.ElementTree as ET
from dateutil.parser import parse       # 날짜 parsing을 위한 라이브러리

def TestSiteInfo(x,y): #Lot, Wafer,Maskset,TestSite,DieRow,DieColumn 정보를 가져오는 함수
    tree = ET.parse(x)                  # xml parsing
    a = tree.find("./TestSiteInfo")     # xml 파일에서 TestSiteInfo 노드를 찾음
    return (a.get(y))                   # 해당 정보 반환

def Date(x):                # 날짜 반환 함수
    tree = ET.parse(x)      # xml parsing
    # xml 파일에서 측정에 대한 정보가 있는 노드를 찾음
    c= tree.find("./ElectroOpticalMeasurements/ModulatorSite/Modulator/PortCombo")
    w =str(parse(c.get("DateStamp")))                               # DateStamp 속성에 있는 측정 날짜 정보를 문자열로 변환
    return (w[0:4]+w[5:7]+w[8:10]+"_"+w[11:13]+w[14:16]+w[17:19])    # yyyyMMdd_HHmmss 형태로 측정 날짜를 반환

def transmission(x):        # 최대 투과율을 반환하는 함수
    tree = ET.parse(x)      # xml parsing
    # xml 파일에서 투과율에 대한 정보가 있는 노드를 찾음
    IL7 = tree.find(".ElectroOpticalMeasurements/ModulatorSite/Modulator[2]/PortCombo/WavelengthSweep/IL")
    IL7 = IL7.text.split(",")       # 투과율 정보를 ','를 기준으로 나눔
    IL7 = list(map(float, IL7))     # 문자열로 된 투과율 정보를 실수형으로 변환
    return max(IL7)                 # 투과율 정보 중 최대값 반환

def Name(x):                # 디바이스 이름을 반환하는 함수
    tree = ET.parse(x)      # xmla parsing
    # xml 파일에서 디바이스 정보가 있는 노드를 찾음
    b= tree.find('./ElectroOpticalMeasurements/ModulatorSite/Modulator/DeviceInfo')
    return (b.get("Name"))  # Name 속성에 해당하는 디바이스 이름을 반환

def Wavelength(x):           # 디바이스 파장 정보를 반환하는 함수
    tree = ET.parse(x)       # xml parsing
    # xml 파일에서 디바이스 정보가 있는 노드를 찾음
    d = tree.find('./ElectroOpticalMeasurements/ModulatorSite/Modulator/DeviceInfo/DesignParameters/DesignParameter[2]')
    return d.text            # 디바이스 파장 정보를 반환