import xml.etree.ElementTree as ET

# i_one
def positive1(x):
    tree = ET.parse(x)              # xml parsing
    # xml에서 Current 태그 내용 가져오기
    c = tree.find(".ElectroOpticalMeasurements/ModulatorSite/Modulator/PortCombo/IVMeasurement/Current")
    y_2 = c.text.split(",")         # 쉼표로 분리된 문자열을 리스트로 변환하고, 각 원소를 실수형으로 변환
    y_list = list(map(float, y_2))
    y_list_1=[]
    # 리스트의 원소들의 절대값을 구해서 새로운 리스트에 저장
    for i in range(len(y_list)):
        g = abs(y_list[i])
        y_list_1.append(g)
    return y_list_1[12]             # 리스트의 13번째 원소 반환

# i_none
def negative1(x):
    tree = ET.parse(x)              # xml parsing
    # xml에서 Current 태그 내용 가져오기
    c = tree.find(".ElectroOpticalMeasurements/ModulatorSite/Modulator/PortCombo/IVMeasurement/Current")
    y_2 = c.text.split(",")         # 쉼표로 분리된 문자열을 리스트로 변환하고, 각 원소를 실수형으로 변환
    y_list = list(map(float, y_2))
    y_list_1=[]
    # 리스트의 원소들의 절대값을 구해서 새로운 리스트에 저장
    for i in range(len(y_list)):
        g = abs(y_list[i])
        y_list_1.append(g)
    return y_list_1[4]              # 리스트의 5번째 원소 반환
