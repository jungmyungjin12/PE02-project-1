# 현재 디렉토리를 포함하여 하위 디렉토리에서 'LMZ'라는 이름을 가진 XML 파일들의 전체 경로를 리스트 형태로 반환하는 코드
import glob
# 특정 경로 패턴과 일치하는 모든 파일의 경로를 리슽트로 반환하고 LMZ 파일만 추출함
all_LMZ = glob.glob(r'.\data\**\*LMZ?.xml',
                    recursive = True)