import os

file_list = []          # 빈 리스트 생성

# 해당 디렉토리 내의 파일 목록을 불러오는 함수 정의
def search(dirname):
    filenames = os.listdir(dirname)     # 디렉토리 내의 모든 파일 및 폴더 목록을 리스트 형태로 반환
    # 파일명이 하나씩 들어있는 리스트에서 파일명을 꺼내어 처리
    for filename in filenames:
        full_filename = os.path.join(dirname, filename)     # 파일명과 디렉토리 경로를 합쳐서 파일의 전체 경로 생성
        ext = os.path.splitext(full_filename)[-1]           # 파일의 확장자가 '.png'인 경우에만 리스트에 추가
        if ext == '.png':
            file_list.append(ext)
