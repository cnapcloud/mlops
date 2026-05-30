# Ray 공식 이미지를 베이스로 사용 (Python 버전 및 CPU/GPU 여부 확인)
FROM rayproject/ray:2.55.1-py312-cpu-aarch64

# 필요한 패키지 설치
# Ray 공식 이미지에는 이미 ray가 포함되어 있습니다.
COPY requirements.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt