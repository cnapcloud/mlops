# Ray 공식 이미지를 베이스로 사용 (Python 버전 및 CPU/GPU 여부 확인)
FROM rayproject/ray:2.55.1-py312-cpu-aarch64

# 작업 디렉토리 설정
WORKDIR /app

# 필요한 패키지 설치
# Ray 공식 이미지에는 이미 ray가 포함되어 있습니다.
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# 소스 코드 복사
COPY ./serve/serve_app.py /app/serve_app.py

# (선택 사항) 환경 변수 설정
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app

# Ray Serve가 사용할 포트 노출
EXPOSE 8000