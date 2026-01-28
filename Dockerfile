# 1. 베이스 이미지 변경
# Python, CUDA, cuDNN, PyTorch, Jupyter 등이 모두 포함된 이미지로 교체
FROM nvcr.io/nvidia/pytorch:25.05-py3

# 2. 시스템 환경 변수 설정 (UTF-8)
# 이 부분은 한국어 환경을 위해 그대로 유지합니다.
ENV LANG=ko_KR.UTF-8
ENV LANGUAGE=ko_KR:ko:en
ENV LC_ALL=ko_KR.UTF-8

# 3. 시스템 의존성 설치 (최소화)
# Python, pip 등은 베이스 이미지에 포함되어 있으므로 설치 목록에서 제외합니다.
# git, vim 같은 유틸리티와 한국어 언어팩만 설치합니다.
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    g++ \
    vim \
    git \
    zip \
    unzip \
    wget \
    curl \
    file \
    language-pack-ko && \
    locale-gen ko_KR.UTF-8 && \
    ldconfig && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/* /tmp/*

# 4. 파이썬 라이브러리 설치
COPY requirements.txt .
RUN pip3 install --timeout=3000 --no-cache-dir -r requirements.txt

# 5. zsh 설치
RUN apt-get update && apt-get install -y zsh && \
    sh -c "$(wget -O- https://github.com/deluan/zsh-in-docker/releases/download/v1.2.1/zsh-in-docker.sh)" -- \
    -x \
    -p https://github.com/zsh-users/zsh-autosuggestions \
    -p https://github.com/zsh-users/zsh-syntax-highlighting && \
    sed -i 's/en_US.UTF-8/ko_KR.UTF-8/g' /root/.zshrc && \
    sed -i 's/en_US:en/ko_KR:ko:en/g' /root/.zshrc

# 6. 주피터 노트북 포트 설정
EXPOSE 8888

# 7. 컨테이너 실행 시 주피터 노트북 자동 실행
# CMD 또한 변경할 필요가 없습니다.
CMD ["jupyter", "notebook", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root"]