FROM python:3.12-slim
LABEL authors="lihb"

RUN pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu --no-cache-dir

COPY requirements.txt /requirements.txt
# 单独安装torch 是因为默认会下载nvidia，这个下下来镜像就太大了
RUN pip install -r /requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple  --no-cache-dir

ARG APP_PATH=/data

COPY ./ ${APP_PATH}/str_to_vector

WORKDIR ${APP_PATH}

EXPOSE 8000


ENTRYPOINT ["python"]

CMD ["/data/str_to_vector/main.py"]
