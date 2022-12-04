# 常规构建命令：docker build -t {my_image_name:tag}
# The base image is python:3.9
FROM python:3.9

LABEL org.xiangci-cgec.image-author="pe653"
LABEL version = "0.1.0"
LABEL description="cgec-pivot's dockerfile."

# 设置文件夹是工作目录
WORKDIR /code

# 拷贝文件夹
COPY requirements.txt /code/requirements.txt

# 安装支持, python库通过阿里云镜像加速
RUN pip3 --default-timeout=60000 install --no-cache-dir --upgrade -i https://mirrors.aliyun.com/pypi/simple/ --trusted-host mirrors.aliyun.com -r /code/requirements.txt

# 预下载model file in huggingface的标准 ./cache/huggingface/hub 目录
# 预下载model file in sentence-transformers的标准 ./cache/torch/sentence-transformers 目录
# RUN python -c "from sentence_transformers import SentenceTransformer; model = SentenceTransformer('uer/sbert-base-chinese-nli')"

# 拷贝频繁变化的代码最后步骤，加速docker image的生成
COPY . /code

# 声明建议暴露的端口(for api-uvicorn)
EXPOSE 7777

# 容器启动入口
# ENTRYPOINT ["uvicorn", "app.main:c_api", "--proxy-headers", "--host", "0.0.0.0", "--port", "9000"]
