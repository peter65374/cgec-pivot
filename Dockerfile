# 常规构建命令：docker build -t {my_image_name:tag}
# The base image is python:3.7
# FROM python:3.7

# 阿里云华东1杭州容器镜像服务加速方案
# docker pull python:3.7
# docker tag python:3.7 registry.cn-hangzhou.aliyuncs.com/xiangci-coach/python:3.7
# docker push registry.cn-hangzhou.aliyuncs.com/xiangci-coach/python:3.7

FROM registry.cn-hangzhou.aliyuncs.com/xiangci-coach/python:3.7

LABEL org.xiangci-coach.image-author="zhangning@xiangci.top"
LABEL version = "1.0"
LABEL description="aut0-cr2t's dockerfile."

# 设置文件夹是工作目录
WORKDIR /code

# 拷贝文件夹
COPY requirements.txt /code/requirements.txt
# COPY .wheel/zh_core_web_trf-3.4.0-py3-none-any.whl /code/.wheel/zh_core_web_trf-3.4.0-py3-none-any.whl

# 安装支持, python库通过阿里云镜像加速
RUN pip3 --default-timeout=60000 install --no-cache-dir --upgrade -i https://mirrors.aliyun.com/pypi/simple/ --trusted-host mirrors.aliyun.com -r /code/requirements.txt

# 预下载model file in huggingface的标准 ./cache/huggingface/hub 目录
# RUN python -c 'from transformers import AutoTokenizer; AutoTokenizer.from_pretrained("uer/sbert-base-chinese-nli")'
# RUN python -c 'from transformers import AutoModel; AutoModel.from_pretrained("uer/sbert-base-chinese-nli")'

# 预下载model file in sentence-transformers的标准 ./cache/torch/sentence-transformers 目录
RUN python -c "from sentence_transformers import SentenceTransformer; model = SentenceTransformer('uer/sbert-base-chinese-nli')"


# download zh_core_web_sm pipeline of spacy
RUN python -m spacy download zh_core_web_sm

# With local file
# RUN pip3 install .wheel/zh_core_web_sm-3.0.0-py3-none-any.whl
# RUN pip3 install .wheel/zh_core_web_sm-3.0.0.tar.gz

# 拷贝频繁变化的代码最后步骤，加速docker image的生成
COPY . /code

# 声明建议暴露的端口(for coach_api-uvicorn)
# EXPOSE 9000

# 容器启动入口
# ENTRYPOINT ["uvicorn", "app.main:coach_api", "--proxy-headers", "--host", "0.0.0.0", "--port", "9000"]
