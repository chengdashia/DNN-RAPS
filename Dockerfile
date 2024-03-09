# 使用官方Python镜像作为基础镜像
FROM python:3.8.5

# 设置工作目录
WORKDIR /app

# 复制包含所有Python依赖的requirements.txt文件到容器中
COPY requirements.txt /app/requirements.txt

# 使用pip安装Python依赖
RUN pip install --no-cache-dir -r requirements.txt

# 复制你的自定义模块和推理脚本到工作目录
COPY model.pth /app/model.pth
COPY Communicator.py /app/Communicator.py
COPY VGG.py /app/VGG.py
COPY main.py /app/main.py

# 运行推理脚本
CMD ["python", "main.py"]