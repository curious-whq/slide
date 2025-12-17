As Miniconda3-py39_24.4.0-0-Linux-x86_64.sh exceeds 100MB, we fail to upload it to gitee.

To build a docker image based on the Dockerfile, we can use "RUN wget https://mirrors.tuna.tsinghua.edu.cn/anaconda/miniconda/Miniconda3-py39_24.4.0-0-Linux-x86_64.sh" to replace "COPY Miniconda3-py39_24.4.0-0-Linux-x86_64.sh .".
