FROM ubuntu:22.04 AS herd7
LABEL "maintainer"="Deheng Yang <dehengyang@outlook.com>"
LABEL "about"="Herdtools7 & Slide project env"

#ARG DEBIAN_FRONTEND=noninteractive
RUN echo 'debconf debconf/frontend select Noninteractive' | debconf-set-selections

RUN apt update && apt-get install -y --no-install-recommends wget ca-certificates apt-utils 

COPY sources.list /etc/apt/sources.list

RUN apt-get update && apt-get install -y pkg-config

RUN apt-get update && apt install -y opam sudo libgmp-dev


RUN useradd --create-home --shell /bin/bash synth
# passwordless sudo access for ASAN and installing extra tools:
RUN echo "apr ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers
# RUN usermod -aG sudo fuzzer
USER root
WORKDIR /home/synth

RUN opam init -a -y --disable-sandboxing
RUN eval `opam config env`
RUN opam repo set-url default https://mirrors.sjtug.sjtu.edu.cn/git/opam-repository.git --all --set-default

#RUN opam update && opam install -y "herdtools7"

RUN sudo apt-get update && sudo apt-get install -y \
  m4 bubblewrap aspcud rsync curl git

# clone
RUN git config --global http.version HTTP/1.1 && \
    git clone https://github.com/herd/herdtools7.git && \
    cd herdtools7 && \
    git checkout 7.56

#RUN git clone https://github.com/herd/herdtools7.git && \
#    cd herdtools7 && \
#    git checkout 7.56

RUN opam pin add herdtools7 7.56 --yes && \
    opam install herdtools7 --yes


RUN wget https://mirrors.tuna.tsinghua.edu.cn/anaconda/miniconda/Miniconda3-py39_24.4.0-0-Linux-x86_64.sh
COPY .pip .pip
# COPY .condarc .
# ENV CONDA_DIR /opt/conda
RUN /bin/bash Miniconda3-py39_24.4.0-0-Linux-x86_64.sh -b -p /opt/conda
ENV PATH=/opt/conda/bin:$PATH
# -b: This flag, when used with the Miniconda installer, tells the installer to run in "batch" mode, which means it won't prompt for input during the installation process.
# -p /opt/conda: This specifies the installation prefix, i.e., the directory where Miniconda3 will be installed. In this case, /opt/conda is the target directory.

RUN  conda config --set show_channel_urls yes && \
 conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main && \
 conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/r && \
 conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/msys2 && \
 conda clean -i -y  && conda info && conda create -n slide python=3.12.9 -y


RUN sudo apt install -y --no-install-recommends openssh-server nano

# RUN sudo ssh-keygen -t rsa -f /etc/ssh/ssh_host_rsa_key && \
#  sudo ssh-keygen -t dsa -f /etc/ssh/ssh_host_dsa_key && \
#  sudo ssh-keygen -t ecdsa -f /etc/ssh/ssh_host_ecdsa_key && \
#  sudo ssh-keygen -t ed25519 -f /etc/ssh/ssh_host_ed25519_key

# RUN sudo service ssh start
RUN sudo ssh-keygen -A
RUN sudo service ssh --full-restart

EXPOSE 22


#CMD ["/usr/sbin/sshd","-D"]

#RUN sudo mkdir /var/run/sshd
#RUN sudo ssh-keygen -A  

#EXPOSE 22

#CMD ["/bin/bash", "-c", "exec /usr/sbin/sshd -D"]
#CMD ["/usr/sbin/sshd", "-D"]

#COPY entrypoint.sh /usr/local/bin/entrypoint.sh

#EXPOSE 22
#CMD ["entrypoint.sh"]


RUN apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y \
    curl git build-essential libssl-dev libffi-dev ca-certificates \
    && rm -rf /var/lib/apt/lists/*


RUN curl -L https://mirror.racket-lang.org/installers/8.2/racket-8.2-x86_64-linux.sh -o racket.sh

RUN chmod +x racket.sh && \
    ./racket.sh --in-place --dest /usr/racket && \
    ln -s /usr/racket/bin/* /usr/local/bin/ && \
    rm racket.sh

#RUN wget https://mirror.racket-lang.org/installers/8.2/racket-8.2-x86_64-linux.tar.gz&& \
#    tar -xzf racket-8.2-x86_64-linux.tar.gz&& \
#    cd racket-8.2-x86_64-linux&& \
#    sudo ./install.sh



ENV PATH="/usr/racket/bin:$PATH"


RUN raco pkg config --set catalogs \
    "https://pkgs.racket-lang.org" \
    "https://download.racket-lang.org/releases/8.2/catalog/" && \
    for i in 1 2 3; do raco pkg install --auto ocelot && break || sleep 5; done


# openssh-client
RUN apt-get update && \
    apt-get install -y --no-install-recommends openssh-client && \
    rm -rf /var/lib/apt/lists/*


# memsynth
WORKDIR /home/synth
RUN git clone https://github.com/rmculpepper/custom-load.git && \
    cd custom-load && \
    raco pkg install

WORKDIR /home/synth
RUN git clone https://github.com/tonyg/racket-rfc6455.git && \
    mv racket-rfc6455 rfc6455 && \
    cd rfc6455 && \
    raco pkg install

WORKDIR /home/synth
RUN git clone https://github.com/emina/rosette.git && \
    cd rosette && \
    git checkout 3.1 && \
    raco pkg install

WORKDIR /home/synth
RUN git clone https://github.com/jamesbornholt/ocelot.git && \
    cd ocelot && \
    raco pkg install

ARG CACHEBUSTER=1

WORKDIR /home/synth

RUN git clone https://github.com/tracesynth/slide.git

RUN sudo apt install -y graphviz-dev && cd slide && git checkout main

RUN cd slide && . /opt/conda/bin/activate && conda activate base && conda activate slide && pip install -r requirements.txt


WORKDIR /home/synth

RUN git clone https://github.com/tracesynth/riscv_memsynth.git
RUN mv riscv_memsynth memsynth