FROM ubuntu:20.04 as base

ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=america/los_angeles

RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

RUN apt-get update -y 
RUN apt-get install -y libsdl1.2-dev \
    libfreetype6-dev \
    python3-pygame \
    git \
    python3 \ 
    python3-pip

WORKDIR work
COPY requirements.txt /work

RUN pip3 install -r /work/requirements.txt

RUN apt-get install -y vim

EXPOSE 8000

RUN echo "cd /src && pip3 install -e . && cd riglib/plexon && pip3 install -e . && cd /src" >> /root/.bashrc