FROM ubuntu:16.04

RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip

RUN pip3 install --upgrade pip setuptools

RUN pip3 install jupyter

COPY bashrc /etc/bash.bashrc
RUN chmod a+rwx /etc/bash.bashrc

RUN mkdir /workdir && chmod a+rwx /workdir
RUN mkdir /workdir/smartstore && chmod a+rwx /workdir/smartstore

WORKDIR /workdir

ENV HOME /workdir
ENV PYTHONUNBUFFERED x

RUN pip3 install numpy pillow
