FROM ubuntu:22.04

SHELL ["/bin/bash", "-exo", "pipefail", "-c"]

RUN echo 'APT::Install-Suggests "0";' >> /etc/apt/apt.conf.d/00-docker \
    && echo 'APT::Install-Recommends "0";' >> /etc/apt/apt.conf.d/00-docker

RUN DEBIAN_FRONTEND=noninteractive \
  apt-get update \
  && apt-get install -y python3 pip python3-tk \
  && rm -rf /var/lib/apt/lists/* \
  && mkdir /root/wheels \
  && mkdir -p /root/toolkit/models

COPY ./rkllm_toolkit-1.1.4-cp310-cp310-linux_x86_64.whl /root/wheels

RUN python3 -m pip install inquirer /root/wheels/rkllm_toolkit-1.1.4-cp310-cp310-linux_x86_64.whl xformers

COPY ./interactive_pipeline.py /root/toolkit/

WORKDIR /root/toolkit

CMD /usr/bin/python3 ./interactive_pipeline.py
