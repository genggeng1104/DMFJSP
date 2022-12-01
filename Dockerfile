# hash:sha256:2cd9a88279759d26be4197fba39d359ba7eff1a7f3b0d25ca182400a715b70b7
FROM registry.codeocean.com/codeocean/miniconda3:4.9.2-cuda11.7.0-cudnn8-ubuntu20.04

ARG DEBIAN_FRONTEND=noninteractive

RUN pip3 install -U --no-cache-dir \
    keras==2.10.0 \
    numpy==1.23.3 \
    tensorflow==2.10.0
