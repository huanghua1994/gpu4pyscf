#!/bin/bash

BASE_IMAGE=nvidia/cuda:12.8.0-devel-ubuntu24.04
IMAGE_TAG=ubuntu24.04-gcc13.1-cuda12.8-PySCF-GPU-MxP-DF

docker build \
    --tag=${USER}:${IMAGE_TAG} \
    --build-arg image=${BASE_IMAGE} \
    --build-arg uid=$(id -u) --build-arg username=$(id -un) \
    --build-arg gid=$(id -g) --build-arg groupname=$(id -gn) \
    -f Dockerfile .
