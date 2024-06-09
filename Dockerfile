FROM continuumio/miniconda3

RUN mkdir -p image_analysis

COPY . /image_analysis
WORKDIR /image_analysis

RUN conda env update --file environment.yml

RUN echo "conda activate image_analysis" >> ~/.bashrc
SHELL ["/bin/bash", "--login", "-c"]

RUN pre-commit install
