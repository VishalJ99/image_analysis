FROM continuumio/miniconda3

RUN mkdir -p image_analysis

COPY . /image_analysis
WORKDIR /image_analysis

RUN conda env update --file environment.yml

RUN echo "conda activate ia_vj279" >> ~/.bashrc
SHELL ["/bin/bash", "--login", "-c"]

# Needed for opencv.
RUN apt-get update && apt-get install -y libgl1-mesa-glx
