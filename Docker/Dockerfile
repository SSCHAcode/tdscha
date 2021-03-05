FROM ubuntu:20.04

WORKDIR /root


ENV PYTHON_PACKAGES="\
    numpy \
    matplotlib \
    scipy \
    ase \
" 

ENV SYSTEM_PACKAGES="\
    build-essential \
    libopenblas-dev \
    liblapack-dev \
    gfortran \
    python3-dev \
    python3-pip \
    libjpeg-dev \
    zlib1g-dev \
"

# Install the package
RUN apt-get update && apt-get -y install $SYSTEM_PACKAGES

# Install with pip the packages
RUN pip3 install --upgrade pip setuptools \
    && pip3 install $PYTHON_PACKAGES

# Install CellConstructor and python-sscha
RUN pip3 install --no-cache-dir CellConstructor python-sscha 

RUN mkdir mount

RUN ["bash"]