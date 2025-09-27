FROM ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential curl ca-certificates wget xz-utils \
    libffi-dev libssl-dev zlib1g-dev libbz2-dev libreadline-dev \
    libsqlite3-dev tk-dev uuid-dev && \
    rm -rf /var/lib/apt/lists/*

RUN apt-get update && apt-get install -y git cmake build-essential

WORKDIR /tmp
ARG PY_VER=3.10.3
RUN wget -q https://www.python.org/ftp/python/${PY_VER}/Python-${PY_VER}.tgz && \
    tar -xzf Python-${PY_VER}.tgz && cd Python-${PY_VER} && \
    ./configure --enable-optimizations --with-lto --prefix=/usr/local && \
    make -j"$(nproc)" && make altinstall && \
    ln -s /usr/local/bin/python3.10 /usr/local/bin/python3 && \
    /usr/local/bin/python3.10 -m ensurepip && \
    /usr/local/bin/python3.10 -m pip install --upgrade pip && \
    cd / && rm -rf /tmp/Python-${PY_VER}*
# sanity check
RUN python3 --version && python3 -m pip --version

# glpkソルバーのインストール
RUN apt-get update && apt-get install -y glpk-utils

# scipソルバーのインストール
WORKDIR /scip_tmp
RUN wget -q https://www.scipopt.org/download/release/scipoptsuite-9.2.3.tgz
RUN tar xzf scipoptsuite-9.2.3.tgz
WORKDIR /scip_tmp/scipoptsuite-9.2.3/scip/build
RUN cmake .. -DAUTOBUILD=ON -DSOPLEX_WITH_THREADS=ON
RUN make -j"$(nproc)" PARASCIP=true && make install

WORKDIR /scip_tmp/scipoptsuite-9.2.3/ug/build
RUN cmake .. -DSCIP_DIR=../../scip/build
RUN make -j"$(nproc)"
RUN make install
