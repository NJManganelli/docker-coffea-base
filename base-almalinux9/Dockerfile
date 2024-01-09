FROM almalinux:9

ARG python
ARG release

ENV PYTHON_VERSION=${python}

RUN yum -y update \
     && yum --allowerasing -y install git curl bzip2 libgfortran which zsh emacs \
     && curl -sSL https://github.com/conda-forge/miniforge/releases/download/23.3.1-1/Mambaforge-Linux-x86_64.sh -o /tmp/mambaforge.sh \
     && bash /tmp/mambaforge.sh -bfp /usr/local \
     && rm -rf /tmp/mambaforge.sh \
     && mamba update mamba \
     && mamba clean --all --yes \
     && rpm -e --nodeps curl bzip2 \
     && yum clean all

# XRootD + CA + VOMS + Coffea
RUN mamba install --yes \
      -c conda-forge -c pytorch \
      python=${PYTHON_VERSION} \
      gxx \
      voms \
      jupyterlab \
      ipympl \
      dask_labextension \
      ca-policy-lcg \
      xrootd \
      fsspec \
      "uproot>=5.2.1" \
      "awkward>=2.5.1" \
      awkward-pandas \
      pyarrow \
      "dask[array]>=2023.12.1" \
      "dask-awkward>=2024.1.0" \
      "dask-histogram>=2023.10.0" \
      numpy \
      coffea=${release} \
      vector xgboost \
      lz4 python-xxhash zstandard h5py \
      pytables \
      python-rapidjson geventhttpclient \
      brotli nomkl \
     && mamba clean -y --all \
     && pip install --no-cache-dir \
      --extra-index-url https://download.pytorch.org/whl/cpu/ \
      --extra-index-url https://google-coral.github.io/py-repo/ \
      fastjet==3.4.1.3 \
      tritonclient[all] \
      torch \
      tflite-runtime \
      onnxruntime \
      fsspec-xrootd \
      graphviz \
     && pip install --no-cache-dir \
      torch-scatter torch-sparse torch-cluster torch-spline-conv -f https://data.pyg.org/whl/torch-1.12.0+cpu.html \
     && pip install --no-cache-dir \
      torch-geometric

# Make a symbolic link between installation /opt/conda/etc/grid-security and actual directory /etc/grid-security
RUN ln -s /usr/local/etc/grid-security /etc/grid-security
