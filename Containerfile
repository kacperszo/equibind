# EquiBind encoder, modern tier.
#
# No DGL: the IEGMN encoder and its featurisation are both ported to PyTorch Geometric,
# which is why this image is an ordinary pip install rather than a conda archaeology
# exercise. The authors' published weights load into it unchanged.
FROM docker.io/library/python:3.13-slim

ENV PIP_NO_CACHE_DIR=1 PYTHONUNBUFFERED=1

# Torch is a build argument so one tree serves both cards. The default is what
# `equibind:latest` holds and what every published number here was produced with: PyPI's
# 2.5.1, whose kernels stop at sm_90. Blackwell (sm_120, the lab server's RTX PRO 6000)
# needs a cu128 build, which still carries sm_75-sm_90 and so runs on every card here:
#   podman build --format=docker -t equibind:cu128 \
#     --build-arg TORCH=2.7.1 --build-arg TORCH_INDEX=https://download.pytorch.org/whl/cu128 .
ARG TORCH=2.5.1
ARG TORCH_INDEX=https://pypi.org/simple
RUN pip install --no-cache-dir torch==${TORCH} --index-url ${TORCH_INDEX}

# Pinned to exactly what `equibind:latest` resolved, so changing TORCH changes torch and
# nothing else. Unpinned, a rebuild silently picks up a newer rdkit — and rdkit is part of
# this model, not a detail around it: it decides which molecule is read, and a difference
# there is a featurisation change that nothing raises. IGN learned that three times over;
# see models/ign/CLAUDE.md.
RUN pip install --no-cache-dir \
    "torch-geometric==2.6.1" \
    "numpy==2.5.2" \
    "scipy==1.18.1" \
    "rdkit==2026.3.5" \
    "biopython==1.88" \
    "biopandas==0.5.2" \
    "tqdm==4.70.0"

RUN python -c "import torch, torch_geometric, rdkit, Bio, biopandas, scipy; print('env ok', torch.__version__)"

WORKDIR /work
COPY . /work
RUN pip install --no-cache-dir --no-deps -e . && \
    python -c "from equibind.models.encoder import EquiBindEncoder; \
from equibind.features.featurize import get_calpha_graph; print('equibind imports ok')"

# The encoder/head boundary, checked at build time. Needs no checkpoint and no data — the
# graphs are synthetic — so unlike test_equivariance.py it can run here, and a boundary that
# only gets checked when somebody remembers is a boundary that drifts.
RUN python test_transfer.py
