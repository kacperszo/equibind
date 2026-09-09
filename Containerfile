# EquiBind encoder, modern tier.
#
# No DGL: the IEGMN encoder and its featurisation are both ported to PyTorch Geometric,
# which is why this image is an ordinary pip install rather than a conda archaeology
# exercise. The authors' published weights load into it unchanged.
FROM docker.io/library/python:3.13-slim

ENV PIP_NO_CACHE_DIR=1 PYTHONUNBUFFERED=1

RUN pip install --no-cache-dir \
    torch==2.5.1 \
    torch-geometric==2.6.1 \
    numpy \
    scipy \
    rdkit \
    biopython \
    biopandas \
    tqdm

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
