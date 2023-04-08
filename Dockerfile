FROM python:3.7.3

COPY requirements.txt /
RUN pip install --upgrade pip
RUN pip install --upgrade setuptools wheel
RUN pip install torch
RUN pip install torch_geometric
RUN pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-1.13.0+cu114.html
RUN pip install -r requirements.txt

# COPY server.py /
# COPY inference_pb2_grpc.py /
# COPY inference_pb2.py /
COPY utils /
COPY model /
COPY main.py /
COPY train.py /
COPY config.py /
COPY dataset.py /
COPY train_test_split.py /
# mount data to /

CMD ["python3", "main.py"]
