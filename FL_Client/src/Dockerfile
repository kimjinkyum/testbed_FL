
FROM nvcr.io/nvidia/l4t-pytorch:r32.5.0-pth1.7-py3

# RUN export PATH=/usr/local/cuda-10.2/bin${PATH:+:${PATH}}
# RUN export LD_LIBRARY_PATH=/usr/local/cuda-10.2/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}


# RUN pip install --upgrade pip3
# RUN pip3 install -U setuptools
# RUN wget https://nvidia.box.com/shared/static/cs3xn3td6sfgtene6jdvsxlr366m2dhq.whl -O torch-1.7.0-cp36-cp36m-linux_aarch64.whl
# RUN pip3 install torch-1.7.0-cp36-cp36m-linux_aarch64.whl
RUN pip3 install flask
COPY mains.py .
COPY init_fl.py .
COPY model.py .
COPY utils.py .

CMD [ "python3", "mains.py" ]

# Configure Python 3.9
#RUN add-apt-repository ppa:deadsnakes/ppa \
#    && apt-get update \
#    && apt-get install -y python${V_PYTHON} python${V_PYTHON}-dev python${V_PYTHON}-venv python${V_PYTHON_MAJOR}-tk \
#    && rm /usr/bin/python \
#    && rm /usr/bin/python3 \
#    && ln -s $(which python${V_PYTHON}) /usr/bin/python \
#    && ln -s $(which python${V_PYTHON}) /usr/bin/python${V_PYTHON_MAJOR} \
#    && curl --silent --show-error https://bootstrap.pypa.io/get-pip.py | python
# Install PyTorch
# COPY torch-1.10.0-cp39-cp39-linux_aarch64.whl /tmp/torch-1.10.0-cp39-cp39-linux_aarch64.whl
# RUN pip install /tmp/torch-1.10.0-cp39-cp39-linux_aarch64.whl
# Install Dependencies
# RUN pip install numpy
# Application




#FROM arm64v8/ubuntu:18.04
#FROM nvcr.io/nvidia/l4t-pytorch:r32.6.1-pth1.9-py3
#RUN apt-get install python3-setuptools
#RUN easy_install3 pip

# ENV LD_LIBRARY_PATH=/usr/lib/aarch64-linux-gnu/tegra

#COPY main.py .
#COPY init_fl.py .
#COPY model.py .
#COPY utils.py .
#RUN pip3 install flask
#ENTRYPOINT ["python3", "main.py"]
