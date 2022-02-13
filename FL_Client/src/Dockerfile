
FROM nvcr.io/nvidia/l4t-pytorch:r32.5.0-pth1.7-py3

RUN pip3 install flask
COPY mains.py .
COPY init_fl.py .
COPY model.py .
COPY utils.py .

CMD [ "python3", "mains.py" ]

