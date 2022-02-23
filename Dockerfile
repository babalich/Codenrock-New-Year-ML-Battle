FROM tensorflow/tensorflow:latest-gpu-jupyter

COPY ./ /app

WORKDIR /app


RUN pip3 install -r requirements.txt

CMD ["python3","run.py"]