FROM pytorch/pytorch:2.0.0-cuda11.8-cudnn8-runtime

WORKDIR /app

COPY requirements.docker.txt /app/requirements.docker.txt

RUN pip install --no-cache-dir -r /app/requirements.docker.txt

COPY . /app

CMD ["python", "start_batch_server.py"]