FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt /app/

RUN pip install --no-cache-dir -r requirements.txt

COPY . /app/

CMD ["python", "-u", "main.py"]

#  docker run -it api-img /bin/bash (open the shell)
# docker run --name con1 --rm -v "/home/sainath/SkimLit/Api:/app" api-img