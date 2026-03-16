FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt /app/

RUN pip install --no-cache-dir -r requirements.txt

COPY . /app/

EXPOSE 8000
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]

#  docker run -it api-img /bin/bash (open the shell)
# docker run --name con1 --rm -v "/home/sainath/SkimLit/Api:/app" api-img
# docker run --name con1 --rm -p 8000:8000 -v "/home/sainath/SkimLit/Api:/app" api:1