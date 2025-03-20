FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt /app/requirements.txt
RUN [ "pip", "install", "-r", "/app/requirements.txt" ]
COPY . /app
EXPOSE 8000
RUN [ "python", "/app/boostrap_db.py" ]
ENTRYPOINT [ "chainlit", "run", "/app/main.py" ]