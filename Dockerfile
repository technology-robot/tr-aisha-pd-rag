FROM python:3.10.12-slim

# Allow statements and log messages to immediately appear in the Cloud Run logs
ENV PYTHONUNBUFFERED 1

# Create and change to the app directory.
WORKDIR /usr/src/app

# Copy application dependency manifests to the container image.
# Copying this separately prevents re-running pip install on every code change.
COPY requirements.txt ./

# Install dependencies.
RUN pip install -r requirements.txt

# Copy local code to the container image.
COPY . ./

CMD exec uvicorn main:app --host 0.0.0.0 --port $PORT --workers $workers
