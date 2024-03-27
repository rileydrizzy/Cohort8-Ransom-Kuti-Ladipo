# 
FROM python:3.11-slim

RUN apt-get update && apt-get install -y libglib2.0-0 && apt-get install -y libgl1-mesa-glx

LABEL maintainer="Ladipo Ipadeola" \
      maintainer_email="ipadeolaoladipo@outlook.com"

ENV APPROOT="/usr/src/app" \
    VERSION="0.1"

WORKDIR ${APPROOT}

COPY . ${APPROOT}

RUN pip install --no-cache-dir -r requirements.txt

ENTRYPOINT ["python", "-m", "streamlit", "run", "app.py"]
