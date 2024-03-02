# 
FROM python:3.11

LABEL maintainer="Ladipo Ipadeola" \
      maintainer_email="ipadeolaoladipo@outlook.com"

ENV APPROOT="/app" \
    VERSION="0.1"

WORKDIR ${APPROOT}

ADD . ${APPROOT}

COPY streamlit_app/ /app

RUN pip install --no-cache-dir -r requirements.txt

ENTRYPOINT ["python", "-m", "streamlit", "run", "app.py"]
