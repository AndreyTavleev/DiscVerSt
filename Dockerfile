FROM ghcr.io/hombit/mesa2py:latest

COPY bin /app/bin/
COPY requirements.txt README.md setup.py /app/
WORKDIR /app

RUN pip3 install -U pip
RUN pip3 install -r requirements.txt
RUN python3 setup.py install
