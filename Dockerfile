FROM ghcr.io/hombit/mesa2py:latest

COPY requirements.txt README.md setup.py /app/

RUN pip3 install -U pip
RUN pip3 install -r requirements.txt

COPY bin /app/bin/

WORKDIR /app


RUN python3 setup.py install
