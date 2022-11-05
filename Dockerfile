FROM ghcr.io/hombit/mesa2py:latest

COPY README.md setup.py pyproject.toml LICENSE.md requirements.txt /app/

RUN pip3 install -U pip setuptools

COPY disc_verst /app/disc_verst/

WORKDIR /app

RUN pip3 install .