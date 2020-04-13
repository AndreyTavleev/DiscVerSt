FROM mesa2py

RUN apt-get update && \
    apt-get install -y texlive-latex-recommended texlive-pictures texlive-latex-extra texlive-fonts-recommended texlive-lang-cyrillic dvipng cm-super && \ 
    apt-get clean -y

COPY /requirements.txt /requirements.txt
RUN pip install -r /requirements.txt

COPY bin /app
WORKDIR /app

RUN mkdir -pv /app/fig
VOLUME /app/fig

CMD ["python", "MAIN.py"]
