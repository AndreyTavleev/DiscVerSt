FROM mesa2py

RUN apt-get update && \
    apt-get install -y texlive-full && \ 
    apt-get clean -y

COPY /requirements.txt /requirements.txt
RUN pip install -r /requirements.txt

COPY Vertical_structure /app
WORKDIR /app

RUN mkdir -pv /app/fig
VOLUME /app/fig

CMD ["python", "vs.py"]
