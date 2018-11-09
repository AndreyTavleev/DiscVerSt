FROM mesa2py

COPY /requirements.txt /requirements.txt
RUN pip install -r /requirements.txt

COPY Vertical_structure /app
WORKDIR /app

CMD ["python", "vs.py"]
