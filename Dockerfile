FROM python:3

ADD Main.py /

RUN pip install -r requirements.txt

CMD [ "python", "./Main.py" ]