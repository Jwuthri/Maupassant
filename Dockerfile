FROM python:3.7.3

RUN mkdir -p /maupassant/src
WORKDIR /maupassant/src

COPY requirements.txt setup.py ./
COPY maupassant ./maupassant/

RUN pip install -r requirements.txt

ENV PYTHONPATH "${PYTHONPATH}:/maupassant/src"
