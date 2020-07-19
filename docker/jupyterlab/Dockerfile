FROM python:3.7.3

RUN mkdir -p /maupassant/src
WORKDIR /maupassant/src

COPY requirements.txt setup.py ./
COPY maupassant ./maupassant/

RUN apt-get update
RUN pip install -r requirements.txt
RUN pip install jupyterlab
RUN python -m nltk.downloader punkt

EXPOSE 8888

ENV PYTHONPATH "${PYTHONPATH}:/maupassant/src"
