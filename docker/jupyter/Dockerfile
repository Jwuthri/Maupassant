FROM nvidia/cuda:10.1-cudnn7-devel-ubuntu18.04
MAINTAINER "Durgesh Mankekar"

RUN apt-get update && apt-get install -y wget ca-certificates \
    git curl vim python3-dev python3-pip
    # libfreetype6-dev libpng12-dev

RUN pip3 install --upgrade pip
RUN pip3 install tensorflow-gpu
RUN pip3 install numpy pandas sklearn matplotlib seaborn jupyter

RUN ["mkdir", "notebooks"]
# COPY jupyter_notebook_config.py /root/.jupyter/
COPY run_jupyter.sh /

# Jupyter and Tensorboard ports
EXPOSE 8888 6006

# Store notebooks in this mounted directory
VOLUME /notebooks

CMD ["/run_jupyter.sh"]
