# https://pythonspeed.com/articles/activate-conda-dockerfile
# FROM continuumio/miniconda3
FROM nvidia/cuda:11.4.0-devel-ubuntu20.04

# create a working directory for docker
WORKDIR /app

# SHELL ["/bin/bash", "--login", "-c"]

# >>> https://hub.docker.com/r/continuumio/miniconda3/dockerfile
ENV LANG=C.UTF-8 LC_ALL=C.UTF-8
ENV PATH /opt/conda/bin:$PATH

RUN apt-get update --fix-missing && \
    apt-get install -y wget bzip2 ca-certificates curl git build-essential && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/* 

RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-py39_4.11.0-Linux-x86_64.sh -O ~/miniconda.sh && \
    /bin/bash ~/miniconda.sh -b -p /opt/conda && \
    rm ~/miniconda.sh && \
    /opt/conda/bin/conda clean -tipsy && \
    ln -s /opt/conda/etc/profile.d/conda.sh /etc/profile.d/conda.sh && \
    echo ". /opt/conda/etc/profile.d/conda.sh" >> ~/.bashrc 
    # && \ echo "conda activate base" >> ~/.bashrc

# ENV TINI_VERSION v0.16.1
# ADD https://github.com/krallin/tini/releases/download/${TINI_VERSION}/tini /usr/bin/tini
# RUN chmod +x /usr/bin/tini

# ENTRYPOINT [ "/usr/bin/tini", "--" ]
# CMD [ "/bin/bash" ]
# <<<

# SHELL [ "/usr/bin/tini", "--", "/bin/bash", "--login", "-c"]

# pull additional dependencies not included in continuumio/miniconda3
# in theory, we could eliminate these if we get conda packaging right
# - build-essential for python setup.py install
# - freeglut3-dev for pyopengl
# - libgtk2.0-0 for wx (NOTE: we also need the glib package 
#   in docker-env.yaml for wx)
# RUN apt-get update && \
#     apt-get -y install build-essential && \
#     apt-get -y install freeglut3-dev && \
#     apt-get -y install libgtk2.0-0
# RUN apt-get update && apt-get -y install build-essential

# create environment
COPY docker-env.yaml .
RUN conda env create -f docker-env.yaml
#RUN conda config --add channels david_baddeley
#RUN conda create -n pyme python=3.7 pyme-depends python-microscopy

# make run commands use the new environment
RUN echo "conda activate pyme" >> ~/.bashrc
# RUN conda activate pyme
SHELL ["conda", "run", "-n", "pyme", "/bin/bash", "--login", "-c"]

RUN pip install pycuda

# ===> Ideally, swap these for proper conda builds

# clone python-microscopy and pymecompress into workdir
# RUN git clone https://github.com/python-microscopy/python-microscopy.git
RUN git clone https://github.com/zacsimile/python-microscopy.git
RUN git clone https://github.com/python-microscopy/pymecompress.git
# RUN git clone https://github.com/inducer/pycuda.git
RUN git clone https://github.com/python-microscopy/pyme-warp-drive.git

# install pymecompress
RUN cd pymecompress && python setup.py develop

# install PYME development version
RUN cd python-microscopy && git checkout dockerize && python setup.py develop

# install pycuda
# RUN cd pycuda && python setup.py develop

# install pyme-warp-drive
RUN cd pyme-warp-drive && python setup.py develop

# make sure it worked, Docker build will crash here if not
# in theory, we could and should eliminate this (it adds
# unnecessary layers to the Docker image)
# RUN echo "test for pyme installation"
# RUN python -c "import PYME"

# make dataserver root
RUN mkdir ~/.PYME
RUN echo $'dataserver-root: "/"\ndataserver-filter: ""' >> ~/.PYME/config.yaml

# launch cluster
ENTRYPOINT ["conda", "run", "--no-capture-output", "-n", "pyme", "PYMECluster"]

# A shell entrypoint may be safer long term...
# COPY server-entrypoint.sh .
# RUN chmod +x server-entrypoint.sh
# ENTRYPOINT ["./server-entrypoint.sh"]
