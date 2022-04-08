# https://pythonspeed.com/articles/activate-conda-dockerfile
FROM continuumio/miniconda3

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
RUN apt-get update && apt-get -y install build-essential

# create a working directory for docker
WORKDIR /app

# create environment
COPY docker-env.yaml .
RUN conda env create -f docker-env.yaml
#RUN conda config --add channels david_baddeley
#RUN conda create -n pyme python=3.7 pyme-depends python-microscopy

# make run commands use the new environment
RUN echo "conda activate pyme" >> ~/.bashrc
SHELL ["/bin/bash", "--login", "-c"]

# ===> Ideally, swap these for proper conda packages

# clone python-microscopy and pymecompress into workdir
# RUN git clone https://github.com/python-microscopy/python-microscopy.git
RUN git clone https://github.com/zacsimile/python-microscopy.git
RUN git clone https://github.com/python-microscopy/pymecompress.git

# install pymecompress
RUN cd pymecompress && python setup.py install

# install PYME development version
RUN cd python-microscopy && git checkout dockerize && python setup.py install

# <===

# make sure it worked, Docker build will crash here if not
# in theory, we could and should eliminate this (it adds
# unnecessary layers to the Docker image)
# SHELL ["conda", "run", "-n", "pyme", "/bin/bash", "-c"]
RUN echo "test for pyme installation"
RUN python -c "import PYME"

# make dataserver root
RUN mkdir ~/.PYME
RUN echo $'dataserver-root: "/"\ndataserver-filter: ""' >> ~/.PYME/config.yaml

# launch cluster
ENTRYPOINT ["conda", "run", "--no-capture-output", "-n", "pyme", "PYMECluster"]

# A shell entrypoint may be safer long term...
# COPY server-entrypoint.sh .
# RUN chmod +x server-entrypoint.sh
# ENTRYPOINT ["./server-entrypoint.sh"]
