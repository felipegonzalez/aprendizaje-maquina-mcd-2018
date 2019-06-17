FROM rocker/verse:3.5.1


RUN cat /etc/os-release 

RUN apt-get update \ 
    && apt-get install -y libudunits2-dev 

RUN \
  apt-get update && \
  apt-get install -y python python-dev python-pip 

RUN \
  pip install --upgrade tensorflow && \
  pip install --upgrade keras scipy h5py pyyaml requests Pillow

RUN r -e 'devtools::install_github("rstudio/reticulate")'
RUN r -e 'devtools::install_github("rstudio/tensorflow")'
RUN r -e 'devtools::install_github("rstudio/keras")'


RUN install2.r --error \
    glmnet