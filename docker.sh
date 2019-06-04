#!/bin/bash
nvidia-docker run  \
  -v "$(pwd)":/workdir \
  -w '/workdir' \
  --name pr1\
  --user 1014:1014\
  -p 8888:8888 \
  kommiu/nlp\
  jupyter notebook --ip=0.0.0.0 --no-browser

