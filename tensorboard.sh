#!/bin/bash
nvidia-docker run  \
  -v "$(pwd)":/workdir \
  -w '/workdir' \
  -d \
  --name tensorboard\
  --user 1014:1014\
  -p 6116:6006 \
  kommiu/pytorch\
  tensorboard --logdir /workdir/runs

