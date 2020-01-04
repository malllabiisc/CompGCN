#!/bin/bash
mkdir log
mkdir checkpoints
mkdir data

unzip data_compressed/FB15k-237.zip -d data
unzip data_compressed/WN18RR.zip -d data