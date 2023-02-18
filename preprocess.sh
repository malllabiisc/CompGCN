#!/bin/bash
mkdir log
mkdir checkpoints
mkdir data

unzip data_compressed/codex-l.zip -d data
unzip data_compressed/codex-m.zip -d data
unzip data_compressed/codex-s.zip -d data
unzip data_compressed/codex-xxs.zip -d data
unzip data_compressed/FB15k-237.zip -d data
unzip data_compressed/WN18RR.zip -d data
