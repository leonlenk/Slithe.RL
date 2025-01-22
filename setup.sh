#!/bin/bash

# Create and activate conda environment
conda create -y -n atari python=3.10
source activate atari

# Install main dependencies
pip install gymnasium[atari,other] pettingzoo==1.24.3 stable-baselines3[extra] matplotlib numpy

# Clone and install Multi-Agent-ALE
git clone https://github.com/Farama-Foundation/Multi-Agent-ALE.git
cd Multi-Agent-ALE
python setup.py build
python setup.py install
