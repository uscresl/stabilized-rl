#!/bin/bash -xe

#python3 -m venv venv
#source venv/bin/activate
pip install -e .
pip install -e ./tianshou
pip install -e ./metaworld
pip install shimmy
pip install envpool
pip install "gym<0.26.0"
pip install "cython<3"
