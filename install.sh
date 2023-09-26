#!/bin/bash -xe

if [ ! -d venv_install ]; then
  python3 -m venv venv_install
fi
source venv_install/bin/activate

pip install -e .
pip install git+https://github.com/krzentner/doexp.git@main
pip install -e ./tianshou
pip install -e ./metaworld
pip install scipy
pip install shimmy
pip install envpool
pip install "gym<0.26.0"
pip install "cython<3"
