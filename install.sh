#!/bin/bash -xe

if [ ! -d venv_install ]; then
  python3 -m venv venv_install
fi
source venv_install/bin/activate

echo "Using python: $(which python)"
pip install setuptools
pip install -e .
pip install -e ./doexp
pip install -e ./tianshou
pip install -e ./metaworld
pip install scipy
pip install shimmy
pip install envpool
pip install "cython<3"
pip install "gym>=0.22.0,<0.26.0"
