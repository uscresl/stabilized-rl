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
pip install "gym==0.22.0"

pushd trust_region_layers_cpp_deps/nlopt
cmake && make .
popd
pip install -e trust-region-layers/cpp_projection
