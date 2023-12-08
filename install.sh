#!/bin/bash -xe

# If .venv doesn't exist, create it
if [ ! -d .venv ]; then
  python3 -m venv .venv
fi

# If not in a virtual env, activate .venv
if [[ "$VIRTUAL_ENV" == "" ]]
then
  source .venv/bin/activate
fi

echo "Using python: $(which python)"
pip install --upgrade pip
poetry install
pip install setuptools
#pip install -e .
#pip install -e ./doexp
#pip install -e ./tianshou
#pip install -e ./metaworld
#pip install scipy
#pip install shimmy
#pip install envpool
#pip install "cython<3"
#pip install "gym==0.22.0"

pushd trust_region_layers_cpp_deps/nlopt
cmake . && make .
popd
pip install -e trust-region-layers/cpp_projection
