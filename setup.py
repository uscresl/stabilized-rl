from setuptools import setup
from setuptools import find_packages

setup(
    name='stabilized-rl',
    package_dir={'': 'src'},
    packages=find_packages(where='src'),
    )
