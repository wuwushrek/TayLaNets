from setuptools import setup, find_packages

with open("README.md", 'r') as f:
    long_description = f.read()

def _parse_requirements(requirements_txt_path):
  with open(requirements_txt_path) as fp:
    return fp.read().splitlines()

setup(
   name='taylanets',
   version='0.0.1',
   description='A module for training deep neural networks, based on Taylor Lagrange expansions with learned mean-value point, to provide accurate solutions of ODEs and learn unkown dynamics from sampled trajectories',
   license="GNU 3.0",
   long_description=long_description,
   author='Franck Djeumou and Cyrus Neary',
   author_email='fdjeumou@utexas.edu, cneary@utexas.edu',
   url="https://github.com/wuwushrek/TayLaNets.git",
   packages=find_packages(),
   # packages=['taylanets'],
   # package_dir={'taylanets': 'taylanets/'},
   install_requires=_parse_requirements('requirements.txt'),
)
