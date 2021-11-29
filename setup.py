from setuptools import setup, find_packages
from codecs import open
from os import path

version = '1.0.0'


here = path.abspath(path.dirname(__file__))

requires_list = []
with open(path.join(here, 'requirements.txt'), encoding='utf-8') as f:
    for line in f:
        requires_list.append(str(line))

extras = {}

all_deps = []
for group_name in extras:
    all_deps += extras[group_name]
extras['all'] = all_deps

long_description = 'TODO.'

setup(
    name='atacom',
    version=version,
    description='Acting on the Tangent Space of the Constraint Manifold',
    long_description=long_description,
    author="Puze Liu",
    author_email='puze@robot-learning.de',
    license='MIT',
    packages=[package for package in find_packages()
              if package.startswith('node')],
    zip_safe=False,
    install_requires=requires_list,
    extras_require=extras,
    classifiers=["Programming Language :: Python :: 3",
                 "License :: OSI Approved :: MIT License",
                 "Operating System :: OS Independent",
                 ]
)
