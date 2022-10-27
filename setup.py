#!/usr/bin/env python3
from setuptools import setup, find_packages

with open('README.md') as file:
    long_description = file.read()

setup(
    name='disc-verst',
    version='1.0',
    author='Andrey Tavleev',
    author_email='tavleev.as15@physics.msu.ru',
    description='Vertical structure of accretion discs',
    long_description=long_description,
    packages=find_packages(include=['disc_verst']),
    install_requires=['numpy', 'scipy', 'matplotlib', 'astropy']
    )
