#!/usr/bin/env python3
from setuptools import setup
import bin

with open('README.md') as file:
    long_description = file.read()

setup(
    name='Vertical-structure-of-accretion-discs',
    version=bin.__version__,
    author='Andrey Tavleev',
    author_email='tavleev.as15@physics.msu.ru',
    description='Vertical structure of accretion discs',
    long_description=long_description,
    package_dir={'': 'bin'},
    py_modules=['vs', 'mesa_vs', 'plots_vs']
    )
