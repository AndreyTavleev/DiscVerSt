from setuptools import setup
import Vertical_structure

with open('README.txt') as file:
    long_description = file.read()

setup(
    name='vs',
    version=Vertical_structure.__version__,
    author='Andrey Tavleev',
    author_email='tavleev.as15@physics.msu.ru',
    description='Vertical structure of accretion discs',
    long_description=long_description,
    package_dir={'': 'Vertical_structure'},
    py_modules=['vs'],
    test_suite='test',
    entry_points={'console_scripts': ['example_structure = vs:main']},
    install_requires=['numpy', 'scipy', 'astropy'],
    dependency_links=['git+https://github.com/hombit/convinstab.git']
)
