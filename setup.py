
from setuptools import setup, find_packages

setup(
    name='gcgridobj',
    version='0.2',
    packages=find_packages(exclude=['tests*']),
    license='MIT',
    description='A package which provides grid objects for working with GEOS-Chem data',
    long_description=open('README.txt').read(),
    install_requires=['numpy'],
    url='https://github.com/sdeastham/gcgridobj',
    author='Sebastian David Eastham',
    author_email='seastham@mit.edu'
)
