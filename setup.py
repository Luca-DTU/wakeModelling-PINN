from setuptools import setup, find_packages

setup(
    name='PINN',
    version='0.0.0',
    packages=find_packages(include=['src', 'src.*'])
)