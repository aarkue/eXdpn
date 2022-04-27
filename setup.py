from setuptools import find_packages, setup


setup(
    name='exdpn',
    packages=find_packages(),
    version='0.0.1',
    description='Tool to mine and evaluate explainable data Petri nets using different classification techniques.',
    setup_requires=['pm4py']
)