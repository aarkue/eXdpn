from setuptools import find_packages, setup


setup(
    name='exdpn',
    packages=find_packages(),
    version='0.0.1',
    description='Tool to mine and evaluate explainable data Petri nets using different classification techniques.',
    install_requires=[
        'pm4py==2.2.20.1',
        'scikit-learn==1.0.2',
        'shap==0.40.0',
        'pandas==1.3.5',
    ],
    test_suite='tests'
    
)