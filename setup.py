from setuptools import setup, find_packages

setup(
    name="ml_evaluator",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "scikit-learn",
        "pandas",
        "numpy"
    ],
    author="Ruslan Mamedov",
    description="Modular ML evaluator for classification and regression models",
)

