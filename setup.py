# setup.py
from setuptools import setup, find_packages
from pathlib import Path

README = Path(__file__).with_name("README.md").read_text(encoding="utf-8")

setup(
    name="bayesrvat",
    version="0.1.0",
    description="Bayesian rare variant association test",
    long_description=README,
    long_description_content_type="text/markdown",
    author="Antonio Nappi & Francesco Paolo Casale",
    license="MIT",
    packages=find_packages(include=["bayesrvat", "bayesrvat.*"]),
    python_requires=">=3.9",
    install_requires=[
        "numpy",
        "pandas",
        "scipy",
    ],
    include_package_data=True,
)
