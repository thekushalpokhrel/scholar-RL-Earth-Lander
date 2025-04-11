from setuptools import setup, find_packages

setup(
    name="earth_lander",
    version="0.1",
    packages=find_packages(),
    install_requires=[...],
    package_dir={'': '.'},  # Add this line
)