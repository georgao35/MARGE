from setuptools import setup, find_packages

setup(
    name="backward",
    version="1.0",
    packages=find_packages(
        exclude=(
            "utils/",
            "data/",
            "custom_datasets/",
            "trl/"
        )
    ),
)
