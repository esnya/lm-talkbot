from setuptools import find_packages, setup

import talkbot

setup(
    name="talkbot",
    version=talkbot.__version__,
    packages=find_packages(include=["talkbot", "talkbot.*"]),
    python_requires=">=3.10,<3.11",
)
