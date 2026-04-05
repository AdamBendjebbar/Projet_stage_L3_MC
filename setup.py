from setuptools import setup, find_packages

setup(
    name="janiParser",
    version="0.1.0",
    author="Jiahua Li, Zeyu Tao",
    description="A tool for parsing a subset of JANI models, " \
        "solving them primarily with the Marmote library, with optional support for MDPToolbox, " \
        "and performing benchmarking.",
    url="https://github.com/bellkilo/pAI2D_mdp_languages",
    packages=find_packages(),
    install_requires=[
        "pymdptoolbox",
        "numpy",
        "scipy"
    ],
    license="MIT License"
)