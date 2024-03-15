from setuptools import setup

install_requires = [
    "hydra-core",
    "jupyterlab",
    "nltk",
    "numpy",
    "pandas",
    "pytest",
    "scipy",
    "seaborn",
    "torch",
    "wandb",
]


setup(
    name="art_ngrams",
    install_requires=install_requires,
    version="0.1",
    scripts=[],
    packages=["art_ngrams"],
)
