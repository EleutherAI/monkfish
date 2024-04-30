from setuptools import setup, find_packages

setup(
    name='catfish',
    version='0.0.1',
    packages=find_packages(),
    install_requires=[
        "ray",
        "jax",
        "fabric",
        "equinox",
        "optax"
    ],
    extras_require={
        'test': ['pytest']
    }
)

