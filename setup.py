from setuptools import setup, find_packages

setup(
    name='monkfish',
    version='0.0.1',
    packages=find_packages(),
    install_requires=[
        "ray",
        "jax",
        "fabric",
        "equinox",
        "optax",
        "google-cloud-storage"
    ],
    extras_require={
        'test': ['pytest']
    }
)

