from setuptools import setup, find_packages

setup(
    name="hazardgan",
    version="0.01",
    author="Alison Peard",
    author_email="alison.peard@gmail.com",
    description="GAN for compound coastal hazard generation.",
    license="GNU v3",
    url="https://github.com/alisonpeard/cycloneGAN",

    packages=find_packages(),
    install_requires=[
        "tensorflow-macos",
        "tensorflow-metal",
        "scipy",
        "pandas",
        "numpy",
        "geopandas",
        "matplotlib"
    ]
)
