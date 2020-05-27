from setuptools import setup, find_packages

version = "0.0.4"

setup(
    name="maupassant",
    version=version,
    license="proprietary",
    packages=find_packages(),
    include_package_data=True,
    zip_safe=False,
)
