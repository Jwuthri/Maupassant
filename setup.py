from setuptools import setup, find_packages

version = "0.0.4"

setup(
    name="maupassant",
    version=version,
    license="proprietary",
    packages=find_packages(),
    include_package_data=True,
    zip_safe=False,
    install_requires=[
        "tensorflow==2.0.1",
        "tensorflow-hub==0.7.0",
        "tensorflow-text==2.0.1",
        "autocorrect==0.4.4",
        "comet-ml==3.1.1",
        "emot==2.1",
        "hdbscan==0.8.25",
        "nltk==3.4.5",
        "pandas==1.0.1",
        "pydot==1.4.1",
        "pytest==5.4.0",
        "scikit-learn==0.20.2",
        "streamlit==0.56.0",
        "umap-learn==0.3.10"
    ]
)
