from setuptools import setup, find_packages

version = "0.0.8"

setup(
    name="maupassant",
    version=version,
    license="proprietary",
    packages=find_packages(),
    include_package_data=True,
    zip_safe=False,
    install_requires=[
        "tensorflow-gpu==2.2.0",
        "tensorflow-hub==0.7.0",
        "tensorflow-text==2.2.0",
        "autocorrect==0.4.4",
        "comet-ml==3.1.1",
        "emot==2.1",
        "nltk==3.4.5",
        "pandas==1.0.1",
        "jupyter==1.0.0",
        "notebook==6.0.3"
        "pydot==1.4.1",
        "pytest==5.4.0",
        "hdbscan==0.8.25",
        "scikit-learn==0.20.2",
        "streamlit==0.56.0",
        "umap-learn==0.3.10",
        "contractions==0.0.24",
        "python-dotenv==0.12.0",
        "ipython==7.13.0"
    ]
)
