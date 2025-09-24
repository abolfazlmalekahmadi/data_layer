from setuptools import setup, find_packages

setup(
    name="ryn",   # Your project/package name
    version="0.1.0",
    packages=find_packages(include=["data", "data.*"]),
    install_requires=[
        "pydantic",
        "boto3",
        "pandas",
        "datasets",
        "openml",
        "kagglehub",
        "fastapi",
        "opencv-python",
        "opencv-python-headless",
        "polars"
    ],
    python_requires=">=3.8",
    description="SDK for data ingestion and platform utilities.",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
)