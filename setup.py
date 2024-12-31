from setuptools import setup, find_packages

setup(
    name="tldw",
    version="0.1.0",
    author="Robert Musser",
    author_email="contact@tldwproject.com",
    description="A short description of your project",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/rmusser01/tldw",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "requests",
    ],
    classifiers=[  # Classifiers for PyPI (optional)
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache 2.0 License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.9",
)

# Dev
pytest-asyncio