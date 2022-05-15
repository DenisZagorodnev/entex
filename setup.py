import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="entex",
    version="1.0.0",
    author="Denis Zagorodnev",
    author_email="denis.zagorodnev@gmail.com",
    description="Text Entity Extractor",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/DenisZagorodnev/entex",
    project_urls={
        "Bug Tracker": "https://github.com/DenisZagorodnev/entex/issues",
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
    package_dir={"": "src2"},
    packages=setuptools.find_packages(where="src"),
    python_requires=">=3.6",
    install_requires=[
        "numpy",
        "sklearn",
        "nltk",
        "pymorphy2",
        "natasha",
        "pandas",
        "emosent-py"
    ],
)