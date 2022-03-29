from setuptools import find_packages, setup


with open("README.md") as readme_file:
    readme = readme_file.read()

setup(
    name="hockey_rink",
    version="0.1.6",
    description="A Python library for plotting hockey rinks with Matplotlib.",
    long_description_content_type="text/markdown",
    long_description=readme,
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Framework :: Matplotlib",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        "Programming Language :: Python :: 3.6",
    ],
    url="https://github.com/the-bucketless/hockey_rink",
    author="The Bucketless",
    author_email="thebucketless@protonmail.com",
    license="GNU General Public License v3 (GPLv3)",
    packages=find_packages(),
    install_requires=["matplotlib", "numpy", "scipy"],
    zip_safe=False,
)
