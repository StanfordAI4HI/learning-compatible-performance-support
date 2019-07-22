import setuptools

# with open("README.md", "r") as fh:
#    long_description = fh.read()


# Make sure custom gym and baselines are already installed.
setuptools.setup(
    name="perflearn",
    version="0.0.1",
    author="Jonathan Bragg",
    author_email="jbragg@cs.stanford.edu",
    description="Shared autonomy methods for high performance and learning",
    # long_description=long_description,
    # long_description_content_type="text/markdown",
    url="https://github.com/jbragg/perflearn",
    packages=setuptools.find_packages(),
    install_requires=[
        'tensorflow>=1.11',
        'baselines @ git+https://github.com/jbragg/baselines.git@universe-turk-assist',
    ],
    # classifiers=[
    #    "Programming Language :: Python :: 3",
    #    "License :: OSI Approved :: MIT License",
    #    "Operating System :: OS Independent",
    # ],
)
