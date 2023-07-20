import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name='ReinForceMate',
    version='0.1.0',
    author=['Silvio Baratto', 'Valeria Insogna', 'Thomas Verardo'],
    author_email='silvio.baratto22@gmail.com',
    description='A simplified version of chess game via RL algorithm',
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/valinsogna/ReinForceMate/",
    classifiers=[
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        "Operating System :: OS Independent", 
        'Programming Language :: Python :: 3',
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Mathematics",
    ],
    packages=setuptools.find_packages(),
    python_requires='>=3.6',
    install_requires=[
        "torch",
        "tqdm",
        "numpy",
        "chess"
    ],
)

