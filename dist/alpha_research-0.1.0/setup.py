from setuptools import setup, find_packages
from Cython.Build import cythonize

setup(
    name="alpha_research",
    version="0.1.0",
    description="Alpha Research Backtesting Framework",
    author="Your Name",
    packages=find_packages(include=["alpha_research", "alpha_research.*"]),
    ext_modules=cythonize(
        ["alpha_research/core.pyx"],  # compile single entrypoint
        compiler_directives={"language_level": "3"},
    ),
    include_package_data=True,  # ensure non-Python files included
    package_data={
        "alpha_research": ["*.pyi"],  # ship stub files for autocomplete
    },
    zip_safe=False,
)
