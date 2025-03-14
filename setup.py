"""
Setup configuration for the Bangladesh Development Simulation Model.
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="bd_macro",
    version="0.1.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="A comprehensive simulation model for Bangladesh's development trajectory",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/bd_macro",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Information Analysis",
    ],
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.21.0",
        "pandas>=1.3.0",
        "scipy>=1.7.0",
        "matplotlib>=3.4.0",
        "seaborn>=0.11.0",
        "pyyaml>=5.4.0",
        "scikit-learn>=0.24.0",
    ],
    extras_require={
        "full": [
            "mesa>=1.1.1",
            "pysd>=3.12.0",
            "folium>=0.14.0",
            "plotly>=5.13.0",
            "dash>=2.9.0",
            "dash-bootstrap-components>=1.4.0",
        ],
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=4.0.0",
            "pytest-mock>=3.10.0",
            "flake8>=6.0.0",
            "black>=23.0.0",
            "isort>=5.12.0",
            "mypy>=1.0.0",
            "sphinx>=6.0.0",
            "sphinx-rtd-theme>=1.3.0",
            "sphinx-autodoc-typehints>=1.24.0",
            "ipython>=8.0.0",
            "jupyter>=1.0.0",
            "jupyterlab>=4.0.0",
            "types-setuptools>=68.0.0",
            "types-requests>=2.31.0",
            "types-PyYAML>=6.0.0",
            "pre-commit>=3.3.0",
            "coverage>=7.0.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "bd-sim=bd_macro.__main__:main",
            "bd-test=bd_macro.tests.run_tests:main",
            "bd-validate=bd_macro.utils.validate:main",
        ],
    },
    package_data={
        "bd_macro": [
            "config/*.yaml",
            "data/*.csv",
            "data/*.json",
            "templates/*.html",
        ],
    },
) 