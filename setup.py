from setuptools import setup, find_packages

setup(
    name="geopolitical_credit_risk",
    version="0.1.0",
    description="Geopolitical risk early warning system for European corporate credit",
    author="Jishnu",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.10",
)
