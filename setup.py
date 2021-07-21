import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="angulargrad",
    version="0.0.1",
    author='S.K. Roy, M.E. Paoletti, J.M. Haut, S.R. Dubey, P. Kar, A. Plaza and B.B. Chaudhuri',
    description='Angulargrad - modified for setup by Ivan Svogor',
    url="https://github.com/isvogor-foi/AngularGrad",
    package_dir={"": "angulargrad"},
    packages=setuptools.find_packages(where="angulargrad"),
    python_requires=">=3.6",
)
