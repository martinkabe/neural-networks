## Create virtual environment

python -m venv venv
virtualenv -p python3 venv

## install one specific package

pip install --default-timeout=100 --trusted-host pypi.org --trusted-host pypi.python.org --trusted-host files.pythonhosted.org numpy

pip install --default-timeout=100 --trusted-host pypi.org --trusted-host pypi.python.org --trusted-host files.pythonhosted.org -U scikit-learn

<!-- autopep8 automatically formats Python code to conform to the PEP 8 style guide. It uses the pycodestyle utility to determine what parts of the code needs to be formatted. autopep8 is capable of fixing most of the formatting issues that can be reported by pycodestyle. -->
<!-- https://pypi.org/project/autopep8/ -->

pip install --default-timeout=100 --trusted-host pypi.org --trusted-host pypi.python.org --trusted-host files.pythonhosted.org -U autopep8

## install all packages from requirements.txt

pip install -r requirements.txt
