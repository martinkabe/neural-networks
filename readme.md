## Create virtual environment

python -m venv venv
virtualenv -p python3 venv

## install one specific package

pip install --default-timeout=100 --trusted-host pypi.org --trusted-host pypi.python.org --trusted-host files.pythonhosted.org numpy

pip install --default-timeout=100 --trusted-host pypi.org --trusted-host pypi.python.org --trusted-host files.pythonhosted.org -U scikit-learn
