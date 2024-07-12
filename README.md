# mph-deep-learning-pipeline
A data pipeline that represents point clouds as multi-parameter persistent homology landscapes for deep learning models

### Python3 virtual environment

Before using the code it is best to setup and start a Python virtual environment in order to avoid potential package clashes using the [requirements](src/requirements.txt) file:

```
# Navigate into the data project directory

# Create a virtual environment
python3 -m venv <env-name>

# Activate virtual environment
source <env-name>/bin/activate

# Install dependencies for code
pip3 install -r requirements.txt

# When finished with virtual environment
deactivate
```