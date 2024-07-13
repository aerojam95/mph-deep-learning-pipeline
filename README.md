# mph-deep-learning-pipeline
A data pipeline that represents point clouds as multi-parameter persistent homology landscapes for deep learning models

## Python3 virtual environment

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

## Installations

### Xming

If the user is a Windows Subsystem for Linux (WSL) user they will need to install [Xming](https://sourceforge.net/projects/xming/) in order to have a viewer to visualise the persistence diagrams generates by [Rivet](https://github.com/rivetTDA/rivet/tree/master?tab=readme-ov-file).

### Rivet

Full instructions for installation of the Rivet appication is provided by the following [documentation](https://rivet.readthedocs.io/en/latest/installing.html) when building the C++ application from the master branch.

### Hera

In order to for PyRivet to do the matching distance computation or bottleneck distances it requires the `bottleneck_dist` application installed from [Hera](https://bitbucket.org/xoltar/hera/src/master/). Hera C++ code needs to be built to be used by the PyRivet package:

1. From the repoisotries root navigate to the root of the Hera submodule and checkout the `c-api` branch of the code as this is the branch needed for PyRivet:

```
cd src/hera
git checkout c-api
```

2. Navigate to `geom_bottleneck` directory

```
cd geom_bottleneck
```

3. Build the `bottleneck_dist` code

```
mkdir build
cd build
cmake ..
make
```

4. Return to root of Hera submodule and navigate to wasserstein directory:

```
cd geom_bottleneck/wasserstein
```

5. Build the `wasserstein_dist` code:

```
mkdir build
cd build
cmake ..
make
```


### PyRivet



### Multi-parameter persistence landscapes



### Ripser

If the user has not used the virtual envionrment then the following python packages need to installed from Pypi in order to do single parameter persistence computations:

```
pip3 install Cython
pip3 install Rpiser
```