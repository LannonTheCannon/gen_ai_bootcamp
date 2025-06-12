# Installing the Python AI Stack Development Environment

*AI Bootcamp: Generative AI and LLMs for Data Scientists*

The 8-Week AI Bootcamp is constantly being updated with new AI features, libraries, and tools. To ensure that you have the latest and greatest features, I recommend that you create the development environment using the `environment_ds4b_301p_dev_installation.yml` file. This file is located in the root of the repository.

**This creates a new conda environment named `ds4b_301p_dev` with the latest versions of the libraries used in the course.**

## Step 1 - Create the Development Environment with Conda

``` bash
conda env create -f development_environment_installation_instructions/environment_ds4b_301p_dev_installation.yml
```

This creates an environment named `ds4b_301p_dev`.

You can check the environment with:

``` bash
conda env list
```

You can see what packages are installed with:

``` bash
conda list
```

## Step 2 - Activate the Environment

In VS Code, you can use the command palette (Ctrl+Shift+P) and type "Python: Select Interpreter" to select the environment you just created. 

Now you can run the code in the clinics. 

## Step 3 - Install Additional Packages

Throughout the training I will add `additional-requirements.txt` files to the repository. These files will contain additional packages that are not included in the `environment_ds4b_301p_dev_installation.yml` file.

You can install these packages with the following command:

``` bash
pip install -r additional-requirements.txt
```

This will install the packages in the current environment.

You can also install these packages manually by running the following command:

``` bash
pip install <package_name>
```
