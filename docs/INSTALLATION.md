# Installation

Welcome to the Installation Guide of `baselines`!

This installation guide has been tested on **Ubuntu 16.04 LTS** with **Python 3.6**.

## 1. Clone Git Repository to Local

First, clone this repository locally. Then, change directory to `baselines/`.

```
git clone https://github.com/seungjaeryanlee/baselines.git
cd baselines
```

## 2. Install Linux Packages

We use OpenCV, which requires some additional library. Install them with the following `apt` command:

```
sudo apt-get update
sudo apt-get install libsm6 libxrender-dev
```

## 3. Install Python Packages

Then, install necessary Python packages using the provided `requirements.txt`.

```
pip install -r requirements.txt
```

## 4. Test Installation

To verify that `baselines` has been installed correctly, try running `main.py`.

```
./main.py
```

## Problems?

Feel free to leave an issue if this procedure did not work!
