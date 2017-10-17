# IMP

Interactive Multiscale Projections

## Overview

The code for this project lives in different modules. Here are the most important ones:

* `operators`: This is where most of the functionality is. For example, several projection methods live in `projections.py`, and sampling methods in `random_sampling.py`.
* `model`: Here, the API for a dataset lives (in `dataset.py`), along with its rendering code (in `dataset_view.py`). There are also some files like `embeddingd.py` and `sampling.py`, which are mostly replaced by the code in `operators`.
* `widgets`: These are the Qt widgets that control the flow of the application, and the UI.

## Requirements

IMP uses [Qt5](https://www1.qt.io/download/), which should be installed manually.
Other than this, its dependencies are managed by `pip3`, Python's package manager.

## Run

First, create a virtual environment and (locally) install the required packages:

```bash
python3 -m venv venv
source ./venv/bin/activate
pip3 install -r requirements.txt
```

Then run with:
```bash
./imp.py
```
