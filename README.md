# ImageRep

[Try it out!](https://www.imagerep.io/)

Here we introduce the 'ImageRep' method for fast phase fraction representativity estimation from a single microstructural image. This is achieved by calculating the Two-Point Correlation (TPC) function of the image, combined with a data-driven analysis of the [MicroLib](https://microlib.io/) dataset. By applying a statistical framework that utilizes both data sources, we can establish the uncertainty in the phase fraction in the image with a given confidence, **and** the image size that would be needed to meet a given target uncertainty. Further details are provided in our [paper](CITATION.cff).

If you use this ImageRep in your research, [please cite us](CITATION.cff).

## Usage:

This method can be used via the [website (imagerep.io)](https://www.imagerep.io/) or as python package - see [`example.ipynb`](example.ipynb).

<p align="center">
    <img src="https://sambasegment.blob.core.windows.net/resources/repr_repo_v2.gif">
</p>

NB: the website may run out of memory for large volumes (>1000x1000x1000) - if this happens run the method locally or contact us

## Limitations:
- **This is not the only source of uncertainty!** Other sources *i.e,* segmentation uncertainty, also contribute and may be larger
- For multi-phase materials, this method estimates the uncertainty in phase-fraction of a single (chosen) phase, counting all the others as a single phase (*i.e,* a binary microstructure)
- Not validated for for images smaller than 200x200 or 200x200x200
- Not validated for large integral ranges/features sizes (>70 px) 
- Not designed for periodic structures
- 'Length needed for target uncertainty' is an intentionally conservative estimate - retry when you have measured the larger sample to see a more accurate estimate of that uncertainty

## Local Installation Instructions

These instructions are for installing and running the method locally. They assume a UNIX enviroment (mac or linux), but adapting for Windows is straightforward. Note you will need 2 terminals, one for the frontend local server and one for the backend local server.

### Preliminaries

Install [npm](https://docs.npmjs.com/downloading-and-installing-node-js-and-npm) (ideally via a manager like [nvm](https://github.com/nvm-sh/nvm)) if you want to run the website. Clone this repo and change directory:
```
git clone https://github.com/tldr-group/Representativity && cd Representativity
```


### Install & run the backend

0. Setup a [virtual environment in Python](https://docs.python.org/3/library/venv.html) and activate it (not necessary but recommended)
1. Install the repo as a local package:

```
pip install -e .
```

**NOTE: this is all you need to do if you wish to use the method via the python package.** To run the website locally, follow the rest of the instructions.

2. With your virtual environment activated, and inside the `representativity/` directory, run

```
python -m flask --app server run
```

The server should now be running on `http://127.0.0.1:500` and listening for requests!


3. If you want to reproduce (all) the figures, you'll need `pytorch` and some additional dependencies. It may be worth using [conda](https://www.anaconda.com/) to install `pytorch` as this will interact correctly with your GPU. Run
```
pip install -r requirements_dev.txt
```


### Install & run the frontend

0. Install the JS libraries needed to build and run the frontend. Install Yarn (and npm first if needed)

```
npm install --g yarn
```

1. Build and run:

```
yarn && yarn start
```

2. Navigate to [`http://localhost:8080/`](http://localhost:8080/) (the browser should do this automatically).

## Testing Instructions

![Tests](https://github.com/tldr-group/Representativity/actions/workflows/tests.yml/badge.svg)

1. Run (with your virtual enviroment activated!)

```
python tests/tests.py
```