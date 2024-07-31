# Representativity

![Tests](https://github.com/tldr-group/Representativity/actions/workflows/tests.yml/badge.svg)

[Try it out!](https://representative9984.z33.web.core.windows.net/)

You take a micrograph of a material. You segment it, and measure the phase fractions. How sure are you that the phase fraction of the whole material is close to your measurements?  We define 'representativity' as
> A microstructure is $(c, d)$-property representative if the measured value of the microstructural property deviates by no more than $d\%$ from the bulk material property, with at least $c\%$ confidence. For example, if $(c,d)=(95,3)$, and the property is phase-fraction, this means we can be $95\%$ confident that the measured phase-fraction is within $3\%$ of the bulk material phase-fraction. 

We introduce the 'ImageRep' model for performing fast phase-fraction representativity estimation from a single microstructural image. This is achieved by estimating the Two-Point Correlation (TPC) function of the image via an FFT. From the TPC the 'Integral Range' can be directly determined - the Integral Range has previously been determined using (slow) statistical methods. We then represent the image as binary squares of length 'Integral Range' which are samples from a Bernoulli distribution with a probability determined by the measured phase fraction. From this we can establish the uncertainty in the phase fraction in the image, **and** the image size that would be needed to meet a given target uncertainty.

## TODO:
- Website name (imagerep.com) (isitrepresentative.com) (howrepresentativeismysample.com)
- fix zoom
- fix 3D
- cls for squares, fix other tests
- update readme/example notebook (add static figure @ top of readme)
- licence: todo
- 

## Website

<p align="center">
    <img src="https://sambasegment.blob.core.windows.net/resources/repr_repo_v2.gif">
</p>

NB: the website may run out of memory for large volumes (>1000x1000x1000) - if this happens run the model locally or contact us

## Local Installation Instructions

These instructions are for installing and running the model locally. They assume a UNIX enviroment (mac or linux), but adapting for Windows is straightforward. Note you will need 2 terminals, one for the frontend local server and one for the backend local server.

### Preliminaries

Install [npm](https://docs.npmjs.com/downloading-and-installing-node-js-and-npm) (ideally via a manager like [nvm](https://github.com/nvm-sh/nvm)). Clone this repo and change directory:
```
git clone https://github.com/tldr-group/Representativity && cd Representativity
```


### Install & run the backend

0. Setup a [virtual environment in Python](https://docs.python.org/3/library/venv.html) and activate it (not necessary but recommended)
1. Install the repo as a local package:

```
pip install -e .
```

2. With your virtual environment activated, and inside the `representativity/` directory, run

```
python -m flask --app server run
```

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

1. Run (with your virtual enviroment activated!)

```
python tests/tests.py
```