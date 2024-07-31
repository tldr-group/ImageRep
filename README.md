# Representativity

You take a micrograph of a material. You segment it, and measure the phase fractions. How sure are you that the phase fraction of the whole material is close to your measurements? 

<p align="center">
    <img src="https://sambasegment.blob.core.windows.net/resources/repr_repo.gif">
</p>

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

The server is now setup and listening for requests from our frontend!

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

Questions:
- Website name (isitrepresentative.com) (howrepresentativeismysample.com)
- model err problem
- refactoring
- cls for squares
- update readme/example notebook
- licnece: todo
