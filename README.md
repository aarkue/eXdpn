# eXdpn #
E**x**plainable **d**ata **P**etri **n**ets
Tool to mine and evaluate explainable data Petri nets using different classification techniques. 

## Docker Deployment ##
The project can be run with Docker.
The easiest way to do so is with the included `docker-compose.yml` file.

### Using Docker Compose ###
1. `docker-compose up`

To force re-creation of the container add the `--build` flag to the `docker-compose` command: `docker-compose up --build`.
The web ui will then be available on port 8080.

### Building the Docker Container ###
1. `docker build .`
2. `docker run -p 8080:5000 <container id>`

The web ui will then be available on port 8080.


## Development ##
1. First, if not done yet, create a virtual env using `python3 -m venv venv`, activate it (see 2.) and install all required packages using `pip install -r requirements.txt`.
2. Activate the environment using `source venv/bin/activate` (or `venv\Scripts\activate.bat` on Windows).
3. Run `python setup.py bdist_wheel` to build the project.
4. It can then be installed using `pip install dist/[wheel name].whl --force-reinstall`.


### UI: Flask Webserver ###
1. Set the FLASK_ENV env. variable `export FLASK_ENV=development` (bash) or `$env:FLASK_ENV = "development"` (powershell)
2. Navigate into the ui/ directory and run `flask run`

### Generating Documentation ###
- Run `pdoc ./exdpn -o ./docs -d google -t ./docs/_templates --footer-text "exdpn - version 0.0.1"`


