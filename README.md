# eXdpn #
E**x**plainable **d**ata **P**etri **n**ets
Tool to mine and evaluate explainable data Petri nets using different classification techniques. 



## Development ##
1. First, if not done yet, create a virtual env using `python3 -m venv venv`, activate it (see 2.) and install all required packages using `pip install -r requirements.txt`.
2. Activate the environment using `source venv/bin/activate` (or `venv\Scripts\activate.bat` on Windows).
3. Run `python setup.py bdist_wheel` to build the project.
4. It can then be installed using `pip install dist/[wheel name].whl --force-reinstall`.


### UI: Flask Webserver ###
1. Set the FLASK_ENV env. variable `export FLASK_ENV=development` (bash) or `$env:FLASK_ENV = "development"` (powershell)
2. Navigate into the ui/ directory and run `flask run`