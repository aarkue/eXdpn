# Development Instructions #

## First Steps ##
1. First, create a virtual environment using `python3 -m venv venv`
2. Activate it with one of:
   - `source venv/bin/activate` 
   - `venv/Scripts/Activate.ps1` (powershell)
   - `venv\Scripts\activate.bat` (windows)
3. Install all packages required for development using `pip install -r requirements.txt`.

## Building and Installing the Package ##
1. Activate the environment using one of:
   - `source venv/bin/activate`
   - `venv/Scripts/Activate.ps1` (powershell)
   - `venv\Scripts\activate.bat` (windows)
2. Run `python -m build` to build the project (Install `build` via `pip install build`).
3. Install it using `pip install dist/[wheel name].whl --force-reinstall`.

## Testing ##
1. Run `python -m unittest discover -s tests -p "*.py"` from the root directory.
- Remember to always test the newest build.

## Uploading to (Test) PyPi ##
1. Update the information in the `setup.cfg` file and build the package.
2. - Run `twine upload -r testpypi dist/*` from the root directory to upload to Test PyPi.
   - Run `twine upload dist/*` from the root directory to upload to PyPi.
- Note that your account needs to be associated with the package.


## UI: Flask Webserver ##
1. Set the FLASK_ENV env. variable:
   -  `export FLASK_ENV=development` (bash) or
   -  `$env:FLASK_ENV = "development"` (powershell)
2. Navigate into the `ui/` directory and run `flask run`

## Generating Documentation ##
- Run `pdoc ./exdpn -o ./docs -d google -t ./docs/_templates --favicon https://aarkue.github.io/eXdpn/favicon.ico --footer-text "exdpn - version <current version number>"` from the root directory.
