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

## UI: Flask Webserver ##
1. Set the FLASK_ENV env. variable:
   -  `export FLASK_ENV=development` (bash) or
   -  `$env:FLASK_ENV = "development"` (powershell)
2. Navigate into the `ui/` directory and run `flask run`

## Generating Documentation ##
- Run `pdoc ./exdpn -o ./docs -d google -t ./docs/_templates --footer-text "exdpn - version 0.0.1"` from the root directory
