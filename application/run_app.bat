@echo off
echo Starting Pollen Classification Web Application...
echo.

:: Check if virtual environment exists
if not exist "venv" (
    echo Creating virtual environment...
    python -m venv venv
    echo.
)

:: Activate virtual environment
echo Activating virtual environment...
call venv\Scripts\activate.bat
echo.

:: Install requirements if needed
echo Installing dependencies...
pip install -r requirements.txt
echo.

:: Start the Flask application
echo Starting Flask application...
echo Open your browser and go to: http://localhost:5000
echo Press Ctrl+C to stop the server
echo.
python app.py

pause
