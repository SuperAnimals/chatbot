@echo off
echo Dang thiet lap moi truong Sandbox...
python -m venv venv
call .\venv\Scripts\activate
pip install -r requirements.txt
echo Da san sang!
pause