@echo off
echo ===================================================
echo Starting Flood-AI Interactive Flask Dashboard...
echo Local URL: http://127.0.0.1:5000
echo ===================================================
python src/api/flask_app.py --host 127.0.0.1 --port 5000
pause
