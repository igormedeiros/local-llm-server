source venv/bin/activate
git pull
uvicorn main:app --host 0.0.0.0 --port 1234
