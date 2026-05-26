# Quick setup for Document Intelligence (Windows PowerShell)
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords')"
Write-Host "Start API: python -m api.main"
Write-Host "Start UI:  cd frontend; npm install; npm run dev"
