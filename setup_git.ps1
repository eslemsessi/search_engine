Write-Host "🚀 STARTING SETUP..." -ForegroundColor Green

# 1. Create requirements.txt
"pandas`nmatplotlib`nscikit-learn`nnltk`nnumpy" | Set-Content requirements.txt -Encoding UTF8

# 2. Create .gitignore
"__pycache__/`n*.pyc`n.DS_Store`nvenv/`n*.csv" | Set-Content .gitignore -Encoding UTF8

# 3. Create README.md
"# Arxiv Search Engine`nA Python search engine project using TF-IDF and Cosine Similarity." | Set-Content README.md -Encoding UTF8

# 4. Initialize Git
git init
git branch -M main
git add .
git commit -m "Complete Project Upload"

# 5. Ask for URL
Write-Host "----------------------------------------------------" -ForegroundColor Yellow
Write-Host "⚠️  PASTE YOUR GITHUB URL BELOW AND PRESS ENTER:" -ForegroundColor Yellow
Write-Host "----------------------------------------------------" -ForegroundColor Yellow
$RepoUrl = Read-Host "Paste URL here"

# 6. Push to GitHub
if ($RepoUrl) {
    git remote remove origin 2>$null
    git remote add origin $RepoUrl
    git push -u origin main
    Write-Host "✅ SUCCESS! Check your GitHub page." -ForegroundColor Green
} else {
    Write-Host "❌ No URL provided. Files saved locally but not pushed." -ForegroundColor Red
}