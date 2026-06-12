# PowerShell Script to Push only HW6 directory to stratup50 repository

Write-Host "=== Git Push HW6 Automation ===" -ForegroundColor Cyan

# 1. Check if git is installed
if (!(Get-Command git -ErrorAction SilentlyContinue)) {
    Write-Error "Git is not installed or not in PATH. Please install Git and try again."
    exit
}

# 2. Initialize new Git repository in HW6 if not exists
if (!(Test-Path .git)) {
    Write-Host "Initializing new Git repository in HW6..." -ForegroundColor Yellow
    git init
    git branch -M main
}

# 3. Add remote origin if not exists, or set URL
$remoteCheck = git remote get-url origin 2>$null
if ($null -eq $remoteCheck) {
    Write-Host "Adding remote origin URL..." -ForegroundColor Yellow
    git remote add origin https://github.com/gogogo137-cmyk/stratup50.git
} else {
    Write-Host "Setting remote origin URL..." -ForegroundColor Yellow
    git remote set-url origin https://github.com/gogogo137-cmyk/stratup50.git
}

# 4. Stage and commit files
Write-Host "Staging HW6 files..." -ForegroundColor Yellow
git add -A
git commit -m "Upload HW6 50 Startups CRISP-DM project"

# 5. Push to main branch
Write-Host "Pushing to GitHub..." -ForegroundColor Yellow
git push -u origin main --force

Write-Host "=== Push Completed Successfully! ===" -ForegroundColor Green
