# Git Repository Setup

## Quick Setup

```bash
cd /Users/tal.darchi/work/core-ai-audio-embeddings/audio-similarity-search

# Initialize git repository
git init

# Add all files
git add .

# Create initial commit
git commit -m "Initial commit: Audio similarity search with VO removal"

# Add remote (replace with your repo URL)
git remote add origin https://github.com/YOUR_USERNAME/audio-similarity-search.git

# Push to remote
git push -u origin main
```

## ⚠️ Large Files Warning

Your repository contains **~527 MB of data**:
- `data/embeddings/audio_embeddings_essentia.parquet` (509 MB)
- `models/essentia_model/discogs_multi_embeddings-effnet-bs64-1.pb` (16 MB)

### Option 1: Include Everything (Simple)
Just push as-is. Works for GitHub (up to 2GB repo size) but uploads will be slow.

### Option 2: Use Git LFS (Recommended for Large Files)
```bash
# Install Git LFS (one-time)
brew install git-lfs  # macOS
# or: sudo apt install git-lfs  # Linux

# Initialize Git LFS
git lfs install

# Track large files
git lfs track "data/**/*.parquet"
git lfs track "models/**/*.pb"

# Add .gitattributes
git add .gitattributes

# Commit and push
git commit -m "Add Git LFS tracking for large files"
git push
```

### Option 3: Exclude Large Files (Smallest Repo)
Uncomment these lines in `.gitignore`:
```
data/
models/
```

Then document how to download the data files separately (S3, Google Drive, etc.)

## Create GitHub Repository

1. Go to https://github.com/new
2. Name: `audio-similarity-search`
3. Description: "Audio similarity search with vocal removal using Essentia and DeepFilterNet"
4. Public or Private (your choice)
5. Don't initialize with README (you already have one)
6. Click "Create repository"
7. Use the commands above with your repo URL

## Delete This File

After setup, delete this file:
```bash
git rm GIT_SETUP.md
git commit -m "Remove setup instructions"
git push
```

