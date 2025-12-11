# Quick Start Guide

## Setup (First Time Only)

1. **Install Python dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Verify everything is in place**:
   ```bash
   # Check data files exist
   ls -lh data/embeddings/audio_embeddings_essentia.parquet
   ls -lh data/processed/slices_catalog.parquet
   ls -lh models/essentia_model/discogs_multi_embeddings-effnet-bs64-1.pb
   ```

## Run Your First Query

```bash
# Basic usage - find 5 similar songs
python find_similar_songs.py --audio your_song.wav

# Get 10 results
python find_similar_songs.py --audio your_song.wav --top-k 10

# With vocal/voice-over removal
python find_similar_songs.py --audio your_song.wav --remove-vo

# Debug mode (see what's happening)
python find_similar_songs.py --audio your_song.wav --debug
```

## Expected Output

```
Query: your_song.wav
Model: Essentia (raw embeddings)
VO Removal: ✅ Applied (signal ratio: 0.3245)
VO-cleaned audio saved to: output/your_song_20241211-143022_vo_cleaned.wav
────────────────────────────────────────────────────────────────────────────────
 # song_id      BPM    genres                                    similarity
────────────────────────────────────────────────────────────────────────────────
 1 123456       120.0  Pop, Dance, Electronic                    0.8542
 2 789012       118.5  Pop, Synth Pop                            0.8321
 3 345678       122.0  Electronic, House                         0.8156
...
```

## Common Issues

**"Audio file not found"**
- Make sure your audio file path is correct
- Try using absolute path: `/full/path/to/song.wav`

**TensorFlow/CUDA warnings**
- These are normal and can be ignored
- The script automatically falls back to CPU if GPU is unavailable

**"Embeddings file not found"**
- Make sure you're running the script from the `audio-similarity-search` folder
- Or provide full paths: `--emb /full/path/to/embeddings.parquet`

## Next Steps

See `README.md` for:
- Complete command-line options
- Advanced usage examples
- Batch processing
- Python API usage
