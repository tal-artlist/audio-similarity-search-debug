# Audio Similarity Search

Find similar songs using raw Essentia audio embeddings. This tool performs audio-based similarity search using pre-computed embeddings and returns the most similar songs from a catalog.

## Features

- **Pure audio-based similarity**: Uses raw Essentia embeddings (no additional features like BPM, genre, or language)
- **Fast inference**: No projection heads or fine-tuned models
- **Smart filtering**: Automatic artist diversity and version deduplication
- **VO removal**: Optional vocal/voice-over removal using DeepFilterNet3
- **Easy to use**: Simple command-line interface

## Requirements

- Python 3.8+
- Audio files in WAV format (or other formats supported by pydub)
- Pre-computed catalog embeddings (included in `data/` folder)

## Installation

1. Install dependencies:

```bash
pip install -r requirements.txt
```

2. Verify installation:

```bash
python find_similar_songs.py --help
```

## Quick Start

### Basic Usage

Find 5 similar songs from your audio file:

```bash
python find_similar_songs.py --audio your_song.wav
```

### Get More Results

Return 10 similar songs:

```bash
python find_similar_songs.py --audio your_song.wav --top-k 10
```

### Disable Filters

For faster results without artist/version filtering:

```bash
python find_similar_songs.py --audio your_song.wav --disable-artist-filter --disable-version-filter
```

### Remove Vocals/Voice-Over

Process audio with vocal removal before similarity search:

```bash
python find_similar_songs.py --audio your_song.wav --remove-vo
```

Adjust the VO detection threshold (lower = more aggressive):

```bash
python find_similar_songs.py --audio your_song.wav --remove-vo --vo-threshold 0.3
```

Specify output directory for VO-cleaned audio:

```bash
python find_similar_songs.py --audio your_song.wav --remove-vo --output-dir cleaned_audio
```

### Debug Mode

Enable detailed logging:

```bash
python find_similar_songs.py --audio your_song.wav --debug
```

## Command-Line Options

| Option | Description | Default |
|--------|-------------|---------|
| `--audio` | Path to query audio file (required) | - |
| `--emb` | Path to embeddings parquet file | `data/embeddings/audio_embeddings_essentia.parquet` |
| `--catalog` | Path to catalog metadata parquet file | `data/processed/slices_catalog.parquet` |
| `--top-k` | Number of similar songs to return | `5` |
| `--max-artist-frequency` | Max songs per artist in results | `3` |
| `--disable-artist-filter` | Disable artist diversity filter | `False` |
| `--disable-version-filter` | Disable version deduplication | `False` |
| `--remove-vo` | Remove vocals/voice-over before similarity search | `False` |
| `--vo-threshold` | Voice detection threshold (0-1, lower = more aggressive) | `0.5` |
| `--output-dir` | Directory to save VO-cleaned audio files | `output` |
| `--debug` | Enable debug logging | `False` |

## Output Format

The script outputs a table with the following columns:

- **#**: Rank (1-N)
- **song_id**: Catalog song ID
- **BPM**: Beats per minute (if available)
- **genres**: Song genres from metadata
- **similarity**: Cosine similarity score (0-1, higher is better)

Example output:

```
Query: test_song.wav
Model: Essentia (raw embeddings)
────────────────────────────────────────────────────────────────────────────────────────────────────
 # song_id      BPM    genres                                                           similarity
────────────────────────────────────────────────────────────────────────────────────────────────────
 1 123456       120.0  Pop, Dance, Electronic                                           0.8542
 2 789012       118.5  Pop, Synth Pop                                                   0.8321
 3 345678       122.0  Electronic, House                                                0.8156
 4 901234       119.0  Dance, Electro Pop                                               0.7998
 5 567890       121.5  Pop, Electronic                                                  0.7889
```

## Data Files

The tool requires the following data files (included):

### Required Files

- `data/embeddings/audio_embeddings_essentia.parquet` (509MB)
  - Pre-computed Essentia embeddings for all catalog songs
  
- `data/processed/slices_catalog.parquet` (1.7MB)
  - Catalog metadata (song IDs, BPM, genres, etc.)

- `models/essentia_model/discogs_multi_embeddings-effnet-bs64-1.pb` (16MB)
  - Pre-trained Essentia model

### Optional Files (for filtering)

- `data/processed/song_artist_mapping.parquet` (297KB)
  - Artist mappings for diversity filtering
  
- `data/processed/song_version_mapping.parquet` (132KB)
  - Version mappings for deduplication

## How It Works

1. **Load Catalog**: Loads pre-computed embeddings for all catalog songs
2. **VO Removal (Optional)**: Removes vocals/voice-over using DeepFilterNet3 if `--remove-vo` is enabled
   - Processes audio through DeepFilterNet3 noise reduction
   - Calculates VO signal ratio to determine if voice is present
   - Saves cleaned audio to output directory
3. **Generate Query Embedding**: Processes your audio file (or VO-cleaned version) through the Essentia model
4. **Compute Similarities**: Calculates cosine similarity between query and catalog
5. **Apply Filters**: 
   - Removes duplicate versions of songs
   - Limits songs per artist for diversity
6. **Return Results**: Returns top-K most similar songs with metadata

## Technical Details

### Model

- **Essentia EffnetDiscogs**: Pre-trained music embedding model
- **Embedding Dimension**: 1280
- **Similarity Metric**: Cosine similarity (L2-normalized vectors)

### Filtering Strategy

1. **Version Deduplication**: Ensures only one version of each song appears
2. **Artist Diversity**: Limits to max 3 songs per artist in top results (customizable)

### Performance

- **Query Time**: ~1-2 seconds per query (depends on catalog size)
- **Memory Usage**: ~2-3GB (loaded catalog embeddings)
- **GPU Support**: Optional (TensorFlow will use GPU if available)

## Troubleshooting

### "Audio file not found"

Make sure the path to your audio file is correct. Use absolute paths if needed:

```bash
python find_similar_songs.py --audio /full/path/to/your_song.wav
```

### "Embeddings file not found"

Verify that the data files are in the correct location:

```bash
ls -lh data/embeddings/audio_embeddings_essentia.parquet
ls -lh data/processed/slices_catalog.parquet
```

### TensorFlow GPU Errors

If you see CUDA/cuDNN errors, the script will automatically fall back to CPU. To force CPU mode:

```bash
CUDA_VISIBLE_DEVICES="" python find_similar_songs.py --audio your_song.wav
```

### Out of Memory

If you run out of memory, try:

1. Close other applications
2. Use a smaller catalog (filter the parquet files)
3. Reduce `--top-k` value

## Advanced Usage

### Custom Embeddings File

Use a different embeddings file:

```bash
python find_similar_songs.py --audio song.wav --emb path/to/custom_embeddings.parquet
```

### Custom Catalog

Use a different catalog metadata file:

```bash
python find_similar_songs.py --audio song.wav --catalog path/to/custom_catalog.parquet
```

### Batch Processing

Process multiple files:

```bash
for file in songs/*.wav; do
    echo "Processing: $file"
    python find_similar_songs.py --audio "$file" --top-k 10
done
```

## API Usage (Python)

You can also use the tool as a Python module:

```python
from models.essentia import EssentiaModel
from utils.result_filters import apply_similarity_result_filters
import numpy as np
import pandas as pd

# Load model
model = EssentiaModel()

# Generate embedding
query_emb = model.get_embeddings("your_song.wav")

# Load catalog embeddings
catalog_df = pd.read_parquet("data/embeddings/audio_embeddings_essentia.parquet")
catalog_embs = np.vstack(catalog_df.embedding.values)

# Compute similarities
similarities = catalog_embs @ query_emb

# Get top results
top_indices = similarities.argsort()[::-1][:10]
```

## License

This tool uses the Essentia library which is released under the Affero GPLv3 license.

## Credits

- **Essentia**: Music Information Retrieval library by Music Technology Group (UPF)
- **EffnetDiscogs**: Pre-trained music embedding model

## Support

For issues or questions, please contact the development team or open an issue in the repository.

---

**Last Updated**: December 2024
