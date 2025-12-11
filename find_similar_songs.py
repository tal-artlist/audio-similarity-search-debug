#!/usr/bin/env python
"""
Simple audio similarity search using raw Essentia embeddings.
No projection heads, no additional features - just pure audio embeddings.
"""

import argparse
import logging
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import torch

# Add current directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from models.essentia import EssentiaModel
from utils.audio_loader import AudioLoader
from utils.result_filters import apply_similarity_result_filters


def load_catalog_embeddings(emb_pq: str, catalog_pq: str) -> tuple[np.ndarray, pd.DataFrame]:
    """Load raw audio embeddings from the catalog."""
    df = pq.read_table(emb_pq).to_pandas()
    
    # Filter for hooks only
    df = df[df.slice_type == "hook"]
    
    # Filter out external sources if present
    if "source" in df.columns:
        df = df[~df.source.str.contains("external", na=False)]
    
    df = df.sort_values("emb_idx")
    
    # Extract raw embeddings
    embeddings = np.vstack([np.asarray(emb, dtype=np.float32) for emb in df.embedding.values])
    
    # Normalize embeddings for cosine similarity
    embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
    
    # Load BPM data for display
    if "bpm" not in df.columns:
        try:
            cat = pd.read_parquet(catalog_pq)
            cat["song_id"] = cat["song_id"].astype(str)
            cat["slice_type"] = cat["slice_type"].fillna("none").astype(str)
            
            df["song_id"] = df["song_id"].astype(str)
            df["slice_type"] = df["slice_type"].fillna("none").astype(str)
            
            df = df.merge(cat[["song_id", "slice_type", "bpm"]], on=["song_id", "slice_type"], how="left")
            logging.info("BPM data loaded for display")
        except Exception as exc:
            logging.warning("Could not load BPM data: %s", exc)
            df["bpm"] = np.nan
    
    logging.info("Loaded %d catalog hooks with %d-dim embeddings", len(df), embeddings.shape[1])
    return embeddings, df


def embed_query(audio_path: str, audio_model: EssentiaModel, remove_vo: bool = False, 
                vo_threshold: float = 0.5, df_model: tuple = None, 
                output_dir: str = "output") -> tuple[np.ndarray, dict]:
    """
    Generate embedding for query audio, optionally with VO removal.
    
    Returns:
        tuple: (embedding, metadata_dict)
    """
    metadata = {
        "vo_removed": False,
        "vo_signal_ratio": None,
        "vo_cleaned_path": None
    }
    
    audio_data_for_embedding = audio_path
    
    # Step 1: VO Removal (if requested)
    if remove_vo:
        if df_model is None:
            logging.warning("⚠️ VO removal requested but DeepFilterNet model not loaded. Skipping VO removal.")
        else:
            try:
                logging.info("🎙️ Starting VO removal process...")
                
                # Load audio file as bytes
                with open(audio_path, 'rb') as f:
                    audio_bytes = f.read()
                
                # Create audio loader and process
                from utils.audio_loader import AudioLoader
                audio_loader = AudioLoader()
                
                # Process with VO threshold
                processed_audio_bytes, is_voice_detected, signal_ratio = audio_loader.process_audio_with_vo_threshold(
                    audio_bytes, vo_threshold, df_model
                )
                
                metadata["vo_signal_ratio"] = float(signal_ratio)
                
                if is_voice_detected:
                    # Save the VO-cleaned audio
                    os.makedirs(output_dir, exist_ok=True)
                    
                    # Create output filename based on input
                    input_filename = Path(audio_path).stem
                    timestamp = pd.Timestamp.now().strftime("%Y%m%d-%H%M%S")
                    vo_cleaned_filename = f"{input_filename}_{timestamp}_vo_cleaned.wav"
                    vo_cleaned_path = os.path.join(output_dir, vo_cleaned_filename)
                    
                    # Write VO-cleaned audio
                    with open(vo_cleaned_path, 'wb') as f:
                        f.write(processed_audio_bytes)
                    
                    logging.info(f"✅ VO removed successfully. Saved to: {vo_cleaned_path}")
                    logging.info(f"📊 VO signal ratio: {signal_ratio:.4f} (threshold: {vo_threshold})")
                    
                    metadata["vo_removed"] = True
                    metadata["vo_cleaned_path"] = vo_cleaned_path
                    
                    # Use VO-cleaned audio for embedding
                    audio_data_for_embedding = vo_cleaned_path
                else:
                    logging.info(f"ℹ️ VO signal ratio ({signal_ratio:.4f}) exceeded threshold ({vo_threshold}). Using original audio.")
                    metadata["vo_removed"] = False
                    
            except Exception as e:
                logging.error(f"❌ VO removal failed: {e}")
                logging.warning("⚠️ Continuing with original audio")
    
    # Step 2: Generate embedding
    emb = np.asarray(audio_model.get_embeddings(audio_data_for_embedding), dtype=np.float32)
    
    # Normalize for cosine similarity
    emb = emb / np.linalg.norm(emb)
    
    print(f"Query embedding: {emb.shape[0]} dimensions")
    print(f"First 10 dimensions: {emb[:10]}")
    
    return emb, metadata


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Find similar songs using raw audio embeddings (Essentia)"
    )
    
    # Input/Output
    parser.add_argument(
        "--audio", 
        required=True, 
        help="Query audio file path"
    )
    parser.add_argument(
        "--emb", 
        default="data/embeddings/audio_embeddings_essentia.parquet",
        help="Embeddings parquet file (default: data/embeddings/audio_embeddings_essentia.parquet)"
    )
    parser.add_argument(
        "--catalog", 
        default="data/processed/slices_catalog.parquet",
        help="Catalog metadata parquet file (default: data/processed/slices_catalog.parquet)"
    )
    
    # Retrieval
    parser.add_argument(
        "--top-k", 
        type=int, 
        default=5,
        help="Number of similar songs to return (default: 5)"
    )
    
    # Result filtering
    parser.add_argument("--max-artist-frequency", type=int, default=3,
                       help="Maximum number of songs per artist in top results (default: 3)")
    parser.add_argument("--disable-artist-filter", action="store_true",
                       help="Disable artist limitation filter")
    parser.add_argument("--disable-version-filter", action="store_true", 
                       help="Disable version deduplication filter")
    
    # VO Removal
    parser.add_argument(
        "--remove-vo",
        action="store_true",
        help="Remove vocals/voice-over using DeepFilterNet before generating embeddings"
    )
    parser.add_argument(
        "--vo-threshold",
        type=float,
        default=0.5,
        help="Voice detection threshold (0-1, default: 0.5). Lower values are more aggressive."
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="output",
        help="Directory to save intermediate VO-cleaned audio files (default: output)"
    )
    
    # Misc
    parser.add_argument(
        "--debug", 
        action="store_true",
        help="Enable debug logging"
    )
    
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    
    # Setup logging
    logging.basicConfig(
        level=logging.DEBUG if args.debug else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s"
    )
    
    # Validate input file
    if not os.path.exists(args.audio):
        logging.error("Audio file not found: %s", args.audio)
        sys.exit(1)
    
    # Validate embeddings file
    if not os.path.exists(args.emb):
        logging.error("Embeddings file not found: %s", args.emb)
        sys.exit(1)
    
    logging.info("Using Essentia model for audio embeddings")
    logging.info("Using embeddings: %s", args.emb)
    
    # Load catalog embeddings
    try:
        catalog_embeddings, catalog_df = load_catalog_embeddings(
            args.emb, args.catalog
        )
    except Exception as e:
        logging.error("Failed to load catalog embeddings: %s", e)
        sys.exit(1)
    
    # Load DeepFilterNet model if VO removal is requested
    df_model = None
    if args.remove_vo:
        try:
            logging.info("🔧 Loading DeepFilterNet model for VO removal...")
            from df import init_df
            import torch
            
            device = "cuda" if torch.cuda.is_available() else "cpu"
            model, df_state, _ = init_df(model_base_dir="DeepFilterNet3")
            model = model.to(device).eval()
            df_model = (model, df_state, device)
            
            logging.info(f"✅ DeepFilterNet model loaded successfully on {device}")
        except Exception as e:
            logging.error(f"❌ Failed to load DeepFilterNet model: {e}")
            logging.warning("⚠️ Continuing without VO removal")
            args.remove_vo = False
    
    # Initialize audio model
    audio_model = EssentiaModel()
    
    # Generate query embedding
    try:
        query_embedding, vo_metadata = embed_query(
            args.audio, 
            audio_model,
            remove_vo=args.remove_vo,
            vo_threshold=args.vo_threshold,
            df_model=df_model,
            output_dir=args.output_dir
        )
    except Exception as e:
        logging.error("Failed to generate query embedding: %s", e)
        sys.exit(1)
    
    # Compute similarities (cosine similarity via dot product of normalized vectors)
    similarities = catalog_embeddings @ query_embedding
    
    # Use ALL candidates for maximum diversity before filtering
    all_indices = similarities.argsort()[::-1]  # All candidates, best first
    all_scores = similarities[all_indices]
    
    # Apply result filters
    catalog_meta = None
    if os.path.exists(args.catalog):
        try:
            catalog_meta = pd.read_parquet(args.catalog)
            catalog_meta["song_id"] = catalog_meta.song_id.astype(str)
        except Exception as e:
            logging.warning("Could not load catalog meta for filtering: %s", e)
    
    # Apply filters to get final top results from ALL candidates
    filtered_indices, filtered_scores = apply_similarity_result_filters(
        all_indices.tolist(),
        all_scores.tolist(),
        catalog_df,
        catalog_meta,
        max_artist_frequency=args.max_artist_frequency,
        apply_version_filter=not args.disable_version_filter,
        apply_artist_filter=not args.disable_artist_filter,
        top_k=args.top_k
    )
    
    # Convert back to numpy arrays for compatibility
    top_indices = np.array(filtered_indices)
    final_scores = np.array(filtered_scores)
    
    # Load metadata for display
    meta = None
    if os.path.exists(args.catalog):
        try:
            meta = pd.read_parquet(args.catalog)
            meta["song_id"] = meta.song_id.astype(str)
        except Exception as e:
            logging.warning("Could not load catalog metadata: %s", e)
    
    # Display results
    print(f"\nQuery: {Path(args.audio).name}")
    print(f"Model: Essentia (raw embeddings)")
    
    # Display VO removal info if applicable
    if args.remove_vo:
        if vo_metadata["vo_removed"]:
            print(f"VO Removal: ✅ Applied (signal ratio: {vo_metadata['vo_signal_ratio']:.4f})")
            print(f"VO-cleaned audio saved to: {vo_metadata['vo_cleaned_path']}")
        else:
            if vo_metadata["vo_signal_ratio"] is not None:
                print(f"VO Removal: ⚠️ Skipped (signal ratio: {vo_metadata['vo_signal_ratio']:.4f} > threshold: {args.vo_threshold})")
            else:
                print(f"VO Removal: ❌ Failed")
    
    print("─" * 100)
    print(f"{'#':>2} {'song_id':<12} {'BPM':<6} {'genres':<65} {'similarity':<10}")
    print("─" * 100)
    
    for rank, idx in enumerate(top_indices, 1):
        row = catalog_df.iloc[idx]
        song_id = str(row.song_id)
        similarity = final_scores[rank-1]
        
        # BPM
        bpm_val = row.get("bpm")
        bpm_display = "n/a" if pd.isna(bpm_val) or bpm_val == 0 else f"{bpm_val:5.1f}"
        
        # Genre info
        genre_display = "Unknown"
        if meta is not None:
            meta_match = meta[meta.song_id == song_id]
            if not meta_match.empty:
                genres = [
                    x for x in [
                        meta_match.iloc[0].get("genre_main_fam1"),
                        meta_match.iloc[0].get("genre_main_fam2"),
                        meta_match.iloc[0].get("genre_group"),
                        meta_match.iloc[0].get("genre_leaf")
                    ] if pd.notna(x)
                ]
                if genres:
                    genre_display = ", ".join(genres)
        
        print(f"{rank:>2} {song_id:<12} {bpm_display:<6} {genre_display:<65} {similarity:.4f}")


if __name__ == "__main__":
    main()
