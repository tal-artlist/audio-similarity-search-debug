import logging
import pandas as pd
from collections import defaultdict
from typing import Dict, Set, List, Tuple, Optional
from pathlib import Path

# Cache instrumental positions for a given hooks_df to avoid recomputation
_INSTRUMENTAL_POS_CACHE: Dict[int, Set[int]] = {}

def _normalize_is_vocals_value(val) -> int:
    """Return 0 for instrumental, 1 for vocals, default to 1 when unknown."""
    try:
        return int(val)
    except Exception:
        if isinstance(val, bool):
            return 1 if val else 0
        if isinstance(val, str):
            s = val.strip().lower()
            if s in ("0", "false", "no", "n", "none", "null", ""):
                return 0
            if s in ("1", "true", "yes", "y"):
                return 1
        return 1

def get_instrumental_positions(hooks_df: pd.DataFrame) -> Set[int]:
    """Compute and cache the row positions that are instrumental (is_vocals == 0)."""
    if 'is_vocals' not in hooks_df.columns:
        return set()
    key = id(hooks_df)
    cached = _INSTRUMENTAL_POS_CACHE.get(key)
    if cached is not None:
        return cached
    vals = hooks_df['is_vocals'].tolist()
    positions: Set[int] = set(i for i, v in enumerate(vals) if _normalize_is_vocals_value(v) == 0)
    _INSTRUMENTAL_POS_CACHE[key] = positions
    logging.info(f"Cached instrumental positions for hooks_df id={key}: {len(positions)} rows.")
    return positions

# Get project root for loading data files
ROOT = Path(__file__).resolve().parent.parent


def load_filtering_data() -> Tuple[Dict[int, int], Dict[int, int]]:
    """
    Load artist and version mappings from stored data.
    
    Returns:
        Tuple of (song_to_artist_map, original_version_map)
    """
    song_to_artist_map = {}
    original_version_map = {}
    
    # Load artist data
    artist_path = ROOT / "data" / "processed" / "song_artist_mapping.parquet"
    if artist_path.exists():
        try:
            artist_df = pd.read_parquet(artist_path)
            song_to_artist_map = build_artist_mapping(artist_df, "song_id", "primary_artist_id")
            logging.info(f"Loaded artist mapping for {len(song_to_artist_map)} songs from {artist_path}")
        except Exception as e:
            logging.warning(f"Could not load artist mapping: {e}")
    else:
        logging.warning(f"Artist mapping file not found: {artist_path}")
    
    # Load version data
    version_path = ROOT / "data" / "processed" / "song_version_mapping.parquet"
    if version_path.exists():
        try:
            version_df = pd.read_parquet(version_path)
            original_version_map = build_version_mapping(version_df, "song_id", "original_version_id")
            logging.info(f"Loaded version mapping for {len(original_version_map)} songs from {version_path}")
        except Exception as e:
            logging.warning(f"Could not load version mapping: {e}")
    else:
        logging.warning(f"Version mapping file not found: {version_path}")
    
    return song_to_artist_map, original_version_map


def build_artist_mapping(song_data: pd.DataFrame, song_id_col: str = "song_id", artist_id_col: str = "artist_id") -> Dict[int, int]:
    """
    Build a mapping from song_id to artist_id.
    
    Args:
        song_data: DataFrame containing song metadata with artist information
        song_id_col: Column name for song IDs
        artist_id_col: Column name for artist IDs (or primary_artist_id)
        
    Returns:
        Dictionary mapping song_id -> artist_id
    """
    song_to_artist_map = {}
    
    for _, row in song_data.iterrows():
        song_id = int(row[song_id_col])
        artist_id = int(row[artist_id_col]) if pd.notna(row[artist_id_col]) else None
        if artist_id is not None:
            song_to_artist_map[song_id] = artist_id
    
    logging.info(f"Built artist mapping for {len(song_to_artist_map)} songs")
    return song_to_artist_map


def build_version_mapping(df: pd.DataFrame, song_id_col: str, original_version_id_col: str) -> Dict[int, int]:
    """
    Build a mapping from song_id to its original_version_id.
    
    Args:
        df: DataFrame with song and version information
        song_id_col: Name of the song ID column
        original_version_id_col: Name of the original version ID column
    
    Returns:
        Dictionary mapping song_id -> original_version_id
    """
    mapping = {}
    valid_rows = df[
        df[song_id_col].notna() & 
        df[original_version_id_col].notna() &
        (df[song_id_col] != '') &
        (df[original_version_id_col] != '')
    ]
    
    for _, row in valid_rows.iterrows():
        try:
            song_id = int(row[song_id_col])
            original_id = int(row[original_version_id_col])
            mapping[song_id] = original_id
        except (ValueError, TypeError):
            # Skip rows with invalid IDs that can't be converted to int
            continue
    
    return mapping


def apply_artist_limit_filter_top_k_only(
    candidates: List[Dict], 
    song_to_artist_map: Dict[int, int], 
    max_artist_frequency: int = 2,
    top_k_for_artist_filtering: int = 5,
    song_id_key: str = "song_id",
    final_top_k: int = None
) -> List[Dict]:
    """
    Filter candidates to limit the number of songs per artist ONLY in the top K results.
    Songs beyond the top K are not subject to artist frequency limits.
    
    Args:
        candidates: List of candidate dictionaries (sorted by similarity, best first)
        song_to_artist_map: Mapping from song_id to artist_id
        max_artist_frequency: Maximum number of songs per artist allowed in top K
        top_k_for_artist_filtering: Only apply artist limits to top K results (e.g., 5)
        song_id_key: Key to access song_id in candidate dictionaries
        final_top_k: Stop when we have this many results (None = no limit)
        
    Returns:
        Filtered list of candidates
    """
    # Split candidates into top K and remaining
    top_candidates = candidates[:top_k_for_artist_filtering]
    remaining_candidates = candidates[top_k_for_artist_filtering:]
    
    # Apply artist filtering only to top candidates
    artist_frequency = defaultdict(int)
    filtered_top_results = []
    
    for candidate in top_candidates:
        song_id = int(candidate[song_id_key])
        artist_id = song_to_artist_map.get(song_id)
        
        # Skip if we've already included max songs from this artist in top results
        if artist_id and artist_frequency[artist_id] >= max_artist_frequency:
            continue
            
        # Include this song
        filtered_top_results.append(candidate)
        
        # Increment counter for this artist
        if artist_id:
            artist_frequency[artist_id] += 1
    
    # Combine filtered top results with unfiltered remaining results
    all_results = filtered_top_results + remaining_candidates
    
    # Apply final top_k limit if specified
    if final_top_k:
        all_results = all_results[:final_top_k]
    
    return all_results


def apply_artist_limit_filter(
    candidates: List[Dict], 
    song_to_artist_map: Dict[int, int], 
    max_artist_frequency: int = 2,
    song_id_key: str = "song_id",
    top_k: int = None
) -> List[Dict]:
    """
    Filter candidates to limit the number of songs per artist in the top results.
    
    Args:
        candidates: List of candidate dictionaries (sorted by similarity, best first)
        song_to_artist_map: Mapping from song_id to artist_id
        max_artist_frequency: Maximum number of songs per artist allowed
        song_id_key: Key to access song_id in candidate dictionaries
        top_k: Stop when we have this many results (None = no limit)
        
    Returns:
        Filtered list of candidates
    """
    artist_frequency = defaultdict(int)
    filtered_results = []
    
    for candidate in candidates:
        song_id = int(candidate[song_id_key])
        artist_id = song_to_artist_map.get(song_id)
        
        # Skip if we've already included max songs from this artist
        if artist_id and artist_frequency[artist_id] >= max_artist_frequency:
            continue
            
        # Include this song
        filtered_results.append(candidate)
        
        # Increment counter for this artist
        if artist_id:
            artist_frequency[artist_id] += 1
            
        # Stop when we have enough results
        if top_k and len(filtered_results) >= top_k:
            break
    
    return filtered_results


def apply_version_deduplication_filter(
    candidates: List[Dict], 
    original_version_map: Dict[int, int],
    song_id_key: str = "song_id",
    top_k: int = None
) -> List[Dict]:
    """
    Filter candidates to remove duplicate versions of the same song.
    
    Args:
        candidates: List of candidate dictionaries (sorted by similarity, best first)
        original_version_map: Mapping from song_id to original_version_id
        song_id_key: Key to access song_id in candidate dictionaries
        top_k: Stop when we have this many results (None = no limit)
        
    Returns:
        Filtered list of candidates
    """
    included_original_ids = set()
    filtered_results = []
    
    for candidate in candidates:
        song_id = int(candidate[song_id_key])
        original_id = original_version_map.get(song_id)
        
        # Skip if this song's original version is already included
        if original_id and original_id in included_original_ids:
            continue
            
        # Skip if this song itself is already tracked as an original
        if song_id in included_original_ids:
            continue
        
        # Include this song
        filtered_results.append(candidate)
        
        # Track which original version we've included
        if original_id:
            included_original_ids.add(original_id)
        else:
            # If no original_id, treat this song as the original
            included_original_ids.add(song_id)
            
        # Stop when we have enough results
        if top_k and len(filtered_results) >= top_k:
            break
    
    return filtered_results


def apply_similarity_result_filters(
    similarity_indices: List[int],
    similarity_scores: List[float],
    hooks_df: pd.DataFrame,
    catalog_df: pd.DataFrame = None,
    max_artist_frequency: int = 2,
    apply_version_filter: bool = True,
    apply_artist_filter: bool = True,
    top_k: int = None,
    query_song_id: int = None
) -> Tuple[List[int], List[float]]:
    """
    High-level function to apply filters to similarity search results.
    
    Args:
        similarity_indices: List of indices into hooks_df (sorted by similarity, best first)
        similarity_scores: Corresponding similarity scores
        hooks_df: DataFrame with song information (must have 'song_id' column)
        catalog_df: Optional catalog DataFrame with artist and version info
        max_artist_frequency: Maximum number of songs per artist allowed
        apply_version_filter: Whether to apply version deduplication
        apply_artist_filter: Whether to apply artist limitation
        top_k: Target number of results after filtering
        query_song_id: ID of the query song (for version filtering)
        
    Returns:
        Tuple of (filtered_indices, filtered_scores)
    """
    if not apply_version_filter and not apply_artist_filter:
        # No filtering requested
        if top_k:
            return similarity_indices[:top_k], similarity_scores[:top_k]
        return similarity_indices, similarity_scores
    
    # Build candidate list
    candidates = []
    for i, (idx, score) in enumerate(zip(similarity_indices, similarity_scores)):
        row = hooks_df.iloc[idx]
        candidates.append({
            'song_id': int(row.song_id),
            'original_index': idx,
            'similarity_score': score,
            'rank': i
        })
    
    # Filter out the query song itself
    if query_song_id is not None:
        initial_count = len(candidates)
        candidates = [c for c in candidates if c['song_id'] != query_song_id]
        if len(candidates) < initial_count:
            logging.info(f"Removed self-match for song ID {query_song_id}.")
    
    # Build filter mappings
    song_to_artist_map = {}
    original_version_map = {}
    
    # Try to load from cached data files
    if apply_artist_filter or apply_version_filter:
        loaded_artist_map, loaded_version_map = load_filtering_data()
        
        if apply_artist_filter and loaded_artist_map:
            song_to_artist_map = loaded_artist_map
            logging.info(f"Using cached artist mapping for {len(song_to_artist_map)} songs")
        
        if apply_version_filter and loaded_version_map:
            original_version_map = loaded_version_map
            logging.info(f"Using cached version mapping for {len(original_version_map)} songs")
    
    # Fall back to catalog_df if needed and available
    if catalog_df is not None:
        if apply_artist_filter and not song_to_artist_map:
            if 'artist_id' in catalog_df.columns:
                song_to_artist_map = build_artist_mapping(catalog_df, "song_id", "artist_id")
            elif 'primary_artist_id' in catalog_df.columns:
                song_to_artist_map = build_artist_mapping(catalog_df, "song_id", "primary_artist_id")
        
        if apply_version_filter and not original_version_map:
            if 'original_version_id' in catalog_df.columns:
                original_version_map = build_version_mapping(catalog_df, "song_id", "original_version_id")
    
    # Apply filters
    filtered_candidates = candidates
    
    # Version filtering first (if query song is known)
    if apply_version_filter and query_song_id is not None and original_version_map:
        query_family_id = original_version_map.get(query_song_id, query_song_id)
        initial_count = len(filtered_candidates)
        filtered_candidates = [
            c for c in filtered_candidates 
            if original_version_map.get(c['song_id'], c['song_id']) != query_family_id
        ]
        if len(filtered_candidates) < initial_count:
            logging.info(f"Removed {initial_count - len(filtered_candidates)} song version(s) related to query song {query_song_id}.")
    
    # Artist filtering (only top 5)
    if apply_artist_filter:
        filtered_candidates = apply_artist_limit_filter_top_k_only(
            filtered_candidates, song_to_artist_map, 
            max_artist_frequency=max_artist_frequency,
            top_k_for_artist_filtering=5,
            final_top_k=top_k
        )
    elif top_k:
        filtered_candidates = filtered_candidates[:top_k]
    
    # Extract filtered indices and scores
    filtered_indices = [c['original_index'] for c in filtered_candidates]
    filtered_scores = [c['similarity_score'] for c in filtered_candidates]
    
    # Log filtering results
    original_count = len(candidates)
    filtered_count = len(filtered_candidates)
    if filtered_count < original_count:
        logging.info(f"Applied result filters: {original_count} -> {filtered_count} results "
                    f"(artist_filter={apply_artist_filter}, version_filter={apply_version_filter})")
    
    return filtered_indices, filtered_scores
