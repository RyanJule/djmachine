# streamlit_app.py
import streamlit as st
import pandas as pd
import numpy as np
import requests
from bs4 import BeautifulSoup
import time
import re
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional
from io import StringIO
import json

# -----------------------
# Page config
# -----------------------
st.set_page_config(page_title="Harmonic Song Analyzer — Camelot + SongData",
                   page_icon="🎵",
                   layout="wide")


# -----------------------
# Data classes
# -----------------------
@dataclass
class Song:
    title: str
    artist: str
    key: str
    tempo: Optional[int] = None
    mood: Optional[str] = None
    notes: Optional[str] = None
    camelot: Optional[str] = None


# -----------------------
# Key / Camelot detection helpers
# -----------------------
# NOTE: order matters: match 10-12 before single digits.
CAMLEOT_REGEX = re.compile(r'\(?\s*(?:1[0-2]|[1-9])\s*[ABab]\s*\)?')
CAMEL_NUMERIC_ONLY = re.compile(r'^\s*\d+(\.\d+)?\s*$')
KEY_NAME_REGEX = re.compile(r'^[A-Ga-g](?:[#♯]|[b♭])?m?$')
KEY_NAME_IN_CELL = re.compile(r'[A-Ga-g](?:[#♯]|[b♭])?\s*(?:major|minor|Major|Minor)?', re.I)


def is_probable_camelot(cell: str) -> bool:
    if not cell or str(cell).strip() == '':
        return False
    s = str(cell).strip()
    if CAMLEOT_REGEX.search(s):
        return True
    if CAMEL_NUMERIC_ONLY.match(s):
        try:
            num = float(s)
            if 1 <= num <= 12:
                return True
        except Exception:
            pass
    return False


def is_probable_keyname(cell: str) -> bool:
    if not cell or str(cell).strip() == '':
        return False
    s = str(cell).strip()
    # More comprehensive key name detection including Unicode symbols
    if re.search(r'^[A-Ga-g](?:[#♯]|[b♭])?\s*(?:minor|major|Minor|Major)?$', s, re.I):
        return True
    return False


def detect_key_column_from_rows(rows: List[List[str]], headers: List[str]) -> Optional[int]:
    """
    Determine best column index that looks like 'key'/'camelot'.
    Prefer a column with header containing 'camelot' first (explicit).
    Otherwise use content-scoring per column (ratio of cells that look like keys/Camelot).
    """
    if not rows:
        return None
    ncols = max(len(r) for r in rows)
    scores = [0.0] * ncols
    counts = [0] * ncols

    for r in rows:
        for j in range(ncols):
            cell = r[j] if j < len(r) else ""
            if cell is None:
                cell = ""
            cs = str(cell).strip()
            if cs == "":
                continue
            counts[j] += 1
            if is_probable_keyname(cs):
                scores[j] += 1

    ratios = [(scores[i] / counts[i]) if counts[i] > 0 else 0.0 for i in range(ncols)]

    # Otherwise consider a 'key' header (only if it actually looks key-like based on content)
    key_indices = [i for i, h in enumerate(headers) if 'key' in h.lower()]
    if key_indices:
        ki = key_indices[0]
        if ki < len(ratios) and ratios[ki] > 0.25:
            return ki

    # final fallback: None
    return None

def extract_key_and_camelot_from_cell(cell: str) -> Tuple[Optional[str], Optional[str]]:
    """
    Given a cell possibly containing both a key and a Camelot code (e.g. 'C (8B)' or '8A'),
    return (keyname, camelot_code).
    """
    if cell is None:
        return None, None
    s = str(cell).strip()
    if not s:
        return None, None

    # Handle Unicode musical symbols
    s = s.replace('♭', 'b').replace('♯', '#')

    # Find Camelot token first (prefer full e.g. '10B', '11A', '8A', '(8B)')
    cam = None
    m = CAMLEOT_REGEX.search(s)
    if m:
        token = m.group(0)
        digits = re.search(r'(?:1[0-2]|[1-9])', token)
        letter = re.search(r'([ABab])', token)
        if digits:
            num = digits.group(0)
            let = (letter.group(1).upper() if letter else 'A')
            cam = f"{int(num)}{let}"

    # Numeric-only cell (like '11') -> return '11' (ambiguous A/B)
    if cam is None and CAMEL_NUMERIC_ONLY.match(s):
        try:
            num = int(float(s))
            if 1 <= num <= 12:
                cam = str(num)
        except Exception:
            pass

    # Find keyname token - improved pattern to handle flats/sharps and major/minor
    keyname = None
    
    # First try: exact match for the whole string
    if re.match(r'^[A-Ga-g](?:[#b])?(?:\s*(?:major|minor|Major|Minor))?$', s, re.I):
        keyname = s
    else:
        # Second try: search within the string for key pattern
        key_pattern = re.search(r'([A-Ga-g])([#b]?)\s*(major|minor|Major|Minor)?', s, re.I)
        if key_pattern:
            note = key_pattern.group(1).upper()
            accidental = key_pattern.group(2) or ''
            mode = key_pattern.group(3) or ''
            
            # Normalize the mode
            if mode.lower() == 'minor':
                mode = 'm'
            elif mode.lower() == 'major':
                mode = ''
            else:
                mode = ''
            
            keyname = f"{note}{accidental}{mode}"

    # Clean up keyname if found
    if keyname:
        keyname = keyname.strip()
        # Normalize case: first letter uppercase, rest as-is for accidentals and 'm'
        if len(keyname) >= 1:
            keyname = keyname[0].upper() + keyname[1:]

    return keyname, cam


# -----------------------
# SongData fetch + parse (header-aware + content detection)
# -----------------------
def extract_spotify_playlist_id(spotify_url: str) -> Optional[str]:
    if not spotify_url:
        return None
    s = spotify_url.strip()
    m = re.search(r'playlist/([A-Za-z0-9]+)', s)
    if m:
        return m.group(1)
    m = re.search(r'spotify:playlist:([A-Za-z0-9]+)', s)
    if m:
        return m.group(1)
    if re.fullmatch(r'[A-Za-z0-9]+', s):
        return s
    return None


def fetch_songdata_playlist(spotify_url: str, timeout: float = 30.0, debug: bool = False) -> Tuple[List[Song], str, Dict]:
    """
    Fetch SongData playlist and detect which column contains keys/camelot.
    Returns (songs, songdata_url, debug_info).
    """
    pid = extract_spotify_playlist_id(spotify_url)
    if not pid:
        raise RuntimeError("Couldn't extract Spotify playlist id from that input.")
    songdata_url = f"https://songdata.io/playlist/{pid}"
    headers = {'User-Agent': 'Mozilla/5.0 (compatible; SongDataFetcher/1.0)'}
    try:
        r = requests.get(songdata_url, headers=headers, timeout=timeout)
    except Exception as e:
        raise RuntimeError(f"Failed to fetch SongData page: {e}")

    if r.status_code != 200:
        raise RuntimeError(f"SongData returned HTTP {r.status_code} for {songdata_url}")

    soup = BeautifulSoup(r.text, 'html.parser')
    tables = soup.find_all('table')
    songs: List[Song] = []
    debug_info = {'headers_seen': [], 'first_rows': [], 'chosen_key_column': None}

    for table in tables:
        headers = []
        thead = table.find('thead')
        if thead:
            headers = [th.get_text(strip=True) for th in thead.find_all('th')]
        else:
            first_row = table.find('tr')
            if first_row:
                headers = [td.get_text(strip=True) for td in first_row.find_all(['th', 'td'])]
        headers = [h.strip() for h in headers if h.strip()]
        if not headers:
            continue

        debug_info['headers_seen'].append(headers)
        lheaders = [h.lower() for h in headers]

        has_title = any('title' in h or 'song' in h or 'track' in h for h in lheaders)
        has_artist = any('artist' in h for h in lheaders)
        if not (has_title and has_artist):
            continue

        # Build row matrix
        tbody = table.find('tbody') or table
        rows_html = tbody.find_all('tr')
        rows_cells: List[List[str]] = []
        sample_rows: List[List[str]] = []
        for row in rows_html:
            cols = row.find_all(['td', 'th'])
            texts = [c.get_text(strip=True) for c in cols]
            texts = [t if t is not None else "" for t in texts]
            rows_cells.append(texts)
            if len(sample_rows) < 6:
                sample_rows.append(texts)
        debug_info['first_rows'] = sample_rows

        # detect key-like column
        key_col_idx = detect_key_column_from_rows(rows_cells, headers)
        debug_info['chosen_key_column'] = key_col_idx

        # map indices for title/artist/tempo
        header_map: Dict[str, int] = {}
        for idx, h in enumerate(lheaders):
            if any(k in h for k in ("title", "song", "track")) and 'title' not in header_map:
                header_map['title'] = idx
            elif 'artist' in h and 'artist' not in header_map:
                header_map['artist'] = idx
            elif ('bpm' in h or 'tempo' in h) and 'tempo' not in header_map:
                header_map['tempo'] = idx

        # produce Song objects; prioritize detected Camelot column data
        for row in rows_cells:
            def get_cell(i: int) -> str:
                try:
                    return row[i]
                except Exception:
                    return ""

            title = get_cell(header_map['title']) if 'title' in header_map and header_map['title'] < len(row) else ""
            artist = get_cell(header_map['artist']) if 'artist' in header_map and header_map['artist'] < len(row) else ""
            tempo_txt = get_cell(header_map['tempo']) if 'tempo' in header_map and header_map['tempo'] < len(row) else ""
            tempo = None
            if tempo_txt:
                m = re.search(r'(\d{2,3})', tempo_txt)
                if m:
                    try:
                        tempo = int(m.group(1))
                    except Exception:
                        tempo = None

            key_val = ""
            camelot_val = None

            # 1) Try detected key column first
            if key_col_idx is not None and key_col_idx < len(row):
                possible = get_cell(key_col_idx)
                kn, cam = extract_key_and_camelot_from_cell(possible)
                if kn:
                    key_val = kn
                if cam:
                    camelot_val = cam

            # 2) Try explicit header columns named 'camelot' or 'key' (if any)
            if (not key_val or key_val == ""):
                for idx, h in enumerate(lheaders):
                    if ('camelot' in h or 'key' in h) and idx < len(row):
                        possible = get_cell(idx)
                        kn, cam = extract_key_and_camelot_from_cell(possible)
                        if kn and not key_val:
                            key_val = kn
                        if cam and not camelot_val:
                            camelot_val = cam

            # 3) fallback: scan row for first cell that looks like a key or camelot
            if not key_val and not camelot_val:
                for idx in range(len(row)):
                    possible = get_cell(idx)
                    kn, cam = extract_key_and_camelot_from_cell(possible)
                    if kn and not key_val:
                        key_val = kn
                    if cam and not camelot_val:
                        camelot_val = cam
                    if key_val or camelot_val:
                        break

            # Compose a friendly key display string and set Song.camelot if we have it
            display_key = ""
            if key_val and camelot_val:
                # if camelot_val is numeric-only (like '11'), try to keep numeric with A/B if possible later;
                # still show combined value for clarity.
                display_key = f"{key_val} ({camelot_val})"
            elif camelot_val:
                display_key = f"{camelot_val}"
            elif key_val:
                display_key = key_val
            else:
                display_key = ""

            s = Song(title=title, artist=artist, key=display_key, tempo=tempo)
            # store detected camelot explicitly when possible
            if camelot_val:
                s.camelot = camelot_val
            songs.append(s)

        if songs:
            break

    # fallback: try JSON-embedded tracks
    if not songs:
        scripts = soup.find_all('script')
        for sc in scripts:
            text = sc.string or ""
            if not text:
                continue
            m = re.search(r'(?:"tracks"|\'tracks\'|tracks)\s*:\s*(\[[^\]]+\])', text, re.IGNORECASE | re.DOTALL)
            if m:
                arr_text = m.group(1)
                try:
                    arr = json.loads(arr_text)
                    for item in arr:
                        title = item.get('title') or item.get('name') or ""
                        artist = item.get('artist') or item.get('artists') or ""
                        if isinstance(artist, list):
                            artist = ", ".join(artist)
                        raw_key = item.get('key') or item.get('camelot') or ""
                        tempo = item.get('bpm') or item.get('tempo') or None
                        kn, cam = extract_key_and_camelot_from_cell(raw_key)
                        display_key = ""
                        if kn and cam:
                            display_key = f"{kn} ({cam})"
                        elif cam:
                            display_key = cam
                        elif kn:
                            display_key = kn
                        songs.append(Song(title=title, artist=artist, key=display_key, tempo=tempo, camelot=cam))
                    if songs:
                        break
                except Exception:
                    continue

    if not songs:
        raise RuntimeError(f"Could not parse songs from SongData page. Inspect: {songdata_url}")

    if debug:
        return songs, songdata_url, debug_info
    return songs, songdata_url, {}


# -----------------------
# Camelot-based HarmonicSequencer (full-featured)
# -----------------------
class HarmonicSequencer:
    def __init__(self):
        # Use standard music theory notation (sharps for major keys, context-appropriate for minors)
        self.camelot_to_key: Dict[str, str] = {
            '1A': 'G#m', '1B': 'B',      # G# minor / B major
            '2A': 'D#m', '2B': 'F#',     # D# minor / F# major  
            '3A': 'A#m', '3B': 'C#',     # A# minor / C# major
            '4A': 'Fm',  '4B': 'G#',     # F minor / G# major
            '5A': 'Cm',  '5B': 'D#',     # C minor / D# major
            '6A': 'Gm',  '6B': 'A#',     # G minor / A# major
            '7A': 'Dm',  '7B': 'F',      # D minor / F major
            '8A': 'Am',  '8B': 'C',      # A minor / C major
            '9A': 'Em',  '9B': 'G',      # E minor / G major
            '10A': 'Bm', '10B': 'D',     # B minor / D major
            '11A': 'F#m', '11B': 'A',    # F# minor / A major
            '12A': 'C#m', '12B': 'E'     # C# minor / E major
        }
        self.alias_to_camelot: Dict[str, str] = self._build_alias_map()
        self.camelot_codes: List[str] = list(self.camelot_to_key.keys())

    def _build_alias_map(self) -> Dict[str, str]:
        def clean(k: str) -> str:
            if not k:
                return ""
            s = str(k).strip()
            # Handle Unicode musical symbols that might appear in data
            s = s.replace('♭', 'b').replace('♯', '#')
            # Remove spaces and case variations in major/minor
            s = s.replace(' Major', '').replace(' major', '')
            s = s.replace(' Minor', 'm').replace(' minor', 'm')
            s = re.sub(r'\s+', '', s)
            return s.lower()

        m: Dict[str, str] = {}
        def add_aliases(aliases: List[str], code: str):
            for a in aliases:
                m[clean(a)] = code

        # 1A/1B - B major / G#m (Ab minor)
        add_aliases(['B', 'B major', 'Cb', 'Cb major'], '1B')
        add_aliases(['G#m', 'G# minor', 'Abm', 'Ab minor'], '1A')
        
        # 2A/2B - F# major / D#m (Eb minor) 
        add_aliases(['F#', 'F# major', 'Gb', 'Gb major'], '2B')
        add_aliases(['D#m', 'D# minor', 'Ebm', 'Eb minor'], '2A')
        
        # 3A/3B - Db major / Bbm (A# minor)
        add_aliases(['Db', 'Db major', 'C#', 'C# major'], '3B')
        add_aliases(['Bbm', 'Bb minor', 'A#m', 'A# minor'], '3A')
        
        # 4A/4B - Ab major / Fm
        add_aliases(['Ab', 'Ab major', 'G#', 'G# major'], '4B')
        add_aliases(['Fm', 'F minor'], '4A')
        
        # 5A/5B - Eb major / Cm
        add_aliases(['Eb', 'Eb major', 'D#', 'D# major'], '5B')
        add_aliases(['Cm', 'C minor'], '5A')
        
        # 6A/6B - Bb major / Gm
        add_aliases(['Bb', 'Bb major', 'A#', 'A# major'], '6B')
        add_aliases(['Gm', 'G minor'], '6A')
        
        # 7A/7B - F major / Dm
        add_aliases(['F', 'F major'], '7B')
        add_aliases(['Dm', 'D minor'], '7A')
        
        # 8A/8B - C major / Am
        add_aliases(['C', 'C major'], '8B')
        add_aliases(['Am', 'A minor'], '8A')
        
        # 9A/9B - G major / Em
        add_aliases(['G', 'G major'], '9B')
        add_aliases(['Em', 'E minor'], '9A')
        
        # 10A/10B - D major / Bm
        add_aliases(['D', 'D major'], '10B')
        add_aliases(['Bm', 'B minor'], '10A')
        
        # 11A/11B - A major / F#m (Gb minor)
        add_aliases(['A', 'A major'], '11B')
        add_aliases(['F#m', 'F# minor', 'Gbm', 'Gb minor'], '11A')
        
        # 12A/12B - E major / C#m (Db minor)
        add_aliases(['E', 'E major'], '12B')
        add_aliases(['C#m', 'C# minor', 'Dbm', 'Db minor'], '12A')

        return m

    def _clean_key_input(self, key: str) -> str:
        if not key:
            return ""
        s = str(key).strip()
        s = s.replace('♭', 'b').replace('♯', '#')
        s = s.replace('Major', '').replace('major', '')
        s = s.replace('Minor', 'm').replace('minor', 'm')
        s = re.sub(r'\s+', '', s)
        return s.lower()

    def key_to_camelot(self, key: str) -> Optional[str]:
        """
        Map many key spellings or combined forms to a single canonical Camelot code.
        Accepts input like 'A', 'A (11B)', '11B', 'C#m', 'Dbm', etc.
        """
        if not key:
            return None
        # Try to parse combined 'K (11B)' style first
        kn, cam = extract_key_and_camelot_from_cell(key)
        if cam:
            # if cam is numeric-only, try to infer A/B from alias map via keyname
            if cam.isdigit() and kn:
                mapped = self.alias_to_camelot.get(self._clean_key_input(kn))
                if mapped:
                    return mapped
                else:
                    return cam  # numeric-only fallback
            return cam
        cleaned = self._clean_key_input(key)
        # direct alias lookup
        if cleaned in self.alias_to_camelot:
            return self.alias_to_camelot[cleaned]
        # numeric-only input
        if CAMEL_NUMERIC_ONLY.match(cleaned):
            try:
                num = int(float(cleaned))
                if 1 <= num <= 12:
                    return str(num)
            except Exception:
                pass
        # As a last attempt, check for a single-letter key that maps to a camelot
        if cleaned and cleaned in self.alias_to_camelot:
            return self.alias_to_camelot[cleaned]
        return None

    def camelot_to_display(self, camelot_code: str) -> str:
        if not camelot_code:
            return "(unknown)"
        if str(camelot_code).isdigit():
            return f"{camelot_code} (unknown A/B)"
        name = self.camelot_to_key.get(camelot_code, camelot_code)
        return f"{name} ({camelot_code})"

    def camelot_neighbors(self, code: str) -> List[str]:
        if not code or code not in self.camelot_to_key:
            return []
        m = re.match(r'(\d+)([AB])', code)
        if not m:
            return []
        num = int(m.group(1))
        letter = m.group(2)
        other_letter = 'A' if letter == 'B' else 'B'
        prev_num = num - 1 if num > 1 else 12
        next_num = num + 1 if num < 12 else 1
        return [f"{num}{other_letter}", f"{prev_num}{letter}", f"{next_num}{letter}"]

    def camelot_score(self, c1: str, c2: str) -> float:
        if not c1 or not c2:
            return 0.2
        if c1 == c2:
            return 1.0
        # numeric-only conservative handling
        if str(c1).isdigit() or str(c2).isdigit():
            try:
                if int(float(c1)) == int(float(c2)):
                    return 0.6
            except Exception:
                pass
            return 0.3
        if c1 not in self.camelot_to_key or c2 not in self.camelot_to_key:
            return 0.2
        n1, l1 = int(re.match(r'(\d+)', c1).group(1)), c1[-1]
        n2, l2 = int(re.match(r'(\d+)', c2).group(1)), c2[-1]
        if n1 == n2 and l1 != l2:
            return 0.95
        diff = min((n1 - n2) % 12, (n2 - n1) % 12)
        if diff == 1:
            return 0.8 if l1 == l2 else 0.6
        if diff == 2:
            return 0.45 if l1 == l2 else 0.35
        return 0.2

    def _distance_score(self, a: str, b: str) -> float:
        ca = self.key_to_camelot(a) or ""
        cb = self.key_to_camelot(b) or ""
        return self.camelot_score(ca, cb)

    # Bridge logic (single, two-step, multi-hop) preserved...
    # Replace the suggest_bridge_keys method in your HarmonicSequencer class with this version:

    # Replace the suggest_bridge_keys method in your HarmonicSequencer class with this version:

    def suggest_bridge_keys(self, key1: str, key2: str, available_keys: Optional[List[str]] = None) -> List[str]:
        c1 = self.key_to_camelot(key1)
        c2 = self.key_to_camelot(key2)
        
        if not c1 or not c2 or c1 == c2:
            return []
        
        # If already harmonically compatible, no bridge needed
        if c2 in self.camelot_neighbors(c1):
            return []
        
        def get_available_marker(code: str) -> str:
            if not available_keys:
                return ""
            available_camelot = [self.key_to_camelot(k) for k in available_keys]
            return " ★" if code in available_camelot else ""
        
        # Simple BFS to find all shortest harmonic paths
        from collections import deque
        
        queue = deque([(c1, [])])  # (current_key, path_so_far)
        visited = {c1}
        all_paths = []
        min_length = float('inf')
        
        while queue:
            current, path = queue.popleft()
            
            # If path is longer than current minimum, skip
            if len(path) > min_length:
                continue
                
            # Check all harmonic neighbors
            for neighbor in self.camelot_neighbors(current):
                if neighbor == c2:
                    # Found destination - this is a complete path
                    final_path = path + [neighbor] if neighbor != c2 else path
                    if len(final_path) < min_length:
                        min_length = len(final_path)
                        all_paths = [final_path]
                    elif len(final_path) == min_length:
                        all_paths.append(final_path)
                elif neighbor not in visited:
                    visited.add(neighbor)
                    new_path = path + [neighbor]
                    queue.append((neighbor, new_path))
        
        # Remove destination from paths and format results
        results = []
        seen_paths = set()
        
        for path in all_paths:
            # Remove destination key if it appears in path
            bridge_path = [step for step in path if step != c2]
            
            if not bridge_path:  # Skip empty paths
                continue
                
            path_key = tuple(bridge_path)
            if path_key in seen_paths:
                continue
            seen_paths.add(path_key)
            
            if len(bridge_path) == 1:
                marker = get_available_marker(bridge_path[0])
                formatted = f"{self.camelot_to_display(bridge_path[0])}{marker}"
            else:
                path_parts = []
                for step in bridge_path:
                    marker = get_available_marker(step)
                    path_parts.append(f"{self.camelot_to_display(step)}{marker}")
                formatted = " -> ".join(path_parts)
            
            results.append(formatted)
        
        return results

    # Sequencing helpers (unchanged)
    def create_harmonic_sequence(self, songs: List[Song]) -> List[Song]:
        if not songs:
            return []
        remaining = songs[:]
        seq = [remaining.pop(0)]
        while remaining:
            last = seq[-1]
            best_idx = 0
            best_score = -1.0
            for i, s in enumerate(remaining):
                score = self._distance_score(last.key, s.key)
                if score > best_score:
                    best_score = score
                    best_idx = i
            seq.append(remaining.pop(best_idx))
        return seq

    def find_mixing_pairs(self, songs: List[Song]) -> List[Tuple[Song, Song, float]]:
        pairs = []
        for i in range(len(songs) - 1):
            a = songs[i]
            b = songs[i + 1]
            pairs.append((a, b, self._distance_score(a.key, b.key)))
        return pairs

    def analyze_song_collection(self, songs: List[Song], available_keys: Optional[List[str]] = None) -> Dict:
        for s in songs:
            # If Song.camelot was set by the fetcher, keep it; otherwise infer from s.key
            if not s.camelot:
                s.camelot = self.key_to_camelot(s.key)
        sequence = self.create_harmonic_sequence(songs)
        mixing_pairs = self.find_mixing_pairs(sequence)
        gaps_and_bridges = []
        existing_keys = [s.key for s in songs]
        for i in range(len(sequence) - 1):
            a = sequence[i]
            b = sequence[i + 1]
            score = self._distance_score(a.key, b.key)
            if score < 0.6:
                suggestions = self.suggest_bridge_keys(a.key, b.key, available_keys=existing_keys)
                gaps_and_bridges.append({'from': a, 'to': b, 'score': score, 'suggestions': suggestions})
        return {'sequence': sequence, 'mixing_pairs': mixing_pairs, 'gaps_and_bridges': gaps_and_bridges}


# -----------------------
# BPM fetch helper (best-effort)
# -----------------------
def fetch_bpm_from_web(title: str, artist: str, timeout: float = 5.0) -> Optional[int]:
    try:
        q = f"{title} {artist} bpm"
        url = f"https://www.google.com/search?q={requests.utils.quote(q)}"
        headers = {'User-Agent': 'Mozilla/5.0 (compatible; BPMFetcher/1.0)'}
        r = requests.get(url, headers=headers, timeout=timeout)
        if r.status_code != 200:
            return None
        text = r.text
        m = re.search(r'(\d{2,3})\s?BPM', text, re.IGNORECASE)
        if m:
            return int(m.group(1))
    except Exception:
        return None
    return None


# -----------------------
# Streamlit UI
# -----------------------
# ...existing code...

st.title("Harmonic Song Analyzer 🎧 — Camelot + SongData")
st.markdown("Analyze playlists, build harmonic play orders, and get bridge-key suggestions (Camelot system).")

left_col, right_col = st.columns([1, 2])

with left_col:
    st.header("Input")
    input_mode = st.radio(
        "Choose input method",
        ["Upload CSV / Paste", "Fetch from Spotify → SongData"],
        key="input_mode_radio"
    )
    uploaded_file = None
    pasted_text = ""
    songdata_input = ""
    sd_timeout = 300
    sd_debug = False
    fetch_songdata_btn = False

    if input_mode == "Upload CSV / Paste":
        # Add unique key if you use a file uploader
        uploaded_file = st.file_uploader("Upload CSV", type=["csv"], key="csv_uploader")
        pasted_text = st.text_area("Paste CSV or song list here", key="csv_paste_area")
    else:
        songdata_input = st.text_input("Spotify playlist URL", key="spotify_url_input")
        sd_timeout = st.slider("SongData fetch timeout (seconds)", min_value=5, max_value=600, value=300, key="sd_timeout_slider")
        sd_debug = st.checkbox("Show SongData fetch debug info", value=False, key="sd_debug_checkbox")
        fetch_songdata_btn = st.button("Fetch SongData playlist", key="fetch_songdata_btn")

    st.markdown("---")
    st.header("Options")
    shuffle_flag = st.checkbox("Shuffle input order before sequencing", value=False, key="shuffle_checkbox")
    auto_bpm = st.checkbox("Attempt to fetch missing tempos (slow)", value=False, key="auto_bpm_checkbox")
    
songs: List[Song] = []

# Uploaded CSV
if uploaded_file:
    try:
        df = pd.read_csv(uploaded_file)
        # detect key column heuristically in uploaded CSV
        def detect_key_col_in_df(df: pd.DataFrame) -> Optional[str]:
            cols = list(df.columns)
            rows = df.fillna("").astype(str).values.tolist()
            idx = detect_key_column_from_rows(rows, cols)
            if idx is not None and idx < len(cols):
                return cols[idx]
            for candidate in ['camelot', 'key', 'Input Key', 'Key', 'Camelot']:
                for c in cols:
                    if candidate.lower() == str(c).strip().lower():
                        return c
            # content-based fallback
            for c in cols:
                series = df[c].astype(str)
                nonempty = (series.str.strip() != "").sum()
                if nonempty == 0:
                    continue
                matches = series.apply(lambda x: is_probable_camelot(x) or is_probable_keyname(x)).sum()
                if (matches / nonempty) > 0.25:
                    return c
            return None

        key_col = detect_key_col_in_df(df)
        for _, r in df.iterrows():
            title = str(r.get('Track') or r.get('Title') or r.get('song') or r.get('Song') or "")
            artist = str(r.get('artist') or r.get('Artist') or "")
            tempo = None
            if 'tempo' in r and pd.notna(r.get('tempo')):
                tempo = r.get('tempo')
            key_display = ""
            camelot_val = None
            if key_col:
                raw = str(r.get(key_col) or "")
                kn, cam = extract_key_and_camelot_from_cell(raw)
                if kn and cam:
                    key_display = f"{kn} ({cam})"
                    camelot_val = cam
                elif cam:
                    key_display = cam
                    camelot_val = cam
                elif kn:
                    key_display = kn
                else:
                    # fallback: try explicit columns
                    key_display = str(r.get('key') or r.get('Key') or "")
            else:
                key_display = str(r.get('key') or r.get('Key') or r.get('Input Key') or "")
            s = Song(title=title, artist=artist, key=key_display, tempo=tempo)
            if camelot_val:
                s.camelot = camelot_val
            songs.append(s)
        st.success(f"Loaded {len(songs)} songs from uploaded CSV")
    except Exception as e:
        st.error(f"Could not read uploaded CSV: {e}")

# Pasted text
if pasted_text and not songs:
    sio = StringIO(pasted_text)
    try:
        df = pd.read_csv(sio, header=None)
        for _, r in df.iterrows():
            parts = [str(x) for x in r if pd.notna(x)]
            if len(parts) >= 3:
                tempo = int(parts[3]) if len(parts) > 3 and str(parts[3]).isdigit() else None
                songs.append(Song(title=parts[0], artist=parts[1], key=parts[2], tempo=tempo))
        st.success(f"Parsed {len(songs)} songs from pasted CSV/text")
    except Exception:
        lines = [l.strip() for l in pasted_text.splitlines() if l.strip()]
        for line in lines:
            parts = [p.strip() for p in re.split(r',|\t|;', line) if p.strip()]
            if len(parts) >= 3:
                tempo = int(parts[3]) if len(parts) > 3 and parts[3].isdigit() else None
                songs.append(Song(title=parts[0], artist=parts[1], key=parts[2], tempo=tempo))
        if songs:
            st.success(f"Parsed {len(songs)} songs from pasted text")

# Fetch from SongData if requested
if input_mode == "Fetch from Spotify → SongData" and fetch_songdata_btn:
    if not songdata_input:
        st.error("Please enter a Spotify playlist URL or URI.")
    else:
        with st.spinner("Fetching SongData playlist (may take some time)..."):
            try:
                fetched_songs, sd_url, debug_info = fetch_songdata_playlist(songdata_input, timeout=sd_timeout, debug=sd_debug)
                songs = fetched_songs
                st.success(f"Fetched {len(songs)} songs from SongData")
                st.markdown(f"[Open SongData playlist page]({sd_url})")
                if sd_debug:
                    st.subheader("SongData debug info")
                    st.write("Detected table headers (first few):")
                    st.write(debug_info.get('headers_seen', []))
                    st.write("Sample parsed rows (first few):")
                    st.write(debug_info.get('first_rows', []))
                    st.write("Chosen key column index:", debug_info.get('chosen_key_column'))
            except Exception as e:
                st.error(f"Failed to fetch/parse SongData playlist: {e}")
                pid = extract_spotify_playlist_id(songdata_input)
                if pid:
                    st.markdown(f"Try opening the SongData page directly: https://songdata.io/playlist/{pid}")

# Right column: results and analysis
with right_col:
    if songs:
        hs = HarmonicSequencer()

        if st.checkbox("Shuffle songs before sequencing", value=False):
            np.random.shuffle(songs)

        if st.checkbox("Attempt to fetch missing tempos (slow)", value=False):
            with st.spinner("Fetching BPMs..."):
                for s in songs:
                    if s.tempo is None or (isinstance(s.tempo, float) and np.isnan(s.tempo)):
                        bpm = fetch_bpm_from_web(s.title, s.artist)
                        if bpm:
                            s.tempo = bpm
                        time.sleep(0.12)

        # Ensure each Song has camelot assigned if possible
        for s in songs:
            if not s.camelot:
                s.camelot = hs.key_to_camelot(s.key)

        analysis = hs.analyze_song_collection(songs, available_keys=[s.key for s in songs])
        sequence = analysis['sequence']
        mixing_pairs = analysis['mixing_pairs']
        gaps_and_bridges = analysis['gaps_and_bridges']

        st.header("Recommended Play Order")
        rows = []
        for i, s in enumerate(sequence):
            rows.append({
                'Order': i + 1,
                'Title': s.title,
                'Artist': s.artist,
                'Camelot': hs.camelot_to_display(s.camelot) if s.camelot else '',
                'Tempo': s.tempo or ''
            })
        df_out = pd.DataFrame(rows)
        st.dataframe(df_out)
        st.download_button("Download recommended order (CSV)", df_out.to_csv(index=False).encode('utf-8'), file_name="recommended_order.csv", mime="text/csv")

        st.header("Mixing Pair Scores")
        mp_rows = []
        for a, b, sc in mixing_pairs:
            a_disp = hs.camelot_to_display(a.camelot) if a.camelot else "(unknown)"
            b_disp = hs.camelot_to_display(b.camelot) if b.camelot else "(unknown)"
            mp_rows.append({'From': f"{a.title} — {a.artist} {a_disp}", 'To': f"{b.title} — {b.artist} {b_disp}", 'Score': round(sc, 2)})
        st.table(pd.DataFrame(mp_rows))

        if gaps_and_bridges:
            st.header("Harmonic Gaps & Bridge Suggestions")
            for gb in gaps_and_bridges:
                a = gb['from']; b = gb['to']
                a_disp = hs.camelot_to_display(a.camelot) if a.camelot else a.key
                b_disp = hs.camelot_to_display(b.camelot) if b.camelot else b.key
                st.subheader(f"{a.title} — {a_disp} → {b.title} — {b_disp}  (score {gb['score']:.2f})")
                st.write("Bridge suggestions:")
                if gb['suggestions']:
                    for s in gb['suggestions']:
                        st.write(f"- {s}")
                else:
                    st.write("- (no suggestions found)")
                st.write("---")
        else:
            st.success("No major harmonic gaps detected — your order flows well!")

        if st.checkbox("Show normalized key mapping (debug)", value=False):
            mapping = []
            for s in sequence:
                mapping.append({'Title': s.title, 'Artist': s.artist, 'Input Key': s.key, 'Detected Camelot': s.camelot or '', 'Camelot Display': hs.camelot_to_display(s.camelot) if s.camelot else ''})
            st.dataframe(pd.DataFrame(mapping))

    else:
        st.info("No songs loaded yet. Upload/paste songs or fetch a SongData playlist using the left panel.")

st.markdown("---")
st.markdown(
    "Built with ❤️ — This version improves SongData parsing: it prefers an explicit 'Camelot' header, "
    "correctly parses 10/11/12 Camelot codes, and composes a readable 'Key (Camelot)' display for each song."
)
