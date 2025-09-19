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
KEY_NAME_REGEX = re.compile(r'^[A-Ga-g](?:#|b)?m?$')
KEY_NAME_IN_CELL = re.compile(r'[A-Ga-g](?:#|b)?(?:\s*(?:m|min|major|minor))?', re.I)


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
    if KEY_NAME_REGEX.match(s):
        return True
    if '(' in s and CAMLEOT_REGEX.search(s) and KEY_NAME_IN_CELL.search(s):
        return True
    if re.search(r'^[A-Ga-g](?:#|b)?\s*(?:minor|major|min|maj)$', s, re.I):
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
            if is_probable_camelot(cs) or is_probable_keyname(cs):
                scores[j] += 1

    ratios = [(scores[i] / counts[i]) if counts[i] > 0 else 0.0 for i in range(ncols)]

    # Prefer explicit "camelot" header if present
    camelot_indices = [i for i, h in enumerate(headers) if 'camelot' in h.lower()]
    if camelot_indices:
        # prefer first camelot column
        return camelot_indices[0]

    # Otherwise consider a 'key' header (only if it actually looks key-like based on content)
    key_indices = [i for i, h in enumerate(headers) if 'key' in h.lower()]
    if key_indices:
        ki = key_indices[0]
        if ki < len(ratios) and ratios[ki] > 0.25:
            return ki

    # fallback to the column with the best ratio (if decent)
    best_idx = int(np.argmax(ratios))
    best_ratio = ratios[best_idx] if best_idx < len(ratios) else 0.0
    if best_ratio >= 0.25:
        return best_idx

    # final fallback: None
    return None


def extract_key_and_camelot_from_cell(cell: str) -> Tuple[Optional[str], Optional[str]]:
    """
    Given a cell possibly containing both a key and a Camelot code (e.g. 'C (8B)' or '8A'),
    return (keyname, camelot_code).
    Camelot normalized to e.g. '8A' or '11B'. If the cell contains only a raw number '11'
    we return camelot as '11' (caller may combine with detected keyname).
    """
    if cell is None:
        return None, None
    s = str(cell).strip()
    if not s:
        return None, None

    # find Camelot token (prefer full e.g. '10B', '11A', '8A', '(8B)')
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

    # numeric only cell (like '11') -> return '11' (ambiguous A/B)
    if cam is None and CAMEL_NUMERIC_ONLY.match(s):
        try:
            num = int(float(s))
            if 1 <= num <= 12:
                cam = str(num)
        except Exception:
            pass

    # find keyname token like 'C', 'C#', 'C#m', 'Dbm', 'Am'
    keyname = None
    if KEY_NAME_REGEX.match(s):
        keyname = s
    else:
        m2 = KEY_NAME_IN_CELL.search(s)
        if m2:
            token = m2.group(0).strip()
            keyname = token

    if keyname:
        keyname = keyname.replace(' ', '')
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
        self.camelot_to_key: Dict[str, str] = {
            '1A': 'G#m', '1B': 'B',
            '2A': 'D#m', '2B': 'F#',
            '3A': 'Bbm', '3B': 'Db',
            '4A': 'Fm', '4B': 'Ab',
            '5A': 'Cm', '5B': 'Eb',
            '6A': 'Gm', '6B': 'Bb',
            '7A': 'Dm', '7B': 'F',
            '8A': 'Am', '8B': 'C',
            '9A': 'Em', '9B': 'G',
            '10A': 'Bm', '10B': 'D',
            '11A': 'F#m', '11B': 'A',
            '12A': 'C#m', '12B': 'E'
        }
        self.alias_to_camelot: Dict[str, str] = self._build_alias_map()
        self.camelot_codes: List[str] = list(self.camelot_to_key.keys())

    def _build_alias_map(self) -> Dict[str, str]:
        def clean(k: str) -> str:
            if not k:
                return ""
            s = str(k).strip()
            s = s.replace('♭', 'b').replace('♯', '#')
            s = s.replace(' Major', '').replace(' minor', 'm').replace(' Minor', 'm').replace(' major', '')
            s = re.sub(r'\s+', '', s)
            return s.lower()

        m: Dict[str, str] = {}
        def add_aliases(aliases: List[str], code: str):
            for a in aliases:
                m[clean(a)] = code

        add_aliases(['B', 'B major', 'Cb'], '1B')
        add_aliases(['G#m', 'Abm', 'G# minor', 'Ab minor'], '1A')
        add_aliases(['F#', 'F# major', 'Gb'], '2B')
        add_aliases(['D#m', 'Ebm', 'D# minor', 'Eb minor'], '2A')
        add_aliases(['Db', 'Db major', 'C#', 'C# major'], '3B')
        add_aliases(['Bbm', 'A#m', 'Bb minor', 'A# minor'], '3A')
        add_aliases(['Ab', 'Ab major', 'G#', 'G# major'], '4B')
        add_aliases(['Fm', 'F minor'], '4A')
        add_aliases(['Eb', 'Eb major', 'D#', 'D# major'], '5B')
        add_aliases(['Cm', 'C minor'], '5A')
        add_aliases(['Bb', 'Bb major', 'A#', 'A# major'], '6B')
        add_aliases(['Gm', 'G minor'], '6A')
        add_aliases(['F', 'F major', 'E#'], '7B')
        add_aliases(['Dm', 'D minor'], '7A')
        add_aliases(['C', 'C major', 'B#'], '8B')
        add_aliases(['Am', 'A minor'], '8A')
        add_aliases(['G', 'G major'], '9B')
        add_aliases(['Em', 'E minor'], '9A')
        add_aliases(['D', 'D major'], '10B')
        add_aliases(['Bm', 'B minor'], '10A')
        add_aliases(['A', 'A major'], '11B')
        add_aliases(['F#m', 'F# minor', 'Gbm'], '11A')
        add_aliases(['E', 'E major'], '12B')
        add_aliases(['C#m', 'C# minor', 'Dbm'], '12A')

        extras = {
            'c': '8B', 'am': '8A', 'g': '9B', 'em': '9A',
            'd': '10B', 'bm': '10A', 'a': '11B', 'f#m': '11A',
            'e': '12B', 'c#m': '12A', 'bb': '6B', 'bbm': '3A'
        }
        for k, v in extras.items():
            m[clean(k)] = v
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
    def suggest_bridge_keys(self, key1: str, key2: str, available_keys: Optional[List[str]] = None,
                            max_hops: int = 4, beam_width: int = 80) -> List[str]:
        c1 = self.key_to_camelot(key1)
        c2 = self.key_to_camelot(key2)
        if available_keys:
            available_camelot = [self.key_to_camelot(k) for k in available_keys]
            available_camelot = [x for x in available_camelot if x]
        else:
            available_camelot = []

        if not c1 or not c2:
            known = c1 or c2
            if known:
                neigh = [n for n in self.camelot_neighbors(known) if n in self.camelot_to_key]
                return [self.camelot_to_display(n) for n in neigh][:6]
            return []

        if self.camelot_score(c1, c2) >= 0.8:
            return []

        all_codes = list(self.camelot_to_key.keys())

        def boost_for_available(code: str) -> float:
            return 0.12 if code in available_camelot else 0.0

        single_candidates: List[Tuple[str, float]] = []
        for cand in all_codes:
            if cand in (c1, c2):
                continue
            score = self.camelot_score(c1, cand) + self.camelot_score(cand, c2) + boost_for_available(cand)
            if score > 1.2:
                single_candidates.append((cand, score))
        single_candidates.sort(key=lambda x: x[1], reverse=True)

        two_step: List[Tuple[Tuple[str, str], float]] = []
        for a in all_codes:
            if a in (c1, c2):
                continue
            s1 = self.camelot_score(c1, a)
            if s1 < 0.35:
                continue
            for b in all_codes:
                if b in (c1, c2, a):
                    continue
                s2 = self.camelot_score(a, b)
                s3 = self.camelot_score(b, c2)
                if s2 < 0.32 or s3 < 0.32:
                    continue
                score = s1 + s2 + s3 + boost_for_available(a) + boost_for_available(b)
                two_step.append(((a, b), score))
        two_step.sort(key=lambda x: x[1], reverse=True)

        def find_chains(max_hops_local: int = max_hops, beam_width_local: int = beam_width,
                        min_edge_score: float = 0.32):
            beams: List[Tuple[List[str], float]] = [([c1], 0.0)]
            results: List[Tuple[List[str], float]] = []
            for depth in range(1, max_hops_local + 1):
                new_beams: List[Tuple[List[str], float]] = []
                for path, score in beams:
                    last = path[-1]
                    for cand in all_codes:
                        if cand in path:
                            continue
                        edge = self.camelot_score(last, cand)
                        if edge < min_edge_score:
                            continue
                        new_score = score + edge + boost_for_available(cand)
                        new_path = path + [cand]
                        if cand == c2:
                            results.append((new_path[1:], new_score))
                        else:
                            new_beams.append((new_path, new_score))
                beams = sorted(new_beams, key=lambda x: x[1], reverse=True)[:beam_width_local]
            results.sort(key=lambda x: x[1], reverse=True)
            return results

        chain_results = find_chains()

        results: List[str] = []
        for cand, score in single_candidates[:6]:
            disp = self.camelot_to_display(cand)
            if disp not in results:
                results.append(disp)

        for (a, b), score in two_step[:6]:
            formatted = f"{self.camelot_to_display(a)} -> {self.camelot_to_display(b)}"
            if formatted not in results:
                results.append(formatted)

        for path, score in chain_results[:6]:
            formatted = " -> ".join(self.camelot_to_display(p) for p in path)
            if formatted not in results:
                results.append(formatted)

        if not results:
            neigh = self.camelot_neighbors(c1)
            for n in neigh:
                if n in self.camelot_to_key:
                    results.append(self.camelot_to_display(n))

        return results[:8]

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
st.title("Harmonic Song Analyzer 🎧 — Camelot + SongData")
st.markdown("Analyze playlists, build harmonic play orders, and get bridge-key suggestions (Camelot system).")

left_col, right_col = st.columns([1, 2])

with left_col:
    st.header("Input")
    input_mode = st.radio("Choose input method", ["Upload CSV / Paste", "Fetch from Spotify → SongData"])
    uploaded_file = None
    pasted_text = ""
    songdata_input = ""
    sd_timeout = 30
    sd_debug = False
    fetch_songdata_btn = False

    if input_mode == "Upload CSV / Paste":
        uploaded_file = st.file_uploader("Upload CSV of songs (title,artist,key,tempo?)", type=["csv"])
        pasted_text = st.text_area("Or paste CSV/text (title,artist,key,tempo)", height=160)
    else:
        songdata_input = st.text_input("Spotify playlist URL or URI", placeholder="https://open.spotify.com/playlist/...")
        sd_timeout = st.number_input("SongData fetch timeout (seconds)", min_value=5, max_value=120, value=30, step=5)
        sd_debug = st.checkbox("Show SongData fetch debug info", value=False)
        fetch_songdata_btn = st.button("Fetch SongData playlist", key="fetch_songdata_btn")

    st.markdown("---")
    st.header("Options")
    shuffle_flag = st.checkbox("Shuffle input order before sequencing", value=False)
    show_normalized = st.checkbox("Show normalized keys & Camelot codes", value=True)
    auto_bpm = st.checkbox("Attempt to fetch missing tempos (slow)", value=False)
    st.markdown("Advanced bridge options:")
    max_bridge_hops = st.slider("Max bridge hops (beam search depth)", min_value=2, max_value=5, value=4)
    beam_width = st.slider("Beam width (search pruning)", min_value=20, max_value=200, value=80)
    st.markdown("---")
    st.header("Export / Save")
    # export in right pane

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
            title = str(r.get('title') or r.get('Title') or r.get('song') or r.get('Song') or "")
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
                'Input Key': s.key,
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
                st.write("Bridge suggestions (single keys first, then sequences):")
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
