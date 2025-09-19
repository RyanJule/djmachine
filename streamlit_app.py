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
# SongData fetch helpers (fixed parser: header-aware)
# -----------------------
def extract_spotify_playlist_id(spotify_url: str) -> Optional[str]:
    """Extract the playlist id from several Spotify URL/URI formats."""
    if not spotify_url:
        return None
    s = spotify_url.strip()
    # common open.spotify.com pattern
    m = re.search(r'playlist/([A-Za-z0-9]+)', s)
    if m:
        return m.group(1)
    # URI format
    m = re.search(r'spotify:playlist:([A-Za-z0-9]+)', s)
    if m:
        return m.group(1)
    # maybe user pasted just the id
    if re.fullmatch(r'[A-Za-z0-9]+', s):
        return s
    return None


def fetch_songdata_playlist(spotify_url: str,
                            timeout: float = 30.0,
                            debug: bool = False) -> Tuple[List[Song], str, Dict]:
    """
    Fetch SongData playlist for a spotify playlist id/url.
    Returns: (list_of_Song, songdata_url, debug_info)
    Raises RuntimeError on parse/fetch failure.
    debug_info contains headers_found and sample_rows when debug=True.
    """
    pid = extract_spotify_playlist_id(spotify_url)
    if not pid:
        raise RuntimeError("Couldn't extract Spotify playlist id from that input. Use a full Spotify playlist URL or URI.")
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
    debug_info = {'headers_seen': [], 'first_rows': []}

    # Heuristic: find the first table that contains Title & Artist columns
    for table in tables:
        headers = []
        thead = table.find('thead')
        if thead:
            headers = [th.get_text(strip=True).lower() for th in thead.find_all('th')]
        else:
            # try first row as header fallback
            first_row = table.find('tr')
            if first_row:
                headers = [td.get_text(strip=True).lower() for td in first_row.find_all(['th', 'td'])]
        headers = [h.strip() for h in headers if h.strip()]
        if not headers:
            continue

        debug_info['headers_seen'].append(headers)

        # Check for presence of song/title and artist columns
        has_title = any('title' in h or 'song' in h or 'track' in h for h in headers)
        has_artist = any('artist' in h for h in headers)
        if not (has_title and has_artist):
            continue

        # Build header map to required fields (title, artist, key, tempo)
        header_map: Dict[str, int] = {}
        for idx, h in enumerate(headers):
            if any(k in h for k in ("title", "song", "track")) and 'title' not in header_map:
                header_map['title'] = idx
            elif 'artist' in h and 'artist' not in header_map:
                header_map['artist'] = idx
            elif ('key' in h or 'camelot' in h) and 'key' not in header_map:
                header_map['key'] = idx
            elif ('bpm' in h or 'tempo' in h) and 'tempo' not in header_map:
                header_map['tempo'] = idx
            # ignore columns like 'added', 'duration', 'time', etc.

        # Parse rows - robustly handle missing columns
        tbody = table.find('tbody') or table
        rows = tbody.find_all('tr')
        sample_rows = []
        for row in rows:
            cols = row.find_all(['td', 'th'])
            if not cols:
                continue

            def get_cell(i: int) -> str:
                try:
                    return cols[i].get_text(strip=True)
                except Exception:
                    return ""

            title = get_cell(header_map.get('title', -1)) if 'title' in header_map else ""
            artist = get_cell(header_map.get('artist', -1)) if 'artist' in header_map else ""
            key = get_cell(header_map.get('key', -1)) if 'key' in header_map else ""
            tempo_txt = get_cell(header_map.get('tempo', -1)) if 'tempo' in header_map else ""
            tempo = None
            if tempo_txt:
                m = re.search(r'(\d{2,3})', tempo_txt)
                if m:
                    tempo = int(m.group(1))

            # Only append rows with at least a title or artist
            if title or artist:
                songs.append(Song(title=title, artist=artist, key=key, tempo=tempo))
            # collect a small sample for debugging
            if len(sample_rows) < 6:
                sample_rows.append([get_cell(i) for i in range(min(len(cols), 8))])

        debug_info['first_rows'] = sample_rows

        if songs:
            break  # stop after first plausible table

    # If not found, attempt to find JSON-embedded track list inside script tags (best-effort)
    if not songs:
        scripts = soup.find_all('script')
        for sc in scripts:
            text = sc.string or ""
            if not text:
                continue
            # naive JSON array extraction for 'tracks' or similar arrays
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
                        key = item.get('key') or item.get('camelot') or ""
                        tempo = item.get('bpm') or item.get('tempo') or None
                        songs.append(Song(title=title, artist=artist, key=key, tempo=tempo))
                    if songs:
                        break
                except Exception:
                    continue

    if not songs:
        raise RuntimeError(f"Could not parse songs from SongData page. Inspect the page at: {songdata_url}")

    if debug:
        return songs, songdata_url, debug_info
    return songs, songdata_url, {}


# -----------------------
# Camelot-based HarmonicSequencer (full-featured)
# -----------------------
class HarmonicSequencer:
    def __init__(self):
        # canonical mapping: Camelot code -> display key
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

        # Build alias map: many user-visible spellings -> canonical Camelot code
        self.alias_to_camelot: Dict[str, str] = self._build_alias_map()
        self.camelot_codes: List[str] = list(self.camelot_to_key.keys())

    def _build_alias_map(self) -> Dict[str, str]:
        def clean(k: str) -> str:
            if not k:
                return ""
            s = str(k).strip()
            s = s.replace('♭', 'b').replace('♯', '#')
            s = s.replace(' Major', '').replace(' major', '')
            s = s.replace(' Minor', 'm').replace(' minor', 'm')
            s = re.sub(r'\s+', '', s)
            return s.lower()

        m: Dict[str, str] = {}

        def add_aliases(aliases: List[str], code: str):
            for a in aliases:
                m[clean(a)] = code

        # fill mapping explicitly for deterministic behavior
        add_aliases(['B', 'B major', 'Cb'], '1B')
        add_aliases(['G#m', 'Abm', 'G# minor', 'Ab minor', 'g#minor', 'abminor'], '1A')

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

        # extras and short forms
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
        if not key:
            return None
        cleaned = self._clean_key_input(key)
        # direct mapping
        if cleaned in self.alias_to_camelot:
            return self.alias_to_camelot[cleaned]
        # try variants with/without trailing 'm'
        if cleaned.endswith('m') and cleaned[:-1] in self.alias_to_camelot:
            return self.alias_to_camelot[cleaned[:-1]]
        if (cleaned + 'm') in self.alias_to_camelot:
            return self.alias_to_camelot[cleaned + 'm']
        return None

    def camelot_to_display(self, camelot_code: str) -> str:
        name = self.camelot_to_key.get(camelot_code, camelot_code)
        return f"{name} ({camelot_code})"

    def camelot_neighbors(self, code: str) -> List[str]:
        if code not in self.camelot_to_key:
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
        """
        Heuristic compatibility score in Camelot-space (0..1).
        """
        if not c1 or not c2:
            return 0.2
        if c1 == c2:
            return 1.0
        if c1 not in self.camelot_to_key or c2 not in self.camelot_to_key:
            return 0.2
        n1, l1 = int(re.match(r'(\d+)', c1).group(1)), c1[-1]
        n2, l2 = int(re.match(r'(\d+)', c2).group(1)), c2[-1]
        if n1 == n2 and l1 != l2:
            return 0.95
        # circular numeric distance
        diff = min((n1 - n2) % 12, (n2 - n1) % 12)
        if diff == 1:
            return 0.8 if l1 == l2 else 0.6
        if diff == 2:
            return 0.45 if l1 == l2 else 0.35
        return 0.2

    # compatibility wrapper used by older functions (keeps API)
    def _distance_score(self, a: str, b: str) -> float:
        ca = self.key_to_camelot(a) or ""
        cb = self.key_to_camelot(b) or ""
        return self.camelot_score(ca, cb)

    # ------------------------------------------------------------------
    # Bridge suggestion: single keys, two-step chains, multi-hop beam search
    # ------------------------------------------------------------------
    def suggest_bridge_keys(self, key1: str, key2: str, available_keys: Optional[List[str]] = None,
                            max_hops: int = 4, beam_width: int = 80) -> List[str]:
        """
        Suggest keys or key sequences to bridge key1 -> key2.
        Returns human readable display strings (e.g. "Am (8A)" or "G#m (1A) -> D#m (2A)").
        If available_keys provided, prefers chains that use those (useful when mixing through existing set).
        """
        c1 = self.key_to_camelot(key1)
        c2 = self.key_to_camelot(key2)
        if available_keys:
            available_camelot = [self.key_to_camelot(k) for k in available_keys]
            available_camelot = [x for x in available_camelot if x]
        else:
            available_camelot = []

        # if mapping missing, be permissive and return neighbors of the known one
        if not c1 or not c2:
            known = c1 or c2
            if known:
                neigh = [n for n in self.camelot_neighbors(known) if n in self.camelot_to_key]
                return [self.camelot_to_display(n) for n in neigh][:6]
            return []

        # if already sufficiently compatible, no bridge needed
        if self.camelot_score(c1, c2) >= 0.8:
            return []

        all_codes = list(self.camelot_to_key.keys())

        def boost_for_available(code: str) -> float:
            return 0.12 if code in available_camelot else 0.0

        # Single-key candidates: cand where score(c1->cand) + score(cand->c2) is high
        single_candidates: List[Tuple[str, float]] = []
        for cand in all_codes:
            if cand in (c1, c2):
                continue
            score = self.camelot_score(c1, cand) + self.camelot_score(cand, c2) + boost_for_available(cand)
            if score > 1.2:
                single_candidates.append((cand, score))
        single_candidates.sort(key=lambda x: x[1], reverse=True)

        # Two-step chains (A -> B)
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

        # Beam-search multi-hop (up to max_hops)
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
                            # store intermediate nodes (exclude starting key)
                            results.append((new_path[1:], new_score))
                        else:
                            new_beams.append((new_path, new_score))
                # prune beams
                beams = sorted(new_beams, key=lambda x: x[1], reverse=True)[:beam_width_local]
            results.sort(key=lambda x: x[1], reverse=True)
            return results

        chain_results = find_chains()

        # Format final suggestions: prefer single candidates, then two-step, then multi-hop chains
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

        # fallback: neighbors of c1
        if not results:
            neigh = self.camelot_neighbors(c1)
            for n in neigh:
                if n in self.camelot_to_key:
                    results.append(self.camelot_to_display(n))

        return results[:8]

    # -----------------------
    # Sequencing helpers
    # -----------------------
    def create_harmonic_sequence(self, songs: List[Song]) -> List[Song]:
        """
        Greedy nearest neighbor sequence to minimize harmonic jumps.
        """
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
    """
    Lightweight attempt to fetch BPM from a web search result snippet.
    Best-effort and unreliable; used as fallback when tempo missing.
    """
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

# Layout: left column for inputs/options, right for results
left_col, right_col = st.columns([1, 2])

with left_col:
    st.header("Input")
    input_mode = st.radio("Choose input method", ["Upload CSV / Paste", "Fetch from Spotify → SongData"])
    uploaded_file = None
    pasted_text = ""
    songdata_url_input = ""
    sd_timeout = 30.0
    sd_debug = False
    if input_mode == "Upload CSV / Paste":
        uploaded_file = st.file_uploader("Upload CSV of songs (title,artist,key,tempo?)", type=["csv"])
        pasted_text = st.text_area("Or paste CSV/text (title,artist,key,tempo)", height=160)
    else:
        songdata_input = st.text_input("Spotify playlist URL or URI", placeholder="https://open.spotify.com/playlist/...")
        sd_timeout = st.number_input("SongData fetch timeout (seconds)", min_value=5, max_value=120, value=30, step=5)
        sd_debug = st.checkbox("Show SongData fetch debug info", value=False)
        if st.button("Fetch SongData playlist"):
            songdata_url_input = songdata_input.strip()

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
    if st.button("Export recommended order as CSV"):
        # placeholder — we'll generate later if songs present
        st.info("Export will appear in results pane after analysis.")


# Load songs from inputs
songs: List[Song] = []

# 1) Upload
if uploaded_file:
    try:
        df = pd.read_csv(uploaded_file)
        for _, r in df.iterrows():
            title = str(r.get('title') or r.get('Title') or r.get('song') or r.get('Song') or "")
            artist = str(r.get('artist') or r.get('Artist') or "")
            key = str(r.get('key') or r.get('Key') or "")
            tempo = r.get('tempo') if 'tempo' in r else None
            songs.append(Song(title=title, artist=artist, key=key, tempo=tempo))
        st.success(f"Loaded {len(songs)} songs from uploaded CSV")
    except Exception as e:
        st.error(f"Could not read uploaded CSV: {e}")

# 2) Paste
if pasted_text and not songs:
    # attempt to parse as CSV-like first
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
        # fallback: line-by-line parsing
        lines = [l.strip() for l in pasted_text.splitlines() if l.strip()]
        for line in lines:
            parts = [p.strip() for p in re.split(r',|\t|;', line) if p.strip()]
            if len(parts) >= 3:
                tempo = int(parts[3]) if len(parts) > 3 and parts[3].isdigit() else None
                songs.append(Song(title=parts[0], artist=parts[1], key=parts[2], tempo=tempo))
        if songs:
            st.success(f"Parsed {len(songs)} songs from pasted text")

# 3) Fetch from SongData (via Spotify playlist id)
if songdata_url_input:
    try:
        with st.spinner("Fetching SongData playlist (this can take a while)..."):
            fetched_songs, sd_url, debug_info = fetch_songdata_playlist(songdata_url_input, timeout=sd_timeout, debug=sd_debug)
            songs = fetched_songs
            st.success(f"Fetched {len(songs)} songs from SongData")
            st.markdown(f"[Open SongData playlist page]({sd_url})")
            if sd_debug:
                st.subheader("SongData debug info")
                st.write("Detected table headers (first few):")
                st.write(debug_info.get('headers_seen', []))
                st.write("Sample parsed rows (first few):")
                st.write(debug_info.get('first_rows', []))
    except Exception as e:
        st.error(f"Failed to fetch/parse SongData playlist: {e}")
        # offer direct SongData link if we could parse an ID
        pid = extract_spotify_playlist_id(songdata_url_input)
        if pid:
            st.markdown(f"Try opening the SongData page directly: https://songdata.io/playlist/{pid}")

# Right column: results and analysis
with right_col:
    if songs:
        hs = HarmonicSequencer()
        # optionally shuffle input songs
        if shuffle_flag:
            np.random.shuffle(songs)

        # attempt to fetch missing tempos if user opted in
        if auto_bpm:
            with st.spinner("Fetching missing BPMs (best-effort, may be slow)..."):
                for s in songs:
                    if s.tempo is None or (isinstance(s.tempo, float) and np.isnan(s.tempo)):
                        bpm = fetch_bpm_from_web(s.title, s.artist)
                        if bpm:
                            s.tempo = bpm
                        time.sleep(0.15)

        # Analyze
        analysis = hs.analyze_song_collection(songs, available_keys=[s.key for s in songs])
        sequence = analysis['sequence']
        mixing_pairs = analysis['mixing_pairs']
        gaps_and_bridges = analysis['gaps_and_bridges']

        # Display Recommended Play Order
        st.header("Recommended Play Order")
        order_rows = []
        for i, s in enumerate(sequence):
            camelot_display = hs.camelot_to_display(s.camelot) if s.camelot else "(unknown)"
            order_rows.append({
                'Order': i + 1,
                'Title': s.title,
                'Artist': s.artist,
                'Input Key': s.key,
                'Camelot': camelot_display,
                'Tempo': s.tempo or ''
            })
        df_order = pd.DataFrame(order_rows)
        st.dataframe(df_order)

        # Export CSV button
        csv_bytes = df_order.to_csv(index=False).encode('utf-8')
        st.download_button("Download recommended order (CSV)", data=csv_bytes, file_name="recommended_order.csv", mime="text/csv")

        # Mixing pair scores
        st.header("Mixing Pair Scores (Camelot-based)")
        mp_rows = []
        for a, b, score in mixing_pairs:
            a_disp = hs.camelot_to_display(a.camelot) if a.camelot else "(unknown)"
            b_disp = hs.camelot_to_display(b.camelot) if b.camelot else "(unknown)"
            mp_rows.append({'From': f"{a.title} — {a.artist} {a_disp}", 'To': f"{b.title} — {b.artist} {b_disp}", 'Score': round(score, 2)})
        st.table(pd.DataFrame(mp_rows))

        # Gaps and bridge suggestions
        if gaps_and_bridges:
            st.header("Harmonic Gaps & Bridge Suggestions")
            for gb in gaps_and_bridges:
                a = gb['from']
                b = gb['to']
                a_disp = hs.camelot_to_display(a.camelot) if a.camelot else a.key
                b_disp = hs.camelot_to_display(b.camelot) if b.camelot else b.key
                st.subheader(f"{a.title} — {a_disp}  →  {b.title} — {b_disp}  (score {gb['score']:.2f})")
                st.write("Bridge suggestions (single keys first, then sequences):")
                if gb['suggestions']:
                    for s in gb['suggestions']:
                        st.write(f"- {s}")
                else:
                    st.write("- (no suggestions found)")
                st.write("---")
        else:
            st.success("No major harmonic gaps detected — your order flows well!")

        # Debug / Inspect mapping table
        if show_normalized:
            st.header("Normalized Keys / Camelot Codes")
            mapping_rows = []
            for s in sequence:
                mapping_rows.append({
                    'Title': s.title,
                    'Artist': s.artist,
                    'Input Key': s.key,
                    'Camelot': s.camelot or '',
                    'Camelot Display': hs.camelot_to_display(s.camelot) if s.camelot else ''
                })
            st.dataframe(pd.DataFrame(mapping_rows))

    else:
        st.info("No songs loaded yet. Upload a CSV, paste songs, or fetch a SongData playlist from Spotify in the left panel.")


# Footer
st.markdown("---")
st.markdown(
    "Built with ❤️ — Harmonic suggestions use the **Camelot** system and attempt to parse SongData playlist pages robustly. "
    "SongData parsing is best-effort: if a page renders its main table with client-side JavaScript, the server-side HTML may not contain the table and parsing will fail. "
    "If parsing fails, try opening the SongData page and exporting CSV (if available), or increase the fetch timeout in the sidebar."
)
