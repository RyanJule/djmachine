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
import base64
import hashlib
import secrets
from urllib.parse import urlencode

# -----------------------
# Helpers (serialization, session-state, oauth safe handling)
# -----------------------
def _ensure_session_keys(*keys):
    for k in keys:
        if k not in st.session_state:
            st.session_state[k] = None

def _serialize_songs(songs):
    try:
        return [s.__dict__ for s in songs]
    except Exception:
        return list(songs)

def _deserialize_songs(lst):
    try:
        return [Song(**d) for d in lst]
    except Exception:
        return lst

def generate_code_verifier():
    """Generate code verifier for PKCE"""
    return base64.urlsafe_b64encode(secrets.token_bytes(32)).decode('utf-8').rstrip('=')

def generate_code_challenge(verifier):
    """Generate code challenge from verifier for PKCE"""
    digest = hashlib.sha256(verifier.encode('utf-8')).digest()
    return base64.urlsafe_b64encode(digest).decode('utf-8').rstrip('=')

def get_spotify_auth_url(client_id: str, redirect_uri: str, code_challenge: str, state: str):
    """Generate Spotify authorization URL"""
    scopes = [
        'playlist-modify-public',
        'playlist-modify-private',
        'playlist-read-private',
        'playlist-read-collaborative'
    ]

    params = {
        'client_id': client_id,
        'response_type': 'code',
        'redirect_uri': redirect_uri,
        'code_challenge_method': 'S256',
        'code_challenge': code_challenge,
        'state': state,
        'scope': ' '.join(scopes)
    }

    return f"https://accounts.spotify.com/authorize?{urlencode(params)}"

def exchange_code_for_token(client_id: str, code: str, redirect_uri: str, code_verifier: str):
    """Exchange authorization code for access token"""
    token_data = {
        'client_id': client_id,
        'grant_type': 'authorization_code',
        'code': code,
        'redirect_uri': redirect_uri,
        'code_verifier': code_verifier,
    }

    response = requests.post(
        'https://accounts.spotify.com/api/token',
        data=token_data,
        headers={'Content-Type': 'application/x-www-form-urlencoded'}
    )

    if response.status_code == 200:
        return response.json()
    else:
        raise Exception(f"Token exchange failed: {response.text}")

def _process_oauth_callback_if_present(spotify_client_id, REDIRECT_URI, exchange_code_for_token):
    """
    Safely process OAuth callback from st.query_params.
    Extracts single-string 'code' and 'state', compares with stored state,
    exchanges code for token, stores token in session_state, sets a session flag,
    clears query params, and triggers a rerun.
    """
    try:
        query_params = st.query_params
    except Exception:
        return

    code_from_qp = query_params.get('code', [None])[0] if query_params else None
    state_from_qp = query_params.get('state', [None])[0] if query_params else None

    if code_from_qp and state_from_qp:
        if st.session_state.get('auth_state') and state_from_qp == st.session_state.get('auth_state') and st.session_state.get('code_verifier'):
            try:
                token_response = exchange_code_for_token(
                    client_id=spotify_client_id,
                    code=code_from_qp,
                    redirect_uri=REDIRECT_URI,
                    code_verifier=st.session_state.get('code_verifier')
                )
                access_token = token_response.get('access_token')
                if access_token:
                    st.session_state.spotify_access_token = access_token
                    st.session_state._oauth_just_completed = True
                    try:
                        st.experimental_set_query_params()
                    except Exception:
                        pass
                    try:
                        st.experimental_rerun()
                    except Exception:
                        pass
            except Exception as e:
                st.error(f"Authentication failed during callback processing: {e}")

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
CAMLEOT_REGEX = re.compile(r'\(?\s*(?:1[0-2]|[1-9])\s*[ABab]\s*\)?')
CAMEL_NUMERIC_ONLY = re.compile(r'^\s*\d+(\.\d+)?\s*$')

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
    if re.search(r'^[A-Ga-g](?:[#♯]|[b♭])?\s*(?:minor|major|Minor|Major)?$', s, re.I):
        return True
    return False

def detect_key_column_from_rows(rows: List[List[str]], headers: List[str]) -> Optional[int]:
    if not rows:
        return None
    ncols = max(len(r) for r in rows)
    scores = [0.0] * ncols
    counts = [0] * ncols
    for r in rows:
        for j in range(ncols):
            cell = r[j] if j < len(r) else ""
            cs = str(cell).strip() if cell is not None else ""
            if cs == "":
                continue
            counts[j] += 1
            if is_probable_keyname(cs):
                scores[j] += 1
    ratios = [(scores[i] / counts[i]) if counts[i] > 0 else 0.0 for i in range(ncols)]
    key_indices = [i for i, h in enumerate(headers) if 'key' in h.lower()]
    if key_indices:
        ki = key_indices[0]
        if ki < len(ratios) and ratios[ki] > 0.25:
            return ki
    return None

def extract_key_and_camelot_from_cell(cell: str) -> Tuple[Optional[str], Optional[str]]:
    if cell is None:
        return None, None
    s = str(cell).strip()
    if not s:
        return None, None
    s = s.replace('♭', 'b').replace('♯', '#')
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
    if cam is None and CAMEL_NUMERIC_ONLY.match(s):
        try:
            num = int(float(s))
            if 1 <= num <= 12:
                cam = str(num)
        except Exception:
            pass
    keyname = None
    if re.match(r'^[A-Ga-g](?:[#b])?(?:\s*(?:major|minor|Major|Minor))?$', s, re.I):
        keyname = s
    else:
        key_pattern = re.search(r'([A-Ga-g])([#b]?)\s*(major|minor|Major|Minor)?', s, re.I)
        if key_pattern:
            note = key_pattern.group(1).upper()
            accidental = key_pattern.group(2) or ''
            mode = key_pattern.group(3) or ''
            if mode.lower() == 'minor':
                mode = 'm'
            elif mode.lower() == 'major':
                mode = ''
            else:
                mode = ''
            keyname = f"{note}{accidental}{mode}"
    if keyname:
        keyname = keyname.strip()
        if len(keyname) >= 1:
            keyname = keyname[0].upper() + keyname[1:]
    return keyname, cam

# -----------------------
# SongData fetch + parse
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
    debug_info = {'headers_seen': [], 'first_rows': [], 'chosen_key_column': None, 'header_id_map': {}}
    def parse_tempo(text: str):
        if not text:
            return None
        m = re.search(r'(\d{2,3})', text)
        if m:
            try:
                return int(m.group(1))
            except Exception:
                return None
        return None
    for table in tables:
        thead = table.find('thead')
        header_ths = []
        if thead:
            header_rows = thead.find_all('tr')
            if header_rows:
                header_ths = header_rows[-1].find_all('th')
        else:
            first_row = table.find('tr')
            if first_row:
                header_ths = first_row.find_all('th')
        header_pos_map: Dict[str, int] = {}
        id_map: Dict[str, int] = {}
        text_map: Dict[str, int] = {}
        header_texts: List[str] = []
        for pos, th in enumerate(header_ths):
            th_id = (th.get('id') or "").strip().lower()
            data_col = th.get('data-column-index')
            th_text = th.get_text(strip=True)
            ltxt = th_text.lower().strip()
            header_texts.append(th_text)
            if th_id:
                id_map[th_id] = pos
            if 'track' in ltxt or 'title' in ltxt or 'song' in ltxt:
                text_map.setdefault('title', pos)
            if 'artist' in ltxt:
                text_map.setdefault('artist', pos)
            if 'camelot' in ltxt:
                text_map.setdefault('camelot', pos)
            if ltxt == 'key' or 'key' in ltxt:
                text_map.setdefault('key', pos)
            if 'bpm' in ltxt or 'tempo' in ltxt:
                text_map.setdefault('tempo', pos)
            if data_col is not None:
                try:
                    header_pos_map[f"colidx_{int(data_col)}"] = pos
                except Exception:
                    pass
        debug_info['headers_seen'].append(header_texts)
        debug_info['header_id_map'].update(id_map)
        found_title = ('track_col' in id_map) or ('title' in text_map) or any('track' in h.lower() or 'title' in h.lower() for h in header_texts)
        found_artist = ('artist_col' in id_map) or ('artist' in text_map) or any('artist' in h.lower() for h in header_texts)
        if not (found_title and found_artist):
            continue
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
        header_field_index: Dict[str, int] = {}
        if 'track_col' in id_map:
            header_field_index['title'] = id_map['track_col']
        if 'artist_col' in id_map:
            header_field_index['artist'] = id_map['artist_col']
        if 'camelot_col' in id_map:
            header_field_index['camelot'] = id_map['camelot_col']
        if 'key_col' in id_map:
            header_field_index['key'] = id_map['key_col']
        if 'bpm_col' in id_map:
            header_field_index['tempo'] = id_map['bpm_col']
        for k, pos in text_map.items():
            if k not in header_field_index:
                header_field_index[k] = pos
        key_col_idx = None
        camelot_col_idx = None
        if 'camelot' in header_field_index:
            camelot_col_idx = header_field_index['camelot']
        if 'key' in header_field_index:
            key_col_idx = header_field_index['key']
        chosen_key_column_for_detection = camelot_col_idx if camelot_col_idx is not None else key_col_idx
        if chosen_key_column_for_detection is None:
            guessed = detect_key_column_from_rows(rows_cells, header_texts)
            if isinstance(guessed, int) and guessed >= 0:
                chosen_key_column_for_detection = guessed
        debug_info['chosen_key_column'] = chosen_key_column_for_detection
        for row in rows_cells:
            def get_cell(i: int) -> str:
                try:
                    return row[i]
                except Exception:
                    return ""
            title = ""
            artist = ""
            tempo_txt = ""
            if 'title' in header_field_index and header_field_index['title'] < len(row):
                title = get_cell(header_field_index['title'])
            else:
                if len(row) > 2:
                    title = get_cell(2)
            if 'artist' in header_field_index and header_field_index['artist'] < len(row):
                artist = get_cell(header_field_index['artist'])
            else:
                if len(row) > 3:
                    artist = get_cell(3)
            if 'tempo' in header_field_index and header_field_index['tempo'] < len(row):
                tempo_txt = get_cell(header_field_index['tempo'])
            else:
                for c in row:
                    if re.search(r'\b\d{2,3}\b', c):
                        tempo_txt = c
                        break
            tempo = parse_tempo(tempo_txt) if 'parse_tempo' in globals() else None
            # define local parse if missing
            if tempo is None:
                try:
                    m = re.search(r'(\d{2,3})', tempo_txt or "")
                    if m:
                        tempo = int(m.group(1))
                except Exception:
                    tempo = None
            camelot_val = None
            key_val = ""
            if 'camelot' in header_field_index and header_field_index['camelot'] < len(row):
                possible = get_cell(header_field_index['camelot'])
                kn, cam = extract_key_and_camelot_from_cell(possible)
                if kn:
                    key_val = kn
                if cam:
                    camelot_val = cam
            if (not key_val or key_val == "") and 'key' in header_field_index and header_field_index['key'] < len(row):
                possible = get_cell(header_field_index['key'])
                kn, cam = extract_key_and_camelot_from_cell(possible)
                if kn and not key_val:
                    key_val = kn
                if cam and not camelot_val:
                    camelot_val = cam
            if (not key_val and not camelot_val) and chosen_key_column_for_detection is not None:
                idx = chosen_key_column_for_detection
                if idx < len(row):
                    possible = get_cell(idx)
                    kn, cam = extract_key_and_camelot_from_cell(possible)
                    if kn and not key_val:
                        key_val = kn
                    if cam and not camelot_val:
                        camelot_val = cam
            if not key_val and not camelot_val:
                for c in row:
                    kn, cam = extract_key_and_camelot_from_cell(c)
                    if kn and not key_val:
                        key_val = kn
                    if cam and not camelot_val:
                        camelot_val = cam
                    if key_val or camelot_val:
                        break
            display_key = ""
            if key_val and camelot_val:
                display_key = f"{key_val} ({camelot_val})"
            elif camelot_val:
                display_key = camelot_val
            elif key_val:
                display_key = key_val
            else:
                display_key = ""
            s = Song(title=title, artist=artist, key=display_key, tempo=tempo)
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
# Camelot-based HarmonicSequencer
# -----------------------
class HarmonicSequencer:
    def __init__(self):
        self.camelot_to_key: Dict[str, str] = {
            '1A': 'G#m', '1B': 'B',
            '2A': 'D#m', '2B': 'F#',
            '3A': 'A#m', '3B': 'C#',
            '4A': 'Fm',  '4B': 'G#',
            '5A': 'Cm',  '5B': 'D#',
            '6A': 'Gm',  '6B': 'A#',
            '7A': 'Dm',  '7B': 'F',
            '8A': 'Am',  '8B': 'C',
            '9A': 'Em',  '9B': 'G',
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
            s = s.replace(' Major', '').replace(' major', '')
            s = s.replace(' Minor', 'm').replace(' minor', 'm')
            s = re.sub(r'\s+', '', s)
            return s.lower()
        m: Dict[str, str] = {}
        def add_aliases(aliases: List[str], code: str):
            for a in aliases:
                m[clean(a)] = code
        add_aliases(['B', 'B major', 'Cb', 'Cb major'], '1B')
        add_aliases(['G#m', 'G# minor', 'Abm', 'Ab minor'], '1A')
        add_aliases(['F#', 'F# major', 'Gb', 'Gb major'], '2B')
        add_aliases(['D#m', 'D# minor', 'Ebm', 'Eb minor'], '2A')
        add_aliases(['Db', 'Db major', 'C#', 'C# major'], '3B')
        add_aliases(['Bbm', 'Bb minor', 'A#m', 'A# minor'], '3A')
        add_aliases(['Ab', 'Ab major', 'G#', 'G# major'], '4B')
        add_aliases(['Fm', 'F minor'], '4A')
        add_aliases(['Eb', 'Eb major', 'D#', 'D# major'], '5B')
        add_aliases(['Cm', 'C minor'], '5A')
        add_aliases(['Bb', 'Bb major', 'A#', 'A# major'], '6B')
        add_aliases(['Gm', 'G minor'], '6A')
        add_aliases(['F', 'F major'], '7B')
        add_aliases(['Dm', 'D minor'], '7A')
        add_aliases(['C', 'C major'], '8B')
        add_aliases(['Am', 'A minor'], '8A')
        add_aliases(['G', 'G major'], '9B')
        add_aliases(['Em', 'E minor'], '9A')
        add_aliases(['D', 'D major'], '10B')
        add_aliases(['Bm', 'B minor'], '10A')
        add_aliases(['A', 'A major'], '11B')
        add_aliases(['F#m', 'F# minor', 'Gbm', 'Gb minor'], '11A')
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
        if not key:
            return None
        kn, cam = extract_key_and_camelot_from_cell(key)
        if cam:
            if cam.isdigit() and kn:
                mapped = self.alias_to_camelot.get(self._clean_key_input(kn))
                if mapped:
                    return mapped
                else:
                    return cam
            return cam
        cleaned = self._clean_key_input(key)
        if cleaned in self.alias_to_camelot:
            return self.alias_to_camelot[cleaned]
        if CAMEL_NUMERIC_ONLY.match(cleaned):
            try:
                num = int(float(cleaned))
                if 1 <= num <= 12:
                    return str(num)
            except Exception:
                pass
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
        prev_num = 12 if num == 1 else num - 1
        next_num = 1 if num == 12 else num + 1
        return [f"{num}{other_letter}", f"{prev_num}{letter}", f"{next_num}{letter}"]

    def camelot_score(self, c1: str, c2: str) -> float:
        if not c1 or not c2:
            return 0.2
        if c1 == c2:
            return 1.0
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

    def suggest_bridge_keys(self, key1: str, key2: str, available_keys: Optional[List[str]] = None) -> List[str]:
        c1 = self.key_to_camelot(key1)
        c2 = self.key_to_camelot(key2)
        if not c1 or not c2 or c1 == c2:
            return []
        if c2 in self.camelot_neighbors(c1):
            return []
        def get_available_marker(code: str) -> str:
            if not available_keys:
                return ""
            available_camelot = [self.key_to_camelot(k) for k in available_keys]
            return " ★" if code in available_camelot else ""
        from collections import deque
        queue = deque([(c1, [])])
        visited = {c1}
        all_paths = []
        min_length = float('inf')
        while queue:
            current, path = queue.popleft()
            if len(path) > min_length:
                continue
            for neighbor in self.camelot_neighbors(current):
                if neighbor == c2:
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
        results = []
        seen_paths = set()
        for path in all_paths:
            bridge_path = [step for step in path if step != c2]
            if not bridge_path:
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

    def create_harmonic_sequence(self, songs: List[Song]) -> List[Song]:
        if not songs:
            return []
        if len(songs) == 1:
            return songs[:]
        position_groups = {}
        unkeyed_songs = []
        for song in songs:
            camelot = self.key_to_camelot(song.key)
            if camelot and camelot in self.camelot_to_key:
                if camelot not in position_groups:
                    position_groups[camelot] = []
                position_groups[camelot].append(song)
            else:
                unkeyed_songs.append(song)
        if not position_groups:
            return songs[:]
        available_positions = list(position_groups.keys())
        def get_harmonic_neighbors(camelot_code):
            m = re.match(r'(\d+)([AB])', camelot_code)
            if not m:
                return []
            num = int(m.group(1))
            letter = m.group(2)
            other_letter = 'A' if letter == 'B' else 'B'
            neighbors = []
            neighbors.append(f"{num}{other_letter}")
            up = num + 1 if num < 12 else 1
            down = num - 1 if num > 1 else 12
            neighbors.append(f"{up}{letter}")
            neighbors.append(f"{down}{letter}")
            neighbors.append(f"{up}{other_letter}")
            neighbors.append(f"{down}{other_letter}")
            return neighbors
        def find_hamiltonian_path_dfs():
            def dfs(path, remaining):
                if not remaining:
                    return path
                if not path:
                    for start in available_positions:
                        result = dfs([start], [pos for pos in available_positions if pos != start])
                        if result:
                            return result
                    return None
                current = path[-1]
                neighbors = get_harmonic_neighbors(current)
                for next_pos in remaining:
                    if next_pos in neighbors:
                        new_remaining = [pos for pos in remaining if pos != next_pos]
                        result = dfs(path + [next_pos], new_remaining)
                        if result:
                            return result
                return None
            return dfs([], available_positions)
        hamiltonian_path = find_hamiltonian_path_dfs()
        if hamiltonian_path:
            sequence = []
            for position in hamiltonian_path:
                sequence.extend(position_groups[position])
            sequence.extend(unkeyed_songs)
            return sequence
        else:
            return songs[:]

    def find_mixing_pairs(self, songs: List[Song]) -> List[Tuple[Song, Song, float]]:
        pairs = []
        for i in range(len(songs) - 1):
            a = songs[i]
            b = songs[i + 1]
            pairs.append((a, b, self._distance_score(a.key, b.key)))
        return pairs

    def analyze_song_collection(self, songs: List[Song], available_keys: Optional[List[str]] = None) -> Dict:
        for s in songs:
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
# BPM fetch helper
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
# Spotify helpers (playlist tracks, reorder)
# -----------------------
def get_playlist_tracks(access_token: str, playlist_id: str):
    headers = {'Authorization': f'Bearer {access_token}'}
    playlist_response = requests.get(f'https://api.spotify.com/v1/playlists/{playlist_id}', headers=headers)
    if playlist_response.status_code != 200:
        raise Exception(f"Failed to fetch playlist: {playlist_response.text}")
    playlist_data = playlist_response.json()
    tracks = []
    url = f'https://api.spotify.com/v1/playlists/{playlist_id}/tracks'
    while url:
        response = requests.get(url, headers=headers)
        if response.status_code != 200:
            raise Exception(f"Failed to fetch tracks: {response.text}")
        data = response.json()
        tracks.extend(data['items'])
        url = data['next']
    return playlist_data, tracks

def reorder_playlist_tracks(access_token: str, playlist_id: str, reorder_operations: list):
    headers = {'Authorization': f'Bearer {access_token}', 'Content-Type': 'application/json'}
    playlist_response = requests.get(f'https://api.spotify.com/v1/playlists/{playlist_id}', headers=headers)
    if playlist_response.status_code != 200:
        raise Exception(f"Failed to get playlist info: {playlist_response.text}")
    snapshot_id = playlist_response.json()['snapshot_id']
    for operation in reorder_operations:
        reorder_data = {'range_start': operation['range_start'], 'insert_before': operation['insert_before'], 'snapshot_id': snapshot_id}
        if 'range_length' in operation:
            reorder_data['range_length'] = operation['range_length']
        response = requests.put(f'https://api.spotify.com/v1/playlists/{playlist_id}/tracks', headers=headers, json=reorder_data)
        if response.status_code != 200:
            raise Exception(f"Failed to reorder tracks: {response.text}")
        snapshot_id = response.json()['snapshot_id']
    return snapshot_id

def calculate_reorder_operations(original_tracks: list, target_order: list):
    target_positions = {}
    for i, song in enumerate(target_order):
        for j, track_item in enumerate(original_tracks):
            track = track_item['track']
            if (track and track.get('name', '').lower() == song.title.lower() and track.get('artists')
                and any(artist.get('name', '').lower() == song.artist.lower() for artist in track['artists'])):
                target_positions[j] = i
                break
    operations = []
    current_positions = list(range(len(original_tracks)))
    for target_pos in range(len(target_order)):
        current_track_original_pos = None
        for orig_pos, current_pos in enumerate(current_positions):
            if orig_pos in target_positions and target_positions[orig_pos] == target_pos:
                current_track_original_pos = orig_pos
                break
        if current_track_original_pos is None:
            continue
        current_pos = current_positions.index(current_track_original_pos)
        if current_pos != target_pos:
            operations.append({'range_start': current_pos, 'insert_before': target_pos if target_pos < current_pos else target_pos + 1, 'range_length': 1})
            track_to_move = current_positions.pop(current_pos)
            insert_pos = target_pos if target_pos < current_pos else target_pos
            current_positions.insert(insert_pos, track_to_move)
    return operations

# -----------------------
# UI / Main flow
# -----------------------
st.title("Harmonic Song Analyzer - Leverages songdata.io and Camelot System for Harmonic Mixing")
st.markdown("Paste the share link for your spotify playlist in order to build a harmonic play order, and get bridge-key suggestions.")

with st.expander("📖 Instructions & Help"):
    st.markdown("""
    - Choose an input method (Spotify→SongData or Upload CSV / Paste).
    - Choose arrangement mode: Key, Tempo, Key + Tempo (Key-priority), Tempo + Key (Tempo-priority with bridge suggestions).
    - Optionally authenticate with Spotify to apply reordering.
    """)

left_col, right_col = st.columns([1, 2])

with left_col:
    st.header("Input")
    input_mode = st.radio("Choose input method", ["Fetch from Spotify → SongData", "Upload CSV / Paste"], key="input_mode_radio")
    uploaded_file = None
    pasted_text = ""
    songdata_input = ""
    sd_timeout = 300
    sd_debug = False
    fetch_songdata_btn = False

    if input_mode == "Upload CSV / Paste":
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

    # Sorting selectbox — now has 4 options
    sort_mode = st.selectbox("Arrange setlist by:", ["Key", "Tempo", "Key + Tempo", "Tempo + Key (Bridged)"], index=0, key="sort_mode_select")

# Ensure session keys exist
for key in ("spotify_access_token", "code_verifier", "auth_state", "cached_songs", "cached_songdata_input", "cached_sequence", "_oauth_just_completed"):
    if key not in st.session_state:
        st.session_state[key] = None

songs: List[Song] = []

# Uploaded CSV
if uploaded_file:
    try:
        df = pd.read_csv(uploaded_file)
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
                st.session_state.cached_songs = _serialize_songs(songs)
                st.session_state.cached_songdata_input = songdata_input
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

# Restore cached songs if present and no current songs
if st.session_state.get('cached_songs') and (songs is None or (isinstance(songs, (list, tuple)) and len(songs) == 0)):
    try:
        songs = _deserialize_songs(st.session_state.cached_songs)
        if st.session_state.cached_songdata_input:
            songdata_input = st.session_state.cached_songdata_input
        st.info(f"Restored {len(songs)} songs from cache")
    except Exception:
        pass

# Right column: analysis and results
with right_col:
    if songs:
        hs = HarmonicSequencer()
        if shuffle_flag:
            np.random.shuffle(songs)
        if auto_bpm:
            with st.spinner("Fetching BPMs..."):
                for s in songs:
                    if s.tempo is None or (isinstance(s.tempo, float) and np.isnan(s.tempo)):
                        bpm = fetch_bpm_from_web(s.title, s.artist)
                        if bpm:
                            s.tempo = bpm
                        time.sleep(0.12)
        for s in songs:
            if not s.camelot:
                s.camelot = hs.key_to_camelot(s.key)

        # Base harmonic analysis
        analysis = hs.analyze_song_collection(songs, available_keys=[s.key for s in songs])
        base_sequence = analysis['sequence']

        arranged = None  # final arranged sequence used below
        # ---- Sorting modes ----
        if sort_mode == "Key":
            arranged = base_sequence

        elif sort_mode == "Tempo":
            def tempo_safe(s: Song):
                t = s.tempo
                return float(t) if (t is not None and not (isinstance(t, float) and np.isnan(t))) else 10**9
            arranged = sorted(songs, key=lambda s: (tempo_safe(s), s.title.lower(), s.artist.lower()))

        elif sort_mode == "Key + Tempo":
            # Key-first: preserve harmonic group order, then tempo-smooth within & between groups
            group_order = []
            groups = {}
            unkeyed = []
            for s in base_sequence:
                cam = hs.key_to_camelot(s.key)
                if cam:
                    if cam not in groups:
                        groups[cam] = []
                        group_order.append(cam)
                    groups[cam].append(s)
                else:
                    unkeyed.append(s)
            # include any songs not in base_sequence
            seen = set((s.title, s.artist) for s in base_sequence)
            for s in songs:
                if (s.title, s.artist) in seen:
                    continue
                cam = hs.key_to_camelot(s.key)
                if cam:
                    if cam not in groups:
                        groups[cam] = []
                        group_order.append(cam)
                    groups[cam].append(s)
                else:
                    unkeyed.append(s)
            def tempo_val(s: Song):
                t = s.tempo
                return float(t) if (t is not None and not (isinstance(t, float) and np.isnan(t))) else None
            for cam in groups:
                groups[cam] = sorted(groups[cam], key=lambda ss: (tempo_val(ss) is None, tempo_val(ss) or 0))
            arranged_sequence = []
            last_tempo = None
            for cam in group_order:
                group = groups.get(cam, [])
                if not arranged_sequence:
                    arranged_sequence.extend(group)
                    last_tempo = tempo_val(arranged_sequence[-1]) if arranged_sequence else last_tempo
                    continue
                if not group:
                    continue
                first_t = tempo_val(group[0])
                last_t = tempo_val(group[-1])
                if last_tempo is None:
                    arranged_sequence.extend(group)
                    last_tempo = tempo_val(arranged_sequence[-1]) if arranged_sequence else last_tempo
                    continue
                jump_keep = abs((first_t or last_tempo) - last_tempo) if first_t is not None else 999
                jump_reverse = abs((last_t or last_tempo) - last_tempo) if last_t is not None else 999
                if jump_reverse < jump_keep:
                    group = list(reversed(group))
                arranged_sequence.extend(group)
                last_t = tempo_val(arranged_sequence[-1]) if arranged_sequence else last_t
                if last_t is not None:
                    last_tempo = last_t
            arranged = arranged_sequence + sorted(unkeyed, key=lambda ss: (tempo_val(ss) is None, tempo_val(ss) or 0))

        else:  # "Tempo + Key (Bridged)"
            # Tempo-first arrangement but require harmonic compatibility as preference (heavy penalty if not)
            songs_with_key = [s for s in songs if hs.key_to_camelot(s.key)]
            unkeyed = [s for s in songs if not hs.key_to_camelot(s.key)]
            def tempo_val(s):
                return s.tempo if s.tempo not in (None, np.nan) else None
            def is_harmonically_compatible(k1, k2):
                if not k1 or not k2:
                    return False
                m1 = re.match(r"(\d{1,2})([AB])", k1)
                m2 = re.match(r"(\d{1,2})([AB])", k2)
                if not m1 or not m2:
                    return False
                n1, l1 = int(m1.group(1)), m1.group(2)
                n2, l2 = int(m2.group(1)), m2.group(2)
                if n1 == n2 and l1 != l2:
                    return True
                if abs((n1 - n2) % 12) == 1 and l1 == l2:
                    return True
                if (n1 == 1 and n2 == 12 or n1 == 12 and n2 == 1) and l1 == l2:
                    return True
                return False
            def transition_cost(a, b):
                tg = abs((tempo_val(a) or 0) - (tempo_val(b) or 0))
                harmonic_penalty = 0 if is_harmonically_compatible(hs.key_to_camelot(a.key), hs.key_to_camelot(b.key)) else 50
                return tg + harmonic_penalty
            remaining = songs_with_key[:]
            arranged_sequence = []
            # start near median tempo for balanced flow
            valid_tempos = [tempo_val(s) for s in remaining if tempo_val(s) is not None]
            if valid_tempos:
                median_tempo = float(np.median(valid_tempos))
                start_index = int(np.argmin([abs((tempo_val(s) or median_tempo) - median_tempo) for s in remaining]))
            else:
                start_index = 0
            arranged_sequence.append(remaining.pop(start_index))
            while remaining:
                last = arranged_sequence[-1]
                next_song = min(remaining, key=lambda s: transition_cost(last, s))
                arranged_sequence.append(next_song)
                remaining.remove(next_song)
            arranged = arranged_sequence + sorted(unkeyed, key=lambda s: tempo_val(s) or 0)
            # Suggest bridges for large gaps
            bridge_suggestions = []
            for i in range(len(arranged) - 1):
                a = arranged[i]
                b = arranged[i + 1]
                a_t = tempo_val(a) or 0
                b_t = tempo_val(b) or 0
                tempo_gap = abs(a_t - b_t)
                a_key = hs.key_to_camelot(a.key)
                b_key = hs.key_to_camelot(b.key)
                if (not is_harmonically_compatible(a_key, b_key)) or tempo_gap > 15:
                    # candidate bridge keys = neighbors of a_key that are compatible with b_key
                    candidate_keys = []
                    if a_key and b_key:
                        neighbors = hs.camelot_neighbors(a_key)
                        for nb in neighbors:
                            if is_harmonically_compatible(nb, b_key):
                                candidate_keys.append(nb)
                    bridge_suggestions.append({
                        "From → To": f"{a.title} → {b.title}",
                        "Tempo Gap (BPM)": tempo_gap,
                        "Suggested Bridge Key(s)": ", ".join(candidate_keys) or "None",
                        "Ideal Bridge Tempo (BPM)": round((a_t + b_t) / 2)
                    })
            if bridge_suggestions:
                st.subheader("🔀 Suggested Bridge Key + Tempo Combinations")
                st.dataframe(pd.DataFrame(bridge_suggestions))

        # final: arranged should be set
        if arranged is None:
            arranged = base_sequence

        # compute mixing pairs & gaps based on arranged sequence
        mixing_pairs = hs.find_mixing_pairs(arranged)
        gaps_and_bridges = []
        existing_keys = [s.key for s in songs]
        for i in range(len(arranged) - 1):
            a = arranged[i]
            b = arranged[i + 1]
            score = hs._distance_score(a.key, b.key)
            if score < 0.6:
                suggestions = hs.suggest_bridge_keys(a.key, b.key, available_keys=existing_keys)
                gaps_and_bridges.append({'from': a, 'to': b, 'score': score, 'suggestions': suggestions})

        # cache arranged sequence for Spotify reordering
        st.session_state.cached_sequence = _serialize_songs(arranged)

        st.header("Recommended Play Order")
        rows = []
        for i, s in enumerate(arranged):
            camelot_display = hs.camelot_to_display(s.camelot) if s.camelot else s.key
            rows.append({'Order': i + 1, 'Title': s.title, 'Artist': s.artist, 'Camelot': camelot_display, 'Tempo': s.tempo})
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
            for s in arranged:
                mapping.append({'Title': s.title, 'Artist': s.artist, 'Input Key': s.key, 'Detected Camelot': s.camelot or '', 'Camelot Display': hs.camelot_to_display(s.camelot) if s.camelot else ''})
            st.dataframe(pd.DataFrame(mapping))

    else:
        st.info("No songs loaded yet. Upload/paste songs or fetch a SongData playlist using the left panel.")

st.markdown("---")
st.header("🎵 Spotify Integration")

SPOTIFY_CLIENT_ID = st.secrets.get("SPOTIFY_CLIENT_ID", "")
REDIRECT_URI = st.secrets.get("REDIRECT_URI", "http://127.0.0.1:8501/")

# Process OAuth callback now that client id and redirect are known
if SPOTIFY_CLIENT_ID:
    try:
        _process_oauth_callback_if_present(SPOTIFY_CLIENT_ID, REDIRECT_URI, exchange_code_for_token)
    except Exception:
        pass

if not SPOTIFY_CLIENT_ID:
    st.warning("""
    **Spotify integration not configured.**
    Add `SPOTIFY_CLIENT_ID` and `REDIRECT_URI` to Streamlit secrets.
    Example secrets.toml:
    ```
    SPOTIFY_CLIENT_ID = "your_client_id_here"
    REDIRECT_URI = "http://127.0.0.1:8501/"
    ```
    """)
    spotify_client_id = None
else:
    spotify_client_id = SPOTIFY_CLIENT_ID
    st.info("Spotify integration is available! Sign in below to reorder your playlists.")

if spotify_client_id and (songs or st.session_state.cached_songs):
    st.markdown("### 🔐 Spotify Authentication")
    query_params = st.query_params
    if 'code' in query_params and 'state' in query_params:
        if (st.session_state.auth_state and query_params['state'] == st.session_state.auth_state and st.session_state.code_verifier):
            try:
                token_response = exchange_code_for_token(client_id=spotify_client_id, code=query_params['code'], redirect_uri=REDIRECT_URI, code_verifier=st.session_state.code_verifier)
                st.session_state.spotify_access_token = token_response['access_token']
                st.success("✅ Successfully authenticated with Spotify!")
                st.query_params.clear()
            except Exception as e:
                st.error(f"Authentication failed: {e}")

    if not st.session_state.spotify_access_token:
        st.info("Authenticate with Spotify to enable playlist reordering")
        if st.button("🎵 Authenticate with Spotify"):
            code_verifier = generate_code_verifier()
            code_challenge = generate_code_challenge(code_verifier)
            auth_state = secrets.token_urlsafe(32)
            st.session_state.code_verifier = code_verifier
            st.session_state.auth_state = auth_state
            auth_url = get_spotify_auth_url(client_id=spotify_client_id, redirect_uri=REDIRECT_URI, code_challenge=code_challenge, state=auth_state)
            st.markdown(f"[Click here to authenticate with Spotify]({auth_url})")
            st.info("After authentication, you'll be redirected back to this page.")
    else:
        st.success("✅ Authenticated with Spotify")
        current_input = songdata_input or st.session_state.cached_songdata_input
        playlist_id = extract_spotify_playlist_id(current_input) if current_input else None
        current_sequence = locals().get('arranged') or (st.session_state.cached_sequence)
        if playlist_id and current_sequence:
            st.markdown("### 🎯 Apply Harmonic Order to Spotify Playlist")
            st.info(f"Playlist ID: {playlist_id}")
            col1, col2 = st.columns([1, 1])
            with col1:
                if st.button("🔄 Reorder Spotify Playlist", type="primary"):
                    try:
                        with st.spinner("Reordering playlist on Spotify..."):
                            playlist_data, current_tracks = get_playlist_tracks(st.session_state.spotify_access_token, playlist_id)
                            st.info(f"Found {len(current_tracks)} tracks in playlist: {playlist_data['name']}")
                            # build target order Song objects from cached_sequence
                            target_order = _deserialize_songs(current_sequence) if isinstance(current_sequence, list) and isinstance(current_sequence[0], dict) else current_sequence
                            operations = calculate_reorder_operations(current_tracks, target_order)
                            if operations:
                                st.info(f"Executing {len(operations)} reorder operations...")
                                new_snapshot = reorder_playlist_tracks(st.session_state.spotify_access_token, playlist_id, operations)
                                st.success(f"✅ Successfully reordered playlist! New snapshot: {new_snapshot}")
                                st.balloons()
                            else:
                                st.info("Playlist is already in the optimal order!")
                    except Exception as e:
                        st.error(f"Failed to reorder playlist: {e}")
                        st.info("Make sure you own the playlist or have collaborative access.")
            with col2:
                if st.button("🔓 Sign Out"):
                    st.session_state.spotify_access_token = None
                    st.session_state.code_verifier = None
                    st.session_state.auth_state = None
                    st.experimental_rerun()
        else:
            if not current_input:
                st.warning("No playlist ID found. Make sure you used Spotify → SongData input method.")
            elif not current_sequence:
                st.warning("No harmonic sequence available. Please analyze a playlist first.")
            else:
                st.warning("Playlist data not available.")
else:
    if (songs or st.session_state.cached_songs) and not spotify_client_id:
        st.info("Configure your Spotify Client ID and REDIRECT_URI in Streamlit secrets to enable playlist reordering.")
    elif not (songs or st.session_state.cached_songs):
        st.info("Fetch a playlist first to enable Spotify integration.")

st.markdown("---")
st.markdown("1.0.0\nThanks to songdata.io for playlist data.\nThanks to Mixed In Key for the Camelot system.")
