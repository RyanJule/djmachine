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

# Set page config
st.set_page_config(
    page_title="Harmonic Song Analyzer — Camelot + SongData fetch",
    page_icon="🎵",
)

# ---- Data classes ----
@dataclass
class Song:
    title: str
    artist: str
    key: str
    tempo: Optional[int] = None
    mood: Optional[str] = None
    notes: Optional[str] = None
    camelot: Optional[str] = None  # filled in after normalization

# ---- SongData fetch helper ----
def extract_spotify_playlist_id(spotify_url: str) -> Optional[str]:
    """Extract the playlist id from several Spotify URL/URI formats."""
    if not spotify_url:
        return None
    s = spotify_url.strip()
    # common patterns:
    # https://open.spotify.com/playlist/<id>?si=...
    m = re.search(r'playlist/([A-Za-z0-9]+)', s)
    if m:
        return m.group(1)
    # spotify:playlist:<id>
    m = re.search(r'spotify:playlist:([A-Za-z0-9]+)', s)
    if m:
        return m.group(1)
    # fallback: if user pasted just the id
    if re.fullmatch(r'[A-Za-z0-9]+', s):
        return s
    return None

def fetch_songdata_playlist(spotify_url: str, timeout: float = 8.0) -> Tuple[List[Song], str]:
    """
    Given a spotify url/uri/id, build the songdata playlist url and attempt to parse the song table.
    Returns (songs_list, songdata_url). Raises RuntimeError with helpful message on failure.
    """
    pid = extract_spotify_playlist_id(spotify_url)
    if not pid:
        raise RuntimeError("Couldn't extract Spotify playlist id from that input. Try an URL like https://open.spotify.com/playlist/<id> or a Spotify URI spotify:playlist:<id>.")
    songdata_url = f"https://songdata.io/playlist/{pid}"
    headers = {'User-Agent': 'Mozilla/5.0'}
    try:
        r = requests.get(songdata_url, headers=headers, timeout=timeout)
    except Exception as e:
        raise RuntimeError(f"Failed to fetch SongData page: {e}")
    if r.status_code != 200:
        raise RuntimeError(f"SongData returned status {r.status_code} for {songdata_url}.")

    soup = BeautifulSoup(r.text, 'html.parser')

    # Look for a table which contains Title & Artist columns (heuristic)
    tables = soup.find_all('table')
    songs: List[Song] = []
    found = False
    for table in tables:
        # find headers
        headers = []
        thead = table.find('thead')
        if thead:
            headers = [th.get_text(strip=True).lower() for th in thead.find_all('th')]
        else:
            # try first row as header
            first_row = table.find('tr')
            if first_row:
                headers = [td.get_text(strip=True).lower() for td in first_row.find_all(['th','td'])]
        # normalize headers
        headers = [h.strip() for h in headers if h.strip()]
        if not headers:
            continue
        # Check if this table looks like a song list
        has_title = any('title' in h or 'song' in h or 'track' in h for h in headers)
        has_artist = any('artist' in h for h in headers)
        if not (has_title and has_artist):
            continue
        # Map header indices to fields
        idx_title = next((i for i,h in enumerate(headers) if ('title' in h or 'song' in h or 'track' in h)), None)
        idx_artist = next((i for i,h in enumerate(headers) if 'artist' in h), None)
        idx_key = next((i for i,h in enumerate(headers) if 'key' in h or 'camelot' in h), None)
        idx_bpm = next((i for i,h in enumerate(headers) if 'bpm' in h or 'tempo' in h), None)

        tbody = table.find('tbody') or table
        rows = tbody.find_all('tr')
        for row in rows:
            cols = row.find_all(['td','th'])
            if not cols or len(cols) <= max(filter(lambda x: x is not None, [idx_title, idx_artist, idx_key, idx_bpm]) or [-1]):
                # skip malformed rows
                continue
            def cell_text(i):
                try:
                    return cols[i].get_text(strip=True)
                except Exception:
                    return ""
            title = cell_text(idx_title) if idx_title is not None else ""
            artist = cell_text(idx_artist) if idx_artist is not None else ""
            key = cell_text(idx_key) if idx_key is not None else ""
            bpm_text = cell_text(idx_bpm) if idx_bpm is not None else ""
            tempo = None
            if bpm_text:
                m = re.search(r'(\d{2,3})', bpm_text)
                if m:
                    tempo = int(m.group(1))
            # Only append rows that have at least title+artist
            if title or artist:
                songs.append(Song(title=title, artist=artist, key=key or "", tempo=tempo))
        if songs:
            found = True
            break

    if not found:
        # try alternate approach: some SongData pages render data in JSON inside a script tag
        # attempt to locate any JSON array of tracks (best-effort)
        scripts = soup.find_all('script')
        for sc in scripts:
            text = sc.string or ""
            if not text:
                continue
            # naive attempt to find a 'tracks' JSON array
            m = re.search(r'(?:"tracks"|\'tracks\'|tracks)\s*:\s*(\[[^\]]+\])', text, re.IGNORECASE | re.DOTALL)
            if m:
                arr_text = m.group(1)
                try:
                    import json
                    arr = json.loads(arr_text)
                    for item in arr:
                        title = item.get('title') or item.get('name') or ""
                        artist = item.get('artist') or item.get('artists') or ""
                        if isinstance(artist, list):
                            artist = ", ".join(artist)
                        key = item.get('key') or item.get('camelot') or ""
                        tempo = item.get('bpm') or item.get('tempo') or None
                        songs.append(Song(title=title, artist=artist, key=key or "", tempo=tempo))
                    if songs:
                        found = True
                        break
                except Exception:
                    pass

    if not songs:
        # nothing parsed
        raise RuntimeError(f"Could not parse songs from SongData page. You can inspect the page here: {songdata_url}")

    return songs, songdata_url

# ---- HarmonicSequencer (Camelot-based) ----
class HarmonicSequencer:
    def __init__(self):
        # Canonical mapping Camelot -> display key
        self.camelot_to_key = {
            '1A': 'G#m',   '1B': 'B',
            '2A': 'D#m',   '2B': 'F#',
            '3A': 'Bbm',   '3B': 'Db',
            '4A': 'Fm',    '4B': 'Ab',
            '5A': 'Cm',    '5B': 'Eb',
            '6A': 'Gm',    '6B': 'Bb',
            '7A': 'Dm',    '7B': 'F',
            '8A': 'Am',    '8B': 'C',
            '9A': 'Em',    '9B': 'G',
            '10A': 'Bm',   '10B': 'D',
            '11A': 'F#m',  '11B': 'A',
            '12A': 'C#m',  '12B': 'E'
        }
        self.alias_to_camelot: Dict[str, str] = self._build_alias_map()
        self.camelot_codes = list(self.camelot_to_key.keys())

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

        m = {}
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
            'c': '8B', 'am': '8A', 'g': '9B', 'em': '9A', 'd': '10B', 'bm': '10A',
            'a': '11B', 'f#m': '11A', 'e': '12B', 'c#m': '12A', 'bb': '6B', 'bbm': '3A'
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
        return s

    def key_to_camelot(self, key: str) -> Optional[str]:
        if not key:
            return None
        cleaned = self._clean_key_input(key).lower()
        if cleaned in self.alias_to_camelot:
            return self.alias_to_camelot[cleaned]
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
        neighbors = []
        other_letter = 'A' if letter == 'B' else 'B'
        neighbors.append(f"{num}{other_letter}")
        prev_num = num - 1 if num > 1 else 12
        next_num = num + 1 if num < 12 else 1
        neighbors.append(f"{prev_num}{letter}")
        neighbors.append(f"{next_num}{letter}")
        return neighbors

    def camelot_score(self, c1: str, c2: str) -> float:
        if c1 == c2:
            return 1.0
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
        if not ca or not cb:
            return 0.25
        return self.camelot_score(ca, cb)

    def suggest_bridge_keys(self, key1: str, key2: str, available_keys: Optional[List[str]] = None) -> List[str]:
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
                neigh = self.camelot_neighbors(known)
                results = [self.camelot_to_display(n) for n in neigh if n in self.camelot_to_key]
                return results[:6]
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
            s = self.camelot_score(c1, cand) + self.camelot_score(cand, c2) + boost_for_available(cand)
            if s > 1.2:
                single_candidates.append((cand, s))
        single_candidates.sort(key=lambda x: x[1], reverse=True)

        two_step: List[Tuple[Tuple[str, str], float]] = []
        for a in all_codes:
            if a in (c1, c2):
                continue
            s1 = self.camelot_score(c1, a)
            if s1 < 0.4:
                continue
            for b in all_codes:
                if b in (c1, c2, a):
                    continue
                s2 = self.camelot_score(a, b)
                s3 = self.camelot_score(b, c2)
                if s2 < 0.35 or s3 < 0.35:
                    continue
                score = s1 + s2 + s3 + boost_for_available(a) + boost_for_available(b)
                two_step.append(((a, b), score))
        two_step.sort(key=lambda x: x[1], reverse=True)

        def find_chains(max_hops=4, beam_width=60, min_edge_score=0.35):
            beams = [([c1], 0.0)]
            results: List[Tuple[List[str], float]] = []
            for depth in range(1, max_hops + 1):
                new_beams = []
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
                beams = sorted(new_beams, key=lambda x: x[1], reverse=True)[:beam_width]
            results.sort(key=lambda x: x[1], reverse=True)
            return results

        chain_results = find_chains(max_hops=4, beam_width=80, min_edge_score=0.32)

        results: List[str] = []
        for cand, score in single_candidates[:4]:
            disp = self.camelot_to_display(cand)
            if disp not in results:
                results.append(disp)

        for (a, b), score in two_step[:4]:
            formatted = f"{self.camelot_to_display(a)} -> {self.camelot_to_display(b)}"
            if formatted not in results:
                results.append(formatted)

        for path, score in chain_results[:6]:
            formatted = " -> ".join(self.camelot_to_display(p) for p in path)
            if formatted not in results:
                results.append(formatted)

        if not results:
            neighbors = self.camelot_neighbors(c1)
            for n in neighbors:
                if n in self.camelot_to_key:
                    results.append(self.camelot_to_display(n))
            if not results:
                alt = f"{int(re.match(r'(\d+)', c1).group(1))}{('A' if c1.endswith('B') else 'B')}"
                if alt in self.camelot_to_key:
                    results.append(self.camelot_to_display(alt))

        return results[:8]

    def create_harmonic_sequence(self, songs: List[Song]) -> List[Song]:
        if not songs:
            return []
        remaining = songs[:]
        seq = [remaining.pop(0)]
        while remaining:
            last = seq[-1]
            best_idx = 0
            best_score = -1
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
            score = self._distance_score(a.key, b.key)
            pairs.append((a, b, score))
        return pairs

    def analyze_song_collection(self, songs: List[Song]) -> Dict:
        for song in songs:
            song.key = song.key or ""
            song.camelot = self.key_to_camelot(song.key)
        sequence = self.create_harmonic_sequence(songs)
        mixing_pairs = self.find_mixing_pairs(songs)
        existing_keys = [s.key for s in songs]
        gaps_and_bridges = []
        for i in range(len(sequence) - 1):
            a = sequence[i]
            b = sequence[i + 1]
            score = self._distance_score(a.key, b.key)
            if score < 0.6:
                suggestions = self.suggest_bridge_keys(a.key, b.key, available_keys=existing_keys)
                gaps_and_bridges.append({
                    'from': a,
                    'to': b,
                    'score': score,
                    'suggestions': suggestions
                })
        return {
            'sequence': sequence,
            'mixing_pairs': mixing_pairs,
            'gaps_and_bridges': gaps_and_bridges
        }

# ---- BPM helper (kept for fallback) ----
def fetch_bpm_from_web(title: str, artist: str) -> Optional[int]:
    try:
        q = f"{title} {artist} bpm"
        url = f"https://www.google.com/search?q={requests.utils.quote(q)}"
        headers = {'User-Agent': 'Mozilla/5.0'}
        r = requests.get(url, headers=headers, timeout=5)
        soup = BeautifulSoup(r.text, 'html.parser')
        text = soup.get_text()
        m = re.search(r'(\d{2,3})\s?BPM', text, re.IGNORECASE)
        if m:
            return int(m.group(1))
    except Exception:
        return None
    return None

# ---- Streamlit UI ----
st.title("Harmonic Song Analyzer 🎧 (Camelot + SongData)")
st.write("Upload a CSV, paste songs, or paste a Spotify playlist URL to fetch a SongData table automatically.")

# Left: input modes
st.sidebar.header("Input")
input_mode = st.sidebar.radio("Choose input method", ["Upload CSV / Paste", "Fetch from Spotify → SongData"])

songs: List[Song] = []

if input_mode == "Upload CSV / Paste":
    uploaded = st.file_uploader("Upload CSV of songs (title,artist,key,tempo?)", type=["csv"])
    text_input = st.text_area("Or paste CSV/text (title,artist,key,tempo)", height=140)
    if uploaded:
        try:
            df = pd.read_csv(uploaded)
            for _, r in df.iterrows():
                title = str(r.get('title') or r.get('Title') or r.get('song') or r.get('Song') or "")
                artist = str(r.get('artist') or r.get('Artist') or "")
                key = str(r.get('key') or r.get('Key') or "")
                tempo = r.get('tempo') if 'tempo' in r else None
                songs.append(Song(title=title, artist=artist, key=key, tempo=tempo))
        except Exception as e:
            st.error(f"Could not read CSV: {e}")
    if text_input and not songs:
        sio = StringIO(text_input)
        try:
            df = pd.read_csv(sio, header=None)
            for _, r in df.iterrows():
                parts = [str(x) for x in r if pd.notna(x)]
                if len(parts) >= 3:
                    songs.append(Song(title=parts[0], artist=parts[1], key=parts[2], tempo=(int(parts[3]) if len(parts) > 3 else None)))
        except Exception:
            lines = [l.strip() for l in text_input.splitlines() if l.strip()]
            for line in lines:
                parts = [p.strip() for p in re.split(r',|\t|;', line) if p.strip()]
                if len(parts) >= 3:
                    t, a, k = parts[0], parts[1], parts[2]
                    tempo = int(parts[3]) if len(parts) > 3 and parts[3].isdigit() else None
                    songs.append(Song(title=t, artist=a, key=k, tempo=tempo))
else:
    st.write("Paste a Spotify playlist URL, open.spotify.com/playlist/<id> or spotify:playlist:<id>")
    spotify_input = st.text_input("Spotify playlist URL or URI")
    fetch_button = st.button("Fetch SongData table for this playlist")
    if fetch_button and spotify_input:
        with st.spinner("Fetching playlist from SongData..."):
            try:
                fetched_songs, sd_url = fetch_songdata_playlist(spotify_input)
                songs = fetched_songs
                st.success(f"Parsed {len(songs)} songs from SongData: {sd_url}")
                st.markdown(f"[Open SongData playlist page]({sd_url})")
            except Exception as e:
                st.error(str(e))
                # offer the user the page link if we could at least build it
                pid = extract_spotify_playlist_id(spotify_input)
                if pid:
                    st.markdown(f"Try opening the generated SongData page: https://songdata.io/playlist/{pid}")

# UI options
st.sidebar.header("Options")
shuffle = st.sidebar.checkbox("Shuffle input order before sequencing", value=False)
show_raw = st.sidebar.checkbox("Show normalized keys & Camelot codes", value=True)

if songs:
    hs = HarmonicSequencer()
    if shuffle:
        np.random.shuffle(songs)

    # attempt to fetch missing tempos only if necessary (light-touch)
    for s in songs:
        if s.tempo is None:
            s.tempo = fetch_bpm_from_web(s.title, s.artist)
            time.sleep(0.15)

    analysis = hs.analyze_song_collection(songs)
    seq = analysis['sequence']
    mixing_pairs = analysis['mixing_pairs']
    gaps_and_bridges = analysis['gaps_and_bridges']

    st.header("Recommended Play Order")
    rows = []
    for i, s in enumerate(seq):
        camelot_display = hs.camelot_to_display(s.camelot) if s.camelot else "(unknown)"
        rows.append({
            'Order': i + 1,
            'Title': s.title,
            'Artist': s.artist,
            'Key (input)': s.key,
            'Camelot': camelot_display,
            'Tempo': s.tempo or ''
        })
    df_display = pd.DataFrame(rows)
    st.dataframe(df_display)

    st.header("Mixing Pair Scores (Camelot-based)")
    mp_rows = []
    for a, b, score in mixing_pairs:
        a_c = hs.camelot_to_display(a.camelot) if a.camelot else "(unknown)"
        b_c = hs.camelot_to_display(b.camelot) if b.camelot else "(unknown)"
        mp_rows.append({
            'From': f"{a.title} - {a.artist} {a_c}",
            'To': f"{b.title} - {b.artist} {b_c}",
            'Score': round(score, 2)
        })
    st.table(pd.DataFrame(mp_rows))

    if gaps_and_bridges:
        st.header("Harmonic Gaps & Bridge Suggestions (Camelot)")
        for gb in gaps_and_bridges:
            a = gb['from']
            b = gb['to']
            a_disp = hs.camelot_to_display(a.camelot) if a.camelot else a.key
            b_disp = hs.camelot_to_display(b.camelot) if b.camelot else b.key
            st.subheader(f"{a.title} — {a_disp}  →  {b.title} — {b_disp} — score {gb['score']:.2f}")
            st.write("Bridge suggestions (single keys first, then sequences; displayed as 'Name (Camelot)'):")
            if gb['suggestions']:
                for s in gb['suggestions']:
                    st.write(f"- {s}")
            else:
                st.write("- (no suggestions found)")
            st.write("---")
    else:
        st.success("No major harmonic gaps detected — your order flows well!")
else:
    st.info("No songs provided yet. Upload/paste songs, or fetch a playlist from Spotify → SongData using the sidebar.")

# ---- Footer / credits ----
st.markdown("---")
st.markdown(
    "Built with ❤️ — Harmonic suggestions use the Camelot system. "
    "SongData fetch is best-effort: some SongData pages render JavaScript and may not expose a parseable server-side table. "
    "If parsing fails, open the SongData link and try exporting CSV from the site then upload it here."
)

