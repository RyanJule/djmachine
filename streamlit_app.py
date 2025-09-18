import streamlit as st
import pandas as pd
import numpy as np
import requests
from bs4 import BeautifulSoup
import time
import re
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional
import plotly.express as px
import plotly.graph_objects as go
from io import StringIO

# Set page config
st.set_page_config(
    page_title="Harmonic Song Analyzer",
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

# ---- Harmonic Sequencer ----
class HarmonicSequencer:
    def __init__(self):
        # Circle of Fifths for major keys
        self.circle_of_fifths = [
            'C', 'G', 'D', 'A', 'E', 'B', 'F#', 'Db', 'Ab', 'Eb', 'Bb', 'F'
        ]
        
        # Relative minor keys
        self.relative_minors = {
            'C': 'Am', 'G': 'Em', 'D': 'Bm', 'A': 'F#m', 'E': 'C#m', 'B': 'G#m',
            'F#': 'D#m', 'Db': 'Bbm', 'Ab': 'Fm', 'Eb': 'Cm', 'Bb': 'Gm', 'F': 'Dm'
        }
        
        # Extended key mappings including minors and alternative notations
        self.key_mappings = self._build_key_mappings()
        
        # Compatibility matrix (major/minor compatibilities, symmetric-ish)
        self.compatibility_matrix = self._build_compatibility_matrix()
    
    def _build_key_mappings(self) -> Dict[str, str]:
        mappings = {}
        # map natural major and minors
        for major, minor in self.relative_minors.items():
            mappings[major] = major
            mappings[minor] = minor
        
        # Add common enharmonics
        enharmonics = {
            'Gb': 'F#', 'Cb': 'B', 'E#': 'F', 'B#': 'C', 'A#': 'Bb', 'D#': 'Eb', 'G#': 'Ab',
            'Bb': 'Bb', 'Eb': 'Eb', 'Ab': 'Ab', 'Db': 'Db', 'Gb': 'F#',
            'F#': 'F#', 'C#': 'C#', 'D#': 'D#', 'G#': 'G#', 'A#': 'A#'
        }
        
        # Also map unicode flat/sharp and long forms
        enharmonics.update({
            'B♭': 'Bb', 'E♭': 'Eb', 'A♭': 'Ab', 'D♭': 'Db', 'G♭': 'Gb',
            'F♯': 'F#', 'C♯': 'C#', 'D♯': 'D#', 'G♯': 'G#', 'A♯': 'A#'
        })
        
        # accept X Major / X Minor textual forms
        for k in list(mappings.keys()):
            mappings[k + " Major"] = mappings[k]
            if k.endswith('m'):
                mappings[k[:-1] + " Minor"] = mappings[k]
                mappings[k[:-1] + "m"] = mappings[k]
            else:
                mappings[k + " Minor"] = mappings.get(k, k) + "m"
        
        mappings.update(enharmonics)
        return mappings
    
    def _normalize_key(self, key: str) -> str:
        key = key.strip().replace(' Major', '').replace(' Minor', 'm')
        return self.key_mappings.get(key, key)
    
    def _build_compatibility_matrix(self) -> Dict[Tuple[str, str], float]:
        # Build a simple heuristic compatibility between keys based on circle distance
        matrix: Dict[Tuple[str, str], float] = {}
        base_keys = list(self.circle_of_fifths)
        minors = list(self.relative_minors.values())
        all_keys = base_keys + minors
        
        # helper to compute distance on circle (0..6)
        def circle_distance(a: str, b: str) -> int:
            try:
                ia = base_keys.index(a)
                ib = base_keys.index(b)
            except ValueError:
                # if minor input, convert to relative major
                ia = None
                ib = None
                for maj, minr in self.relative_minors.items():
                    if minr == a:
                        ia = base_keys.index(maj)
                    if minr == b:
                        ib = base_keys.index(maj)
                # if still not found, fallback maximum distance
                if ia is None or ib is None:
                    return 6
            d = abs(ia - ib)
            return min(d, 12 - d)
        
        for a in all_keys:
            for b in all_keys:
                if a == b:
                    matrix[(a, b)] = 1.0
                    continue
                # convert minor to relative major for distance purposes
                ar = a[:-1] if a.endswith('m') else a
                br = b[:-1] if b.endswith('m') else b
                # default distance
                try:
                    dist = circle_distance(ar, br)
                except Exception:
                    dist = 6
                # base compatibility by distance
                if dist == 0:
                    comp = 1.0
                elif dist == 1:
                    comp = 0.9
                elif dist == 2:
                    comp = 0.75
                elif dist == 3:
                    comp = 0.6
                elif dist == 4:
                    comp = 0.4
                elif dist == 5:
                    comp = 0.25
                else:
                    comp = 0.2
                # minor/major relationship boosts
                if a.endswith('m') and (not b.endswith('m')):
                    # minor to its relative major
                    if self.relative_minors.get(br) == a:
                        comp = max(comp, 0.95)
                if b.endswith('m') and (not a.endswith('m')):
                    if self.relative_minors.get(ar) == b:
                        comp = max(comp, 0.95)
                # symmetric
                matrix[(a, b)] = comp
        return matrix
    
    def _get_circle_position(self, key: str) -> Optional[int]:
        # Handle minor keys by converting to relative major
        if key.endswith('m'):
            for major, minor in self.relative_minors.items():
                if minor == key:
                    key = major
                    break
        try:
            return self.circle_of_fifths.index(key)
        except ValueError:
            return None
    
    def _distance_score(self, a: str, b: str) -> float:
        # Return a compatibility-like score between 0 and 1
        a = self._normalize_key(a)
        b = self._normalize_key(b)
        if a == b:
            return 1.0
        if (a, b) in self.compatibility_matrix:
            return self.compatibility_matrix[(a, b)]
        # fallback - compute by circle position
        pa = self._get_circle_position(a)
        pb = self._get_circle_position(b)
        if pa is None or pb is None:
            return 0.3
        distance = abs(pa - pb) % 12
        if distance == 0:
            return 1.0
        elif distance == 1:
            return 0.9
        elif distance == 2:
            return 0.7
        elif distance == 3:
            return 0.4
        elif distance == 4:  # Minor third
            return 0.6
        elif distance == 5:  # Perfect fourth
            return 0.8
        elif distance == 6:  # Opposite on circle
            return 0.3
        else:
            return 0.5

    def _to_major(self, key: str) -> Optional[str]:
        """Convert a minor (e.g. 'Am') to its relative major ('C'), or return major unchanged."""
        key = self._normalize_key(key)
        if key.endswith('m'):
            # Build reverse mapping from relative_minors
            for major, minor in self.relative_minors.items():
                if minor == key:
                    return major
            return None
        return key

    def _to_minor(self, key: str) -> Optional[str]:
        """Convert a major (e.g. 'C') to its relative minor ('Am'), or return minor unchanged."""
        key = self._normalize_key(key)
        if not key.endswith('m'):
            return self.relative_minors.get(key)
        return key

    def suggest_bridge_keys(self, key1: str, key2: str, available_keys: Optional[List[str]] = None) -> List[str]:
        """
        Suggest keys that can bridge between key1 -> key2.
        Returns a list of human-readable bridge suggestions (single keys or chains like 'C -> Am' or 'C -> G -> Em').
        If available_keys is provided, prefer sequences that use those keys (useful if you want to mix *through* songs in the set).
        """
        key1 = self._normalize_key(key1)
        key2 = self._normalize_key(key2)
        if available_keys:
            available_keys = [self._normalize_key(k) for k in available_keys]
        else:
            available_keys = []

        # If already compatible enough, nothing needed
        direct_compat = self.compatibility_matrix.get((key1, key2), 0)
        if direct_compat > 0.65:
            return []

        # Compose universe of candidate keys (majors + relative minors)
        all_keys = list(self.circle_of_fifths) + list(self.relative_minors.values())

        # Helper to get compatibility with safety default
        def compat(a, b):
            return self.compatibility_matrix.get((self._normalize_key(a), self._normalize_key(b)), 0.0)

        candidates = set()

        # 1) Add obvious candidates: relative major/minor of each endpoint
        rm1 = self._to_major(key1)
        rm2 = self._to_major(key2)
        if rm1:
            candidates.add(self._normalize_key(rm1))
            m1 = self._to_minor(rm1)
            if m1:
                candidates.add(self._normalize_key(m1))
        if rm2:
            candidates.add(self._normalize_key(rm2))
            m2 = self._to_minor(rm2)
            if m2:
                candidates.add(self._normalize_key(m2))

        # 2) Add keys that are near either key on the circle (neighbors/2-steps)
        pos1 = self._get_circle_position(key1)
        pos2 = self._get_circle_position(key2)
        if pos1 is not None:
            for offset in (-2, -1, 1, 2):
                idx = (pos1 + offset) % 12
                candidates.add(self.circle_of_fifths[idx])
        if pos2 is not None:
            for offset in (-2, -1, 1, 2):
                idx = (pos2 + offset) % 12
                candidates.add(self.circle_of_fifths[idx])

        # 3) Add any key that has reasonable compatibility with both endpoints
        for k in all_keys:
            k_norm = self._normalize_key(k)
            if k_norm == key1 or k_norm == key2:
                continue
            if compat(key1, k_norm) > 0.45 and compat(k_norm, key2) > 0.45:
                candidates.add(k_norm)

        # Score single-key candidates by combined compatibility
        single_suggestions = []
        for k in candidates:
            score = compat(key1, k) + compat(k, key2)
            if score > 0:  # some usefulness
                # small boost to keys that exist in available_keys (so we encourage mixing through set)
                if k in available_keys:
                    score += 0.15
                single_suggestions.append((k, score))
        single_suggestions.sort(key=lambda x: x[1], reverse=True)

        # --- Beam search for multi-hop chains (up to max_hops) ---
        def find_chains(max_hops: int = 4, beam_width: int = 60, min_edge_compat: float = 0.45):
            """
            Beam-search style pathfinder that finds sequences from key1 to key2 (excluding key1 in formatted result).
            Returns list of (path_list, score) where path_list includes intermediate keys and ends with key2.
            """
            beams = [([key1], 0.0)]
            results = []

            for depth in range(1, max_hops + 1):
                new_beams = []
                for path, score in beams:
                    last = path[-1]
                    # consider neighbors from all_keys
                    for k in all_keys:
                        k_norm = self._normalize_key(k)
                        if k_norm in path:
                            continue
                        edge_c = compat(last, k_norm)
                        if edge_c < min_edge_compat:
                            continue
                        # boost nodes that are in available_keys
                        node_boost = 0.12 if k_norm in available_keys else 0.0
                        new_score = score + edge_c + node_boost
                        new_path = path + [k_norm]
                        if k_norm == key2:
                            # record the full path (excluding starting key for display)
                            results.append((new_path[1:], new_score))
                        else:
                            new_beams.append((new_path, new_score))
                # prune and continue
                beams = sorted(new_beams, key=lambda x: x[1], reverse=True)[:beam_width]
            # sort results best-first
            results.sort(key=lambda x: x[1], reverse=True)
            return results

        chain_results = find_chains(max_hops=4, beam_width=80, min_edge_compat=0.45)

        # Format results: single suggestions first, then chain suggestions
        results: List[str] = []
        # Add up to 4 top single-key suggestions
        for k, score in single_suggestions[:4]:
            if k not in (key1, key2) and k not in results:
                results.append(k)

        # Add chain suggestions from the dedicated chain search
        # Format chains like 'A -> B' or 'A -> B -> C'
        for path, score in chain_results[:6]:
            formatted = " -> ".join(path)
            if formatted not in results:
                results.append(formatted)

        # If still nothing, fall back to simple two-step try (keeps prior behavior)
        if not results:
            # build two-step chain suggestions (previous approach)
            chain_suggestions = []
            for a in all_keys:
                a_norm = self._normalize_key(a)
                if a_norm in (key1, key2):
                    continue
                if compat(key1, a_norm) < 0.5:
                    continue
                for b in all_keys:
                    b_norm = self._normalize_key(b)
                    if b_norm in (key1, key2, a_norm):
                        continue
                    if compat(a_norm, b_norm) < 0.5:
                        continue
                    if compat(b_norm, key2) < 0.5:
                        continue
                    score = compat(key1, a_norm) + compat(a_norm, b_norm) + compat(b_norm, key2)
                    # small boosts for available_keys
                    if a_norm in available_keys:
                        score += 0.1
                    if b_norm in available_keys:
                        score += 0.1
                    chain_suggestions.append(((a_norm, b_norm), score))
            chain_suggestions.sort(key=lambda x: x[1], reverse=True)
            for (a, b), score in chain_suggestions[:4]:
                formatted = f"{a} -> {b}"
                if formatted not in results:
                    results.append(formatted)

        # Fallback: if nothing found, suggest stepping halfway around the circle (median key)
        if not results:
            if pos1 is not None and pos2 is not None:
                # Compute halfway position (clockwise)
                dist = (pos2 - pos1) % 12
                half = (pos1 + dist // 2) % 12
                fallback = self.circle_of_fifths[half]
                results.append(fallback)

        # Limit output to 8 suggestions for UI clarity
        return results[:8]
    
    def create_harmonic_sequence(self, songs: List[Song]) -> List[Song]:
        # Simple reordering to minimize harmonic jumps: greedy nearest neighbor
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
            song.key = self._normalize_key(song.key)
        
        sequence = self.create_harmonic_sequence(songs)
        mixing_pairs = self.find_mixing_pairs(songs)
        
        # Prepare a list of keys that exist in the set so suggest_bridge_keys can prefer paths
        existing_keys = [s.key for s in songs]
        
        gaps_and_bridges = []
        for i in range(len(sequence) - 1):
            a = sequence[i]
            b = sequence[i + 1]
            score = self._distance_score(a.key, b.key)
            if score < 0.6:
                # pass the existing keys so the bridge suggestion can prefer using them
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

# ---- Scraping helpers (song metadata fetchers) ----
def fetch_bpm_from_web(title: str, artist: str) -> Optional[int]:
    # naive web scrape for bpm from a site like songbpm.com
    try:
        q = f"{title} {artist} bpm"
        url = f"https://www.google.com/search?q={requests.utils.quote(q)}"
        headers = {'User-Agent': 'Mozilla/5.0'}
        r = requests.get(url, headers=headers, timeout=5)
        soup = BeautifulSoup(r.text, 'html.parser')
        # try to find bpm numbers
        text = soup.get_text()
        m = re.search(r'(\d{2,3})\s?BPM', text, re.IGNORECASE)
        if m:
            return int(m.group(1))
    except Exception:
        return None
    return None

# ---- Streamlit UI ----
st.title("Harmonic Song Analyzer 🎧")
st.write("Upload a CSV or paste a list of songs to analyze harmonic flow and suggest bridge keys/sequences.")

uploaded = st.file_uploader("Upload CSV of songs (title,artist,key,tempo?)", type=["csv"])
text_input = st.text_area("Or paste CSV/text (title,artist,key,tempo)", height=120)

songs: List[Song] = []

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
    # try to parse simple CSV lines
    sio = StringIO(text_input)
    try:
        df = pd.read_csv(sio, header=None)
        for _, r in df.iterrows():
            parts = [str(x) for x in r if pd.notna(x)]
            if len(parts) >= 3:
                songs.append(Song(title=parts[0], artist=parts[1], key=parts[2], tempo=(int(parts[3]) if len(parts) > 3 else None)))
    except Exception:
        # fallback simple line parser
        lines = [l.strip() for l in text_input.splitlines() if l.strip()]
        for line in lines:
            parts = [p.strip() for p in re.split(r',|\t|;', line) if p.strip()]
            if len(parts) >= 3:
                t, a, k = parts[0], parts[1], parts[2]
                tempo = int(parts[3]) if len(parts) > 3 and parts[3].isdigit() else None
                songs.append(Song(title=t, artist=a, key=k, tempo=tempo))

st.sidebar.header("Options")
shuffle = st.sidebar.checkbox("Shuffle input order before sequencing", value=False)
show_raw = st.sidebar.checkbox("Show normalized keys", value=True)

if songs:
    hs = HarmonicSequencer()
    if shuffle:
        np.random.shuffle(songs)
    
    # attempt to fetch missing tempos
    for s in songs:
        if s.tempo is None:
            s.tempo = fetch_bpm_from_web(s.title, s.artist)
            time.sleep(0.2)
    
    analysis = hs.analyze_song_collection(songs)
    seq = analysis['sequence']
    mixing_pairs = analysis['mixing_pairs']
    gaps_and_bridges = analysis['gaps_and_bridges']
    
    st.header("Recommended Play Order")
    rows = []
    for i, s in enumerate(seq):
        rows.append({
            'Order': i + 1,
            'Title': s.title,
            'Artist': s.artist,
            'Key': s.key,
            'Tempo': s.tempo or ''
        })
    df_display = pd.DataFrame(rows)
    st.dataframe(df_display)
    
    st.header("Mixing Pair Scores")
    mp_rows = []
    for a, b, score in mixing_pairs:
        mp_rows.append({
            'From': f"{a.title} - {a.artist} ({a.key})",
            'To': f"{b.title} - {b.artist} ({b.key})",
            'Score': round(score, 2)
        })
    st.table(pd.DataFrame(mp_rows))
    
    if gaps_and_bridges:
        st.header("Harmonic Gaps & Bridge Suggestions")
        for gb in gaps_and_bridges:
            a = gb['from']
            b = gb['to']
            st.subheader(f"{a.title} ({a.key}) → {b.title} ({b.key}) — score {gb['score']:.2f}")
            st.write("Bridge suggestions (single keys, then multi-step sequences — sequences prefer keys already in your set):")
            for s in gb['suggestions']:
                st.write(f"- {s}")
            st.write("---")
    else:
        st.success("No major harmonic gaps detected — your order flows well!")
else:
    st.info("No songs provided yet. Upload a CSV or paste songs in the textbox.")

# ---- Footer / credits ----
st.markdown("---")
st.markdown("Built with ❤️ — Harmonic suggestions are heuristic-based to help DJs and playlist curators. "
            "Use your ears and musical judgement; these are suggestions, not rules.")

