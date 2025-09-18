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
    page_title="Harmonic Song Analyzer — Camelot",
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

# ---- Harmonic Sequencer (Camelot-based) ----
class HarmonicSequencer:
    def __init__(self):
        # Canonical display name for each Camelot code (one canonical key per encoding)
        # We choose familiar spellings for display: majors use common names, minors use the usual minor names.
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

        # Build alias map: many possible textual spellings -> one canonical Camelot code.
        # This ensures "all keys are mapped to just one possible encoding".
        self.alias_to_camelot: Dict[str, str] = self._build_alias_map()

        # Precompute numeric neighbors for Camelot codes (1..12)
        # no need for a separate compatibility matrix; we'll compute on the fly using rules
        self.camelot_codes = list(self.camelot_to_key.keys())

    def _build_alias_map(self) -> Dict[str, str]:
        # Normalize function for alias keys (lowercase, normalize unicode accidentals, strip)
        def clean(k: str) -> str:
            if not k:
                return ""
            s = str(k).strip()
            s = s.replace('♭', 'b').replace('♯', '#')
            s = s.replace(' Major', '').replace(' major', '')
            s = s.replace(' Minor', 'm').replace(' minor', 'm')
            s = re.sub(r'\s+', '', s)
            s = s.replace('mM', 'm')
            return s.lower()

        # Manual mapping: map common spellings (majors and minors, flats/sharps/variants) to Camelot codes.
        # This is intentionally explicit so a wide variety of user inputs map deterministically to one encoding.
        m = {}
        # Helper to add multiple aliases
        def add_aliases(aliases: List[str], code: str):
            for a in aliases:
                m[clean(a)] = code

        # 1B = B major ; 1A = G# minor (Ab minor enharmonic)
        add_aliases(['B', 'B major', 'Cb'], '1B')
        add_aliases(['G#m', 'Abm', 'G# minor', 'Ab minor', 'G#m'], '1A')

        # 2B = F# major ; 2A = D# minor (Eb minor enharmonic)
        add_aliases(['F#', 'F# major', 'Gb'], '2B')
        add_aliases(['D#m', 'Ebm', 'D# minor', 'Eb minor', 'D#m'], '2A')

        # 3B = Db major ; 3A = Bbm
        add_aliases(['Db', 'Db major', 'C#', 'C# major'], '3B')
        add_aliases(['Bbm', 'A#m', 'Bbminor', 'A# minor', 'Bb m', 'A#m'], '3A')

        # 4B = Ab major ; 4A = Fm
        add_aliases(['Ab', 'Ab major', 'G#', 'G# major'], '4B')
        add_aliases(['Fm', 'F minor'], '4A')

        # 5B = Eb major ; 5A = Cm
        add_aliases(['Eb', 'Eb major', 'D#', 'D# major'], '5B')
        add_aliases(['Cm', 'C minor'], '5A')

        # 6B = Bb major ; 6A = Gm
        add_aliases(['Bb', 'Bb major', 'A#', 'A# major'], '6B')
        add_aliases(['Gm', 'G minor'], '6A')

        # 7B = F major ; 7A = Dm
        add_aliases(['F', 'F major', 'E#'], '7B')
        add_aliases(['Dm', 'D minor'], '7A')

        # 8B = C major ; 8A = Am
        add_aliases(['C', 'C major', 'B#'], '8B')
        add_aliases(['Am', 'A minor'], '8A')

        # 9B = G major ; 9A = Em
        add_aliases(['G', 'G major'], '9B')
        add_aliases(['Em', 'E minor'], '9A')

        # 10B = D major ; 10A = Bm
        add_aliases(['D', 'D major'], '10B')
        add_aliases(['Bm', 'B minor'], '10A')

        # 11B = A major ; 11A = F#m
        add_aliases(['A', 'A major'], '11B')
        add_aliases(['F#m', 'F# minor', 'Gbm'], '11A')

        # 12B = E major ; 12A = C#m
        add_aliases(['E', 'E major'], '12B')
        add_aliases(['C#m', 'C# minor', 'Dbm'], '12A')

        # Some additional helpful aliases and common notations (with spaces, lower/upper case variants)
        extras = {
            'c': '8B', 'am': '8A', 'g': '9B', 'em': '9A', 'd': '10B', 'bm': '10A',
            'a': '11B', 'f#m': '11A', 'e': '12B', 'c#m': '12A', 'bb': '6B', 'bbm': '3A'
        }
        for k, v in extras.items():
            m[clean(k)] = v

        # Finally return the alias map
        return m

    def _clean_key_input(self, key: str) -> str:
        # tiny helper to produce normalized key string used for alias lookup
        if not key:
            return ""
        s = str(key).strip()
        s = s.replace('♭', 'b').replace('♯', '#')
        s = s.replace('Major', '').replace('major', '')
        s = s.replace('Minor', 'm').replace('minor', 'm')
        s = re.sub(r'\s+', '', s)
        return s

    def key_to_camelot(self, key: str) -> Optional[str]:
        """Return the Camelot code (e.g. '8A', '8B') for a given key string, or None if unknown."""
        if not key:
            return None
        cleaned = self._clean_key_input(key).lower()
        if cleaned in self.alias_to_camelot:
            return self.alias_to_camelot[cleaned]
        # Last-ditch attempts to handle awkward inputs:
        # try with/without trailing 'm'
        if cleaned.endswith('m') and cleaned[:-1] in self.alias_to_camelot:
            return self.alias_to_camelot[cleaned[:-1]]
        if (cleaned + 'm') in self.alias_to_camelot:
            return self.alias_to_camelot[cleaned + 'm']
        return None

    def camelot_to_display(self, camelot_code: str) -> str:
        """Return human-friendly 'Name (Camelot)' for display, e.g. 'Am (8A)'."""
        name = self.camelot_to_key.get(camelot_code, camelot_code)
        return f"{name} ({camelot_code})"

    # ---- Camelot compatibility / scoring rules ----
    def camelot_neighbors(self, code: str) -> List[str]:
        """
        Return immediate safe neighbors for a Camelot code:
        - same number, opposite letter (A<->B)
        - adjacent numbers +/-1 with same letter
        """
        if code not in self.camelot_to_key:
            return []
        # parse
        m = re.match(r'(\d+)([AB])', code)
        if not m:
            return []
        num = int(m.group(1))
        letter = m.group(2)
        neighbors = []
        # same number opposite letter
        other_letter = 'A' if letter == 'B' else 'B'
        neighbors.append(f"{num}{other_letter}")
        # adjacent numbers (wrap around 1..12)
        prev_num = num - 1 if num > 1 else 12
        next_num = num + 1 if num < 12 else 1
        neighbors.append(f"{prev_num}{letter}")
        neighbors.append(f"{next_num}{letter}")
        return neighbors

    def camelot_score(self, c1: str, c2: str) -> float:
        """
        Heuristic compatibility score between two Camelot codes (0..1).
        Rules (typical DJ practice):
         - same exact code: 1.0
         - same number A<->B (relative major/minor): 0.95
         - adjacent number, same letter (e.g., 8A -> 9A): 0.8
         - adjacent number but A/B swap (less ideal): 0.6
         - two-step adjacency (±2 same letter): 0.4
         - otherwise small baseline 0.2
        """
        if c1 == c2:
            return 1.0
        # validate codes
        if c1 not in self.camelot_to_key or c2 not in self.camelot_to_key:
            return 0.2
        n1, l1 = int(re.match(r'(\d+)', c1).group(1)), c1[-1]
        n2, l2 = int(re.match(r'(\d+)', c2).group(1)), c2[-1]
        if n1 == n2 and l1 != l2:
            return 0.95
        # compute circular distance
        diff = min((n1 - n2) % 12, (n2 - n1) % 12)
        if diff == 1:
            return 0.8 if l1 == l2 else 0.6
        if diff == 2:
            return 0.45 if l1 == l2 else 0.35
        return 0.2

    # ---- Distance wrapper used by other logic (keeps previous API) ----
    def _distance_score(self, a: str, b: str) -> float:
        ca = self.key_to_camelot(a) or ""
        cb = self.key_to_camelot(b) or ""
        if not ca or not cb:
            # if unknown mapping, fallback to conservative low compatibility
            return 0.25
        return self.camelot_score(ca, cb)

    # ---- Bridge suggestion (single keys, two-step chains, multi-hop chains) ----
    def suggest_bridge_keys(self, key1: str, key2: str, available_keys: Optional[List[str]] = None) -> List[str]:
        """
        Suggest camelot-keyed bridges between key1 -> key2.
        Returns display strings like 'Am (8A)' or 'G#m (1A) -> D#m (2A)'.
        If available_keys provided, we prefer chains that use those Camelot encodings.
        """

        c1 = self.key_to_camelot(key1)
        c2 = self.key_to_camelot(key2)
        if available_keys:
            available_camelot = [self.key_to_camelot(k) for k in available_keys]
            available_camelot = [x for x in available_camelot if x]
        else:
            available_camelot = []

        # If either mapping unknown, try to be permissive: return a minimal suggestion based on any partial map
        if not c1 or not c2:
            # if one of the keys unknown, try to suggest using the known one
            known = c1 or c2
            if known:
                # suggest neighbors of the known key
                neigh = self.camelot_neighbors(known)
                results = [self.camelot_to_display(n) for n in neigh if n in self.camelot_to_key]
                return results[:6]
            return []

        # If already compatible enough, no bridge needed
        if self.camelot_score(c1, c2) >= 0.8:
            return []

        # Universe of all camelot codes
        all_codes = list(self.camelot_to_key.keys())

        # Helper to boost scores for using available keys
        def boost_for_available(code: str) -> float:
            return 0.12 if code in available_camelot else 0.0

        # 1) Single-key candidates: any single Camelot code that reasonably connects both endpoints
        single_candidates: List[Tuple[str, float]] = []
        for cand in all_codes:
            if cand in (c1, c2):
                continue
            s = self.camelot_score(c1, cand) + self.camelot_score(cand, c2) + boost_for_available(cand)
            # threshold: a single candidate should provide sensible combined compatibility.
            if s > 1.2:  # tuned threshold (two 0.6 edges or better)
                single_candidates.append((cand, s))
        single_candidates.sort(key=lambda x: x[1], reverse=True)

        # 2) Two-step chains (A -> B)
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

        # 3) Multi-hop beam search (up to 4 hops) using Camelot adjacency scoring
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
                            # store the intermediate nodes (excluding starting key)
                            results.append((new_path[1:], new_score))
                        else:
                            new_beams.append((new_path, new_score))
                beams = sorted(new_beams, key=lambda x: x[1], reverse=True)[:beam_width]
            results.sort(key=lambda x: x[1], reverse=True)
            return results

        chain_results = find_chains(max_hops=4, beam_width=80, min_edge_score=0.32)

        # Format output: prefer single candidates, then two-step, then multi-hop
        results: List[str] = []

        # Add top single candidates (displayed as 'Name (Code)')
        for cand, score in single_candidates[:4]:
            disp = self.camelot_to_display(cand)
            if disp not in results:
                results.append(disp)

        # Add top two-step chains
        for (a, b), score in two_step[:4]:
            formatted = f"{self.camelot_to_display(a)} -> {self.camelot_to_display(b)}"
            if formatted not in results:
                results.append(formatted)

        # Add multi-hop chain results (format chain of codes into display names)
        for path, score in chain_results[:6]:
            formatted = " -> ".join(self.camelot_to_display(p) for p in path)
            if formatted not in results:
                results.append(formatted)

        # Fallback: if nothing found (rare), return neighbors of c1 that move toward c2 numerically
        if not results:
            # attempt simple neighbor-based guidance
            neighbors = self.camelot_neighbors(c1)
            for n in neighbors:
                if n in self.camelot_to_key:
                    results.append(self.camelot_to_display(n))
            # as final fallback, show the direct number-match (A/B swap)
            if not results and f"{int(re.match(r'(\d+)', c1).group(1))}{('A' if c1.endswith('B') else 'B')}" in self.camelot_to_key:
                alt = f"{int(re.match(r'(\d+)', c1).group(1))}{('A' if c1.endswith('B') else 'B')}"
                results.append(self.camelot_to_display(alt))

        return results[:8]

    # ---- Sequencing and analysis (uses Camelot distance) ----
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
        # Normalize and assign camelot codes to each song
        for song in songs:
            song.key = song.key or ""
            song.camelot = self.key_to_camelot(song.key)
            # if we couldn't map, leave None — downstream logic will handle it
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

# ---- Scraping helpers (song metadata fetchers) ----
def fetch_bpm_from_web(title: str, artist: str) -> Optional[int]:
    # naive web scrape for bpm from a site like songbpm.com
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
st.title("Harmonic Song Analyzer 🎧 (Camelot)")
st.write("Upload a CSV or paste a list of songs to analyze harmonic flow and suggest bridge keys/sequences using the Camelot system.")

uploaded = st.file_uploader("Upload CSV of songs (title,artist,key,tempo?)", type=["csv"])
text_input = st.text_area("Or paste CSV/text (title,artist,key,tempo)", height=140)

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
show_raw = st.sidebar.checkbox("Show normalized keys & Camelot codes", value=True)

if songs:
    hs = HarmonicSequencer()
    if shuffle:
        np.random.shuffle(songs)

    # attempt to fetch missing tempos
    for s in songs:
        if s.tempo is None:
            s.tempo = fetch_bpm_from_web(s.title, s.artist)
            time.sleep(0.18)

    analysis = hs.analyze_song_collection(songs)
    seq = analysis['sequence']
    mixing_pairs = analysis['mixing_pairs']
    gaps_and_bridges = analysis['gaps_and_bridges']

    st.header("Recommended Play Order")
    rows = []
    for i, s in enumerate(seq):
        camelot_display = hs.camelot_to_display(s.camelot) if s.camelot else f"(unknown)"
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
    st.info("No songs provided yet. Upload a CSV or paste songs in the textbox.")

# ---- Footer / credits ----
st.markdown("---")
st.markdown(
    "Built with ❤️ — Harmonic suggestions now use the **Camelot** system. "
    "Each input key is deterministically mapped to a single Camelot code so suggestions are consistent. "
    "These are heuristics to help mixing — always trust your ears."
)
