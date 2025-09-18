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
    layout="wide",
    initial_sidebar_state="expanded"
)

@dataclass
class Song:
    name: str
    artist: str
    key: str
    bpm: int
    
    def __str__(self):
        return f"{self.name} by {self.artist} ({self.key}, {self.bpm} BPM)"

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
        
        # Harmonic compatibility matrix (0-1 scale)
        self.compatibility_matrix = self._build_compatibility_matrix()
    
    def _build_key_mappings(self) -> Dict[str, str]:
        mappings = {}
        
        # Major keys
        for key in self.circle_of_fifths:
            mappings[key] = key
            mappings[key + 'maj'] = key
            mappings[key + 'major'] = key
        
        # Minor keys
        for major, minor in self.relative_minors.items():
            mappings[minor] = minor
            mappings[minor.replace('m', 'min')] = minor
            mappings[minor.replace('m', 'minor')] = minor
        
        # Enharmonic equivalents
        enharmonics = {
            'C#': 'Db', 'D#': 'Eb', 'F#': 'Gb', 'G#': 'Ab', 'A#': 'Bb',
            'Gb': 'F#', 'C#m': 'Dbm', 'D#m': 'Ebm', 'G#m': 'Abm', 'A#m': 'Bbm',
            'B♭': 'Bb', 'E♭': 'Eb', 'A♭': 'Ab', 'D♭': 'Db', 'G♭': 'Gb',
            'F♯': 'F#', 'C♯': 'C#', 'D♯': 'D#', 'G♯': 'G#', 'A♯': 'A#'
        }
        
        for alt, standard in enharmonics.items():
            mappings[alt] = standard
            
        return mappings
    
    def _normalize_key(self, key: str) -> str:
        key = key.strip().replace(' Major', '').replace(' Minor', 'm')
        return self.key_mappings.get(key, key)
    
    def _build_compatibility_matrix(self) -> Dict[Tuple[str, str], float]:
        matrix = {}
        all_keys = list(self.circle_of_fifths) + list(self.relative_minors.values())
        
        for key1 in all_keys:
            for key2 in all_keys:
                matrix[(key1, key2)] = self._calculate_compatibility(key1, key2)
        
        return matrix
    
    def _calculate_compatibility(self, key1: str, key2: str) -> float:
        if key1 == key2:
            return 1.0
        
        pos1 = self._get_circle_position(key1)
        pos2 = self._get_circle_position(key2)
        
        if pos1 is None or pos2 is None:
            return 0.3
        
        distance = min(abs(pos1 - pos2), 12 - abs(pos1 - pos2))
        
        if distance == 1:  # Adjacent keys (perfect fifth)
            return 0.9
        elif distance == 2:  # Two steps (major second)
            return 0.7
        elif distance == 3:  # Tritone
            return 0.4
        elif distance == 4:  # Minor third
            return 0.6
        elif distance == 5:  # Perfect fourth
            return 0.8
        elif distance == 6:  # Opposite on circle
            return 0.3
        else:
            return 0.5
    
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
    
    def _bpm_compatibility(self, bpm1: int, bpm2: int) -> float:
        ratio = max(bpm1, bpm2) / min(bpm1, bpm2)
        
        if ratio <= 1.05:
            return 1.0
        elif ratio <= 1.1:
            return 0.8
        elif ratio <= 1.2:
            return 0.6
        elif ratio <= 1.5:
            return 0.4
        elif ratio == 2.0:
            return 0.7
        elif abs(ratio - 1.5) < 0.1:
            return 0.6
        else:
            return 0.2
    
    def create_harmonic_sequence(self, songs: List[Song]) -> List[Song]:
        if not songs:
            return []
        
        # Normalize keys
        for song in songs:
            song.key = self._normalize_key(song.key)
        
        start_song = self._find_best_starting_song(songs)
        sequence = [start_song]
        remaining = [s for s in songs if s != start_song]
        
        while remaining:
            current_key = sequence[-1].key
            best_song = None
            best_score = -1
            
            for song in remaining:
                compatibility = self.compatibility_matrix.get((current_key, song.key), 0.3)
                bpm_compat = self._bpm_compatibility(sequence[-1].bpm, song.bpm)
                total_score = compatibility * 0.7 + bmp_compat * 0.3
                
                if total_score > best_score:
                    best_score = total_score
                    best_song = song
            
            if best_song:
                sequence.append(best_song)
                remaining.remove(best_song)
            else:
                sequence.extend(remaining)
                break
        
        return sequence
    
    def _find_best_starting_song(self, songs: List[Song]) -> Song:
        best_song = songs[0]
        best_score = 0
        
        for song in songs:
            score = sum(self.compatibility_matrix.get((song.key, other.key), 0.3) 
                       for other in songs if other != song)
            if score > best_score:
                best_score = score
                best_song = song
        
        return best_song
    
    def find_mixing_pairs(self, songs: List[Song], min_compatibility: float = 0.6) -> List[Tuple[Song, Song, float]]:
        pairs = []
        
        for i, song1 in enumerate(songs):
            for song2 in songs[i+1:]:
                key_compat = self.compatibility_matrix.get((song1.key, song2.key), 0.3)
                bmp_compat = self._bmp_compatibility(song1.bpm, song2.bpm)
                total_compat = key_compat * 0.6 + bmp_compat * 0.4
                
                if total_compat >= min_compatibility:
                    pairs.append((song1, song2, total_compat))
        
        return sorted(pairs, key=lambda x: x[2], reverse=True)
    
    def suggest_bridge_keys(self, key1: str, key2: str) -> List[str]:
        key1 = self._normalize_key(key1)
        key2 = self._normalize_key(key2)
        
        if self.compatibility_matrix.get((key1, key2), 0) > 0.6:
            return []
        
        bridge_keys = []
        all_keys = list(self.circle_of_fifths) + list(self.relative_minors.values())
        
        for bridge_key in all_keys:
            if (self.compatibility_matrix.get((key1, bridge_key), 0) > 0.6 and 
                self.compatibility_matrix.get((bridge_key, key2), 0) > 0.6):
                bridge_keys.append(bridge_key)
        
        bridge_keys.sort(key=lambda k: (
            self.compatibility_matrix.get((key1, k), 0) + 
            self.compatibility_matrix.get((k, key2), 0)
        ), reverse=True)
        
        return bridge_keys[:3]
    
    def analyze_song_collection(self, songs: List[Song]) -> Dict:
        for song in songs:
            song.key = self._normalize_key(song.key)
        
        sequence = self.create_harmonic_sequence(songs)
        mixing_pairs = self.find_mixing_pairs(songs)
        
        gaps_and_bridges = []
        for i in range(len(sequence) - 1):
            key1, key2 = sequence[i].key, sequence[i+1].key
            compatibility = self.compatibility_matrix.get((key1, key2), 0)
            
            if compatibility < 0.6:
                bridges = self.suggest_bridge_keys(key1, key2)
                gaps_and_bridges.append({
                    'from_song': sequence[i],
                    'to_song': sequence[i+1],
                    'compatibility': compatibility,
                    'suggested_bridges': bridges
                })
        
        key_counts = {}
        for song in songs:
            key_counts[song.key] = key_counts.get(song.key, 0) + 1
        
        return {
            'total_songs': len(songs),
            'harmonic_sequence': sequence,
            'mixing_pairs': mixing_pairs[:10],
            'gaps_and_bridges': gaps_and_bridges,
            'key_distribution': key_counts,
            'average_compatibility': np.mean([
                self.compatibility_matrix.get((sequence[i].key, sequence[i+1].key), 0)
                for i in range(len(sequence) - 1)
            ]) if len(sequence) > 1 else 0
        }

class SongDataScraper:
    def __init__(self):
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
    
    def scrape_songdata_playlist(self, url: str) -> List[Song]:
        try:
            if not url.startswith('http'):
                url = 'https://' + url
            
            with st.spinner('Fetching playlist data...'):
                time.sleep(1)  # Be respectful to the server
                response = requests.get(url, headers=self.headers, timeout=30)
                response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            title_elem = soup.find('h1', id='h1')
            playlist_title = title_elem.text.strip() if title_elem else "Unknown Playlist"
            
            table = soup.find('table', id='table_chart')
            if not table:
                st.error("Could not find song table on the page")
                return []
            
            songs = []
            tbody = table.find('tbody', id='table_body')
            if not tbody:
                st.error("Could not find table body")
                return []
            
            rows = tbody.find_all('tr', class_='table_object')
            
            progress_bar = st.progress(0)
            for i, row in enumerate(rows):
                try:
                    song = self._parse_songdata_row(row)
                    if song:
                        songs.append(song)
                    progress_bar.progress((i + 1) / len(rows))
                except Exception as e:
                    continue
            
            progress_bar.empty()
            st.success(f"Successfully scraped {len(songs)} songs from '{playlist_title}'")
            return songs
            
        except requests.RequestException as e:
            st.error(f"Error fetching webpage: {e}")
            return []
        except Exception as e:
            st.error(f"Error parsing webpage: {e}")
            return []
    
    def _parse_songdata_row(self, row) -> Optional[Song]:
        try:
            track_cell = row.find('td', class_='table_name')
            if not track_cell:
                return None
            
            track_link = track_cell.find('a')
            song_name = track_link.text.strip() if track_link else track_cell.text.strip()
            
            artist_cell = row.find('td', class_='table_artist')
            if not artist_cell:
                return None
            artist = artist_cell.text.strip()
            
            key_cell = row.find('td', class_='table_key')
            if not key_cell:
                return None
            key = self._clean_key(key_cell.text.strip())
            
            bpm_cell = row.find('td', class_='table_bpm')
            if not bpm_cell:
                return None
            bpm = self._clean_bpm(bpm_cell.text.strip())
            
            if not song_name or not artist or not key or bpm <= 0:
                return None
            
            return Song(
                name=song_name,
                artist=artist,
                key=key,
                bpm=bmp
            )
            
        except Exception:
            return None
    
    def _clean_key(self, key_str: str) -> str:
        if not key_str:
            return ""
        
        key = key_str.strip()
        key = re.sub(r'^(key of |in )', '', key, flags=re.IGNORECASE)
        key = re.sub(r'( major| minor| maj| min)$', lambda m: 'm' if 'min' in m.group().lower() else '', key, flags=re.IGNORECASE)
        key = key.replace('♯', '#').replace('♭', 'b')
        key = key.replace(' flat', 'b').replace('-flat', 'b')
        key = key.replace(' sharp', '#').replace('-sharp', '#')
        
        return key.strip()
    
    def _clean_bpm(self, bpm_value) -> int:
        if not bmp_value:
            return 0
        
        if isinstance(bmp_value, str):
            bmp_match = re.search(r'(\d+(?:\.\d+)?)', str(bmp_value))
            if bpm_match:
                bpm_value = float(bpm_match.group(1))
            else:
                return 0
        
        try:
            bpm = int(float(bpm_value))
            return max(0, bpm)
        except (ValueError, TypeError):
            return 0

def load_csv_data(uploaded_file) -> List[Song]:
    """Load songs from uploaded CSV file"""
    try:
        df = pd.read_csv(uploaded_file)
        
        st.write("CSV Columns detected:", list(df.columns))
        
        # Auto-detect columns
        name_col = None
        artist_col = None
        key_col = None
        bpm_col = None
        
        for col in df.columns:
            col_lower = col.lower().strip()
            if any(term in col_lower for term in ['name', 'song', 'title', 'track']):
                name_col = col
            elif any(term in col_lower for term in ['artist', 'performer', 'by']):
                artist_col = col
            elif any(term in col_lower for term in ['key', 'musical', 'tonality']):
                key_col = col
            elif any(term in col_lower for term in ['bpm', 'tempo', 'beats']):
                bpm_col = col
        
        if not all([name_col, artist_col, key_col, bpm_col]):
            st.error("Could not auto-detect all required columns. Please ensure your CSV has columns for: song name, artist, key, and BPM")
            return []
        
        st.success(f"Auto-detected columns: Name='{name_col}', Artist='{artist_col}', Key='{key_col}', BPM='{bpm_col}'")
        
        songs = []
        for _, row in df.iterrows():
            try:
                name = str(row[name_col]).strip()
                artist = str(row[artist_col]).strip()
                key = str(row[key_col]).strip()
                bpm = int(float(row[bpm_col]))
                
                if name and artist and key and bpm > 0:
                    songs.append(Song(name, artist, key, bpm))
            except:
                continue
        
        return songs
        
    except Exception as e:
        st.error(f"Error reading CSV: {e}")
        return []

def create_visualizations(analysis_results):
    """Create visualizations for the analysis results"""
    
    # Key Distribution
    if analysis_results['key_distribution']:
        st.subheader("🎵 Key Distribution")
        
        keys = list(analysis_results['key_distribution'].keys())
        counts = list(analysis_results['key_distribution'].values())
        
        fig = px.bar(x=keys, y=counts, 
                    title="Distribution of Musical Keys",
                    labels={'x': 'Musical Key', 'y': 'Number of Songs'})
        fig.update_layout(xaxis_tickangle=-45)
        st.plotly_chart(fig, use_container_width=True)
    
    # BPM Distribution
    songs = analysis_results['harmonic_sequence']
    if songs:
        st.subheader("⚡ BPM Distribution")
        
        bpms = [song.bpm for song in songs]
        fig = px.histogram(x=bpms, nbins=20, 
                          title="Distribution of BPM (Beats Per Minute)",
                          labels={'x': 'BPM', 'y': 'Number of Songs'})
        st.plotly_chart(fig, use_container_width=True)
    
    # Compatibility Heatmap for top mixing pairs
    if analysis_results['mixing_pairs']:
        st.subheader("🔥 Top Mixing Compatibility")
        
        pairs_data = []
        for song1, song2, compatibility in analysis_results['mixing_pairs'][:10]:
            pairs_data.append({
                'Song 1': f"{song1.name} ({song1.key})",
                'Song 2': f"{song2.name} ({song2.key})",
                'Compatibility': compatibility
            })
        
        df_pairs = pd.DataFrame(pairs_data)
        st.dataframe(df_pairs, use_container_width=True)

def main():
    st.title("🎵 Harmonic Song Analyzer")
    st.markdown("Analyze your playlists for optimal harmonic flow and DJ mixing recommendations")
    
    # Sidebar
    st.sidebar.title("📊 Input Options")
    input_method = st.sidebar.radio(
        "Choose your input method:",
        ["SongData.io URL", "Upload CSV File", "Manual Entry"]
    )
    
    songs = []
    
    if input_method == "SongData.io URL":
        st.sidebar.markdown("### 🌐 SongData.io Scraping")
        url = st.sidebar.text_input(
            "Enter songdata.io playlist URL:",
            placeholder="https://songdata.io/playlist/your-playlist-id"
        )
        
        if st.sidebar.button("Scrape Playlist"):
            if url:
                scraper = SongDataScraper()
                songs = scraper.scrape_songdata_playlist(url)
            else:
                st.error("Please enter a valid URL")
    
    elif input_method == "Upload CSV File":
        st.sidebar.markdown("### 📁 CSV Upload")
        uploaded_file = st.sidebar.file_uploader(
            "Choose a CSV file", 
            type="csv",
            help="CSV should contain columns: song name, artist, key, BPM"
        )
        
        if uploaded_file is not None:
            songs = load_csv_data(uploaded_file)
    
    elif input_method == "Manual Entry":
        st.sidebar.markdown("### ✏️ Manual Entry")
        st.sidebar.info("Add songs one by one using the form below")
        
        # Initialize session state for manual songs
        if 'manual_songs' not in st.session_state:
            st.session_state.manual_songs = []
        
        with st.form("add_song_form"):
            col1, col2 = st.columns(2)
            
            with col1:
                song_name = st.text_input("Song Name")
                key = st.selectbox("Key", [
                    "C", "C#", "Db", "D", "D#", "Eb", "E", "F", 
                    "F#", "Gb", "G", "G#", "Ab", "A", "A#", "Bb", "B",
                    "Am", "A#m", "Bbm", "Bm", "Cm", "C#m", "Dbm", "Dm",
                    "D#m", "Ebm", "Em", "Fm", "F#m", "Gbm", "Gm", "G#m", "Abm"
                ])
            
            with col2:
                artist = st.text_input("Artist")
                bpm = st.number_input("BPM", min_value=60, max_value=200, value=120)
            
            if st.form_submit_button("Add Song"):
                if song_name and artist:
                    new_song = Song(song_name, artist, key, int(bpm))
                    st.session_state.manual_songs.append(new_song)
                    st.success(f"Added: {new_song}")
                else:
                    st.error("Please fill in song name and artist")
        
        songs = st.session_state.manual_songs
        
        if songs:
            st.subheader("Current Songs")
            for i, song in enumerate(songs):
                col1, col2 = st.columns([4, 1])
                col1.write(f"{i+1}. {song}")
                if col2.button("Remove", key=f"remove_{i}"):
                    st.session_state.manual_songs.pop(i)
                    st.rerun()
    
    # Analysis Section
    if songs:
        st.header("🎯 Analysis Results")
        
        # Basic stats
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Songs", len(songs))
        
        keys = [song.key for song in songs]
        unique_keys = len(set(keys))
        col2.metric("Unique Keys", unique_keys)
        
        bpms = [song.bpm for song in songs]
        avg_bpm = sum(bpms) / len(bpms)
        col3.metric("Average BPM", f"{avg_bpm:.0f}")
        
        bmp_range = max(bpms) - min(bpms)
        col4.metric("BPM Range", f"{bpm_range}")
        
        # Run harmonic analysis
        with st.spinner("Analyzing harmonic relationships..."):
            sequencer = HarmonicSequencer()
            analysis = sequencer.analyze_song_collection(songs)
        
        # Display results
        st.subheader("🎼 Optimal Harmonic Sequence")
        st.info(f"Average Harmonic Compatibility: {analysis['average_compatibility']:.2f}")
        
        sequence_data = []
        for i, song in enumerate(analysis['harmonic_sequence'], 1):
            sequence_data.append({
                'Order': i,
                'Song': song.name,
                'Artist': song.artist,
                'Key': song.key,
                'BPM': song.bpm
            })
        
        df_sequence = pd.DataFrame(sequence_data)
        st.dataframe(df_sequence, use_container_width=True)
        
        # Export sequence
        if st.button("📋 Copy Sequence to Clipboard"):
            sequence_text = "\n".join([f"{i}. {song}" for i, song in enumerate(analysis['harmonic_sequence'], 1)])
            st.code(sequence_text)
        
        # Mixing pairs
        if analysis['mixing_pairs']:
            st.subheader("🔀 Top Mixing Pairs")
            
            for i, (song1, song2, score) in enumerate(analysis['mixing_pairs'][:5], 1):
                with st.expander(f"#{i} - Compatibility: {score:.2f}"):
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.write("**Song 1:**")
                        st.write(f"🎵 {song1.name}")
                        st.write(f"👤 {song1.artist}")
                        st.write(f"🎹 {song1.key}")
                        st.write(f"⚡ {song1.bpm} BPM")
                    
                    with col2:
                        st.write("**Song 2:**")
                        st.write(f"🎵 {song2.name}")
                        st.write(f"👤 {song2.artist}")
                        st.write(f"🎹 {song2.key}")
                        st.write(f"⚡ {song2.bpm} BPM")
        
        # Bridge suggestions
        if analysis['gaps_and_bridges']:
            st.subheader("🌉 Suggested Bridge Keys")
            
            for gap in analysis['gaps_and_bridges']:
                st.write(f"**Gap:** {gap['from_song'].key} → {gap['to_song'].key} (Compatibility: {gap['compatibility']:.2f})")
                st.write(f"*From:* {gap['from_song'].name} *To:* {gap['to_song'].name}")
                if gap['suggested_bridges']:
                    st.write(f"**Suggested bridge keys:** {', '.join(gap['suggested_bridges'])}")
                st.divider()
        
        # Visualizations
        create_visualizations(analysis)
    
    else:
        st.info("👆 Choose an input method from the sidebar to get started!")
        
        # Show example
        st.subheader("📋 Example")
        st.write("Here's what the analysis looks like with sample data:")
        
        example_songs = [
            Song("Blinding Lights", "The Weeknd", "Fm", 171),
            Song("Watermelon Sugar", "Harry Styles", "F", 95),
            Song("Levitating", "Dua Lipa", "Bm", 103),
            Song("Good 4 U", "Olivia Rodrigo", "Ab", 166),
        ]
        
        example_data = []
        for i, song in enumerate(example_songs, 1):
            example_data.append({
                'Order': i,
                'Song': song.name,
                'Artist': song.artist,
                'Key': song.key,
                'BPM': song.bpm
            })
        
        st.dataframe(pd.DataFrame(example_data), use_container_width=True)

if __name__ == "__main__":
    main()