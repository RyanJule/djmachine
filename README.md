# Harmonic Song Analyzer

A Streamlit application that analyzes music playlists using the Camelot Wheel system to create optimal harmonic sequences for DJ mixing and music curation.

## Features

- **Automatic Key Detection**: Parses musical keys from various formats including sharps, flats, and Unicode symbols
- **Camelot Wheel Integration**: Maps keys to the industry-standard Camelot Wheel system (1A-12B)
- **Harmonic Sequencing**: Creates optimal play orders that minimize harmonic gaps between songs
- **Bridge Suggestions**: Identifies missing keys that would create smooth transitions between songs
- **Multiple Input Sources**: 
  - Upload CSV files with song data
  - Paste text/CSV data directly
  - Fetch playlists from SongData.io using Spotify URLs

## Installation

1. Clone this repository:
```bash
git clone https://github.com/yourusername/harmonic-song-analyzer.git
cd harmonic-song-analyzer
```

2. Install required dependencies:
```bash
pip install -r requirements.txt
```

3. Run the Streamlit application:
```bash
streamlit run streamlit_app.py
```

## Usage

### Input Methods

**CSV Upload/Paste**: Include columns for Title, Artist, and Key. The app automatically detects key columns.

**SongData Integration**: Enter a Spotify playlist URL to fetch harmonic data from SongData.io.

### Key Formats Supported

- Standard notation: `C`, `Am`, `F#`, `Bb`
- With mode labels: `C Major`, `A minor`
- Unicode symbols: `C♯`, `D♭`
- Combined format: `C (8B)`, `Am (8A)`

### Output

- **Recommended Play Order**: Songs arranged for optimal harmonic flow
- **Mixing Pair Scores**: Harmonic compatibility between consecutive songs
- **Bridge Suggestions**: Keys to add for smoother transitions between problematic pairs

## The Camelot Wheel

The app uses the Camelot Wheel system where keys are mapped to positions 1A-12B:

```
12A (C♯m) - 12B (E)
11A (F♯m) - 11B (A)
10A (Bm)  - 10B (D)
9A  (Em)  - 9B  (G)
8A  (Am)  - 8B  (C)
7A  (Dm)  - 7B  (F)
6A  (Gm)  - 6B  (B♭)
5A  (Cm)  - 5B  (E♭)
4A  (Fm)  - 4B  (G♯)
3A (A♯m)  - 3B  (C♯)
2A (D♯m)  - 2B  (F♯)
1A (G♯m)  - 1B  (B)
```

### Harmonic Rules

Smooth transitions occur between:
- Same position (identical keys)
- Adjacent numbers (7A ↔ 8A, 12A ↔ 1A)
- Same number, different letter (7A ↔ 7B)
- Diagonal moves (7A ↔ 6B, 7A ↔ 8B)

## Algorithm

The sequencing algorithm uses graph traversal to find paths through the Camelot Wheel that visit all available keys using only harmonic transitions. When perfect harmonic sequences aren't possible, it identifies the optimal bridge keys to complete the path.

## Requirements

- Python 3.7+
- streamlit
- pandas
- numpy
- requests
- beautifulsoup4

## Contributing

Pull requests welcome. For major changes, please open an issue first to discuss proposed modifications.

## License

MIT License - see LICENSE file for details.