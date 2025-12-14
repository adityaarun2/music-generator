Placeholder folder for dataset metadata and intermediate artifacts used by the music generation pipeline.

## Structure
```
artifacts/
├── artist_genres.csv       # Artist → genre mapping
├── index.parquet           # Indexed metadata for all MIDI files
├── splits.json             # Train / val / test split definitions
└── tokenized/              # Tokenized dataset artifacts (generated)
```
- **artist_genres.csv** - Maps artist names to their corresponding musical genres
- **index.parquet** - Contains indexed metadata for all MIDI files in the dataset
- **splits.json** - Defines the train/validation/test split configurations
- **tokenized/** - Directory containing tokenized dataset outputs (generated during preprocessing)