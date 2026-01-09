# Emji (Rust)

On-device emoji semantic lookup CLI powered by vector search

Rust rewrite of [Emji](https://github.com/subpath/Emji) with pure Rust vector search, batch processing, and multiple model support.

## What it does

A CLI that uses sentence-transformers ONNX models with pure Rust vector search to help you find the emoji you're looking for. Faster, lighter, and uses batch processing for efficient indexing.

## Installation

### Pre-built binaries

Download the latest release for your platform from [Releases](https://github.com/subpath/emji-rs/releases):

**Linux (x86_64)**
```bash
curl -L https://github.com/subpath/emji-rs/releases/releases/latest/download/emji-linux-x86_64.tar.gz | tar xz
sudo mv emji /usr/local/bin/
```

**macOS (Apple Silicon)**
```bash
curl -L https://github.com/subpath/emji-rs/releases/latest/download/emji-macos-aarch64.tar.gz | tar xz
sudo mv emji /usr/local/bin/
```

**macOS (Intel)**
```bash
curl -L https://github.com/subpath/emji-rs/releases/latest/download/emji-macos-x86_64.tar.gz | tar xz
sudo mv emji /usr/local/bin/
```

### From source

```bash
git clone https://github.com/your-repo/emji-rust.git
cd emji-rust
cargo build --release
cargo install --path .
```

### Data

All data is stored under `~/.emji`:

- Model files: `~/.emji/model_*.onnx`
- Tokenizer files: `~/.emji/tokenizer_*.json`
- Emoji data: `~/.emji/shortnames.json`
- Optional override: `~/.emji/shortnames_override.json` (takes precedence if present)
- Vector index: `~/.emji/emoji_index_*.db` (per model)
- Config: `~/.emji/.config`

On first run, you'll be prompted to choose a model. The CLI will then automatically download the model, tokenizer, and emoji data.

## Quick Start

1. **First run** - choose your model:
   ```bash
   emji
   ```

   Available models:
   - `minilm` - Fast and lightweight, 22MB (recommended)
   - `minilm-l12` - Higher quality, 120MB
   - `multilingual` - 50+ languages, 470MB

2. **Search for emojis**:
   ```bash
   emji happy birthday
   emji coffee break
   emji celebration party
   ```

3. **Select and copy**: Choose from the interactive list, and the emoji will be copied to your clipboard!

4. **Force rebuild index**:
   ```bash
   emji --build-index
   ```

5. **Cleanup**:
   ```bash
   emji --cleanup
   ```

## Usage

### Commands

- `emji <text>` - Search for emojis matching your description
- `emji --build-index` - Force rebuild the semantic search index
- `emji --cleanup` - Delete all Emji data and config under `~/.emji`
- `emji --show-stats` - Show emoji popularity statistics
- `emji --list-models` - List available embedding models
- `emji --use-model <model>` - Switch to a different model
- `emji --show-model` - Show current model configuration

**Options:**

- `-n <number>`: Number of results to return (default: 3)

### Examples

```bash
# Find celebration emojis
emji party celebration

# Find food-related emojis
emji delicious food

# Get more results
emji animals -n 5

# Switch to multilingual model
emji --use-model multilingual

# Rebuild index after switching models
emji --build-index
```

## How It Works

1. **Model Selection**: On first run, choose from MiniLM (fast), MiniLM-L12 (accurate), or Multilingual (50+ languages)
2. **Automatic Setup**: Downloads the selected model, tokenizer, and emoji data
3. **Batch Processing**: Embeddings are generated in batches of 32 for 10-20x faster indexing
4. **Pure Rust Vector Search**: Fast cosine similarity search without external dependencies
5. **Personalized Re-ranking**: Results blend cosine similarity with historical CTR, controlled by `ALPHA`
6. **Interactive Selection**: Best matches presented in an interactive menu

## Configuration

JSON config at `~/.emji/.config`:

```json
{
  "ALPHA": 0.2,
  "MODEL_NAME": "minilm",
  "ENCODER_TYPE": "minilm",
  "MODEL_URL": "https://huggingface.co/...",
  "TOKENIZER_URL": "https://huggingface.co/...",
  "EMOJI_URL": "https://gist.githubusercontent.com/..."
}
```

- `ALPHA`: Balance between similarity and CTR (0.0-1.0, default: 0.2)
- `MODEL_NAME`: Current model (`minilm`, `minilm-l12`, `multilingual`)
- Switch models with `--use-model` (requires rebuilding index)

To override emojis, place your file at `~/.emji/shortnames_override.json`.

## Performance

Compared to Python version:

- **Startup**: ~50% faster (no Python interpreter)
- **Indexing**: 10-20x faster (batch processing)
- **Memory**: ~60% lower
- **Binary**: ~15MB statically linked

## Development

```bash
# Run with debug info
cargo run -- happy birthday

# Run tests
cargo test

# Format code
cargo fmt

# Build release
cargo build --release
```

## License

Licensed under the Apache License 2.0. See LICENSE for details.

## Contributing

Contributions welcome! Please submit a Pull Request.
