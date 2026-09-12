# Storage and downloads

`cherenkov paths` prints the resolved paths.

| Contents | Default path | Override |
| --- | --- | --- |
| Checkpoints and packed stores | `~/.local/share/cherenkov` | `XDG_DATA_HOME` |
| Transfer scratch | `~/.cache/cherenkov` | `XDG_CACHE_HOME` |
| Server configuration | `~/.config/cherenkov/cherenkov.toml` | `XDG_CONFIG_HOME` |

These defaults also apply on macOS. Each XDG override must be absolute;
Cherenkov appends `cherenkov` to it. Empty or relative values are ignored.
Resolving paths does not create directories. Downloading and packing create
the required parent directories.

`--root DIR` places data in `DIR`, scratch in `DIR/scratch`, and configuration
in `DIR/cherenkov.toml`. Put it after the subcommand:

```sh
cherenkov paths --root /Volumes/Models/cherenkov
cherenkov download
cherenkov pack
cherenkov serve
```

An explicit model path overrides the managed default. `serve` reads the default
config if present, or the file selected by `--config`. Its `[server].root` can
change model lookup; a CLI root takes precedence. Other subcommands do not read
TOML. See [server configuration](server-config.md) for path resolution.

## Layout

```text
data/
  downloads/                       Hugging Face blobs and snapshot links
  models/<owner>/<repo>/<commit>/   source-model directory
    config.json
    tokenizer.json
    model*.safetensors
    packed/                        runnable packed store
      config.json
      tokenizer.json
      manifest.json
      dense.bin
      ngram.bin
      experts.bin
      experts2.bin + manifest2.json
      experts3.bin + manifest3.json
```

Model directories link to downloaded blobs. Clearing scratch leaves models
and packed stores intact. Stores coexist and are not automatically evicted.

## Download

The default checkpoint is `Sawfwair/Qwen3.8-Flash-Next-MLX-4bit` at
`6cc9bbc0fae9ce26b7670b3ed1e26d557c154506`. Branches and tags passed to
`download --revision` are resolved to full commits. Other downloads print
their model path; pass it to `pack`, inference, or `serve` to use them.

`download --metadata-only` fetches config and tokenizer files without weights.
A later full download reuses them. The downloader validates the architecture,
reads shard names from the safetensors index, and checks available disk space.
This checkpoint needs about 104 GB for source weights and another 104 GB for
base packing.

Downloads use the Rust `hf-hub` client and Xet. Supply credentials through
`HF_TOKEN`, an existing Hugging Face login, or `download --hf-token`.
`HF_TOKEN` avoids exposing a token in command arguments. Tokens are not saved
in Cherenkov config or manifests.

Xet scratch uses `scratch/xet/`, unless `HF_XET_CACHE` is set. Credential
lookup follows `HF_TOKEN`, `HF_TOKEN_PATH`, and `HF_HOME`. Authentication grants
account access and rate limits; it does not guarantee faster transfers.

## Pack

```sh
cherenkov pack /path/to/model
cherenkov pack /path/to/model --experts 2,3
```

Packing defaults to `MODEL_DIR/packed`. `--output DIR` selects another runnable
store directory. An existing packed directory can also be passed as input.
Omit the model path to use the managed checkpoint.

`--experts` accepts 4, 3, or 2, separated by commas, spaces, or repeated flags.
The default is 4. Low-bit targets require the Q4 base, built first if absent.
Missing variants share one pass through the base records. Valid stores are
reused; unselected stores are left intact. Allow about 39 GB for 2-bit and
54 GB for 3-bit in addition to the base store.

The packer checks disk space and publishes manifests after flushing output.
Inference also builds missing low-bit stores. `--repack` during inference
rebuilds the selected low-bit store.

The server's prefix cache and sessions are held in RAM. They have no disk store.
