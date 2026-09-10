# Storage and downloads

`cherenkov paths` prints the resolved locations.

| Lifetime | Default location | Contents |
| --- | --- | --- |
| Durable data | `~/.local/share/cherenkov` | Downloaded checkpoints and generated model stores |
| Disposable scratch | `~/.cache/cherenkov` | Xet transfer scratch/cache |
| Editable configuration | `~/.config/cherenkov/cherenkov.toml` | Server policy and defaults |

These XDG defaults apply on macOS too. `XDG_DATA_HOME`, `XDG_CACHE_HOME`, and
`XDG_CONFIG_HOME` override the respective base directories; Cherenkov appends
`cherenkov` to each. Empty or relative overrides are ignored. Resolving paths
does not create directories.

`--root /some/directory` puts durable data directly under that root, scratch
under `scratch/`, and config at `cherenkov.toml`. Supply it after the subcommand.
`[server].root` can relocate server model lookup too; a CLI root wins. Relative
roots in TOML are relative to the TOML file.

```sh
cherenkov paths
cherenkov paths --root /Volumes/Models/cherenkov
cherenkov download
cherenkov pack
cherenkov serve
```

`serve` reads the single default config file if it exists. `--config` selects
another file. There is no directory search. Explicit local model paths take
precedence over the managed model default. `download`, `pack`, and `paths` use
`--root` or the XDG locations directly; they do not read `[server].root` from TOML.

## Durable layout

```text
data/
  downloads/                       Hugging Face blobs and snapshot links
  models/<owner>/<repo>/<commit>/   runnable source-model directory
    config.json                    links to completed download blobs
    tokenizer.json
    model*.safetensors
    packed/                        generated aligned weights
      config.json                  small runtime metadata copies
      tokenizer.json
      manifest.json
      dense.bin
      ngram.bin
      experts.bin
      experts2.bin + manifest2.json
      experts3.bin + manifest3.json
```

Model directories are keyed by the full Hub commit. Downloads and generated
stores are durable: clearing scratch never removes a model or triggers a
104 GB redownload. The model directory links to the Hub blobs instead of
copying the source weights a second time. The 2-bit and 3-bit stores coexist.
There is no automatic disk-store eviction.

The default is the measured `Sawfwair/Qwen3.8-Flash-Next-MLX-4bit` checkpoint
at commit `6cc9bbc0fae9ce26b7670b3ed1e26d557c154506`. Its config declares
`qwen4_exp` with a `qwen4_exp_text` text model. A branch or tag supplied to
`download --revision` is resolved to a full commit before fetching files.
Downloading another repository/revision prints its model directory; supply
that directory to `pack`, inference or `serve` to use it. It does not silently
change the default model.

`download --metadata-only` fetches config/tokenizer metadata without weight
shards. A subsequent full download reuses those files. Metadata alone cannot
be packed or used for inference. The downloader validates the architecture
before downloading weights and takes shard names from the safetensors index.
It checks required disk space before large transfers; source downloads and
base packing each need about 104 GB for this checkpoint.

Packing retains `MODEL_DIR/packed` for local models. A custom `pack --output`
directory includes its runtime metadata and can be passed directly to inference.
`pack` reuses a completed base store instead of overwriting it. Select expert
targets with `pack --experts 2,3` (commas, spaces, or repeated `--experts`);
the default is `4`. Low-bit targets require the Q4 base, built first if needed.
Cherenkov checks the combined additional disk requirement before conversion.
Valid existing variants are reused, and unselected variants are untouched.
You can also pass an existing packed directory directly to `pack` to add
variants there. `--output` selects the store directory, not a file name.
`--repack` during inference rebuilds only the selected low-bit store.

## Authentication and transfers

Downloads use the native Rust `hf-hub` client with built-in Xet transfers;
no Python installation or external `hf` command is required. Supply a token
through `HF_TOKEN`, an existing Hugging Face login, or `download --hf-token`.
Tokens are not written into Cherenkov TOML, model manifests or control responses.
Using `HF_TOKEN` avoids putting the token in shell command history or arguments.

Xet scratch defaults to the resolved scratch directory's `xet/` child.
An explicit `HF_XET_CACHE` remains an upstream developer override. Hugging Face
credential lookup retains its standard `HF_TOKEN`, `HF_TOKEN_PATH` and `HF_HOME`
behavior; Cherenkov does not copy credentials into its data root.

The prefix cache is separate: it stores inference checkpoints in RAM and
uses the server's byte, entry and idle limits. It has no disk directory.
The local control socket remains in its private per-user runtime directory.
