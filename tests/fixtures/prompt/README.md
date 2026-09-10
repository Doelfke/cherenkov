# Checkpoint template references

`chat_template.jinja` is an unchanged copy from
[the checkpoint](https://huggingface.co/Sawfwair/Qwen3.8-Flash-Next-MLX-4bit/blob/6cc9bbc0fae9ce26b7670b3ed1e26d557c154506/chat_template.jinja).
Runtime rendering loads the model directory's template; this copy is test data.

`references.json` records template/tokenizer SHA-256 hashes, generator versions,
and 20 input contexts with expected rendered text, token IDs or template errors.
The outputs were generated independently with Transformers 4.57.6, Jinja2 3.1.6
and Python tokenizers 0.22.2, using only the local checkpoint.

For each context, Transformers' `_compile_jinja_template` rendered the original
source. Successful nonempty cases were also checked with
`AutoTokenizer.apply_chat_template`, both with `tokenize=False` and
`tokenize=True`. Tokenization uses `add_special_tokens=False`. Empty-message
validation goes directly through the template because the Transformers wrapper
indexes the first message before rendering.

To regenerate, use the recorded checkpoint and versions with each saved
`context`; replace `text`, `token_ids` or `error` with those oracle outputs.
Do not derive expected outputs from Cherenkov's renderer.

Cargo tests read these fixtures directly and need no Python installation.
Byte/error comparisons always run. Token-ID comparisons additionally need
`CHERENKOV_MODEL_DIR` pointing at the matching checkpoint tokenizer; no weights
or GPU are needed for that test. Hash checks detect stale reference inputs.
