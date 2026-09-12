# Prompt fixtures

`chat_template.jinja` is copied unchanged from the
[checkpoint](https://huggingface.co/Sawfwair/Qwen3.8-Flash-Next-MLX-4bit/blob/6cc9bbc0fae9ce26b7670b3ed1e26d557c154506/chat_template.jinja).
`references.json` contains 20 contexts, expected text or errors, token IDs,
input hashes, and generator versions.

References were generated with Transformers 4.57.6, Jinja2 3.1.6, and tokenizers
0.22.2. To regenerate:

1. Use the recorded checkpoint and versions.
2. Render each saved `context` with Transformers' `_compile_jinja_template`.
3. Check nonempty successful cases with `AutoTokenizer.apply_chat_template`,
   using both `tokenize=False` and `tokenize=True`.
4. Replace the expected text, token IDs, or error. Use `add_special_tokens=False`
   for tokenization. Do not use Cherenkov to generate reference outputs.

Empty-message cases call the template directly because the Transformers wrapper
indexes the first message before rendering.

Cargo tests need no Python. Text and error checks always run. Token checks
require `CHERENKOV_MODEL_DIR` with the matching tokenizer; weights and GPU are
not needed. Hash checks reject stale inputs.
