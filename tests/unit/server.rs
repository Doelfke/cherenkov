use super::response::Response;
use super::*;

#[test]
fn request_values_override_reloadable_defaults() {
    let defaults = Defaults {
        max_tokens: 128,
        stream: true,
        include_usage: true,
        no_eos: false,
    };
    let r = parse_request(&json!({"prompt":"hello"}), ApiKind::Completion, &defaults).unwrap();
    assert_eq!(r.max_tokens, 128);
    assert!(r.stream && r.include_usage);
    let r = parse_request(
        &json!({"prompt":"hello", "max_tokens":7, "stream":false,
        "stream_options":{"include_usage":false}}),
        ApiKind::Completion,
        &defaults,
    )
    .unwrap();
    assert_eq!(r.max_tokens, 7);
    assert!(!r.stream && !r.include_usage);
}

#[test]
fn chat_template_preserves_roles_and_defaults() {
    let r = parse_request(&json!({"messages":[{"role":"system","content":"Be brief."},{"role":"user","content":"Hi"}],"stream":true,"stream_options":{"include_usage":true},"max_completion_tokens":17}),ApiKind::Chat,&Defaults::default()).unwrap();
    assert_eq!(
        r.prompt,
        "<|im_start|>system\nBe brief.<|im_end|>\n<|im_start|>user\nHi<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
    );
    assert_eq!(r.max_tokens, 17);
    assert!(r.stream && r.include_usage);
    assert_eq!(
        parse_request(
            &json!({"prompt":"raw"}),
            ApiKind::Completion,
            &Defaults::default()
        )
        .unwrap()
        .prompt,
        "raw"
    );
}

#[test]
fn reject_unsupported_generation_instead_of_ignoring_it() {
    for extra in [
        json!({"temperature":0.7}),
        json!({"n":2}),
        json!({"max_tokens":0}),
        json!({"max_tokens":-1}),
        json!({"stream":"yes"}),
        json!({"tools":[]}),
        json!({"stop":["x"]}),
        json!({"model":"other"}),
        json!({"response_format":{"type":"json_object"}}),
    ] {
        let mut request = json!({"prompt":"test"});
        request
            .as_object_mut()
            .unwrap()
            .extend(extra.as_object().unwrap().clone());
        assert!(
            parse_request(&request, ApiKind::Completion, &Defaults::default()).is_err(),
            "{request}"
        );
    }
    assert!(parse_request(&json!({"messages":[]}), ApiKind::Chat, &Defaults::default()).is_err());
    assert!(
        parse_request(
            &json!({"messages":[{"role":"user","content":[{"type":"image_url"}]}]}),
            ApiKind::Chat,
            &Defaults::default()
        )
        .is_err()
    );
}

#[test]
fn bounded_header_reader_rejects_truncation_and_long_lines() {
    let mut n = 4;
    assert!(line(&mut std::io::Cursor::new(b"abcdef\n"), &mut n).is_err());
    let mut n = 16;
    assert!(line(&mut std::io::Cursor::new(b"missing newline"), &mut n).is_err());
    let mut n = 16;
    assert_eq!(
        line(&mut std::io::Cursor::new(b"ok\r\n"), &mut n).unwrap(),
        "ok"
    );
}

#[test]
fn response_wrapper_preserves_endpoint_and_stream_formats() {
    // Full response expectations keep protocol field omissions visible: chat has
    // no logprobs/text, completion has no message/delta, and only chat sends a role.
    let cases = [
        (
            ApiKind::Chat,
            "chatcmpl-123-7",
            "chat.completion",
            "chat.completion.chunk",
            json!({"index":0,"message":{"role":"assistant","content":"Hello \u{e9}"},"finish_reason":"stop"}),
            vec![
                json!({"index":0,"delta":{"role":"assistant","content":""},"finish_reason":null}),
                json!({"index":0,"delta":{"content":"Hello "},"finish_reason":null}),
                json!({"index":0,"delta":{"content":"\u{e9}"},"finish_reason":null}),
                json!({"index":0,"delta":{},"finish_reason":"stop"}),
            ],
        ),
        (
            ApiKind::Completion,
            "cmpl-123-7",
            "text_completion",
            "text_completion",
            json!({"index":0,"text":"Hello \u{e9}","logprobs":null,"finish_reason":"stop"}),
            vec![
                json!({"index":0,"text":"Hello ","logprobs":null,"finish_reason":null}),
                json!({"index":0,"text":"\u{e9}","logprobs":null,"finish_reason":null}),
                json!({"index":0,"text":"","logprobs":null,"finish_reason":"stop"}),
            ],
        ),
    ];
    let usage = json!({"prompt_tokens":5,"prompt_tokens_details":{"cached_tokens":2},"completion_tokens":3,"total_tokens":8});
    for (kind, id, object, chunk_object, final_choice, chunks) in cases {
        for streaming in [false, true] {
            for include_usage in [false, true] {
                let request = Request {
                    prompt: String::new(),
                    stable_prefix: String::new(),
                    message_prefix: String::new(),
                    max_tokens: 8,
                    stream: streaming,
                    include_usage,
                    no_eos: false,
                };
                let mut output = Vec::new();
                let mut response = Response::new(&mut output, kind, 7, 123, &request);
                response.start().unwrap();
                response.text("Hello ").unwrap();
                response.text("\u{e9}").unwrap();
                response
                    .finish("Hello \u{e9}", "stop", usage.clone())
                    .unwrap();
                let wire = String::from_utf8(output).unwrap();
                let (headers, body) = wire.split_once("\r\n\r\n").unwrap();
                assert!(headers.starts_with("HTTP/1.1 200 OK\r\n"));
                if streaming {
                    assert!(headers.contains("Content-Type: text/event-stream"));
                    let expected =
                        expected_stream(id, chunk_object, &chunks, include_usage.then_some(&usage));
                    assert_eq!(body, expected);
                } else {
                    assert!(headers.contains("Content-Type: application/json"));
                    assert!(headers.contains(&format!("Content-Length: {}\r\n", body.len())));
                    assert_eq!(
                        serde_json::from_str::<Value>(body).unwrap(),
                        json!({"id":id,"object":object,"created":123,"model":"cherenkov","choices":[final_choice],"usage":usage})
                    );
                }
            }
        }
    }
}

fn expected_stream(
    id: &str,
    chunk_object: &str,
    chunks: &[Value],
    usage: Option<&Value>,
) -> String {
    let mut expected = String::new();
    for choice in chunks {
        let chunk = json!({"id":id,"object":chunk_object,"created":123,"model":"cherenkov","choices":[choice]});
        expected.push_str(&format!("data: {chunk}\n\n"));
    }
    if let Some(usage) = usage {
        let chunk = json!({"id":id,"object":chunk_object,"created":123,"model":"cherenkov","choices":[],"usage":usage});
        expected.push_str(&format!("data: {chunk}\n\n"));
    }
    expected.push_str("data: [DONE]\n\n");
    expected
}

fn preparation_tokenizer() -> ChatTokenizer {
    use tokenizers::{
        Tokenizer, models::wordlevel::WordLevel, pre_tokenizers::whitespace::Whitespace,
    };
    let model = WordLevel::builder()
        .vocab(
            [("[UNK]", 0), ("hello", 1), ("world", 2), ("last", 3)]
                .into_iter()
                .map(|(word, id)| (word.to_owned(), id))
                .collect(),
        )
        .unk_token("[UNK]".into())
        .build()
        .unwrap();
    let mut inner = Tokenizer::new(model);
    inner.with_pre_tokenizer(Some(Whitespace));
    ChatTokenizer {
        inner,
        im_end: 4,
        endoftext: 5,
    }
}

#[test]
fn preparation_keeps_validated_tokens_prefix_boundaries_and_eos_policy() {
    let tok = preparation_tokenizer();
    let options = Options {
        max_ctx: 7,
        ..Options::default()
    };
    let mut request = parse_request(
        &json!({"prompt":"hello world last", "max_tokens":2}),
        ApiKind::Completion,
        &Defaults {
            no_eos: true,
            ..Defaults::default()
        },
    )
    .unwrap();
    // The incomplete word retokenizes differently: only the first token matches.
    request.stable_prefix = "hello wor".into();
    request.message_prefix = "hello world".into();
    let prepared = request.prepare(&tok, &options, 2).unwrap();
    assert_eq!(prepared.ids, vec![1, 2, 3]);
    assert_eq!(prepared.boundaries, vec![1, 2]);
    assert_eq!(prepared.request.max_tokens, 2);
    assert!(prepared.options.no_eos);
    assert!(!options.no_eos);
}

#[test]
fn preparation_rejects_empty_prompts_and_output_or_context_overflow() {
    let tok = preparation_tokenizer();
    for (prompt, output_limit, context, expected) in [
        ("", 2, 7, "at least one token"),
        ("hello world last", 1, 7, "output policy"),
        ("hello world last", 2, 6, "exceeding --max-ctx"),
    ] {
        let request = parse_request(
            &json!({"prompt":prompt, "max_tokens":2, "stream":true}),
            ApiKind::Completion,
            &Defaults::default(),
        )
        .unwrap();
        let options = Options {
            max_ctx: context,
            ..Options::default()
        };
        let error = request.prepare(&tok, &options, output_limit).err().unwrap();
        assert!(error.to_string().contains(expected), "{error}");
    }
}
