use super::*;

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
                let mut output = Vec::new();
                let mut response =
                    Response::for_writer(&mut output, kind, id, 123, streaming, include_usage);

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
