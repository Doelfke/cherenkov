use super::*;
use serde::Deserialize;
use sha2::{Digest, Sha256};

#[derive(Deserialize)]
struct References {
    template_sha256: String,
    tokenizer_sha256: String,
    cases: Vec<Case>,
}

#[derive(Deserialize)]
struct Case {
    id: String,
    context: Value,
    text: Option<String>,
    #[serde(default)]
    token_ids: Vec<u32>,
    error: Option<String>,
}

fn references() -> References {
    serde_json::from_str(include_str!("../fixtures/prompt/references.json"))
        .expect("Transformers reference fixtures")
}

fn sha256(bytes: &[u8]) -> String {
    Sha256::digest(bytes)
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect()
}

#[test]
fn checkpoint_template_matches_transformers_reference_bytes_and_errors() {
    let references = references();
    let template = fixture_template();
    let source = include_bytes!("../fixtures/prompt/chat_template.jinja");
    assert_eq!(sha256(source), references.template_sha256);
    for case in references.cases {
        let result = template.render_context(&case.context);
        if let Some(expected) = case.error {
            let error = result.expect_err(&case.id);
            assert!(
                format!("{error:#}").contains(&expected),
                "{}: {error:#}",
                case.id
            );
            continue;
        }
        assert_eq!(result.unwrap(), case.text.unwrap(), "{}", case.id);
    }
}

#[test]
fn renderer_matches_reference_token_ids_with_the_checkpoint_tokenizer() {
    let Some(model) = std::env::var_os("CHERENKOV_MODEL_DIR") else {
        eprintln!("set CHERENKOV_MODEL_DIR to verify reference token IDs");
        return;
    };
    let path = Path::new(&model).join("tokenizer.json");
    let references = references();
    let bytes = std::fs::read(&path).unwrap();
    assert_eq!(sha256(&bytes), references.tokenizer_sha256);
    let tokenizer = tokenizers::Tokenizer::from_file(path).unwrap();
    let template = fixture_template();
    for case in references
        .cases
        .into_iter()
        .filter(|case| case.error.is_none())
    {
        let rendered = template.render_context(&case.context).unwrap();
        let encoded = tokenizer.encode(rendered, false).unwrap();
        assert_eq!(encoded.get_ids(), case.token_ids, "{}", case.id);
    }
}

#[test]
fn chat_adapter_preserves_reference_bytes_and_safe_unicode_prefixes() {
    let template = fixture_template();
    for case in references()
        .cases
        .into_iter()
        .filter(|case| case.error.is_none())
    {
        if case.context["enable_thinking"] != false
            || case.context["add_generation_prompt"] != true
            || case.context.get("preserve_thinking").is_some()
        {
            continue;
        }
        let messages = case.context["messages"].as_array().unwrap();
        let prompt = template.chat(messages).unwrap();
        assert_eq!(Some(&prompt.text), case.text.as_ref(), "{}", case.id);
        for boundary in prompt.boundaries {
            assert!(prompt.text.is_char_boundary(boundary), "{}", case.id);
        }
        assert!(prompt.boundaries[0] <= prompt.boundaries[1]);
    }
}

#[test]
fn cli_and_chat_share_the_loaded_template_and_developer_alias() {
    let template = fixture_template();
    let cli = template.user("  Describe a café.  ").unwrap();
    let chat = template
        .chat(&[json!({"role":"user", "content":"  Describe a café.  "})])
        .unwrap();
    assert_eq!(cli.text, chat.text);
    assert_eq!(cli.boundaries, chat.boundaries);
    let mut messages = vec![
        json!({"role":"developer", "content":"Be brief."}),
        json!({"role":"user", "content":"Hi"}),
    ];
    let developer = template.chat(&messages).unwrap();
    messages[0]["role"] = json!("system");
    assert_eq!(developer.text, template.chat(&messages).unwrap().text);
}

#[test]
fn loading_prefers_the_jinja_file_and_supports_embedded_templates() {
    let directory = tempfile::tempdir().unwrap();
    assert!(ChatTemplate::load(directory.path()).unwrap().is_none());
    let config = directory.path().join("tokenizer_config.json");
    for template in [
        json!("embedded {{ messages[0].content }}"),
        json!([{"name":"default", "template":"embedded {{ messages[0].content }}"}]),
    ] {
        std::fs::write(&config, json!({"chat_template":template}).to_string()).unwrap();
        let loaded = ChatTemplate::load(directory.path()).unwrap().unwrap();
        assert_eq!(loaded.user("Hello").unwrap().text, "embedded Hello");
    }
    std::fs::write(
        directory.path().join("chat_template.jinja"),
        "file {{ messages[0].content }}",
    )
    .unwrap();
    let loaded = ChatTemplate::load(directory.path()).unwrap().unwrap();
    assert_eq!(loaded.user("Hello").unwrap().text, "file Hello");
    std::fs::write(
        directory.path().join("chat_template.jinja"),
        "{% invalid %}",
    )
    .unwrap();
    assert!(ChatTemplate::load(directory.path()).is_err());
}

#[test]
fn raw_prompt_has_no_template_or_cache_boundaries() {
    let prompt = Prompt::raw("plain text".into());
    assert_eq!(prompt.text, "plain text");
    assert_eq!(prompt.boundaries, [0, 0]);
}
