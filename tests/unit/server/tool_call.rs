//! Unit tests for the Qwen tool-call output parser and streaming decoder.
//!
//! Marker strings are assembled from the module constants so the tag
//! literals never appear verbatim here.

use super::*;
use serde_json::json;
use std::sync::Arc;

fn contract() -> Arc<ToolCallOutputContract> {
    let tool = json!({
        "function": {
            "name": "get_weather",
            "parameters": {
                "properties": {
                    "city": {"type": "string"},
                    "days": {"type": "integer"},
                    "count": {"type": "integer"},
                    "precip": {"type": ["boolean", "null"]}
                }
            }
        }
    });

    ToolCallOutputContract::from_tools(&[tool])
}

fn arg(name: &str, value: &str) -> String {
    format!("{PARAM_OPEN}{name}>\n{value}\n{PARAM_CLOSE}\n")
}

fn call(name: &str, body: &str) -> String {
    format!("{TOOL_OPEN}\n{FUNCTION_OPEN}{name}>\n{body}{FUNCTION_CLOSE}\n{TOOL_CLOSE}\n")
}

#[test]
fn text_without_marker_stays_content() {
    let parsed = parse_qwen_tool_call_output("Plain answer, no calls.", &contract(), false);

    assert!(!parsed.is_tool_call_response);
    assert_eq!(parsed.content, "Plain answer, no calls.");
    assert!(parsed.tool_calls.is_empty());
    assert!(!parsed.diagnostics.marker_seen);
    assert_eq!(parsed.diagnostics.fallback_reason, FallbackReason::None);
}

#[test]
fn single_tool_call_yields_structured_arguments() {
    let text = format!(
        "Let me check that.\n{}",
        call(
            "get_weather",
            &format!("{}{}", arg("city", "Paris"), arg("days", "3"))
        )
    );
    let parsed = parse_qwen_tool_call_output(&text, &contract(), false);

    assert!(parsed.is_tool_call_response);
    assert_eq!(parsed.content, "Let me check that.");
    assert_eq!(parsed.tool_calls.len(), 1);

    let tool_call = &parsed.tool_calls[0];

    assert_eq!(tool_call.name, "get_weather");
    assert_eq!(tool_call.arguments["city"], json!("Paris"));
    assert_eq!(tool_call.arguments["days"], json!(3));
    assert_eq!(tool_call.arguments_json, "{\"city\":\"Paris\",\"days\":3}");
    assert_eq!(parsed.diagnostics.fallback_reason, FallbackReason::None);
    assert_eq!(parsed.diagnostics.structured_call_count, 1);
}

#[test]
fn adjacent_tool_calls_all_parse() {
    let text = format!(
        "{}{}",
        call("get_weather", &arg("city", "Rome")),
        call("get_weather", &arg("precip", "true"))
    );
    let parsed = parse_qwen_tool_call_output(&text, &contract(), false);

    assert!(parsed.is_tool_call_response);
    assert_eq!(parsed.tool_calls.len(), 2);
    assert_eq!(parsed.tool_calls[0].arguments["city"], json!("Rome"));
    assert_eq!(parsed.tool_calls[1].arguments["precip"], json!(true));
    assert_eq!(parsed.diagnostics.structured_call_count, 2);
}

#[test]
fn normalization_honors_declared_types() {
    let body = format!(
        "{}{}{}",
        arg("days", "1.5"),
        arg("count", ""),
        arg("precip", "TRUE")
    );
    let parsed = parse_qwen_tool_call_output(&call("get_weather", &body), &contract(), false);

    assert!(parsed.is_tool_call_response);

    let tool_call = &parsed.tool_calls[0];

    // An out-of-scope number stays structured and is counted as a mismatch.
    assert_eq!(tool_call.arguments["days"], json!(1.5));
    assert_eq!(parsed.diagnostics.schema_mismatch_arguments, 1);
    // An empty non-string value is omitted entirely.
    assert!(!tool_call.arguments.contains_key("count"));
    assert_eq!(parsed.diagnostics.empty_arguments_omitted, 1);
    // Boolean literals normalize to JSON booleans.
    assert_eq!(tool_call.arguments["precip"], json!(true));
}

#[test]
fn empty_string_parameter_emits_empty_string() {
    let parsed =
        parse_qwen_tool_call_output(&call("get_weather", &arg("city", "")), &contract(), false);

    assert!(parsed.is_tool_call_response);
    assert_eq!(parsed.tool_calls[0].arguments["city"], json!(""));
    assert_eq!(parsed.diagnostics.empty_arguments_omitted, 0);
}

#[test]
fn nested_parameter_markup_stays_inside_the_value() {
    let nested = format!("{PARAM_OPEN}inner>\nx\n{PARAM_CLOSE}");
    let parsed = parse_qwen_tool_call_output(
        &call("get_weather", &arg("city", &nested)),
        &contract(),
        false,
    );

    assert!(parsed.is_tool_call_response);
    assert_eq!(parsed.tool_calls[0].arguments["city"], json!(nested));
}

#[test]
fn trailing_suffix_fails_strict_and_recovers_tolerant() {
    let text = format!(
        "Short answer.\n{}done.",
        call("get_weather", &arg("city", "Oslo"))
    );
    let strict = parse_qwen_tool_call_output(&text, &contract(), false);

    assert!(!strict.is_tool_call_response);
    assert_eq!(strict.content, text);
    assert_eq!(
        strict.diagnostics.fallback_reason,
        FallbackReason::TrailingContent
    );

    let tolerant = parse_qwen_tool_call_output(&text, &contract(), true);

    assert!(tolerant.is_tool_call_response);
    assert_eq!(tolerant.content, "Short answer.");
    assert_eq!(tolerant.tool_calls.len(), 1);
    assert_eq!(
        tolerant.diagnostics.fallback_reason,
        FallbackReason::TruncatedTail
    );
}

#[test]
fn truncated_tail_falls_back_strict_and_recovers_tolerant() {
    // Complete parameter, but the closing tags are cut off at the budget.
    let text = format!(
        "{TOOL_OPEN}\n{FUNCTION_OPEN}get_weather>\n{}\n",
        arg("city", "Turin")
    );

    // Strict mode requires the closing tags; the budget cut is a hard
    // structural failure.
    let strict = parse_qwen_tool_call_output(&text, &contract(), false);

    assert!(!strict.is_tool_call_response);
    assert_eq!(strict.content, text);
    assert_eq!(
        strict.diagnostics.fallback_reason,
        FallbackReason::MalformedStructure
    );

    // Tolerant mode recovers the complete call and discards the truncated tail.
    let tolerant = parse_qwen_tool_call_output(&text, &contract(), true);

    assert!(tolerant.is_tool_call_response);
    assert_eq!(tolerant.tool_calls[0].arguments["city"], json!("Turin"));
    assert_eq!(
        tolerant.diagnostics.fallback_reason,
        FallbackReason::TruncatedTail
    );
}

#[test]
fn tolerant_recovers_missing_gt_after_function_name() {
    let text = format!(
        "{TOOL_OPEN}\n{FUNCTION_OPEN}get_weather\n{param}{FUNCTION_CLOSE}\n{TOOL_CLOSE}\n",
        param = arg("city", "Lyon")
    );

    let strict = parse_qwen_tool_call_output(&text, &contract(), false);

    assert!(!strict.is_tool_call_response);

    let tolerant = parse_qwen_tool_call_output(&text, &contract(), true);

    assert!(tolerant.is_tool_call_response);
    assert_eq!(tolerant.tool_calls[0].name, "get_weather");
    assert_eq!(tolerant.tool_calls[0].arguments["city"], json!("Lyon"));
}

#[test]
fn duplicate_parameter_fails_strict_and_keeps_earlier_calls_tolerant() {
    let text = format!(
        "{}{}",
        call("get_weather", &arg("city", "Lisbon")),
        call(
            "get_weather",
            &format!("{}{}", arg("city", "Madrid"), arg("city", "Seville"))
        )
    );

    let strict = parse_qwen_tool_call_output(&text, &contract(), false);

    assert!(!strict.is_tool_call_response);
    assert_eq!(
        strict.diagnostics.fallback_reason,
        FallbackReason::DuplicateParameter
    );

    let tolerant = parse_qwen_tool_call_output(&text, &contract(), true);

    assert!(tolerant.is_tool_call_response);
    assert_eq!(tolerant.tool_calls.len(), 1);
    assert_eq!(tolerant.tool_calls[0].arguments["city"], json!("Lisbon"));
}

#[test]
fn undeclared_tool_name_falls_back() {
    let text = call("unknown_tool", &arg("city", "Kyiv"));

    let parsed = parse_qwen_tool_call_output(&text, &contract(), false);

    assert!(!parsed.is_tool_call_response);
    assert_eq!(parsed.content, text);
    assert_eq!(
        parsed.diagnostics.fallback_reason,
        FallbackReason::UndeclaredTool
    );

    let tolerant = parse_qwen_tool_call_output(&text, &contract(), true);

    assert!(!tolerant.is_tool_call_response);
}

#[test]
fn decoder_holds_back_a_marker_and_finishes_into_calls() {
    let region = format!(
        "{TOOL_OPEN}\n{FUNCTION_OPEN}get_weather>\n{}{FUNCTION_CLOSE}\n{TOOL_CLOSE}",
        arg("city", "Bonn")
    );
    let mut decoder = ToolCallOutputDecoder::new(contract(), false);

    assert_eq!(decoder.feed("Done. "), "Done.");

    // The marker completes mid-feed; everything from the marker on is held
    // back. The whitespace buffered from the previous feed is flushed with
    // the first visible byte here.
    let visible = decoder.feed(&format!("Sure: {region}"));

    assert_eq!(visible, " Sure:");

    let terminal = decoder.finish();

    assert_eq!(terminal.tool_calls.len(), 1);
    assert_eq!(terminal.tool_calls[0].name, "get_weather");
    assert_eq!(terminal.tool_calls[0].arguments["city"], json!("Bonn"));
    assert!(terminal.content.is_empty());
}

#[test]
fn decoder_restores_held_bytes_without_a_completed_marker() {
    // A pending marker prefix is restored verbatim at the end of the stream.
    let mut decoder = ToolCallOutputDecoder::new(contract(), false);

    assert_eq!(decoder.feed(&format!("Hi {}", &TOOL_OPEN[..8])), "Hi");

    let terminal = decoder.finish();

    assert!(terminal.tool_calls.is_empty());
    assert_eq!(terminal.content.as_str(), format!(" {}", &TOOL_OPEN[..8]));

    let mut decoder = ToolCallOutputDecoder::new(contract(), false);

    assert_eq!(decoder.feed("Hi \n"), "Hi");

    let terminal = decoder.finish();

    assert!(terminal.tool_calls.is_empty());
    assert_eq!(terminal.content, " \n");
}
