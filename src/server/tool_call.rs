//! Qwen tool-call output parsing.
//!
//! The model emits each call as XML-style markup: an outer tool marker, a
//! function element carrying the name, and one parameter element per
//! argument. Argument text is untyped; a terminal contract built from the
//! request's tool schemas normalizes it into JSON values. A type mismatch
//! stays a structured call for consumer validation; only a structure or
//! identity failure returns the whole marker region to ordinary content.

use serde_json::{Map, Value, from_str};
use std::sync::Arc;

#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub(crate) enum FallbackReason {
    #[default]
    None,
    MalformedStructure,
    DuplicateParameter,
    InvalidToolName,
    UndeclaredTool,
    TrailingContent,
    /// Tolerant recovery discarded a trailing suffix that followed an
    /// otherwise complete call. A structured response was still produced.
    TruncatedTail,
}

/// Upper bound on model-emitted function names. Request-side names use a
/// shorter grammar on the wire.
const MAX_TOOL_NAME_LENGTH: usize = 128;

// JSON Schema type bits.
const T_NULL: u8 = 1 << 0;
const T_BOOLEAN: u8 = 1 << 1;
const T_INTEGER: u8 = 1 << 2;
const T_NUMBER: u8 = 1 << 3;
const T_STRING: u8 = 1 << 4;
const T_OBJECT: u8 = 1 << 5;
const T_ARRAY: u8 = 1 << 6;

// Marker literals are assembled at compile time so the plain tag strings do
// not appear verbatim in source, diagnostics, or tool traffic.
const TOOL_OPEN: &str = concat!("<", "tool_call>", "");
const TOOL_CLOSE: &str = concat!("</", "tool_call", ">");
const FUNCTION_OPEN: &str = concat!("<", "function=", "");
const FUNCTION_CLOSE: &str = concat!("</", "function", ">");
const PARAM_OPEN: &str = concat!("<", "parameter=", "");
const PARAM_CLOSE: &str = concat!("</", "parameter", ">");

fn is_ws(c: char) -> bool {
    c == ' ' || c == '\t' || c == '\r' || c == '\n'
}

fn trim_ws(text: &str) -> &str {
    text.trim_matches(is_ws)
}

fn valid_function_name(name: &str) -> bool {
    !name.is_empty()
        && name.len() <= MAX_TOOL_NAME_LENGTH
        && name
            .bytes()
            .all(|b| b.is_ascii_alphanumeric() || b == b'_' || b == b'-')
}

fn schema_bit(name: &str) -> Option<u8> {
    Some(match name {
        "null" => T_NULL,
        "boolean" => T_BOOLEAN,
        "integer" => T_INTEGER,
        "number" => T_NUMBER,
        "string" => T_STRING,
        "object" => T_OBJECT,
        "array" => T_ARRAY,
        _ => return None,
    })
}

/// Record the supported top-level types from a `type` keyword.
fn compile_direct_types(t: &Value) -> Option<u8> {
    if let Some(name) = t.as_str() {
        return schema_bit(name);
    }

    let members = t.as_array()?;

    if members.is_empty() {
        return None;
    }

    let mut bits = 0;

    for member in members {
        bits |= schema_bit(member.as_str()?)?;
    }

    (bits != 0).then_some(bits)
}

/// Compile the type set from a property schema's `type`, `anyOf`, or `oneOf`.
fn compile_schema_types(schema: &Value) -> Option<u8> {
    let object = schema.as_object()?;

    if let Some(ty) = object.get("type") {
        return compile_direct_types(ty);
    }

    let has_any_of = object.get("anyOf").is_some();
    let has_one_of = object.get("oneOf").is_some();

    if has_any_of == has_one_of {
        return None;
    }

    let alternatives = object
        .get(if has_any_of { "anyOf" } else { "oneOf" })?
        .as_array()?;

    if alternatives.is_empty() {
        return None;
    }

    let mut combined = 0;

    for alternative in alternatives {
        combined |= compile_schema_types(alternative)?;
    }

    (combined != 0).then_some(combined)
}

#[derive(Debug, Clone)]
pub(crate) struct Parameter {
    name: String,
    /// Declared types; zero keeps the legacy text policy.
    types: u8,
}

#[derive(Debug, Clone)]
pub(crate) struct Tool {
    name: String,
    parameters: Vec<Parameter>,
    unambiguous: bool,
}

fn same_tool(a: &Tool, b: &Tool) -> bool {
    a.parameters.len() == b.parameters.len()
        && a.parameters
            .iter()
            .zip(&b.parameters)
            .all(|(x, y)| x.name == y.name && x.types == y.types)
}

/// Parameter names and declared types gathered from the request's tool
/// schemas. Guidance only: schemas never validate the whole call.
#[derive(Debug, Clone, Default)]
pub(crate) struct ToolCallOutputContract {
    tools: Vec<Tool>,
    /// Model-emitted names must match a declared tool. Set whenever tool
    /// definitions are present.
    pub(crate) enforce_declared_names: bool,
}

impl ToolCallOutputContract {
    /// Build the terminal contract from shaped tool definitions. A name
    /// declared with conflicting schemas loses type guidance and falls back
    /// to legacy text.
    pub(crate) fn from_tools(tools: &[Value]) -> Arc<Self> {
        let mut contract = ToolCallOutputContract {
            enforce_declared_names: true,
            tools: Vec::new(),
        };

        for definition in tools {
            let Some(function) = definition.get("function").and_then(Value::as_object) else {
                continue;
            };
            let Some(name) = function.get("name").and_then(Value::as_str) else {
                continue;
            };

            let mut tool = Tool {
                name: name.to_owned(),
                parameters: Vec::new(),
                unambiguous: true,
            };

            let parameters = function
                .get("parameters")
                .and_then(Value::as_object)
                .and_then(|schema| schema.get("properties"))
                .and_then(Value::as_object);

            if let Some(parameters) = parameters {
                for (param_name, property) in parameters {
                    tool.parameters.push(Parameter {
                        name: param_name.to_owned(),
                        types: compile_schema_types(property).unwrap_or(0),
                    });
                }
            }

            if let Some(existing) = contract.tools.iter_mut().find(|t| t.name == tool.name) {
                if existing.unambiguous && !same_tool(existing, &tool) {
                    existing.parameters.clear();
                    existing.unambiguous = false;
                }
            } else {
                contract.tools.push(tool);
            }
        }

        Arc::new(contract)
    }

    /// Structural guidance for a call. `None` when the name is unknown or was
    /// declared with conflicting schemas.
    fn find_tool(&self, name: &str) -> Option<&Tool> {
        self.tools
            .iter()
            .find(|tool| tool.name == name && tool.unambiguous)
    }

    fn is_declared(&self, name: &str) -> bool {
        self.tools.iter().any(|tool| tool.name == name)
    }
}

// -- Value normalization -----------------------------------------------------

/// `1e5` is `100000`; integer classification must respect the exponent.
fn json_number_is_integer(number: &str) -> bool {
    let bytes = number.as_bytes();
    let mut pos = if bytes.first() == Some(&b'-') { 1 } else { 0 };

    if pos >= bytes.len() {
        return false;
    }

    let integer_begin = pos;

    while pos < bytes.len() && bytes[pos].is_ascii_digit() {
        pos += 1;
    }

    let integer_end = pos;
    let (mut fraction_begin, mut fraction_end) = (pos, pos);

    if pos < bytes.len() && bytes[pos] == b'.' {
        fraction_begin = pos + 1;
        pos += 1;

        while pos < bytes.len() && bytes[pos].is_ascii_digit() {
            pos += 1;
        }

        fraction_end = pos;
    }

    let (mut exponent_negative, mut exponent_value) = (false, 0usize);

    if pos < bytes.len() && (bytes[pos] == b'e' || bytes[pos] == b'E') {
        pos += 1;

        if pos < bytes.len() && (bytes[pos] == b'+' || bytes[pos] == b'-') {
            exponent_negative = bytes[pos] == b'-';
            pos += 1;
        }

        let cap = bytes.len();

        while pos < bytes.len() && bytes[pos].is_ascii_digit() {
            let digit = (bytes[pos] - b'0') as usize;

            if exponent_value != cap {
                if exponent_value > cap / 10 || (exponent_value == cap / 10 && digit > cap % 10) {
                    exponent_value = cap;
                } else {
                    exponent_value = exponent_value * 10 + digit;
                }
            }

            pos += 1;
        }
    }

    if integer_begin == integer_end || pos != bytes.len() {
        return false;
    }

    let mut coefficient_is_zero = true;
    let mut trailing_zeros = 0;

    for i in integer_begin..integer_end {
        if bytes[i] == b'0' {
            trailing_zeros += 1;
        } else {
            coefficient_is_zero = false;
            trailing_zeros = 0;
        }
    }

    for i in fraction_begin..fraction_end {
        if bytes[i] == b'0' {
            trailing_zeros += 1;
        } else {
            coefficient_is_zero = false;
            trailing_zeros = 0;
        }
    }

    if coefficient_is_zero {
        return true;
    }

    let fraction_digits = fraction_end - fraction_begin;

    if !exponent_negative {
        return if exponent_value >= fraction_digits {
            true
        } else {
            fraction_digits - exponent_value <= trailing_zeros
        };
    }

    if exponent_value > trailing_zeros {
        return false;
    }

    fraction_digits <= trailing_zeros - exponent_value
}

#[derive(Copy, Clone, PartialEq, Eq)]
enum JsonValueKind {
    Null,
    Boolean,
    Integer,
    Number,
    String,
    Object,
    Array,
}

fn classify_json_value(value: &str, kind: &mut JsonValueKind) -> bool {
    if value.is_empty() || from_str::<Value>(value).is_err() {
        return false;
    }

    match value.as_bytes()[0] {
        b'n' => *kind = JsonValueKind::Null,
        b't' | b'f' => *kind = JsonValueKind::Boolean,
        b'"' => *kind = JsonValueKind::String,
        b'{' => *kind = JsonValueKind::Object,
        b'[' => *kind = JsonValueKind::Array,
        b if b == b'-' || b.is_ascii_digit() => {
            *kind = if json_number_is_integer(value) {
                JsonValueKind::Integer
            } else {
                JsonValueKind::Number
            };
        }
        _ => return false,
    }

    true
}

fn admits_value(types: u8, kind: JsonValueKind) -> bool {
    match kind {
        JsonValueKind::Null => types & T_NULL != 0,
        JsonValueKind::Boolean => types & T_BOOLEAN != 0,
        JsonValueKind::Integer => types & (T_INTEGER | T_NUMBER) != 0,
        JsonValueKind::Number => types & T_NUMBER != 0,
        JsonValueKind::String => types & T_STRING != 0,
        JsonValueKind::Object => types & T_OBJECT != 0,
        JsonValueKind::Array => types & T_ARRAY != 0,
    }
}

fn encode_json_string(value: &str) -> String {
    Value::String(value.to_owned()).to_string()
}

/// Parameter values are framed by newlines; a string-typed parameter keeps
/// them, other types trim.
fn strip_parameter_framing(text: &str) -> &str {
    let start = if text.starts_with("\r\n") {
        2
    } else if text.starts_with('\n') {
        1
    } else {
        0
    };
    let mut end = text.len();

    if end >= start + 2 && text.ends_with("\r\n") {
        end -= 2;
    } else if end > start && text.ends_with('\n') {
        end -= 1;
    }

    &text[start..end]
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Disposition {
    Emitted,
    Omitted,
    SchemaMismatch,
}

fn normalize_declared_parameter(encoded_value: &str, types: u8) -> (Disposition, String) {
    let framed = strip_parameter_framing(encoded_value);

    if types & T_STRING != 0 {
        return (Disposition::Emitted, encode_json_string(framed));
    }

    let value = trim_ws(framed);

    if value.is_empty() {
        return (Disposition::Omitted, String::new());
    }

    let mut kind = JsonValueKind::String;

    if classify_json_value(value, &mut kind) {
        let disposition = if admits_value(types, kind) {
            Disposition::Emitted
        } else {
            Disposition::SchemaMismatch
        };
        return (disposition, value.to_owned());
    }

    if types & T_BOOLEAN != 0 {
        if value.eq_ignore_ascii_case("true") {
            return (Disposition::Emitted, "true".to_owned());
        }

        if value.eq_ignore_ascii_case("false") {
            return (Disposition::Emitted, "false".to_owned());
        }
    }

    (Disposition::SchemaMismatch, encode_json_string(framed))
}

fn normalize_parameter(
    encoded_value: &str,
    parameter: Option<&Parameter>,
) -> (Disposition, String) {
    if let Some(parameter) = parameter.filter(|p| p.types != 0) {
        return normalize_declared_parameter(encoded_value, parameter.types);
    }

    let value = trim_ws(encoded_value);

    if from_str::<Value>(value).is_ok() {
        (Disposition::Emitted, value.to_owned())
    } else {
        (Disposition::Emitted, encode_json_string(value))
    }
}

// -- Structural parsing ------------------------------------------------------

#[derive(Debug)]
struct RawParameter {
    name: String,
    value: String,
}

#[derive(Debug)]
struct RawToolCall {
    name: String,
    parameters: Vec<RawParameter>,
}

pub(crate) struct GeneratedToolCall {
    pub name: String,
    /// Decoded argument object, in emission order. The wire and session
    /// history use `arguments_json` / re-serialization; the decoded form is
    /// retained for consumers and qualification.
    #[allow(dead_code)]
    pub arguments: Map<String, Value>,
    /// Canonical serialized form for the wire and for fallback assembly.
    pub arguments_json: String,
}

/// Server-minted wire shape for one decoded model tool call.
pub(crate) struct WireToolCall {
    pub id: String,
    pub name: String,
    pub arguments: String,
}

#[derive(Debug, Default, Clone, Copy)]
pub(crate) struct ToolCallParseDiagnostics {
    /// Whether the terminal marker was seen; informational for qualification.
    #[allow(dead_code)]
    pub marker_seen: bool,
    pub structured_call_count: u32,
    pub empty_arguments_omitted: u32,
    pub schema_mismatch_arguments: u32,
    pub fallback_reason: FallbackReason,
}

pub(crate) struct ParsedToolCallOutput {
    pub is_tool_call_response: bool,
    /// Pre-marker content, or the whole input on fallback. The streaming
    /// decoder holds the region from the marker on, so it only reconstructs
    /// held bytes when a marker is present.
    #[allow(dead_code)]
    pub content: String,
    pub tool_calls: Vec<GeneratedToolCall>,
    pub diagnostics: ToolCallParseDiagnostics,
}

fn fallback(text: &str, diagnostics: ToolCallParseDiagnostics) -> ParsedToolCallOutput {
    ParsedToolCallOutput {
        is_tool_call_response: false,
        content: text.to_owned(),
        tool_calls: Vec::new(),
        diagnostics,
    }
}

fn normalize_raw_tool_call(
    raw: &RawToolCall,
    contract: &ToolCallOutputContract,
    diagnostics: &mut ToolCallParseDiagnostics,
) -> GeneratedToolCall {
    let tool = contract.find_tool(&raw.name);
    let mut arguments = Map::new();
    let mut arguments_json = String::from("{");
    let mut first = true;

    for parameter in &raw.parameters {
        let declared = tool
            .map(|tool| tool.parameters.iter().find(|p| p.name == parameter.name))
            .flatten();
        let mut normalized = normalize_parameter(&parameter.value, declared);

        // An undeclared parameter name on a declared tool stays structured,
        // flagged as a schema mismatch for consumer validation.
        if tool.is_some() && declared.is_none() {
            normalized.0 = Disposition::SchemaMismatch;
        }

        if normalized.0 == Disposition::Omitted {
            diagnostics.empty_arguments_omitted += 1;
            continue;
        }

        if normalized.0 == Disposition::SchemaMismatch {
            diagnostics.schema_mismatch_arguments += 1;
        }

        if !first {
            arguments_json.push(',');
        }

        first = false;
        let value = from_str::<Value>(&normalized.1).unwrap_or(Value::Null);

        arguments_json.push_str(&encode_json_string(&parameter.name));
        arguments_json.push(':');
        arguments_json.push_str(&value.to_string());
        arguments.insert(parameter.name.clone(), value);
    }

    arguments_json.push('}');

    GeneratedToolCall {
        name: raw.name.clone(),
        arguments,
        arguments_json,
    }
}

struct RegionParser<'a> {
    text: &'a str,
    contract: &'a ToolCallOutputContract,
    tolerant: bool,
}

impl<'a> RegionParser<'a> {
    fn parse(&self, calls: &mut Vec<RawToolCall>) -> FallbackReason {
        let mut pos = 0;

        loop {
            skip_ws(self.text, &mut pos);

            if pos == self.text.len() {
                return if calls.is_empty() {
                    FallbackReason::MalformedStructure
                } else {
                    FallbackReason::None
                };
            }

            if !starts_at(self.text, pos, TOOL_OPEN) {
                // In tolerant mode a trailing suffix after one or more complete
                // calls is discarded rather than failing the whole output.
                if self.tolerant && !calls.is_empty() {
                    return FallbackReason::TruncatedTail;
                }

                return if calls.is_empty() {
                    FallbackReason::MalformedStructure
                } else {
                    FallbackReason::TrailingContent
                };
            }

            let mut call = RawToolCall {
                name: String::new(),
                parameters: Vec::new(),
            };
            let failure = self.parse_tool_call(&mut pos, &mut call);

            if failure == FallbackReason::None {
                calls.push(call);
                continue;
            }

            // Tolerant mode only: once at least one complete call has been
            // recovered, a trailing malformed suffix or a malformed second call
            // is discarded rather than failing the whole output.
            if self.tolerant && !calls.is_empty() {
                return FallbackReason::TruncatedTail;
            }

            // Tolerant mode only: recover a single truncated final call. Here
            // the call's name and every parsed parameter are complete, but the
            // closing tag was cut off or trailing content follows; keep the
            // recovered call.
            if failure == FallbackReason::TruncatedTail && calls.is_empty() {
                calls.push(call);
                return FallbackReason::TruncatedTail;
            }

            return failure;
        }
    }

    fn consume(&self, pos: &mut usize, token: &str) -> bool {
        if !starts_at(self.text, *pos, token) {
            return false;
        }

        *pos += token.len();
        true
    }

    /// True when nothing but whitespace remains after `pos`.
    fn at_region_end(&self, pos: usize) -> bool {
        let mut at = pos;

        while at < self.text.len() && is_ws(self.text.as_bytes()[at] as char) {
            at += 1;
        }

        at == self.text.len()
    }

    fn parse_tool_call(&self, pos: &mut usize, call: &mut RawToolCall) -> FallbackReason {
        if !self.consume(pos, TOOL_OPEN) {
            return FallbackReason::MalformedStructure;
        }

        skip_ws(self.text, pos);
        let failure = self.parse_function(pos, call);

        if failure != FallbackReason::None {
            return failure;
        }

        skip_ws(self.text, pos);

        if self.consume(pos, TOOL_CLOSE) {
            return FallbackReason::None;
        }

        // In tolerant mode a complete call may be followed by explanatory text,
        // or the model may have stopped at its budget before the closing tag.
        // The strict parser treats a missing close tag as a structural failure.
        if self.tolerant {
            FallbackReason::TruncatedTail
        } else {
            FallbackReason::MalformedStructure
        }
    }

    fn parse_function(&self, pos: &mut usize, call: &mut RawToolCall) -> FallbackReason {
        if !self.consume(pos, FUNCTION_OPEN) {
            return FallbackReason::MalformedStructure;
        }

        let name_begin = *pos;
        let mut name_end = self.text[name_begin..].find('>').map(|r| name_begin + r);
        let mut ws_boundary = false;

        // Tolerant mode: the model sometimes drops the '>' after the function
        // name (for example when a parameter element follows immediately on
        // the next line). Recover by scanning the identifier run and accepting
        // it when whitespace separates it from the next '<' or the end of the
        // region.
        if self.tolerant {
            let mut scan = name_begin;

            while scan < self.text.len() && scan - name_begin < MAX_TOOL_NAME_LENGTH {
                let byte = self.text.as_bytes()[scan];

                if !byte.is_ascii_alphanumeric() && byte != b'_' && byte != b'-' {
                    break;
                }

                scan += 1;
            }

            if scan > name_begin
                && scan < self.text.len()
                && is_ws(self.text.as_bytes()[scan] as char)
                && name_end.map(|end| scan < end).unwrap_or(true)
            {
                let mut after = scan;

                while after < self.text.len() && is_ws(self.text.as_bytes()[after] as char) {
                    after += 1;
                }

                if after >= self.text.len() || self.text[after..].starts_with('<') {
                    name_end = Some(scan);
                    ws_boundary = true;
                }
            }
        }

        let Some(end) = name_end else {
            return FallbackReason::InvalidToolName;
        };

        if end == name_begin {
            return FallbackReason::InvalidToolName;
        }

        let name = &self.text[name_begin..end];
        call.name = name.to_owned();

        if !valid_function_name(name) {
            return FallbackReason::InvalidToolName;
        }

        if self.contract.enforce_declared_names && !self.contract.is_declared(name) {
            return FallbackReason::UndeclaredTool;
        }

        *pos = if ws_boundary { end } else { end + 1 };

        loop {
            skip_ws(self.text, pos);

            if self.consume(pos, FUNCTION_CLOSE) {
                return FallbackReason::None;
            }

            // Tolerant mode only: the region is exhausted after the last
            // complete parameter, so a missing closing tag is a truncation,
            // not a structural failure.
            if self.tolerant && self.at_region_end(*pos) {
                return FallbackReason::TruncatedTail;
            }

            let failure = self.parse_parameter(pos, call);

            if failure != FallbackReason::None {
                return failure;
            }
        }
    }

    fn parse_parameter(&self, pos: &mut usize, call: &mut RawToolCall) -> FallbackReason {
        if !self.consume(pos, PARAM_OPEN) {
            return FallbackReason::MalformedStructure;
        }

        let name_begin = *pos;
        let Some(rel) = self.text[name_begin..].find('>') else {
            return FallbackReason::MalformedStructure;
        };
        let name_end = name_begin + rel;

        if name_end == name_begin {
            return FallbackReason::MalformedStructure;
        }

        let name = &self.text[name_begin..name_end];

        if call.parameters.iter().any(|p| p.name == name) {
            return FallbackReason::DuplicateParameter;
        }

        let value_begin = name_end + 1;
        let Some(value_end) = self.find_parameter_close(value_begin) else {
            return FallbackReason::MalformedStructure;
        };

        call.parameters.push(RawParameter {
            name: name.to_owned(),
            value: self.text[value_begin..value_end].to_owned(),
        });
        *pos = value_end + PARAM_CLOSE.len();
        FallbackReason::None
    }

    fn find_parameter_open_before(&self, scan: usize, limit: usize) -> Option<usize> {
        let mut candidate = self.text[scan..].find(PARAM_OPEN).map(|r| scan + r);

        while let Some(at) = candidate {
            if at >= limit {
                break;
            }

            let name_begin = at + PARAM_OPEN.len();
            let Some(name_end) = self.text[name_begin..].find('>').map(|r| name_begin + r) else {
                candidate = self.text[at + 1..].find(PARAM_OPEN).map(|r| at + 1 + r);
                continue;
            };

            if name_end < limit && name_end != name_begin {
                return Some(name_end + 1);
            }

            candidate = self.text[at + 1..].find(PARAM_OPEN).map(|r| at + 1 + r);
        }

        None
    }

    /// Depth-aware search for the closing tag belonging to a value; nested
    /// parameter opens inside the value count as deeper levels.
    fn find_parameter_close(&self, value_begin: usize) -> Option<usize> {
        let mut depth = 1;
        let mut scan = value_begin;

        loop {
            let Some(rel) = self.text[scan..].find(PARAM_CLOSE) else {
                return None;
            };
            let close = scan + rel;
            let open_end = self.find_parameter_open_before(scan, close);

            if let Some(open_end) = open_end {
                depth += 1;
                scan = open_end;
                continue;
            }

            depth -= 1;

            if depth == 0 {
                return Some(close);
            }

            scan = close + PARAM_CLOSE.len();
        }
    }
}

fn skip_ws(text: &str, pos: &mut usize) {
    while *pos < text.len() && is_ws(text.as_bytes()[*pos] as char) {
        *pos += 1;
    }
}

fn starts_at(text: &str, pos: usize, prefix: &str) -> bool {
    text.get(pos..pos + prefix.len())
        .is_some_and(|slice| slice == prefix)
}

/// Parse the Qwen XML-like tool-call format. Content before the first marker
/// is retained as ordinary text; a malformed marker region falls back to the
/// whole input, marker included. Tolerant mode recovers complete calls from a
/// truncated or suffixed region.
pub(crate) fn parse_qwen_tool_call_output(
    text: &str,
    contract: &ToolCallOutputContract,
    tolerant: bool,
) -> ParsedToolCallOutput {
    let Some(first) = text.find(TOOL_OPEN) else {
        return ParsedToolCallOutput {
            is_tool_call_response: false,
            content: text.to_owned(),
            tool_calls: Vec::new(),
            diagnostics: ToolCallParseDiagnostics::default(),
        };
    };

    let mut out = ParsedToolCallOutput {
        is_tool_call_response: false,
        content: trim_ws(&text[..first]).to_owned(),
        tool_calls: Vec::new(),
        diagnostics: ToolCallParseDiagnostics {
            marker_seen: true,
            ..Default::default()
        },
    };

    let parser = RegionParser {
        text: &text[first..],
        contract,
        tolerant,
    };
    let mut raw_calls = Vec::new();
    let failure = parser.parse(&mut raw_calls);

    if failure == FallbackReason::TruncatedTail {
        // A truncated tail after a complete call was discarded; the recovered
        // calls stand.
        out.diagnostics.fallback_reason = failure;
    } else if failure != FallbackReason::None {
        out.diagnostics.fallback_reason = failure;
        return fallback(text, out.diagnostics);
    }

    for raw in &raw_calls {
        out.tool_calls
            .push(normalize_raw_tool_call(raw, contract, &mut out.diagnostics));
    }

    out.diagnostics.structured_call_count = out.tool_calls.len() as u32;
    out.is_tool_call_response = true;
    out
}

// -- Streaming decoder -------------------------------------------------------

pub(crate) struct Terminal {
    /// Held bytes that were never published by `feed`.
    pub content: String,
    pub tool_calls: Vec<GeneratedToolCall>,
    pub diagnostics: ToolCallParseDiagnostics,
}

/// Incrementally publishes bytes that are provably outside a possible
/// terminal Qwen tool-call suffix. Bytes that could belong to a marker are
/// held back; at terminal time, valid calls are retained structurally and
/// malformed output is restored verbatim.
pub(crate) struct ToolCallOutputDecoder {
    contract: Arc<ToolCallOutputContract>,
    tolerant: bool,
    trailing_whitespace: String,
    tool_region: String,
    marker_prefix: usize,
    saw_tool_marker: bool,
}

impl ToolCallOutputDecoder {
    pub(crate) fn new(contract: Arc<ToolCallOutputContract>, tolerant: bool) -> Self {
        Self {
            contract,
            tolerant,
            trailing_whitespace: String::new(),
            tool_region: String::new(),
            marker_prefix: 0,
            saw_tool_marker: false,
        }
    }

    /// Feed decoded text and return the visible (non-tool-call) prefix.
    pub(crate) fn feed(&mut self, text: &str) -> String {
        if text.is_empty() {
            return String::new();
        }

        if self.saw_tool_marker {
            self.tool_region.push_str(text);
            return String::new();
        }

        let marker = TOOL_OPEN.as_bytes();
        let mut visible = String::new();
        let mut cursor = 0;

        for c in text.chars() {
            if self.marker_prefix > 0 {
                if c == marker[self.marker_prefix] as char {
                    self.marker_prefix += 1;

                    if self.marker_prefix == marker.len() {
                        // The terminal marker completed: hold the buffered
                        // whitespace, the marker, and everything after it.
                        self.tool_region = std::mem::take(&mut self.trailing_whitespace);
                        self.tool_region.push_str(TOOL_OPEN);
                        self.tool_region.push_str(&text[cursor + c.len_utf8()..]);
                        self.marker_prefix = 0;
                        self.saw_tool_marker = true;
                        return visible;
                    }

                    cursor += c.len_utf8();
                    continue;
                }

                // The prefix was ordinary text: publish it and re-handle `c`.
                visible.push_str(&self.trailing_whitespace);
                self.trailing_whitespace.clear();
                visible.push_str(&TOOL_OPEN[..self.marker_prefix]);
                self.marker_prefix = 0;
            }

            if c == '<' {
                self.marker_prefix = 1;
            } else if is_ws(c) {
                self.trailing_whitespace.push(c);
            } else {
                visible.push_str(&self.trailing_whitespace);
                self.trailing_whitespace.clear();
                visible.push(c);
            }

            cursor += c.len_utf8();
        }

        visible
    }

    /// Close the stream and interpret the held terminal region.
    pub(crate) fn finish(self) -> Terminal {
        let parsed = parse_qwen_tool_call_output(&self.tool_region, &self.contract, self.tolerant);

        if self.saw_tool_marker && parsed.is_tool_call_response {
            return Terminal {
                content: String::new(),
                tool_calls: parsed.tool_calls,
                diagnostics: parsed.diagnostics,
            };
        }

        let mut content = self.trailing_whitespace;

        content.push_str(&TOOL_OPEN[..self.marker_prefix]);
        content.push_str(&self.tool_region);

        Terminal {
            content,
            tool_calls: Vec::new(),
            diagnostics: parsed.diagnostics,
        }
    }
}

#[cfg(test)]
#[path = "../../tests/unit/server/tool_call.rs"]
mod tests;
