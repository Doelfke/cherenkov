//! Bounded HTTP input and JSON/SSE wire framing.

use crate::units::BYTES_PER_KIB;
use anyhow::{Context, Result, ensure};
use serde_json::{Value, json};
use std::io::{BufRead, BufReader, Read, Write};
use std::net::TcpStream;

fn line(reader: &mut impl BufRead, remaining: &mut usize) -> Result<String> {
    let mut bytes = Vec::new();
    let n = reader
        .take((*remaining + 1) as u64)
        .read_until(b'\n', &mut bytes)?;

    ensure!(
        n > 0 && n <= *remaining && bytes.ends_with(b"\n"),
        "invalid or oversized HTTP headers"
    );

    *remaining -= n;

    Ok(String::from_utf8(bytes)?
        .trim_end_matches(['\r', '\n'])
        .to_owned())
}

pub(super) fn read_http(
    stream: &mut TcpStream,
    max_body: usize,
) -> Result<(String, String, Vec<u8>)> {
    let mut reader = BufReader::new(stream.try_clone()?);
    let mut remaining = 16 * BYTES_PER_KIB;
    let first = line(&mut reader, &mut remaining)?;
    let parts: Vec<_> = first.split_whitespace().collect();

    ensure!(
        parts.len() == 3 && parts[2].starts_with("HTTP/1."),
        "invalid request line"
    );

    let mut length = None;
    let mut expect = false;

    loop {
        let h = line(&mut reader, &mut remaining)?;

        if h.is_empty() {
            break;
        }

        let (key, value) = h.split_once(':').context("invalid header")?;

        match key.to_ascii_lowercase().as_str() {
            "content-length" => {
                ensure!(length.is_none(), "duplicate Content-Length");

                length = Some(
                    value
                        .trim()
                        .parse::<usize>()
                        .context("invalid Content-Length")?,
                );
            }
            "transfer-encoding" => {
                anyhow::bail!("Transfer-Encoding is unsupported; send Content-Length")
            }
            "expect" => {
                ensure!(
                    value.trim().eq_ignore_ascii_case("100-continue"),
                    "unsupported Expect"
                );

                expect = true;
            }
            _ => {}
        }
    }

    let length = length.unwrap_or(0);

    ensure!(
        length <= max_body,
        "request body exceeds server limit of {max_body} bytes"
    );

    if expect {
        stream.write_all(b"HTTP/1.1 100 Continue\r\n\r\n")?;
        stream.flush()?;
    }

    let mut body = vec![0; length];

    reader.read_exact(&mut body)?;

    Ok((parts[0].to_owned(), parts[1].to_owned(), body))
}

pub(super) fn respond(out: &mut impl Write, status: u16, body: &Value) -> Result<()> {
    let bytes = serde_json::to_vec(body)?;
    let reason = match status {
        200 => "OK",
        400 => "Bad Request",
        404 => "Not Found",
        409 => "Conflict",
        499 => "Client Closed Request",
        503 => "Service Unavailable",
        _ => "Internal Server Error",
    };

    write!(
        out,
        "HTTP/1.1 {status} {reason}\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
        bytes.len()
    )?;
    out.write_all(&bytes)?;
    out.flush()?;

    Ok(())
}

pub(super) fn error(out: &mut TcpStream, status: u16, message: &str) -> Result<()> {
    respond(
        out,
        status,
        &json!({"error":{"message":message,"type":if status >= 500 {"server_error"} else {"invalid_request_error"},"param":null,"code":null}}),
    )
}

pub(super) fn sse(out: &mut impl Write, body: &Value) -> Result<()> {
    writeln!(out, "data: {}\n", serde_json::to_string(body)?)?;
    out.flush()?;

    Ok(())
}

#[cfg(test)]
#[path = "../../tests/unit/server/http.rs"]
mod tests;
