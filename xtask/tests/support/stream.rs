use super::*;
use std::io::{BufRead, BufReader};

pub struct EventStream {
    reader: BufReader<TcpStream>,
}

impl EventStream {
    pub fn open(address: &str, body: &Value) -> Result<Self> {
        let mut stream = TcpStream::connect(address)?;

        stream.set_read_timeout(Some(Duration::from_secs(180)))?;

        let bytes = serde_json::to_vec(body)?;

        write!(
            stream,
            "POST /v1/chat/completions HTTP/1.1\r\nHost: {address}\r\nContent-Length: {}\r\n\r\n",
            bytes.len()
        )?;
        stream.write_all(&bytes)?;

        let mut reader = BufReader::new(stream);
        let mut line = String::new();

        reader.read_line(&mut line)?;
        ensure!(line.starts_with("HTTP/1.1 200"), "{line}");

        loop {
            line.clear();
            ensure!(reader.read_line(&mut line)? > 0, "stream ended in headers");

            if line == "\r\n" {
                return Ok(Self { reader });
            }
        }
    }

    pub fn next(&mut self) -> Result<Option<Value>> {
        let mut line = String::new();

        loop {
            line.clear();
            ensure!(
                self.reader.read_line(&mut line)? > 0,
                "stream ended before DONE"
            );

            let Some(data) = line.trim().strip_prefix("data: ") else {
                continue;
            };

            if data == "[DONE]" {
                return Ok(None);
            }

            return Ok(Some(serde_json::from_str(data)?));
        }
    }

    pub fn disconnect(self) -> Result<()> {
        self.reader.get_ref().shutdown(std::net::Shutdown::Both)?;

        Ok(())
    }
}
