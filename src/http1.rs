use bytes::{Buf, Bytes, BytesMut};
use http::header::{HeaderMap, HeaderName, HeaderValue, CONTENT_LENGTH, TRANSFER_ENCODING};
use http::{Method, StatusCode};
use serde::{Deserialize, Serialize};

const MAX_HEADERS: usize = 256;
const MAX_CHUNK_LINE: usize = 4096;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RequestHead {
    pub method: Method,
    pub target: String,
    pub headers: HeaderMap,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ResponseHead {
    pub status: StatusCode,
    pub headers: HeaderMap,
}

impl RequestHead {
    pub fn path(&self) -> &str {
        self.target.split('?').next().unwrap_or_default()
    }

    pub fn encode(&self) -> Vec<u8> {
        let mut bytes = format!("{} {} HTTP/1.1\r\n", self.method, self.target).into_bytes();
        append_headers(&self.headers, &mut bytes);
        bytes
    }

    pub fn parse(bytes: &[u8]) -> Result<Option<(RequestHead, usize)>, String> {
        let mut header_slots = [httparse::EMPTY_HEADER; MAX_HEADERS];
        let mut request = httparse::Request::new(&mut header_slots);
        let length = match request.parse(bytes) {
            Ok(httparse::Status::Complete(length)) => length,
            Ok(httparse::Status::Partial) => return Ok(None),
            Err(error) => return Err(format!("invalid HTTP request head: {error}")),
        };
        let method = Method::from_bytes(request.method.unwrap_or_default().as_bytes())
            .map_err(|error| format!("invalid HTTP method: {error}"))?;
        Ok(Some((
            RequestHead {
                method,
                target: request.path.unwrap_or("/").to_string(),
                headers: header_map(request.headers)?,
            },
            length,
        )))
    }
}

impl ResponseHead {
    pub fn new(status: StatusCode) -> ResponseHead {
        ResponseHead {
            status,
            headers: HeaderMap::new(),
        }
    }

    pub fn encode(&self) -> Vec<u8> {
        let reason = self.status.canonical_reason().unwrap_or("");
        let mut bytes = format!("HTTP/1.1 {} {reason}\r\n", self.status.as_u16()).into_bytes();
        append_headers(&self.headers, &mut bytes);
        bytes
    }

    pub fn parse(bytes: &[u8]) -> Result<Option<(ResponseHead, usize)>, String> {
        let mut header_slots = [httparse::EMPTY_HEADER; MAX_HEADERS];
        let mut response = httparse::Response::new(&mut header_slots);
        let length = match response.parse(bytes) {
            Ok(httparse::Status::Complete(length)) => length,
            Ok(httparse::Status::Partial) => return Ok(None),
            Err(error) => return Err(format!("invalid HTTP response head: {error}")),
        };
        let status = StatusCode::from_u16(response.code.unwrap_or_default())
            .map_err(|error| format!("invalid HTTP status: {error}"))?;
        Ok(Some((
            ResponseHead {
                status,
                headers: header_map(response.headers)?,
            },
            length,
        )))
    }
}

fn header_map(headers: &[httparse::Header<'_>]) -> Result<HeaderMap, String> {
    let mut map = HeaderMap::with_capacity(headers.len());
    for header in headers {
        let name = HeaderName::from_bytes(header.name.as_bytes())
            .map_err(|error| format!("invalid header name '{}': {error}", header.name))?;
        let value = HeaderValue::from_bytes(header.value)
            .map_err(|error| format!("invalid value of header '{name}': {error}"))?;
        map.append(name, value);
    }
    Ok(map)
}

fn append_headers(headers: &HeaderMap, bytes: &mut Vec<u8>) {
    for (name, value) in headers {
        bytes.extend_from_slice(name.as_str().as_bytes());
        bytes.extend_from_slice(b": ");
        bytes.extend_from_slice(value.as_bytes());
        bytes.extend_from_slice(b"\r\n");
    }
    bytes.extend_from_slice(b"\r\n");
}

fn is_transport_header(name: &HeaderName) -> bool {
    matches!(
        name.as_str(),
        "connection"
            | "keep-alive"
            | "proxy-authenticate"
            | "proxy-authorization"
            | "proxy-connection"
            | "te"
            | "trailer"
            | "transfer-encoding"
            | "upgrade"
            | "content-length"
            | "host"
            | "expect"
    )
}

pub fn without_transport_headers(headers: &HeaderMap) -> HeaderMap {
    let mut result = HeaderMap::with_capacity(headers.len());
    for (name, value) in headers {
        if !is_transport_header(name) {
            result.append(name.clone(), value.clone());
        }
    }
    result
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum BodyDecoder {
    Length { remaining: u64 },
    Chunked(ChunkedState),
    UntilClose,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum ChunkedState {
    Size,
    Data { remaining: u64 },
    DataEnd,
    Trailers,
    Done,
}

#[derive(Debug, PartialEq, Eq)]
pub enum Decoded {
    Data(Bytes),
    End,
    Incomplete,
}

impl BodyDecoder {
    pub fn for_response(head: &ResponseHead, is_head_request: bool) -> Result<BodyDecoder, String> {
        let has_no_body = is_head_request
            || head.status.is_informational()
            || head.status == StatusCode::NO_CONTENT
            || head.status == StatusCode::NOT_MODIFIED;
        if has_no_body {
            return Ok(BodyDecoder::Length { remaining: 0 });
        }
        let final_transfer_coding = head
            .headers
            .get_all(TRANSFER_ENCODING)
            .iter()
            .flat_map(|value| value.as_bytes().split(|byte| *byte == b','))
            .map(<[u8]>::trim_ascii)
            .rfind(|coding| !coding.is_empty());
        match final_transfer_coding {
            Some(coding) if coding.eq_ignore_ascii_case(b"chunked") => {
                return Ok(BodyDecoder::Chunked(ChunkedState::Size))
            }
            Some(_) => return Ok(BodyDecoder::UntilClose),
            None => {}
        }
        match head.headers.get(CONTENT_LENGTH) {
            Some(value) => value
                .to_str()
                .ok()
                .and_then(|text| text.trim().parse().ok())
                .map(|remaining| BodyDecoder::Length { remaining })
                .ok_or_else(|| "invalid content-length in the engine response".to_string()),
            None => Ok(BodyDecoder::UntilClose),
        }
    }

    pub fn content_length(&self) -> Option<u64> {
        match self {
            BodyDecoder::Length { remaining } => Some(*remaining),
            _ => None,
        }
    }

    pub fn decode(&mut self, buffer: &mut BytesMut) -> Result<Decoded, String> {
        match self {
            BodyDecoder::Length { remaining: 0 } => Ok(Decoded::End),
            BodyDecoder::Length { remaining } => {
                if buffer.is_empty() {
                    return Ok(Decoded::Incomplete);
                }
                let length = buffer.len().min(*remaining as usize);
                *remaining -= length as u64;
                Ok(Decoded::Data(buffer.split_to(length).freeze()))
            }
            BodyDecoder::UntilClose => {
                if buffer.is_empty() {
                    return Ok(Decoded::Incomplete);
                }
                Ok(Decoded::Data(buffer.split().freeze()))
            }
            BodyDecoder::Chunked(state) => decode_chunked(state, buffer),
        }
    }

    pub fn end_of_input(&self) -> Result<(), String> {
        match self {
            BodyDecoder::UntilClose
            | BodyDecoder::Length { remaining: 0 }
            | BodyDecoder::Chunked(ChunkedState::Done) => Ok(()),
            _ => Err("the engine closed the connection before the end of the response".to_string()),
        }
    }
}

fn decode_chunked(state: &mut ChunkedState, buffer: &mut BytesMut) -> Result<Decoded, String> {
    loop {
        match *state {
            ChunkedState::Size => {
                let Some(line) = take_line(buffer)? else {
                    return Ok(Decoded::Incomplete);
                };
                let size_text = line.split(|byte| *byte == b';').next().unwrap_or_default();
                let size_text = std::str::from_utf8(size_text)
                    .map_err(|_| "invalid chunk size in the engine response".to_string())?;
                let size = u64::from_str_radix(size_text.trim(), 16)
                    .map_err(|_| "invalid chunk size in the engine response".to_string())?;
                *state = match size {
                    0 => ChunkedState::Trailers,
                    remaining => ChunkedState::Data { remaining },
                };
            }
            ChunkedState::Data { remaining } => {
                if buffer.is_empty() {
                    return Ok(Decoded::Incomplete);
                }
                let length = buffer.len().min(remaining as usize);
                let left = remaining - length as u64;
                *state = match left {
                    0 => ChunkedState::DataEnd,
                    remaining => ChunkedState::Data { remaining },
                };
                return Ok(Decoded::Data(buffer.split_to(length).freeze()));
            }
            ChunkedState::DataEnd => {
                if buffer.len() < 2 {
                    return Ok(Decoded::Incomplete);
                }
                if &buffer[..2] != b"\r\n" {
                    return Err("missing CRLF after a chunk in the engine response".to_string());
                }
                buffer.advance(2);
                *state = ChunkedState::Size;
            }
            ChunkedState::Trailers => {
                let Some(line) = take_line(buffer)? else {
                    return Ok(Decoded::Incomplete);
                };
                if line.is_empty() {
                    *state = ChunkedState::Done;
                }
            }
            ChunkedState::Done => return Ok(Decoded::End),
        }
    }
}

fn take_line(buffer: &mut BytesMut) -> Result<Option<Bytes>, String> {
    match buffer.windows(2).position(|window| window == b"\r\n") {
        Some(position) => {
            let line = buffer.split_to(position).freeze();
            buffer.advance(2);
            Ok(Some(line))
        }
        None if buffer.len() > MAX_CHUNK_LINE => {
            Err("chunk line too long in the engine response".to_string())
        }
        None => Ok(None),
    }
}

pub fn error_json(message: &str) -> String {
    serde_json::json!({ "error": { "message": message, "type": "proxy_error" } }).to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn decode_byte_by_byte(mut decoder: BodyDecoder, input: &[u8]) -> (Vec<u8>, bool, BodyDecoder) {
        let mut buffer = BytesMut::new();
        let mut body = Vec::new();
        let mut is_complete = false;
        for byte in input {
            buffer.extend_from_slice(&[*byte]);
            loop {
                match decoder.decode(&mut buffer).unwrap() {
                    Decoded::Data(data) => body.extend_from_slice(&data),
                    Decoded::End => {
                        is_complete = true;
                        break;
                    }
                    Decoded::Incomplete => break,
                }
            }
        }
        (body, is_complete, decoder)
    }

    fn response_head(headers: &[(&'static str, &str)]) -> ResponseHead {
        let mut head = ResponseHead::new(StatusCode::OK);
        for (name, value) in headers {
            head.headers
                .append(*name, http::HeaderValue::from_str(value).unwrap());
        }
        head
    }

    #[test]
    fn chunked_body_split_at_every_byte() {
        let decoder =
            BodyDecoder::for_response(&response_head(&[("transfer-encoding", "chunked")]), false)
                .unwrap();
        let input = b"5;name=value\r\nhello\r\n7\r\n, world\r\n0\r\nx-trailer: 1\r\n\r\n";
        let (body, is_complete, decoder) = decode_byte_by_byte(decoder, input);
        assert_eq!(body, b"hello, world");
        assert!(is_complete);
        assert!(decoder.end_of_input().is_ok());
    }

    #[test]
    fn content_length_body() {
        let decoder =
            BodyDecoder::for_response(&response_head(&[("content-length", "5")]), false).unwrap();
        assert_eq!(decoder.content_length(), Some(5));
        let (body, is_complete, _) = decode_byte_by_byte(decoder, b"hello");
        assert_eq!(body, b"hello");
        assert!(is_complete);
    }

    #[test]
    fn truncated_bodies() {
        let chunked = BodyDecoder::Chunked(ChunkedState::Size);
        let (_, is_complete, chunked) = decode_byte_by_byte(chunked, b"5\r\nhel");
        assert!(!is_complete);
        assert!(chunked.end_of_input().is_err());
        let length = BodyDecoder::Length { remaining: 5 };
        let (_, _, length) = decode_byte_by_byte(length, b"hel");
        assert!(length.end_of_input().is_err());
        assert!(BodyDecoder::UntilClose.end_of_input().is_ok());
    }

    #[test]
    fn bodiless_responses() {
        let head = response_head(&[("content-length", "100")]);
        assert_eq!(
            BodyDecoder::for_response(&head, true),
            Ok(BodyDecoder::Length { remaining: 0 })
        );
        let no_content = ResponseHead::new(StatusCode::NO_CONTENT);
        assert_eq!(
            BodyDecoder::for_response(&no_content, false),
            Ok(BodyDecoder::Length { remaining: 0 })
        );
    }

    #[test]
    fn transfer_coding_lists() {
        let framing = |headers: &[(&'static str, &str)]| {
            BodyDecoder::for_response(&response_head(headers), false).unwrap()
        };
        assert_eq!(
            framing(&[
                ("transfer-encoding", "gzip, chunked"),
                ("content-length", "3")
            ]),
            BodyDecoder::Chunked(ChunkedState::Size)
        );
        assert_eq!(
            framing(&[
                ("transfer-encoding", "gzip"),
                ("transfer-encoding", "Chunked")
            ]),
            BodyDecoder::Chunked(ChunkedState::Size)
        );
        assert_eq!(
            framing(&[
                ("transfer-encoding", "chunked, gzip"),
                ("content-length", "3")
            ]),
            BodyDecoder::UntilClose
        );
    }

    #[test]
    fn invalid_chunk_size() {
        let mut decoder = BodyDecoder::Chunked(ChunkedState::Size);
        let mut buffer = BytesMut::from(&b"zz\r\n"[..]);
        assert!(decoder.decode(&mut buffer).is_err());
    }
}
