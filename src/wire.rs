use bytes::{Buf, BufMut, Bytes, BytesMut};

pub const MAX_FRAME_PAYLOAD: usize = 16 * 1024 * 1024;
pub const BODY_CHUNK_SIZE: usize = 64 * 1024;
const FRAME_HEADER_SIZE: usize = 5;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FrameKind {
    RequestHead,
    RequestBody,
    RequestEnd,
    ResponseHead,
    ResponseBody,
    ResponseEnd,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Frame {
    pub kind: FrameKind,
    pub payload: Bytes,
}

impl FrameKind {
    fn code(self) -> u8 {
        match self {
            FrameKind::RequestHead => 1,
            FrameKind::RequestBody => 2,
            FrameKind::RequestEnd => 3,
            FrameKind::ResponseHead => 4,
            FrameKind::ResponseBody => 5,
            FrameKind::ResponseEnd => 6,
        }
    }

    fn from_code(code: u8) -> Option<FrameKind> {
        Some(match code {
            1 => FrameKind::RequestHead,
            2 => FrameKind::RequestBody,
            3 => FrameKind::RequestEnd,
            4 => FrameKind::ResponseHead,
            5 => FrameKind::ResponseBody,
            6 => FrameKind::ResponseEnd,
            _ => return None,
        })
    }
}

pub fn encode_frame(kind: FrameKind, payload: &[u8], output: &mut BytesMut) {
    assert!(
        payload.len() <= MAX_FRAME_PAYLOAD,
        "oversized frame payload"
    );
    output.reserve(FRAME_HEADER_SIZE + payload.len());
    output.put_u8(kind.code());
    output.put_u32(payload.len() as u32);
    output.put_slice(payload);
}

pub fn decode_frame(buffer: &mut BytesMut) -> Result<Option<Frame>, String> {
    if buffer.len() < FRAME_HEADER_SIZE {
        return Ok(None);
    }
    let kind = FrameKind::from_code(buffer[0])
        .ok_or_else(|| format!("invalid frame kind {}", buffer[0]))?;
    let length = u32::from_be_bytes(
        buffer[1..FRAME_HEADER_SIZE]
            .try_into()
            .expect("short frame header"),
    ) as usize;
    if length > MAX_FRAME_PAYLOAD {
        return Err(format!("frame payload of {length} bytes exceeds the limit"));
    }
    if buffer.len() < FRAME_HEADER_SIZE + length {
        return Ok(None);
    }
    buffer.advance(FRAME_HEADER_SIZE);
    Ok(Some(Frame {
        kind,
        payload: buffer.split_to(length).freeze(),
    }))
}

pub fn request_end_offset(buffer: &[u8]) -> Result<Option<usize>, String> {
    let mut offset = 0;
    while buffer.len() >= offset + FRAME_HEADER_SIZE {
        let kind = FrameKind::from_code(buffer[offset])
            .ok_or_else(|| format!("invalid frame kind {}", buffer[offset]))?;
        let length = u32::from_be_bytes(
            buffer[offset + 1..offset + FRAME_HEADER_SIZE]
                .try_into()
                .expect("short frame header"),
        ) as usize;
        if length > MAX_FRAME_PAYLOAD {
            return Err(format!("frame payload of {length} bytes exceeds the limit"));
        }
        if buffer.len() < offset + FRAME_HEADER_SIZE + length {
            return Ok(None);
        }
        offset += FRAME_HEADER_SIZE + length;
        if kind == FrameKind::RequestEnd {
            return Ok(Some(offset));
        }
    }
    Ok(None)
}

pub fn encode_body_frames(kind: FrameKind, body: &[u8], output: &mut BytesMut) {
    for chunk in body.chunks(BODY_CHUNK_SIZE) {
        encode_frame(kind, chunk, output);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn frame_round_trip_with_partial_input() {
        let mut encoded = BytesMut::new();
        encode_frame(
            FrameKind::RequestHead,
            b"GET / HTTP/1.1\r\n\r\n",
            &mut encoded,
        );
        encode_frame(FrameKind::RequestEnd, b"", &mut encoded);
        let mut buffer = BytesMut::new();
        let mut frames = Vec::new();
        for byte in encoded {
            buffer.put_u8(byte);
            while let Some(frame) = decode_frame(&mut buffer).unwrap() {
                frames.push(frame);
            }
        }
        assert_eq!(
            frames,
            vec![
                Frame {
                    kind: FrameKind::RequestHead,
                    payload: Bytes::from_static(b"GET / HTTP/1.1\r\n\r\n")
                },
                Frame {
                    kind: FrameKind::RequestEnd,
                    payload: Bytes::new()
                }
            ]
        );
        assert!(buffer.is_empty());
    }

    #[test]
    fn invalid_frame_rejection() {
        let mut unknown_kind = BytesMut::from(&[9u8, 0, 0, 0, 0][..]);
        assert!(decode_frame(&mut unknown_kind).is_err());
        let mut oversized = BytesMut::from(&[2u8, 0xff, 0xff, 0xff, 0xff][..]);
        assert!(decode_frame(&mut oversized).is_err());
    }
}
