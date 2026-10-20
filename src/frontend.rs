use std::convert::Infallible;
use std::future::Future;
use std::io;
use std::net::SocketAddr;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::Duration;

use bytes::{Bytes, BytesMut};
use http_body_util::channel::{Channel, Sender};
use http_body_util::combinators::BoxBody;
use http_body_util::{BodyExt, Full};
use hyper::body::Incoming;
use hyper::server::conn::http1;
use hyper::service::service_fn;
use hyper::{Request, Response, StatusCode};
use hyper_util::rt::{TokioIo, TokioTimer};
use hyper_util::server::graceful::{GracefulShutdown, Watcher};
use log::{info, warn};
use tokio::io::{AsyncRead, AsyncReadExt, AsyncWrite, AsyncWriteExt};
use tokio::net::unix::{OwnedReadHalf, OwnedWriteHalf};
use tokio::net::{TcpListener, UnixStream};
use tokio::sync::oneshot;
use tokio::task::JoinHandle;

use crate::config::GeneralConfig;
use crate::http1::{error_json, without_transport_headers, RequestHead, ResponseHead};
use crate::utils::AppendFile;
use crate::wire::{decode_frame, encode_body_frames, encode_frame, Frame, FrameKind};

type BoxError = Box<dyn std::error::Error + Send + Sync>;
type ResponseBody = BoxBody<Bytes, BoxError>;

const ACCEPT_ERROR_BACKOFF: Duration = Duration::from_millis(50);
const RESPONSE_BODY_BUFFER_FRAMES: usize = 16;
const MIDDLE_END_BACKLOG_TIMEOUT: Duration = Duration::from_secs(10);
const MIDDLE_END_BACKLOG_RETRY: Duration = Duration::from_millis(10);

#[derive(Clone)]
pub enum Transport {
    Plain,
    #[cfg(feature = "tls")]
    Tls(tokio_rustls::TlsAcceptor),
}

#[derive(Clone)]
pub struct RequestDump {
    file: AppendFile,
    request_count: Arc<AtomicU64>,
}

enum DumpedBody<'a> {
    Complete(&'a [u8]),
    Incomplete(&'a [u8]),
    NotRead,
}

struct RelayContext {
    middle_end_socket: PathBuf,
    request_dump: Option<RequestDump>,
}

pub async fn run(
    config: &GeneralConfig,
    request_dump_path: Option<&Path>,
    shutdown: impl Future<Output = ()>,
) -> Result<(), String> {
    let (transport, scheme) = match config.frontend.tls.clone() {
        None => (Transport::Plain, "http"),
        #[cfg(feature = "tls")]
        Some(tls_config) => (Transport::Tls(tls::acceptor(&tls_config)?), "https"),
    };
    let (request_dump, dump_writer) = match request_dump_path {
        Some(path) => {
            let (request_dump, dump_writer) = RequestDump::open(path)?;
            warn!(
                "frontend appends every request (unredacted) to '{}'",
                path.display()
            );
            (Some(request_dump), Some(dump_writer))
        }
        None => (None, None),
    };
    let listener = TcpListener::bind(config.frontend.listen)
        .await
        .map_err(|error| format!("failed to listen on {}: {error}", config.frontend.listen))?;
    let address = listener
        .local_addr()
        .map_err(|error| format!("failed to read the frontend address: {error}"))?;
    info!(
        "frontend listening on {scheme}://{address}, middle end socket {}",
        config.middle_end_socket.display()
    );
    let drain_timeout = config.process.shutdown_drain_timeout_seconds.duration();
    serve(
        listener,
        transport,
        config.middle_end_socket.clone(),
        request_dump,
        drain_timeout,
        shutdown,
    )
    .await;
    if let Some(dump_writer) = dump_writer {
        if tokio::time::timeout(drain_timeout, dump_writer)
            .await
            .is_err()
        {
            warn!("frontend shutdown: the request dump was not complete after {drain_timeout:?}");
        }
    }
    Ok(())
}

pub async fn serve(
    listener: TcpListener,
    transport: Transport,
    middle_end_socket: PathBuf,
    request_dump: Option<RequestDump>,
    drain_timeout: Duration,
    shutdown: impl Future<Output = ()>,
) {
    let context = Arc::new(RelayContext {
        middle_end_socket,
        request_dump,
    });
    let graceful = GracefulShutdown::new();
    tokio::pin!(shutdown);
    loop {
        let (tcp_stream, peer) = tokio::select! {
            _ = &mut shutdown => break,
            accepted = listener.accept() => match accepted {
                Ok(accepted) => accepted,
                Err(error) => {
                    warn!("frontend accept failed ({error}); retrying in {ACCEPT_ERROR_BACKOFF:?}");
                    tokio::time::sleep(ACCEPT_ERROR_BACKOFF).await;
                    continue;
                }
            },
        };
        let watcher = graceful.watcher();
        let context = context.clone();
        let transport = transport.clone();
        tokio::spawn(async move {
            match transport {
                Transport::Plain => serve_connection(tcp_stream, context, peer, watcher).await,
                #[cfg(feature = "tls")]
                Transport::Tls(acceptor) => {
                    match tokio::time::timeout(tls::HANDSHAKE_TIMEOUT, acceptor.accept(tcp_stream))
                        .await
                    {
                        Ok(Ok(tls_stream)) => {
                            serve_connection(tls_stream, context, peer, watcher).await
                        }
                        Ok(Err(error)) => {
                            warn!("TLS handshake failed ({error}); dropping connection")
                        }
                        Err(_) => warn!("TLS handshake timed out; dropping connection"),
                    }
                }
            }
        });
    }
    drop(listener);
    info!(
        "frontend shutdown: draining {} connection(s)",
        graceful.count()
    );
    if tokio::time::timeout(drain_timeout, graceful.shutdown())
        .await
        .is_err()
    {
        warn!("frontend shutdown: drain timed out after {drain_timeout:?}");
    }
}

async fn serve_connection<Stream>(
    stream: Stream,
    context: Arc<RelayContext>,
    peer: SocketAddr,
    watcher: Watcher,
) where
    Stream: AsyncRead + AsyncWrite + Unpin + Send + 'static,
{
    let service = service_fn(move |request| relay(request, context.clone(), peer));
    let mut builder = http1::Builder::new();
    builder.timer(TokioTimer::new());
    let connection = builder.serve_connection(TokioIo::new(stream), service);
    let _ = watcher.watch(connection).await;
}

async fn relay(
    request: Request<Incoming>,
    context: Arc<RelayContext>,
    peer: SocketAddr,
) -> Result<Response<ResponseBody>, Infallible> {
    let dump = context
        .request_dump
        .clone()
        .map(|request_dump| (request_dump, dumped_head(&request)));
    let middle_end = match connect_middle_end(&context.middle_end_socket).await {
        Ok(middle_end) => middle_end,
        Err(error) => {
            warn!(
                "failed to connect to the middle end at '{}': {error}",
                context.middle_end_socket.display()
            );
            if let Some((request_dump, head)) = dump {
                request_dump.record(peer, &head, DumpedBody::NotRead);
            }
            return Ok(error_response(
                StatusCode::BAD_GATEWAY,
                "middle end unavailable",
            ));
        }
    };
    let (reader, writer) = middle_end.into_split();
    let (parts, body) = request.into_parts();
    let head = RequestHead {
        method: parts.method,
        target: parts
            .uri
            .path_and_query()
            .map_or("/", |target| target.as_str())
            .to_string(),
        headers: without_transport_headers(&parts.headers),
    };
    let (completion_sender, completion_receiver) = oneshot::channel::<()>();
    tokio::spawn(async move {
        let mut writer = writer;
        let mut body_copy = dump.as_ref().map(|_| Vec::new());
        let is_body_complete = send_request(&mut writer, head, body, &mut body_copy).await;
        if let (Some((request_dump, head)), Some(body_copy)) = (dump, body_copy) {
            let body = if is_body_complete {
                DumpedBody::Complete(&body_copy)
            } else {
                DumpedBody::Incomplete(&body_copy)
            };
            request_dump.record(peer, &head, body);
        }
        let _ = completion_receiver.await;
    });

    let mut frames = FrameReader {
        reader,
        buffer: BytesMut::new(),
    };
    let head = match frames.next().await {
        Ok(Some(Frame {
            kind: FrameKind::ResponseHead,
            payload,
        })) => ResponseHead::parse(&payload)
            .ok()
            .flatten()
            .map(|(head, _)| head),
        _ => None,
    };
    let Some(head) = head else {
        return Ok(error_response(
            StatusCode::BAD_GATEWAY,
            "middle end closed the connection without a response",
        ));
    };
    let (sender, body) = Channel::<Bytes, io::Error>::new(RESPONSE_BODY_BUFFER_FRAMES);
    tokio::spawn(async move {
        relay_response_body(frames, sender).await;
        drop(completion_sender);
    });
    let mut response = Response::new(body.map_err(BoxError::from).boxed());
    *response.status_mut() = head.status;
    *response.headers_mut() = head.headers;
    Ok(response)
}

async fn connect_middle_end(socket: &PathBuf) -> io::Result<UnixStream> {
    let deadline = tokio::time::Instant::now() + MIDDLE_END_BACKLOG_TIMEOUT;
    loop {
        match UnixStream::connect(socket).await {
            Err(error)
                if error.kind() == io::ErrorKind::WouldBlock
                    && tokio::time::Instant::now() < deadline =>
            {
                tokio::time::sleep(MIDDLE_END_BACKLOG_RETRY).await;
            }
            result => return result,
        }
    }
}

async fn send_request(
    writer: &mut OwnedWriteHalf,
    head: RequestHead,
    mut body: Incoming,
    body_copy: &mut Option<Vec<u8>>,
) -> bool {
    let mut output = BytesMut::new();
    encode_frame(FrameKind::RequestHead, &head.encode(), &mut output);
    if writer.write_all(&output).await.is_err() {
        return false;
    }
    while let Some(frame) = body.frame().await {
        let Ok(frame) = frame else {
            return false;
        };
        let Some(data) = frame.data_ref() else {
            continue;
        };
        if let Some(body_copy) = body_copy {
            body_copy.extend_from_slice(data);
        }
        output.clear();
        encode_body_frames(FrameKind::RequestBody, data, &mut output);
        if writer.write_all(&output).await.is_err() {
            return false;
        }
    }
    output.clear();
    encode_frame(FrameKind::RequestEnd, b"", &mut output);
    let _ = writer.write_all(&output).await;
    true
}

impl RequestDump {
    pub fn open(path: &Path) -> Result<(RequestDump, JoinHandle<()>), String> {
        let (file, writer) = AppendFile::open(path, "request dump")?;
        let request_dump = RequestDump {
            file,
            request_count: Arc::default(),
        };
        Ok((request_dump, writer))
    }

    fn record(&self, peer: SocketAddr, head: &[u8], body: DumpedBody) {
        let number = self.request_count.fetch_add(1, Ordering::Relaxed) + 1;
        let (body_description, body_bytes): (String, &[u8]) = match body {
            DumpedBody::Complete(bytes) => (format!("body {} bytes", bytes.len()), bytes),
            DumpedBody::Incomplete(bytes) => (
                format!("body incomplete after {} bytes", bytes.len()),
                bytes,
            ),
            DumpedBody::NotRead => ("body not read".to_string(), &[]),
        };
        let mut record = format!(
            "=== {:.3} request {number} from {peer}, {body_description}\n",
            jiff::Timestamp::now()
        )
        .into_bytes();
        record.extend_from_slice(head);
        record.extend_from_slice(body_bytes);
        record.push(b'\n');
        self.file.append(record);
    }
}

fn dumped_head(request: &Request<Incoming>) -> Vec<u8> {
    let mut head = format!(
        "{} {} {:?}\n",
        request.method(),
        request.uri(),
        request.version()
    )
    .into_bytes();
    for (name, value) in request.headers() {
        head.extend_from_slice(name.as_str().as_bytes());
        head.extend_from_slice(b": ");
        head.extend_from_slice(value.as_bytes());
        head.push(b'\n');
    }
    head.push(b'\n');
    head
}

async fn relay_response_body(mut frames: FrameReader, mut sender: Sender<Bytes, io::Error>) {
    loop {
        match frames.next().await {
            Ok(Some(Frame {
                kind: FrameKind::ResponseBody,
                payload,
            })) => {
                if sender.send_data(payload).await.is_err() {
                    return;
                }
            }
            Ok(Some(Frame {
                kind: FrameKind::ResponseEnd,
                ..
            })) => return,
            Ok(_) => {
                sender.abort(io::Error::other("middle end response ended early"));
                return;
            }
            Err(error) => {
                sender.abort(io::Error::other(error));
                return;
            }
        }
    }
}

struct FrameReader {
    reader: OwnedReadHalf,
    buffer: BytesMut,
}

impl FrameReader {
    async fn next(&mut self) -> Result<Option<Frame>, String> {
        loop {
            if let Some(frame) = decode_frame(&mut self.buffer)? {
                return Ok(Some(frame));
            }
            let length = self
                .reader
                .read_buf(&mut self.buffer)
                .await
                .map_err(|error| format!("failed to read from the middle end: {error}"))?;
            if length == 0 {
                return Ok(None);
            }
        }
    }
}

fn error_response(status: StatusCode, message: &str) -> Response<ResponseBody> {
    let mut response = Response::new(
        Full::new(Bytes::from(error_json(message)))
            .map_err(|never| match never {})
            .boxed(),
    );
    *response.status_mut() = status;
    response.headers_mut().insert(
        hyper::header::CONTENT_TYPE,
        hyper::header::HeaderValue::from_static("application/json"),
    );
    response
}

#[cfg(feature = "tls")]
mod tls {
    use std::net::{IpAddr, Ipv4Addr, Ipv6Addr};
    use std::sync::Arc;
    use std::time::Duration;

    use rcgen::{CertificateParams, KeyPair, SanType};
    use rustls::pki_types::pem::PemObject;
    use rustls::pki_types::{CertificateDer, PrivateKeyDer, PrivatePkcs8KeyDer};
    use rustls::ServerConfig;
    use tokio_rustls::TlsAcceptor;

    use crate::config::TlsConfig;

    pub const HANDSHAKE_TIMEOUT: Duration = Duration::from_secs(10);

    pub fn acceptor(config: &TlsConfig) -> Result<TlsAcceptor, String> {
        let (certificate, key) = match config {
            TlsConfig::Files {
                certificate,
                private_key,
            } => (
                CertificateDer::from_pem_file(certificate).map_err(|error| {
                    format!(
                        "failed to read TLS certificate '{}': {error}",
                        certificate.display()
                    )
                })?,
                PrivateKeyDer::from_pem_file(private_key).map_err(|error| {
                    format!(
                        "failed to read TLS private key '{}': {error}",
                        private_key.display()
                    )
                })?,
            ),
            TlsConfig::SelfSigned(_) => self_signed_certificate()?,
        };
        let server_config = ServerConfig::builder()
            .with_no_client_auth()
            .with_single_cert(vec![certificate], key)
            .map_err(|error| format!("invalid TLS configuration: {error}"))?;
        Ok(TlsAcceptor::from(Arc::new(server_config)))
    }

    fn self_signed_certificate() -> Result<(CertificateDer<'static>, PrivateKeyDer<'static>), String>
    {
        let key_pair = KeyPair::generate()
            .map_err(|error| format!("failed to generate a TLS key pair: {error}"))?;
        let mut params = CertificateParams::new(vec!["localhost".to_string()])
            .map_err(|error| format!("failed to create certificate parameters: {error}"))?;
        params.subject_alt_names.extend([
            SanType::IpAddress(IpAddr::V4(Ipv4Addr::LOCALHOST)),
            SanType::IpAddress(IpAddr::V6(Ipv6Addr::LOCALHOST)),
        ]);
        let certificate = params
            .self_signed(&key_pair)
            .map_err(|error| format!("failed to generate a self-signed certificate: {error}"))?;
        let key = PrivateKeyDer::Pkcs8(PrivatePkcs8KeyDer::from(key_pair.serialize_der()));
        Ok((certificate.der().clone(), key))
    }
}

#[cfg(test)]
mod tests {
    use std::net::SocketAddr;
    use std::path::Path;

    use http::header::{HeaderValue, CONTENT_TYPE};
    use http::Method;
    use hyper::client::conn::http1 as client_http1;
    use tokio::net::{TcpStream, UnixListener};
    use tokio::sync::mpsc;

    use super::*;

    struct FakeMiddleEnd {
        socket: PathBuf,
        connections: mpsc::UnboundedReceiver<UnixStream>,
    }

    impl FakeMiddleEnd {
        fn start(directory: &Path) -> FakeMiddleEnd {
            let socket = directory.join("middle-end.sock");
            let listener = UnixListener::bind(&socket).unwrap();
            let (sender, connections) = mpsc::unbounded_channel();
            tokio::spawn(async move {
                while let Ok((stream, _)) = listener.accept().await {
                    if sender.send(stream).is_err() {
                        return;
                    }
                }
            });
            FakeMiddleEnd {
                socket,
                connections,
            }
        }

        async fn connection(&mut self) -> UnixStream {
            self.connections.recv().await.unwrap()
        }
    }

    async fn start_frontend(middle_end_socket: PathBuf, transport: Transport) -> SocketAddr {
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let address = listener.local_addr().unwrap();
        tokio::spawn(serve(
            listener,
            transport,
            middle_end_socket,
            None,
            Duration::from_secs(1),
            std::future::pending(),
        ));
        address
    }

    async fn received_request(stream: &mut UnixStream) -> (RequestHead, Vec<u8>) {
        let mut buffer = BytesMut::new();
        let mut head = None;
        let mut body = Vec::new();
        loop {
            while let Some(frame) = decode_frame(&mut buffer).unwrap() {
                match frame.kind {
                    FrameKind::RequestHead => {
                        head = Some(RequestHead::parse(&frame.payload).unwrap().unwrap().0)
                    }
                    FrameKind::RequestBody => body.extend_from_slice(&frame.payload),
                    FrameKind::RequestEnd => return (head.unwrap(), body),
                    other => panic!("unexpected frame {other:?}"),
                }
            }
            assert_ne!(stream.read_buf(&mut buffer).await.unwrap(), 0);
        }
    }

    async fn send_frames(stream: &mut UnixStream, frames: &[(FrameKind, &[u8])]) {
        let mut output = BytesMut::new();
        for (kind, payload) in frames {
            encode_frame(*kind, payload, &mut output);
        }
        stream.write_all(&output).await.unwrap();
    }

    fn event_stream_head() -> Vec<u8> {
        let mut head = ResponseHead::new(StatusCode::OK);
        head.headers
            .insert(CONTENT_TYPE, HeaderValue::from_static("text/event-stream"));
        head.encode()
    }

    async fn client_request<Stream>(
        stream: Stream,
        request: Request<Full<Bytes>>,
    ) -> hyper::Response<hyper::body::Incoming>
    where
        Stream: AsyncRead + AsyncWrite + Unpin + Send + 'static,
    {
        let (mut sender, connection) = client_http1::handshake(TokioIo::new(stream)).await.unwrap();
        tokio::spawn(connection);
        sender.send_request(request).await.unwrap()
    }

    fn chat_request(address: SocketAddr) -> Request<Full<Bytes>> {
        Request::builder()
            .method(Method::POST)
            .uri("/v1/chat/completions?trace=1")
            .header("host", address.to_string())
            .header("authorization", "Bearer sk-test")
            .header("x-priority", "batch")
            .body(Full::new(Bytes::from_static(b"{\"stream\":true}")))
            .unwrap()
    }

    #[tokio::test]
    async fn request_and_response_relay() {
        let directory = tempfile::tempdir().unwrap();
        let mut middle_end = FakeMiddleEnd::start(directory.path());
        let address = start_frontend(middle_end.socket.clone(), Transport::Plain).await;
        let client = tokio::spawn(async move {
            let stream = TcpStream::connect(address).await.unwrap();
            let response = client_request(stream, chat_request(address)).await;
            let status = response.status();
            let content_type = response.headers()[CONTENT_TYPE].clone();
            let body = response.into_body().collect().await.unwrap().to_bytes();
            (status, content_type, body)
        });

        let mut connection = middle_end.connection().await;
        let (head, body) = received_request(&mut connection).await;
        assert_eq!(head.method, Method::POST);
        assert_eq!(head.target, "/v1/chat/completions?trace=1");
        assert_eq!(head.headers["authorization"], "Bearer sk-test");
        assert_eq!(head.headers["x-priority"], "batch");
        assert!(head.headers.get("content-length").is_none());
        assert_eq!(body, b"{\"stream\":true}");
        send_frames(
            &mut connection,
            &[
                (FrameKind::ResponseHead, &event_stream_head()),
                (FrameKind::ResponseBody, b"data: 1\n\n"),
                (FrameKind::ResponseBody, b"data: 2\n\n"),
                (FrameKind::ResponseEnd, b""),
            ],
        )
        .await;

        let (status, content_type, body) = client.await.unwrap();
        assert_eq!(status, StatusCode::OK);
        assert_eq!(content_type, "text/event-stream");
        assert_eq!(body, Bytes::from_static(b"data: 1\n\ndata: 2\n\n"));
    }

    #[tokio::test]
    async fn middle_end_unavailable() {
        let directory = tempfile::tempdir().unwrap();
        let dump_path = directory.path().join("requests.dump");
        let (request_dump, dump_writer) = RequestDump::open(&dump_path).unwrap();
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let address = listener.local_addr().unwrap();
        let (stop_sender, stop) = oneshot::channel::<()>();
        let frontend = tokio::spawn(serve(
            listener,
            Transport::Plain,
            directory.path().join("missing.sock"),
            Some(request_dump),
            Duration::from_secs(1),
            async {
                let _ = stop.await;
            },
        ));
        let stream = TcpStream::connect(address).await.unwrap();
        let response = client_request(stream, chat_request(address)).await;
        assert_eq!(response.status(), StatusCode::BAD_GATEWAY);
        drop(response);
        stop_sender.send(()).unwrap();
        frontend.await.unwrap();
        dump_writer.await.unwrap();
        let dump = std::fs::read_to_string(&dump_path).unwrap();
        assert!(dump.lines().next().unwrap().ends_with(", body not read"));
        assert!(dump.contains("\nPOST /v1/chat/completions?trace=1 HTTP/1.1\n"));
    }

    #[tokio::test]
    async fn truncated_response_abort() {
        let directory = tempfile::tempdir().unwrap();
        let mut middle_end = FakeMiddleEnd::start(directory.path());
        let address = start_frontend(middle_end.socket.clone(), Transport::Plain).await;
        let client = tokio::spawn(async move {
            let stream = TcpStream::connect(address).await.unwrap();
            let response = client_request(stream, chat_request(address)).await;
            response
                .into_body()
                .collect()
                .await
                .map(|body| body.to_bytes())
        });
        let mut connection = middle_end.connection().await;
        received_request(&mut connection).await;
        send_frames(
            &mut connection,
            &[
                (FrameKind::ResponseHead, &event_stream_head()),
                (FrameKind::ResponseBody, b"data: 1\n\n"),
            ],
        )
        .await;
        drop(connection);
        assert!(client.await.unwrap().is_err());
    }

    #[tokio::test]
    async fn client_disconnect_closes_middle_end_connection() {
        let directory = tempfile::tempdir().unwrap();
        let mut middle_end = FakeMiddleEnd::start(directory.path());
        let address = start_frontend(middle_end.socket.clone(), Transport::Plain).await;
        let client = tokio::spawn(async move {
            let stream = TcpStream::connect(address).await.unwrap();
            client_request(stream, chat_request(address)).await
        });
        let mut connection = middle_end.connection().await;
        received_request(&mut connection).await;
        let mut probe = [0u8; 1];
        let still_open =
            tokio::time::timeout(Duration::from_millis(200), connection.read(&mut probe)).await;
        assert!(still_open.is_err());
        client.abort();
        let closed = tokio::time::timeout(Duration::from_secs(5), connection.read(&mut probe))
            .await
            .unwrap()
            .unwrap();
        assert_eq!(closed, 0);
    }

    #[cfg(feature = "tls")]
    #[tokio::test]
    async fn https_relay() {
        let directory = tempfile::tempdir().unwrap();
        let key_pair = rcgen::KeyPair::generate().unwrap();
        let certificate = rcgen::CertificateParams::new(vec!["localhost".to_string()])
            .unwrap()
            .self_signed(&key_pair)
            .unwrap();
        std::fs::write(directory.path().join("certificate.pem"), certificate.pem()).unwrap();
        std::fs::write(directory.path().join("key.pem"), key_pair.serialize_pem()).unwrap();
        let acceptor = tls::acceptor(&crate::config::TlsConfig::Files {
            certificate: directory.path().join("certificate.pem"),
            private_key: directory.path().join("key.pem"),
        })
        .unwrap();
        let mut middle_end = FakeMiddleEnd::start(directory.path());
        let address = start_frontend(middle_end.socket.clone(), Transport::Tls(acceptor)).await;

        let mut roots = rustls::RootCertStore::empty();
        roots.add(certificate.der().clone()).unwrap();
        let client_config = rustls::ClientConfig::builder()
            .with_root_certificates(roots)
            .with_no_client_auth();
        let connector = tokio_rustls::TlsConnector::from(Arc::new(client_config));
        let client = tokio::spawn(async move {
            let tcp_stream = TcpStream::connect(address).await.unwrap();
            let server_name = rustls::pki_types::ServerName::try_from("localhost").unwrap();
            let tls_stream = connector.connect(server_name, tcp_stream).await.unwrap();
            let response = client_request(tls_stream, chat_request(address)).await;
            response.into_body().collect().await.unwrap().to_bytes()
        });
        let mut connection = middle_end.connection().await;
        received_request(&mut connection).await;
        send_frames(
            &mut connection,
            &[
                (
                    FrameKind::ResponseHead,
                    &ResponseHead::new(StatusCode::OK).encode(),
                ),
                (FrameKind::ResponseBody, b"{}"),
                (FrameKind::ResponseEnd, b""),
            ],
        )
        .await;
        assert_eq!(client.await.unwrap(), Bytes::from_static(b"{}"));
    }
}
