use std::collections::VecDeque;
use std::path::PathBuf;
use std::time::Duration;

use bytes::Buf;
use log::{error, info, warn};
use serde::{Deserialize, Serialize};
use tokio::net::UnixStream;
use tokio::sync::{mpsc, oneshot};
use tokio::time::Instant;

use super::scheduler::Scheduler;
use super::{quiesce_deadline, Quiesce};
use crate::engine_protocol::{Command, CommandLine, EngineLine};
use crate::quality_of_service::Rank;
use crate::utils::Timestamp;
use crate::utils::{BufferedConnection, ConnectionSnapshot, ReadResult};

const RECONNECT_INTERVAL: Duration = Duration::from_secs(1);

pub enum LinkMessage {
    Command(Command),
    GrantStart(oneshot::Sender<Rank>),
}

pub struct ControlLink {
    rank: Rank,
    socket: PathBuf,
    scheduler: Scheduler,
    messages: mpsc::UnboundedReceiver<LinkMessage>,
    connection: Option<BufferedConnection<UnixStream>>,
    next_seq: u64,
    acknowledged_count: u64,
    pending_grants: VecDeque<(Option<u64>, oneshot::Sender<Rank>)>,
    is_failing: bool,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct ControlLinkSnapshot {
    pub socket: PathBuf,
    pub next_seq: u64,
    pub acknowledged_count: u64,
    pub connection: Option<ConnectionSnapshot>,
}

impl ControlLink {
    pub fn new(
        rank: Rank,
        socket: PathBuf,
        scheduler: Scheduler,
        messages: mpsc::UnboundedReceiver<LinkMessage>,
    ) -> ControlLink {
        ControlLink {
            rank,
            socket,
            scheduler,
            messages,
            connection: None,
            next_seq: 0,
            acknowledged_count: 0,
            pending_grants: VecDeque::new(),
            is_failing: false,
        }
    }

    pub fn restore(
        rank: Rank,
        snapshot: ControlLinkSnapshot,
        scheduler: Scheduler,
        messages: mpsc::UnboundedReceiver<LinkMessage>,
    ) -> ControlLink {
        let connection = snapshot.connection.and_then(|connection| {
            BufferedConnection::restore(connection)
                .map_err(|error| {
                    warn!("rank {rank}: failed to adopt the control connection: {error}")
                })
                .ok()
        });
        let acknowledged_count = match connection {
            Some(_) => snapshot.acknowledged_count,
            None => snapshot.next_seq,
        };
        ControlLink {
            connection,
            next_seq: snapshot.next_seq,
            acknowledged_count,
            ..ControlLink::new(rank, snapshot.socket, scheduler, messages)
        }
    }

    pub async fn run(mut self, mut quiesce: Quiesce) -> ControlLink {
        if self.connection.is_some() {
            self.resynchronize();
        }
        loop {
            let outcome = tokio::select! {
                biased;
                deadline = quiesce_deadline(&mut quiesce) => {
                    self.flush_until(deadline).await;
                    return self;
                }
                outcome = self.step() => outcome,
            };
            if let Err(error) = outcome {
                warn!(
                    "rank {}: engine control socket failed: {error}; reconnecting",
                    self.rank
                );
                self.disconnect();
            }
        }
    }

    async fn step(&mut self) -> Result<(), String> {
        if self.connection.is_none() {
            self.connect().await;
            return Ok(());
        }
        let connection = self
            .connection
            .as_mut()
            .expect("missing control connection");
        tokio::select! {
            message = self.messages.recv() => {
                if let Some(message) = message {
                    self.accept_message(message);
                }
                Ok(())
            }
            result = connection.readable() => {
                let result = result
                    .and_then(|()| connection.read_available())
                    .map_err(|error| format!("read failed: {error}"))?;
                if result == ReadResult::EndOfStream {
                    return Err("the engine closed the connection".to_string());
                }
                self.process_lines()
            }
            result = connection.writable(), if connection.has_pending_writes() => {
                result
                    .and_then(|()| connection.write_available())
                    .map_err(|error| format!("write failed: {error}"))
            }
        }
    }

    async fn connect(&mut self) {
        match UnixStream::connect(&self.socket).await {
            Ok(stream) => {
                self.is_failing = false;
                info!(
                    "rank {}: connected to the engine control socket '{}'",
                    self.rank,
                    self.socket.display()
                );
                self.connection = Some(BufferedConnection::new(stream));
                self.resynchronize();
            }
            Err(error) => {
                if !self.is_failing {
                    warn!(
                        "rank {}: failed to connect to the engine control socket '{}': {error}; new requests on this rank wait; retrying every {RECONNECT_INTERVAL:?}",
                        self.rank,
                        self.socket.display()
                    );
                    self.is_failing = true;
                }
                let retry = tokio::time::sleep(RECONNECT_INTERVAL);
                tokio::pin!(retry);
                loop {
                    tokio::select! {
                        _ = &mut retry => return,
                        message = self.messages.recv() => match message {
                            Some(message) => self.accept_message(message),
                            None => std::future::pending().await,
                        },
                    }
                }
            }
        }
    }

    fn resynchronize(&mut self) {
        let commands = self.scheduler.lock().resynchronization_commands(self.rank);
        for command in commands {
            self.write_command(command);
        }
        for (required, _) in &mut self.pending_grants {
            *required = Some(self.next_seq);
        }
        self.grant_acknowledged();
    }

    fn accept_message(&mut self, message: LinkMessage) {
        match message {
            LinkMessage::Command(command) => self.write_command(command),
            LinkMessage::GrantStart(permit) => {
                let required = self.connection.as_ref().map(|_| self.next_seq);
                self.pending_grants.push_back((required, permit));
                self.grant_acknowledged();
            }
        }
    }

    fn write_command(&mut self, command: Command) {
        let Some(connection) = &mut self.connection else {
            return;
        };
        let line = CommandLine {
            seq: self.next_seq,
            command,
        };
        self.next_seq += 1;
        connection.write_buffer.extend_from_slice(&line.encode());
    }

    fn grant_acknowledged(&mut self) {
        while let Some((required, _)) = self.pending_grants.front() {
            if required.is_none_or(|required| required > self.acknowledged_count) {
                break;
            }
            let (_, permit) = self.pending_grants.pop_front().expect("missing grant");
            let _ = permit.send(self.rank);
        }
    }

    fn disconnect(&mut self) {
        self.connection = None;
        self.acknowledged_count = self.next_seq;
        for (required, _) in &mut self.pending_grants {
            *required = None;
        }
    }

    fn process_lines(&mut self) -> Result<(), String> {
        loop {
            let Some(connection) = &mut self.connection else {
                return Ok(());
            };
            let buffer = &mut connection.read_buffer;
            let Some(position) = buffer.iter().position(|byte| *byte == b'\n') else {
                return Ok(());
            };
            let line = buffer.split_to(position);
            buffer.advance(1);
            match EngineLine::parse(&line) {
                Ok(EngineLine::Event(event)) => {
                    self.scheduler
                        .lock()
                        .engine_event(self.rank, event, Timestamp::now());
                }
                Ok(EngineLine::Success { seq, .. }) => {
                    self.acknowledged_count = self.acknowledged_count.max(seq + 1);
                    self.grant_acknowledged();
                }
                Ok(EngineLine::Failure { seq, error }) => {
                    return Err(format!("the engine rejected command {seq}: {error}"));
                }
                Err(error) => return Err(error),
            }
        }
    }

    async fn flush_until(&mut self, deadline: Instant) {
        self.encode_queued_messages();
        if let Some(connection) = &mut self.connection {
            let _ = tokio::time::timeout_at(deadline, connection.flush()).await;
        }
    }

    fn encode_queued_messages(&mut self) {
        while let Ok(message) = self.messages.try_recv() {
            match message {
                LinkMessage::Command(command) => self.write_command(command),
                LinkMessage::GrantStart(permit) => self.pending_grants.push_back((None, permit)),
            }
        }
    }

    pub fn snapshot(mut self) -> ControlLinkSnapshot {
        self.encode_queued_messages();
        let rank = self.rank;
        let connection = self.connection.and_then(|connection| {
            connection
                .snapshot()
                .map_err(|error| {
                    error!("rank {rank}: failed to hand over the control connection: {error}")
                })
                .ok()
        });
        ControlLinkSnapshot {
            socket: self.socket,
            next_seq: self.next_seq,
            acknowledged_count: self.acknowledged_count,
            connection,
        }
    }
}
