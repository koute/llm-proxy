mod control_link;
pub mod handover;
mod request_flow;
mod routing;
mod scheduler;
mod scheduling_log;

use std::collections::HashMap;
use std::future::Future;
use std::num::NonZeroUsize;
use std::os::fd::OwnedFd;
use std::path::{Path, PathBuf};
use std::pin::Pin;
use std::process::Stdio;
use std::sync::Arc;
use std::time::Duration;

use log::{error, info, warn};
use nix::sys::signal::SigmaskHow;
use tokio::io::AsyncWriteExt;
use tokio::net::UnixListener;
use tokio::sync::{mpsc, watch};
use tokio::task::JoinSet;
use tokio::time::Instant;

use crate::config::{Config, ConfigTexts};
use crate::quality_of_service::{Rank, Settings, USAGE_BUCKET};
use crate::utils::HandoverFd;
use crate::utils::Timestamp;
use control_link::{ControlLink, ControlLinkSnapshot};
use handover::{
    change_signal_mask, Handover, ListenerSnapshot, RestartPlan, HANDOVER_FORMAT_VERSION,
    RESTART_SIGNALS,
};
use request_flow::{FlowContext, RequestFlow};
use scheduler::{ProxyIdCounter, Scheduler};
use scheduling_log::{ProcessStop, SchedulingLog, SchedulingLogWriter};

const TICK_INTERVAL: Duration = Duration::from_millis(250);
const STATUS_INTERVAL: Duration = Duration::from_secs(1);
const ACCEPT_ERROR_BACKOFF: Duration = Duration::from_millis(50);
const PREFLIGHT_TIMEOUT: Duration = Duration::from_secs(60);

type Quiesce = watch::Receiver<Option<Instant>>;
type PreflightCheck<'a> = Pin<Box<dyn Future<Output = Result<ConfigTexts, String>> + Send + 'a>>;

pub enum Outcome {
    Shutdown,
    Restart(Box<Handover>),
}

pub struct Controls<'a, Shutdown> {
    pub shutdown: Shutdown,
    pub restart_requests: &'a mut mpsc::UnboundedReceiver<()>,
    pub restart_plan: Option<&'a RestartPlan>,
}

struct MiddleEnd {
    context: Arc<FlowContext>,
    listener: UnixListener,
    links: JoinSet<ControlLink>,
    flows: JoinSet<Option<RequestFlow>>,
    background: JoinSet<()>,
    quiesce_sender: watch::Sender<Option<Instant>>,
    scheduling_log_writer: SchedulingLogWriter,
}

fn bind_listener(path: &Path) -> Result<UnixListener, String> {
    if path.exists() {
        if std::os::unix::net::UnixStream::connect(path).is_ok() {
            return Err(format!(
                "another middle end is listening on '{}'",
                path.display()
            ));
        }
        std::fs::remove_file(path).map_err(|error| {
            format!(
                "failed to remove the stale socket '{}': {error}",
                path.display()
            )
        })?;
    }
    UnixListener::bind(path)
        .map_err(|error| format!("failed to listen on '{}': {error}", path.display()))
}

async fn quiesce_deadline(quiesce: &mut Quiesce) -> Instant {
    let deadline = match quiesce.wait_for(Option::is_some).await {
        Ok(deadline) => *deadline,
        Err(_) => None,
    };
    match deadline {
        Some(deadline) => deadline,
        None => std::future::pending().await,
    }
}

pub async fn serve(
    config: Config,
    handover: Option<Box<Handover>>,
    controls: Controls<'_, impl Future<Output = ()>>,
) -> Result<Outcome, String> {
    let Controls {
        shutdown,
        restart_requests,
        restart_plan,
    } = controls;
    let mut middle_end = MiddleEnd::new(config, handover)?;
    tokio::pin!(shutdown);
    let mut preflight: Option<PreflightCheck> = None;
    let mut is_preflight_outdated = false;
    loop {
        tokio::select! {
            _ = &mut shutdown => {
                middle_end.drain().await;
                return Ok(Outcome::Shutdown);
            }
            Some(()) = restart_requests.recv() => match restart_plan {
                None => {
                    let texts = middle_end.context.config.texts.clone();
                    return Ok(middle_end.restart(texts, &mut shutdown).await);
                }
                Some(_) if preflight.is_some() => {
                    info!("restart requested again; the pre-flight check runs again with the current files");
                    is_preflight_outdated = true;
                }
                Some(plan) => {
                    info!("restart requested; running the pre-flight check");
                    preflight = Some(Box::pin(preflight_check(plan)));
                }
            },
            result = async { preflight.as_mut().expect("missing pre-flight check").await }, if preflight.is_some() => {
                preflight = None;
                if is_preflight_outdated {
                    is_preflight_outdated = false;
                    preflight = restart_plan.map(|plan| Box::pin(preflight_check(plan)) as PreflightCheck);
                } else {
                    match result {
                        Ok(texts) => return Ok(middle_end.restart(texts, &mut shutdown).await),
                        Err(message) => error!("restart aborted: {message}"),
                    }
                }
            }
            accepted = middle_end.listener.accept() => match accepted {
                Ok((stream, _)) => {
                    let flow = RequestFlow::new(stream, &middle_end.context);
                    middle_end.spawn_flow(flow);
                }
                Err(error) => {
                    warn!("middle end accept failed ({error}); retrying in {ACCEPT_ERROR_BACKOFF:?}");
                    tokio::time::sleep(ACCEPT_ERROR_BACKOFF).await;
                }
            },
        }
        while middle_end.flows.try_join_next().is_some() {}
    }
}

impl MiddleEnd {
    fn new(config: Config, handover: Option<Box<Handover>>) -> Result<MiddleEnd, String> {
        let is_adopted = handover.is_some();
        let (listener, proxy_ids, previous_links, previous_flows) = match handover {
            Some(handover) => {
                let Handover {
                    listener,
                    proxy_ids,
                    control_links,
                    flows,
                    ..
                } = *handover;
                (listener, proxy_ids, control_links, flows)
            }
            None => (None, ProxyIdCounter::new(), Vec::new(), Vec::new()),
        };
        let socket = &config.general.middle_end_socket;
        let listener = adopt_listener(socket, listener)?;
        info!("middle end listening on '{}'", socket.display());
        log_configuration(&config);
        let (scheduling_log, scheduling_log_writer) =
            SchedulingLog::open(config.general.scheduling_log.as_deref());
        scheduling_log.record(SchedulingLog::start_event(
            is_adopted,
            &config.rank_capacities,
            config
                .general
                .quality_of_service
                .swap_hysteresis_seconds
                .duration(),
            &config.accounts.classes,
        ));
        let sockets = control_sockets(&config);
        let mut snapshots_by_socket: HashMap<PathBuf, ControlLinkSnapshot> = previous_links
            .into_iter()
            .map(|snapshot| (snapshot.socket.clone(), snapshot))
            .collect();
        let (senders, receivers): (Vec<_>, Vec<_>) =
            sockets.iter().map(|_| mpsc::unbounded_channel()).unzip();
        let quality_of_service = &config.general.quality_of_service;
        let scheduler = Scheduler::new(
            proxy_ids,
            Settings {
                rank_capacities: config.rank_capacities.clone(),
                swap_hysteresis: quality_of_service.swap_hysteresis_seconds.duration(),
                usage_half_life: USAGE_BUCKET * quality_of_service.usage_half_life_minutes,
            },
            config.general.limits.max_queued_requests,
            senders,
            scheduling_log.clone(),
        );
        let links: Vec<ControlLink> = receivers
            .into_iter()
            .zip(sockets)
            .enumerate()
            .map(|(index, (messages, socket))| {
                let rank = Rank(index);
                match snapshots_by_socket.remove(&socket) {
                    Some(snapshot) => {
                        ControlLink::restore(rank, snapshot, scheduler.clone(), messages)
                    }
                    None => ControlLink::new(rank, socket, scheduler.clone(), messages),
                }
            })
            .collect();

        let context = FlowContext {
            config: Arc::new(config),
            scheduler,
            scheduling_log,
        };
        let previous_flow_count = previous_flows.len();
        let mut flows = Vec::with_capacity(previous_flow_count);
        for snapshot in previous_flows {
            match RequestFlow::restore(snapshot, &context) {
                Ok(flow) => flows.push(flow),
                Err(error) => error!("failed to adopt a request: {error}"),
            }
        }
        context.scheduler.lock().schedule(Timestamp::now());
        if is_adopted {
            info!(
                "restart complete: adopted {} of {previous_flow_count} request(s); {}",
                flows.len(),
                context.scheduler.lock().status().total
            );
        }

        let scheduler = context.scheduler.clone();
        let (quiesce_sender, quiesce) = watch::channel(None);
        let mut middle_end = MiddleEnd {
            context: Arc::new(context),
            listener,
            links: JoinSet::new(),
            flows: JoinSet::new(),
            background: JoinSet::new(),
            quiesce_sender,
            scheduling_log_writer,
        };
        for link in links {
            middle_end.links.spawn(link.run(quiesce.clone()));
        }
        for flow in flows {
            middle_end.spawn_flow(flow);
        }
        middle_end.background.spawn(tick_loop(scheduler.clone()));
        middle_end.background.spawn(status_loop(scheduler));
        Ok(middle_end)
    }

    fn spawn_flow(&mut self, flow: RequestFlow) {
        let context = self.context.clone();
        let quiesce = self.quiesce_sender.subscribe();
        self.flows
            .spawn(async move { flow.run(&context, quiesce).await });
    }

    async fn drain(&mut self) {
        let drain_timeout = self
            .context
            .config
            .general
            .process
            .shutdown_drain_timeout_seconds
            .duration();
        while self.flows.try_join_next().is_some() {}
        info!(
            "middle end shutdown: draining {} request(s)",
            self.flows.len()
        );
        let drain = async { while self.flows.join_next().await.is_some() {} };
        if tokio::time::timeout(drain_timeout, drain).await.is_err() {
            warn!("middle end shutdown: drain timed out after {drain_timeout:?}");
        }
        if let Ok(address) = self.listener.local_addr() {
            if let Some(path) = address.as_pathname() {
                let _ = std::fs::remove_file(path);
            }
        }
        self.context
            .scheduling_log
            .stop(
                ProcessStop::Shutdown,
                &mut self.scheduling_log_writer,
                drain_timeout,
            )
            .await;
    }

    async fn restart(
        mut self,
        config_texts: ConfigTexts,
        shutdown: &mut Pin<&mut impl Future<Output = ()>>,
    ) -> Outcome {
        change_signal_mask(SigmaskHow::SIG_BLOCK, &RESTART_SIGNALS);
        let is_shutdown_pending = tokio::select! {
            biased;
            _ = shutdown.as_mut() => true,
            _ = std::future::ready(()) => false,
        };
        if is_shutdown_pending {
            self.drain().await;
            return Outcome::Shutdown;
        }
        info!("restart: quiescing {} request(s)", self.flows.len());
        Outcome::Restart(Box::new(self.handover(config_texts).await))
    }

    async fn handover(mut self, config_texts: ConfigTexts) -> Handover {
        let config = &self.context.config;
        let flush_timeout = config
            .general
            .process
            .restart_flush_timeout_seconds
            .duration();
        let _ = self
            .quiesce_sender
            .send(Some(Instant::now() + flush_timeout));
        self.background.abort_all();
        let mut links = Vec::new();
        while let Some(result) = self.links.join_next().await {
            match result {
                Ok(link) => links.push(link),
                Err(error) => error!("control link failed during the restart: {error}"),
            }
        }
        let mut flows = Vec::new();
        while let Some(result) = self.flows.join_next().await {
            match result {
                Ok(Some(flow)) => flows.push(flow),
                Ok(None) => {}
                Err(error) => error!("request failed during the restart: {error}"),
            }
        }
        let flows = flows
            .into_iter()
            .filter_map(|flow| {
                flow.snapshot(&self.context)
                    .map_err(|error| error!("failed to hand over a request: {error}"))
                    .ok()
            })
            .collect();
        let proxy_ids = {
            let mut scheduler = self.context.scheduler.lock();
            scheduler.release_abandoned(Timestamp::now());
            scheduler.log_handover();
            scheduler.proxy_ids()
        };
        self.context
            .scheduling_log
            .stop(
                ProcessStop::Restart,
                &mut self.scheduling_log_writer,
                flush_timeout,
            )
            .await;
        let control_links = links.into_iter().map(ControlLink::snapshot).collect();
        let listener = self
            .listener
            .into_std()
            .and_then(|listener| HandoverFd::new(OwnedFd::from(listener)))
            .map_err(|error| error!("failed to hand over the middle end socket: {error}"))
            .ok()
            .map(|fd| ListenerSnapshot {
                fd,
                path: config.general.middle_end_socket.clone(),
            });
        Handover {
            format_version: HANDOVER_FORMAT_VERSION,
            config_texts,
            listener,
            proxy_ids,
            control_links,
            flows,
        }
    }
}

fn control_sockets(config: &Config) -> Vec<PathBuf> {
    config
        .backend
        .iter()
        .flat_map(|backend| backend.ranks.iter().map(|rank| rank.control_socket.clone()))
        .collect()
}

fn adopt_listener(path: &Path, snapshot: Option<ListenerSnapshot>) -> Result<UnixListener, String> {
    let Some(snapshot) = snapshot else {
        return bind_listener(path);
    };
    let adopted = snapshot.fd.adopt().and_then(|fd| {
        let listener = std::os::unix::net::UnixListener::from(fd);
        listener.set_nonblocking(true)?;
        UnixListener::from_std(listener)
    });
    match adopted {
        Ok(listener) if snapshot.path == path => Ok(listener),
        Ok(previous) => {
            info!(
                "middle end socket moved from '{}' to '{}'",
                snapshot.path.display(),
                path.display()
            );
            Ok(bind_listener(path).unwrap_or_else(|message| {
                error!("{message}; keeping '{}'", snapshot.path.display());
                previous
            }))
        }
        Err(error) => {
            error!("failed to adopt the middle end socket: {error}; binding it again");
            bind_listener(path)
        }
    }
}

async fn preflight_check(plan: &RestartPlan) -> Result<ConfigTexts, String> {
    let paths = &plan.config_paths;
    let texts = ConfigTexts::read(paths)?;
    let mut child = tokio::process::Command::new(&plan.executable)
        .arg("preflight")
        .arg("--general")
        .arg(&paths.general)
        .arg("--accounts")
        .arg(&paths.accounts)
        .arg("--backend")
        .arg(&paths.backend)
        .arg("--handover-format")
        .arg(HANDOVER_FORMAT_VERSION.to_string())
        .stdin(Stdio::piped())
        .kill_on_drop(true)
        .spawn()
        .map_err(|error| {
            format!(
                "failed to run the pre-flight check '{}': {error}",
                plan.executable.display()
            )
        })?;
    let mut input = Vec::new();
    serde_json::to_writer(&mut input, &texts).expect("unserializable configuration texts");
    let mut stdin = child.stdin.take().expect("missing pre-flight stdin");
    let check = async move {
        stdin.write_all(&input).await.map_err(|error| {
            format!("failed to send the configuration to the pre-flight check: {error}")
        })?;
        drop(stdin);
        child
            .wait()
            .await
            .map_err(|error| format!("the pre-flight check failed: {error}"))
    };
    let status = tokio::time::timeout(PREFLIGHT_TIMEOUT, check)
        .await
        .map_err(|_| format!("the pre-flight check timed out after {PREFLIGHT_TIMEOUT:?}"))??;
    if !status.success() {
        return Err(format!("the pre-flight check failed ({status})"));
    }
    Ok(texts)
}

async fn tick_loop(scheduler: Scheduler) {
    let mut interval = tokio::time::interval(TICK_INTERVAL);
    interval.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Delay);
    loop {
        interval.tick().await;
        scheduler.lock().tick(Timestamp::now());
    }
}

async fn status_loop(scheduler: Scheduler) {
    let mut interval = tokio::time::interval(STATUS_INTERVAL);
    interval.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Delay);
    let mut previous_requests = 0;
    loop {
        interval.tick().await;
        let status = scheduler.lock().status();
        if status.total.requests > 0 || previous_requests > 0 {
            info!("status: {}", status.total);
            for class in &status.classes {
                info!("status:   {class}");
            }
        }
        previous_requests = status.total.requests;
    }
}

fn log_configuration(config: &Config) {
    let Some(backend) = &config.backend else {
        warn!("no backend configured; requests get 503 until a backend file exists and the middle end restarts");
        return;
    };
    info!(
        "backend: {} serving '{}', {} rank(s), parallel request caps per rank {:?}",
        backend.engine,
        backend.model,
        backend.ranks.len(),
        config.rank_capacities
    );
    for (rank, rank_config) in backend.ranks.iter().enumerate() {
        info!(
            "rank {rank}: upstream {}, control socket '{}', data parallel rank {:?}",
            rank_config.upstream,
            rank_config.control_socket.display(),
            rank_config.data_parallel_rank
        );
    }
    for class in &config.accounts.classes {
        info!(
            "class '{}': priority {}, idle only {}, cost {}, soft cap {}, max parallel per user {}",
            class.name,
            class.priority,
            class.idle_only,
            class.cost,
            number_or_none(class.soft_cap),
            number_or_none(class.max_parallel_per_user)
        );
    }
}

fn number_or_none(number: Option<NonZeroUsize>) -> String {
    number.map_or("none".to_string(), |number| number.to_string())
}
