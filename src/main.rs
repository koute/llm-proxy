use std::path::{Path, PathBuf};
use std::process::ExitCode;

use clap::{Parser, Subcommand};
use log::{error, info};
use nix::sys::signal::SigmaskHow;
use tokio::signal::unix::{signal, SignalKind};
use tokio::sync::mpsc;

use llm_proxy::benchmark::{self, BenchmarkOptions};
use llm_proxy::config::{Config, ConfigPaths, ConfigTexts, GeneralConfig};
use llm_proxy::frontend;
use llm_proxy::middle_end::handover::{
    change_signal_mask, close_inherited_descriptors, exec, handover_from_environment,
    write_handover, RestartPlan, HANDOVER_FORMAT_VERSION, RESTART_SIGNALS, TERMINATION_SIGNALS,
};
use llm_proxy::middle_end::{self, Controls, Outcome};

#[derive(Parser)]
#[command(
    version,
    about = "QoS proxy for OpenAI/Anthropic-compatible LLM engines (vLLM, sglang)"
)]
struct Cli {
    #[arg(
        long,
        global = true,
        help = "Log QoS decisions (sets the log level to debug)"
    )]
    verbose: bool,
    #[command(subcommand)]
    command: CliCommand,
}

#[derive(Subcommand)]
enum CliCommand {
    #[command(
        about = "Run the middle end: QoS scheduling and the connection to the engine; SIGUSR1 restarts it in place"
    )]
    Serve(ConfigPaths),
    #[command(about = "Run the HTTP/HTTPS frontend in front of the middle end")]
    Frontend {
        #[arg(long, default_value = "config/general.json")]
        general: PathBuf,
        #[arg(
            long,
            help = "Append every request (headers and body, unredacted, API keys included) to this file"
        )]
        dump_requests: Option<PathBuf>,
    },
    #[command(
        about = "Benchmark a vLLM or sglang engine through its control socket and write the backend file"
    )]
    Benchmark(BenchmarkOptions),
    #[command(about = "Validate the configuration files and exit")]
    CheckConfig(ConfigPaths),
    #[command(hide = true)]
    Preflight {
        #[command(flatten)]
        paths: ConfigPaths,
        #[arg(long)]
        handover_format: u32,
    },
}

fn main() -> ExitCode {
    change_signal_mask(SigmaskHow::SIG_BLOCK, &RESTART_SIGNALS);
    let cli = Cli::parse();
    let default_level = if cli.verbose { "debug" } else { "info" };
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or(default_level))
        .init();
    let result = match cli.command {
        CliCommand::Serve(paths) => serve(paths),
        CliCommand::Frontend {
            general,
            dump_requests,
        } => run_frontend(&general, dump_requests.as_deref()),
        CliCommand::Benchmark(options) => {
            change_signal_mask(SigmaskHow::SIG_UNBLOCK, &TERMINATION_SIGNALS);
            runtime().and_then(|runtime| runtime.block_on(benchmark::run(options)).map(|_| ()))
        }
        CliCommand::CheckConfig(paths) => {
            ConfigTexts::read(&paths).and_then(|texts| check_config(&texts, &paths))
        }
        CliCommand::Preflight {
            paths,
            handover_format,
        } => preflight(&paths, handover_format),
    };
    match result {
        Ok(()) => ExitCode::SUCCESS,
        Err(message) => {
            error!("{message}");
            ExitCode::FAILURE
        }
    }
}

fn check_config(texts: &ConfigTexts, paths: &ConfigPaths) -> Result<(), String> {
    let config = Config::parse(texts, paths)?;
    let backend = if config.backend.is_some() {
        "with"
    } else {
        "without"
    };
    info!("configuration valid ({backend} backend)");
    Ok(())
}

fn preflight(paths: &ConfigPaths, handover_format: u32) -> Result<(), String> {
    if handover_format != HANDOVER_FORMAT_VERSION {
        return Err(format!(
            "this binary reads handover format {HANDOVER_FORMAT_VERSION}, not {handover_format}"
        ));
    }
    let texts = serde_json::from_reader(std::io::stdin().lock())
        .map_err(|error| format!("invalid configuration texts on stdin: {error}"))?;
    check_config(&texts, paths)
}

fn runtime() -> Result<tokio::runtime::Runtime, String> {
    tokio::runtime::Runtime::new()
        .map_err(|error| format!("failed to start the async runtime: {error}"))
}

fn serve(paths: ConfigPaths) -> Result<(), String> {
    let (config, mut handover) = match handover_from_environment() {
        None => (Config::load(&paths)?, None),
        Some(Ok(handover)) => match Config::parse(&handover.config_texts, &paths) {
            Ok(config) => (config, Some(Box::new(handover))),
            Err(message) => {
                error!("{message}; starting without the previous requests");
                drop(handover);
                (Config::load(&paths)?, None)
            }
        },
        Some(Err(message)) => {
            error!("{message}; starting without the previous requests");
            close_inherited_descriptors();
            (Config::load(&paths)?, None)
        }
    };
    let plan = RestartPlan {
        executable: executable_path()?,
        arguments: std::env::args_os().collect(),
        config_paths: paths,
    };
    let runtime = runtime()?;
    let (restart_sender, mut restart_requests) = mpsc::unbounded_channel();
    runtime.block_on(async {
        let mut restart_signal = signal(SignalKind::user_defined1())
            .map_err(|error| format!("failed to install the SIGUSR1 handler: {error}"))?;
        tokio::spawn(async move {
            while restart_signal.recv().await.is_some() {
                if restart_sender.send(()).is_err() {
                    return;
                }
            }
        });
        Ok::<_, String>(())
    })?;
    loop {
        let previous_handover = handover.take();
        let outcome = runtime.block_on(async {
            let shutdown = termination_signal()?;
            change_signal_mask(SigmaskHow::SIG_UNBLOCK, &RESTART_SIGNALS);
            let controls = Controls {
                shutdown,
                restart_requests: &mut restart_requests,
                restart_plan: Some(&plan),
            };
            middle_end::serve(config.clone(), previous_handover, controls).await
        })?;
        match outcome {
            Outcome::Shutdown => return Ok(()),
            Outcome::Restart(next_handover) => {
                let message = match write_handover(&next_handover) {
                    Ok(handover_file) => exec(&plan, &handover_file),
                    Err(error) => format!("failed to write the handover state: {error}"),
                };
                error!("restart failed: {message}; continuing in this process with the previous configuration");
                handover = Some(next_handover);
            }
        }
    }
}

fn executable_path() -> Result<PathBuf, String> {
    let invocation = std::env::args_os().next().map(PathBuf::from);
    let resolved = invocation.and_then(|path| {
        if path.components().count() > 1 {
            return std::path::absolute(path).ok();
        }
        let search_path = std::env::var_os("PATH")?;
        std::env::split_paths(&search_path)
            .map(|directory| directory.join(&path))
            .find(|candidate| candidate.is_file())
    });
    match resolved {
        Some(path) => Ok(path),
        None => std::env::current_exe()
            .map_err(|error| format!("failed to find the current executable: {error}")),
    }
}

fn run_frontend(general_path: &Path, request_dump_path: Option<&Path>) -> Result<(), String> {
    let text = std::fs::read_to_string(general_path)
        .map_err(|error| format!("failed to read '{}': {error}", general_path.display()))?;
    let general = GeneralConfig::parse(&text)
        .map_err(|error| format!("invalid '{}': {error}", general_path.display()))?;
    runtime()?.block_on(async {
        let shutdown = termination_signal()?;
        change_signal_mask(SigmaskHow::SIG_UNBLOCK, &TERMINATION_SIGNALS);
        frontend::run(&general, request_dump_path, shutdown).await
    })
}

fn termination_signal() -> Result<impl std::future::Future<Output = ()>, String> {
    let mut terminate = signal(SignalKind::terminate())
        .map_err(|error| format!("failed to install the SIGTERM handler: {error}"))?;
    let mut interrupt = signal(SignalKind::interrupt())
        .map_err(|error| format!("failed to install the SIGINT handler: {error}"))?;
    Ok(async move {
        tokio::select! {
            _ = terminate.recv() => {}
            _ = interrupt.recv() => {}
        }
    })
}
