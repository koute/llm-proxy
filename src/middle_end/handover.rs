use std::ffi::{CString, OsString};
use std::fs::File;
use std::io::{self, BufReader, Seek, SeekFrom};
use std::os::fd::{AsRawFd, FromRawFd, OwnedFd, RawFd};
use std::os::unix::ffi::OsStrExt;
use std::path::PathBuf;

use nix::fcntl::{fcntl, FcntlArg, FdFlag};
use nix::sys::memfd::{memfd_create, MFdFlags};
use nix::sys::signal::{pthread_sigmask, SigmaskHow, Signal};
use serde::{Deserialize, Serialize};

use super::control_link::ControlLinkSnapshot;
use super::request_flow::FlowSnapshot;
use super::scheduler::ProxyIdCounter;
use crate::config::{ConfigPaths, ConfigTexts};
use crate::utils::HandoverFd;

pub const HANDOVER_FORMAT_VERSION: u32 = 5;
const HANDOVER_VARIABLE: &str = "LLM_PROXY_HANDOVER_FD";
pub const RESTART_SIGNALS: [Signal; 3] = [Signal::SIGUSR1, Signal::SIGTERM, Signal::SIGINT];
pub const TERMINATION_SIGNALS: [Signal; 2] = [Signal::SIGTERM, Signal::SIGINT];

#[derive(Debug, Serialize, Deserialize)]
pub struct Handover {
    pub format_version: u32,
    pub config_texts: ConfigTexts,
    pub listener: Option<ListenerSnapshot>,
    pub proxy_ids: ProxyIdCounter,
    pub control_links: Vec<ControlLinkSnapshot>,
    pub flows: Vec<FlowSnapshot>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct ListenerSnapshot {
    pub fd: HandoverFd,
    pub path: PathBuf,
}

#[derive(Clone, Debug)]
pub struct RestartPlan {
    pub executable: PathBuf,
    pub arguments: Vec<OsString>,
    pub config_paths: ConfigPaths,
}

pub fn write_handover(handover: &Handover) -> io::Result<OwnedFd> {
    let mut file = File::from(memfd_create("llm-proxy-handover", MFdFlags::empty())?);
    serde_json::to_writer(&mut file, handover)?;
    file.seek(SeekFrom::Start(0))?;
    Ok(OwnedFd::from(file))
}

pub fn handover_from_environment() -> Option<Result<Handover, String>> {
    let value = std::env::var_os(HANDOVER_VARIABLE)?;
    std::env::remove_var(HANDOVER_VARIABLE);
    let read = || {
        let fd: RawFd = value
            .to_str()
            .and_then(|text| text.parse().ok())
            .ok_or_else(|| format!("invalid {HANDOVER_VARIABLE} value {value:?}"))?;
        let file = File::from(unsafe { OwnedFd::from_raw_fd(fd) });
        fcntl(&file, FcntlArg::F_SETFD(FdFlag::FD_CLOEXEC)).map_err(|error| {
            format!("failed to adopt the handover file descriptor {fd}: {error}")
        })?;
        let handover: Handover = serde_json::from_reader(BufReader::new(file))
            .map_err(|error| format!("invalid handover state: {error}"))?;
        if handover.format_version != HANDOVER_FORMAT_VERSION {
            return Err(format!(
                "handover format {} differs from the supported format {HANDOVER_FORMAT_VERSION}",
                handover.format_version
            ));
        }
        Ok(handover)
    };
    Some(read())
}

pub fn close_inherited_descriptors() {
    unsafe {
        nix::libc::syscall(nix::libc::SYS_close_range, 3u32, u32::MAX, 0u32);
    }
}

pub fn change_signal_mask(how: SigmaskHow, signals: &[Signal]) {
    let _ = pthread_sigmask(how, Some(&signals.iter().copied().collect()), None);
}

pub fn exec(plan: &RestartPlan, handover: &OwnedFd) -> String {
    let c_string = |bytes: &[u8]| CString::new(bytes).map_err(|error| error.to_string());
    let prepared = || -> Result<(CString, Vec<CString>, Vec<CString>), String> {
        let path = c_string(plan.executable.as_os_str().as_bytes())?;
        let arguments = plan
            .arguments
            .iter()
            .map(|argument| c_string(argument.as_bytes()))
            .collect::<Result<_, _>>()?;
        let mut environment: Vec<CString> = std::env::vars_os()
            .filter(|(name, _)| name != HANDOVER_VARIABLE)
            .map(|(name, value)| {
                let mut entry = name.as_bytes().to_vec();
                entry.push(b'=');
                entry.extend_from_slice(value.as_bytes());
                c_string(&entry)
            })
            .collect::<Result<_, _>>()?;
        environment.push(c_string(
            format!("{HANDOVER_VARIABLE}={}", handover.as_raw_fd()).as_bytes(),
        )?);
        Ok((path, arguments, environment))
    };
    let (path, arguments, environment) = match prepared() {
        Ok(prepared) => prepared,
        Err(error) => return format!("invalid restart command: {error}"),
    };
    if let Err(error) = fcntl(handover, FcntlArg::F_SETFD(FdFlag::empty())) {
        return format!("failed to share the handover file descriptor: {error}");
    }
    change_signal_mask(SigmaskHow::SIG_BLOCK, &RESTART_SIGNALS);
    let error = nix::unistd::execve(&path, &arguments, &environment).unwrap_err();
    format!("execve of '{}' failed: {error}", plan.executable.display())
}
