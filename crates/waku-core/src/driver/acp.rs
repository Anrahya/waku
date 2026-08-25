//! Agent Client Protocol transport backed by the official Rust SDK.
//!
//! The SDK owns JSON-RPC framing, request IDs, response routing, cancellation,
//! unknown-method errors, stdio lifetime, and protocol type validation. Waku
//! only adapts typed ACP messages to its provider-neutral [`DriverEvent`]s.

mod renoa_config;

use std::collections::HashMap;
use std::path::Path;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::thread;
use std::time::Duration;

use agent_client_protocol::schema::ProtocolVersion;
use agent_client_protocol::schema::v1::{
    CancelNotification, ClientCapabilities, ContentBlock, ContentChunk, ImageContent,
    Implementation, InitializeRequest, InitializeResponse, LoadSessionRequest, MessageId,
    NewSessionRequest, PermissionOptionKind, PromptRequest, PromptResponse, RequestId,
    RequestPermissionOutcome, RequestPermissionRequest, RequestPermissionResponse,
    ResumeSessionRequest, SelectedPermissionOutcome, SessionConfigKind, SessionConfigOption,
    SessionConfigOptionCategory, SessionConfigSelectOptions, SessionId, SessionModeId,
    SessionModeState, SessionNotification, SessionUpdate, SetSessionConfigOptionRequest,
    SetSessionModeRequest, StopReason, TextContent, ToolCall as AcpToolCall, ToolCallContent,
    ToolCallStatus, ToolCallUpdate, ToolKind,
};
use agent_client_protocol::{
    AcpAgent, AcpAgentConfig, Agent, Client, ConnectionTo, Handled, LineDirection, Responder,
    UntypedMessage,
};
use anyhow::{Context as _, anyhow};
use parking_lot::Mutex;
use serde_json::{Map, Value, json};
use waku_protocol::TurnPrompt;
use waku_protocol::replay::{
    ProviderReplay, ReplayAssistantMessage, ReplayItem, ReplaySegment, ReplayTool,
    ReplayUserMessage,
};

use super::activity;
use crate::driver::{
    DriverControl, DriverEventSender, DriverEventSink, DriverStartOptions, SessionOptions,
};
use crate::model::{
    ActivityKind, DriverEvent, InteractionMode, PermissionOption, ProviderKind,
    ProviderResumeCursor, RuntimeMode, UserInputAnswer, UserInputOption, UserInputQuestion,
};

enum CommandMessage {
    Prompt(TurnPrompt),
    Steer(String),
    Cancel,
    Respond {
        request_id: String,
        option_id: String,
    },
    RespondUserInput {
        request_id: String,
        answers: Vec<UserInputAnswer>,
    },
    Options(SessionOptions),
    Shutdown,
}

pub struct AcpDriver {
    commands: smol::channel::Sender<CommandMessage>,
    replay_gate: smol::channel::Sender<ReplayGate>,
    replay_acknowledged: AtomicBool,
    replay_cancelled: Arc<AtomicBool>,
    supports_steer: bool,
    mode: RuntimeMode,
    interaction_mode: InteractionMode,
    computer_use: Option<super::support::HeadlessComputerUseRuntime>,
}

#[derive(Clone, Copy)]
enum ReplayGate {
    Committed,
    Aborted,
}

/// Per-provider launch details. Everything after process launch is ACP.
struct AcpLaunch {
    args: Vec<String>,
    env: Vec<(String, String)>,
}

fn launch_for(provider: ProviderKind) -> anyhow::Result<AcpLaunch> {
    match provider {
        ProviderKind::Cursor => Ok(AcpLaunch {
            args: vec!["acp".into()],
            env: Vec::new(),
        }),
        ProviderKind::Grok => Ok(AcpLaunch {
            args: vec!["agent".into(), "stdio".into()],
            env: vec![("GROK_OAUTH2_REFERRER".into(), "waku".into())],
        }),
        ProviderKind::Fx => Ok(AcpLaunch {
            args: vec!["acp".into()],
            env: Vec::new(),
        }),
        ProviderKind::Kimi => Ok(AcpLaunch {
            args: vec!["acp".into()],
            env: Vec::new(),
        }),
        ProviderKind::OpenCode => Ok(AcpLaunch {
            args: vec!["acp".into()],
            env: Vec::new(),
        }),
        ProviderKind::Renoa => Ok(AcpLaunch {
            args: vec!["acp".into()],
            env: Vec::new(),
        }),
        _ => Err(anyhow!(
            "{} does not speak the Agent Client Protocol",
            provider.display_name()
        )),
    }
}

impl AcpDriver {
    pub fn start(
        provider: ProviderKind,
        options: DriverStartOptions,
        events: DriverEventSender,
    ) -> anyhow::Result<Self> {
        let DriverStartOptions {
            binary,
            cwd,
            mode,
            interaction_mode,
            model,
            reasoning_effort,
            service_tier: _,
            context_window: _,
            agent_preset: _,
            computer_use_enabled,
            provider_cursor,
        } = options;
        let fork_context = match &provider_cursor {
            Some(ProviderResumeCursor::Cursor { fork_context, .. }) => fork_context.clone(),
            _ => None,
        };
        let resume_session_id = match provider_cursor {
            Some(cursor) if cursor.provider() == provider => {
                let id = cursor.native_id();
                if id.is_empty() {
                    if provider == ProviderKind::Renoa {
                        return Err(anyhow!("cannot resume Renoa without a durable session id"));
                    }
                    None
                } else {
                    Some(id.to_owned())
                }
            }
            Some(cursor) => {
                return Err(anyhow!(
                    "cannot resume {} from a {} cursor",
                    provider.display_name(),
                    cursor.provider().display_name()
                ));
            }
            None => None,
        };

        let launch = launch_for(provider)?;
        let computer_use = (provider == ProviderKind::Grok && computer_use_enabled)
            .then(|| super::support::HeadlessComputerUseRuntime::start(provider, events.clone()))
            .transpose()?;
        let grok_title_home = computer_use
            .as_ref()
            .and_then(super::support::HeadlessComputerUseRuntime::grok_home)
            .map(ToOwned::to_owned);
        let stderr_lines = Arc::new(Mutex::new(Vec::<String>::new()));
        let agent = sdk_agent(
            &binary,
            &cwd,
            launch,
            computer_use.as_ref().map(|runtime| &runtime.config),
            stderr_lines.clone(),
        )?;
        let (commands, command_rx) = smol::channel::unbounded();
        let (replay_gate, replay_gate_rx) = smol::channel::bounded(1);
        let replay_cancelled = Arc::new(AtomicBool::new(false));
        let thread_replay_cancelled = replay_cancelled.clone();
        let provider_name = provider.display_name();
        let thread_events = events.clone();

        thread::Builder::new()
            .name(format!("waku-{}-acp", provider.id()))
            .spawn(move || {
                if let Err(error) = crate::command_env::unblock_sigchld_for_current_thread() {
                    let _ = thread_events.send(DriverEvent::Error(format!(
                        "{provider_name}: failed to normalize the provider signal mask: {error}"
                    )));
                    let _ = thread_events.send(DriverEvent::ProcessExited);
                    return;
                }
                let result = smol::block_on(run_sdk_connection(
                    agent,
                    provider,
                    cwd,
                    mode,
                    interaction_mode,
                    model,
                    reasoning_effort,
                    resume_session_id,
                    fork_context,
                    grok_title_home,
                    command_rx,
                    replay_gate_rx,
                    thread_replay_cancelled,
                    thread_events.clone(),
                ));
                if let Err(error) = result {
                    let stderr = super::support::provider_stderr_error(stderr_lines.lock().clone());
                    let detail = stderr.unwrap_or_else(|| error.to_string());
                    let _ = thread_events
                        .send(DriverEvent::Error(format!("{provider_name}: {detail}")));
                }
                let _ = thread_events.send(DriverEvent::ProcessExited);
            })
            .with_context(|| format!("failed to start {provider_name} ACP runtime"))?;

        Ok(Self {
            commands,
            replay_gate,
            replay_acknowledged: AtomicBool::new(false),
            replay_cancelled,
            supports_steer: !matches!(provider, ProviderKind::Fx | ProviderKind::Renoa),
            mode,
            interaction_mode,
            computer_use,
        })
    }
}

fn sdk_agent(
    binary: &Path,
    cwd: &Path,
    mut launch: AcpLaunch,
    computer_use: Option<&super::support::HeadlessComputerUseConfig>,
    stderr_lines: Arc<Mutex<Vec<String>>>,
) -> anyhow::Result<AcpAgent> {
    let binary = binary
        .to_str()
        .ok_or_else(|| anyhow!("the ACP executable path is not valid UTF-8"))?;
    let cwd = cwd
        .to_str()
        .ok_or_else(|| anyhow!("the ACP working directory is not valid UTF-8"))?;
    let (computer_args, computer_env) =
        super::support::grok_computer_use_launch_configuration(computer_use);
    launch.args.extend(computer_args);
    let mut environment = crate::command_env::shell_environment()
        .into_iter()
        .map(|(name, value)| {
            (
                name.to_string_lossy().into_owned(),
                value.to_string_lossy().into_owned(),
            )
        })
        .collect::<Vec<_>>();
    environment.append(&mut launch.env);
    environment.extend(computer_env);

    // `AcpAgentConfig` deliberately contains only argv and environment. On
    // Unix, `env -C` supplies the session cwd without a shell, preserving
    // exact argument boundaries and the SDK's process-group lifecycle
    // management. Windows has no `env -C`; the binary is launched directly
    // and ACP still carries the session cwd on session/new and session/load.
    let config = if cfg!(unix) {
        let mut args = vec!["-C".to_owned(), cwd.to_owned(), binary.to_owned()];
        args.extend(launch.args);
        AcpAgentConfig::new("/usr/bin/env")
            .args(args)
            .envs(environment)
    } else {
        AcpAgentConfig::new(binary)
            .args(launch.args)
            .envs(environment)
    };
    Ok(AcpAgent::new(config).with_debug(move |line, direction| {
        if direction != LineDirection::Stderr || line.trim().is_empty() {
            return;
        }
        let mut lines = stderr_lines.lock();
        if lines.len() == 128 {
            lines.remove(0);
        }
        lines.push(line.to_owned());
    }))
}

type PermissionResponder = Responder<RequestPermissionResponse>;
type PendingPermissions = Arc<Mutex<HashMap<String, PermissionResponder>>>;

#[derive(Clone, Copy)]
enum AcpUserInputKind {
    Cursor,
    Xai,
}

struct PendingAcpUserInput {
    kind: AcpUserInputKind,
    params: Value,
    responder: Responder<Value>,
}

type PendingAcpUserInputs = Arc<Mutex<HashMap<String, PendingAcpUserInput>>>;

#[derive(Default)]
struct PendingPrompts(Vec<PendingPrompt>);

struct PendingPrompt {
    request_id: RequestId,
    extension_id: Option<String>,
    session_id: String,
}

impl PendingPrompts {
    fn insert(&mut self, request_id: RequestId, extension_id: Option<String>, session_id: String) {
        self.0.push(PendingPrompt {
            request_id,
            extension_id,
            session_id,
        });
    }

    fn is_empty(&self) -> bool {
        self.0.is_empty()
    }

    fn settle_request(&mut self, request_id: &RequestId) -> bool {
        let Some(index) = self
            .0
            .iter()
            .position(|prompt| &prompt.request_id == request_id)
        else {
            return false;
        };
        self.0.remove(index);
        self.0.is_empty()
    }

    fn settle_extension(&mut self, session_id: &str, extension_id: Option<&str>) -> bool {
        let Some(index) = self.0.iter().position(|prompt| {
            prompt.session_id == session_id
                && extension_id
                    .is_none_or(|extension_id| prompt.extension_id.as_deref() == Some(extension_id))
        }) else {
            return false;
        };
        self.0.remove(index);
        self.0.is_empty()
    }
}

type PendingPromptRequests = Arc<Mutex<PendingPrompts>>;

#[allow(clippy::too_many_arguments)]
async fn run_sdk_connection(
    agent: AcpAgent,
    provider: ProviderKind,
    cwd: std::path::PathBuf,
    mode: RuntimeMode,
    interaction_mode: InteractionMode,
    model: Option<String>,
    reasoning_effort: Option<String>,
    resume_session_id: Option<String>,
    fork_context: Option<String>,
    grok_title_home: Option<std::path::PathBuf>,
    commands: smol::channel::Receiver<CommandMessage>,
    replay_gate: smol::channel::Receiver<ReplayGate>,
    replay_cancelled: Arc<AtomicBool>,
    events: DriverEventSender,
) -> agent_client_protocol::Result<()> {
    let suppress_session_updates = Arc::new(AtomicBool::new(false));
    let renoa_load_capture: Arc<Mutex<Option<RenoaReplayCapture>>> = Arc::new(Mutex::new(None));
    let stream_state = Arc::new(Mutex::new(AcpStreamState::default()));
    let pending_permissions: PendingPermissions = Arc::new(Mutex::new(HashMap::new()));
    let pending_user_inputs: PendingAcpUserInputs = Arc::new(Mutex::new(HashMap::new()));
    let prompt_requests = Arc::new(Mutex::new(PendingPrompts::default()));
    let title_refresh = super::title_refresh::NativeTitleRefresh::default();
    let auto_approve = mode != RuntimeMode::Ask;

    Client
        .builder()
        .name("waku")
        .on_receive_notification(
            {
                let events = events.clone();
                let suppress_session_updates = suppress_session_updates.clone();
                let renoa_load_capture = renoa_load_capture.clone();
                let stream_state = stream_state.clone();
                async move |notification: SessionNotification, _connection| {
                    if let Some(capture) = renoa_load_capture.lock().as_mut() {
                        capture.observe(&notification.update);
                    } else if !suppress_session_updates.load(Ordering::Acquire) {
                        handle_session_update(
                            provider,
                            notification,
                            &events,
                            &mut stream_state.lock(),
                        )?;
                    }
                    Ok(())
                }
            },
            agent_client_protocol::on_receive_notification!(),
        )
        .on_receive_notification(
            {
                let events = events.clone();
                let prompt_requests = prompt_requests.clone();
                let grok_title_home = grok_title_home.clone();
                let title_refresh = title_refresh.clone();
                async move |notification: UntypedMessage, _connection| {
                    if notification.method() == "_x.ai/session/prompt_complete" {
                        if let Some(session_id) = finish_xai_prompt_complete(
                            notification.params(),
                            &prompt_requests,
                            &events,
                        ) {
                            start_grok_title_refresh(
                                grok_title_home.as_deref(),
                                &session_id,
                                &title_refresh,
                                events.clone(),
                            );
                        }
                    }
                    Ok(())
                }
            },
            agent_client_protocol::on_receive_notification!(),
        )
        .on_receive_request(
            {
                let events = events.clone();
                let pending_permissions = pending_permissions.clone();
                async move |request: RequestPermissionRequest, responder, _connection| {
                    handle_permission_request(
                        request,
                        responder,
                        auto_approve,
                        &pending_permissions,
                        &events,
                    )
                }
            },
            agent_client_protocol::on_receive_request!(),
        )
        .on_receive_request(
            {
                let events = events.clone();
                let pending = pending_user_inputs.clone();
                async move |request: UntypedMessage, responder, _connection| {
                    let kind = match request.method() {
                        "cursor/ask_question" => AcpUserInputKind::Cursor,
                        "_x.ai/ask_user_question" | "x.ai/ask_user_question" => {
                            AcpUserInputKind::Xai
                        }
                        _ => {
                            return Ok(Handled::No {
                                message: (request, responder),
                                retry: false,
                            });
                        }
                    };
                    let request_id = responder.id().to_string();
                    let params = match kind {
                        AcpUserInputKind::Cursor => request.params().clone(),
                        AcpUserInputKind::Xai => {
                            unwrap_xai_question_params(request.params()).clone()
                        }
                    };
                    let questions = match kind {
                        AcpUserInputKind::Cursor => cursor_user_input_questions(&params),
                        AcpUserInputKind::Xai => xai_user_input_questions(&params),
                    };
                    if questions.is_empty() {
                        responder.respond(cancelled_user_input_response(kind))?;
                        return Ok(Handled::Yes);
                    }
                    pending.lock().insert(
                        request_id.clone(),
                        PendingAcpUserInput {
                            kind,
                            params,
                            responder,
                        },
                    );
                    if events
                        .send(DriverEvent::UserInputRequested {
                            request_id: request_id.clone(),
                            questions,
                        })
                        .is_err()
                        && let Some(pending) = pending.lock().remove(&request_id)
                    {
                        let _ = pending
                            .responder
                            .respond(cancelled_user_input_response(pending.kind));
                    }
                    Ok(Handled::Yes)
                }
            },
            agent_client_protocol::on_receive_request!(),
        )
        .connect_with(agent, async move |connection: ConnectionTo<Agent>| {
            let mut client_capabilities = ClientCapabilities::new().terminal(false);
            if provider == ProviderKind::Cursor {
                // Cursor only exposes its parameterized model controls to
                // clients that opt in. Waku applies the returned config option
                // ids rather than assuming Cursor's private ids stay stable.
                let mut meta = Map::new();
                meta.insert("parameterizedModelPicker".to_owned(), Value::Bool(true));
                client_capabilities = client_capabilities.meta(meta);
            }
            let initialize = connection
                .send_request(
                    InitializeRequest::new(ProtocolVersion::V1)
                        .client_capabilities(client_capabilities)
                        .client_info(Implementation::new("waku", env!("CARGO_PKG_VERSION"))),
                )
                .block_task()
                .await?;
            let (session_id, modes, config_options, replay) = establish_session(
                &connection,
                provider,
                &initialize,
                resume_session_id.as_deref(),
                &cwd,
                &suppress_session_updates,
                &renoa_load_capture,
            )
            .await?;

            if let Some(mode_id) = desired_mode(provider, modes.as_ref(), mode, interaction_mode) {
                // Mode selection is opportunistic: an agent can advertise a
                // mode but reject a later transition without invalidating the
                // session itself.
                let _ = connection
                    .send_request(SetSessionModeRequest::new(session_id.clone(), mode_id))
                    .block_task()
                    .await;
            }
            let native_session_id = session_id.to_string();
            // The complete replay transaction lands in bounded fragments
            // before the session marker. Renoa prompt handling remains gated
            // below until the daemon acknowledges durable reconciliation, and
            // no individual wire frame grows with the transcript.
            let replay_events = replay
                .as_ref()
                .map(|loaded| prepare_replay_events(&loaded.replay))
                .transpose()
                .map_err(replay_transport_error)?;
            if let Some(replay_events) = replay_events {
                for event in replay_events {
                    events.send(event).map_err(|_| {
                        replay_transport_error(ReplayTransportError::ConsumerClosed)
                    })?;
                }
            }
            let connected = DriverEvent::Connected {
                provider_cursor: Some(ProviderResumeCursor::from_session_id(
                    provider,
                    native_session_id.clone(),
                )),
            };
            if replay.is_some() {
                events
                    .send(connected)
                    .map_err(|_| replay_transport_error(ReplayTransportError::ConsumerClosed))?;
                match replay_gate.recv().await {
                    Ok(ReplayGate::Committed) if !replay_cancelled.load(Ordering::Acquire) => {}
                    Ok(ReplayGate::Committed | ReplayGate::Aborted) => return Ok(()),
                    Err(_) => {
                        return Err(replay_transport_error(
                            ReplayTransportError::AcknowledgementClosed,
                        ));
                    }
                }
            } else {
                let _ = events.send(connected);
            }

            if let Some(usage) = replay.as_ref().and_then(|loaded| loaded.usage) {
                events
                    .send(DriverEvent::UsageUpdated {
                        context_tokens: usage.context_tokens,
                        context_window: usage.context_window,
                    })
                    .map_err(|_| replay_transport_error(ReplayTransportError::ConsumerClosed))?;
            }

            let mut current_model = model;
            let mut current_reasoning_effort = reasoning_effort;
            let mut current_config_options = config_options.unwrap_or_default();
            if provider == ProviderKind::Renoa {
                current_config_options = renoa_config::apply(
                    &connection,
                    &session_id,
                    current_config_options,
                    current_model.as_deref(),
                    current_reasoning_effort.as_deref(),
                )
                .await
                .map_err(renoa_config_error)?;
            } else {
                apply_model(
                    &connection,
                    provider,
                    &session_id,
                    Some(&current_config_options),
                    current_model.as_deref(),
                    current_reasoning_effort.as_deref(),
                    &events,
                )
                .await;
            }
            let mut fork_context = fork_context;

            while let Ok(command) = commands.recv().await {
                match command {
                    CommandMessage::Prompt(turn) => {
                        let text = fork_context
                            .take()
                            .map(|context| {
                                crate::cursor_session::prompt_with_fork_context(
                                    &context,
                                    &turn.prompt,
                                )
                            })
                            .unwrap_or(turn.prompt);
                        let _ = events.send(DriverEvent::TurnStarted);
                        if let Err(error) = send_prompt(
                            &connection,
                            &session_id,
                            text,
                            prompt_extension_id(provider, Some(turn.id)),
                            &prompt_requests,
                            &events,
                            provider,
                            &native_session_id,
                            grok_title_home.clone(),
                            title_refresh.clone(),
                            stream_state.clone(),
                        ) {
                            let _ = events.send(DriverEvent::Error(error.to_string()));
                            let _ = events.send(DriverEvent::TurnFinished {
                                success: false,
                                summary: None,
                            });
                        }
                    }
                    CommandMessage::Steer(text) => {
                        if provider == ProviderKind::Renoa {
                            let _ = events.send(DriverEvent::SteerRejected {
                                message: text,
                                reason: format!(
                                    "{} does not support steering.",
                                    provider.display_name()
                                ),
                            });
                            continue;
                        }
                        if prompt_requests.lock().is_empty() {
                            let _ = events.send(DriverEvent::SteerRejected {
                                message: text,
                                reason: format!(
                                    "{} has no active turn to steer.",
                                    provider.display_name()
                                ),
                            });
                            continue;
                        }
                        match send_prompt(
                            &connection,
                            &session_id,
                            text.clone(),
                            prompt_extension_id(provider, None),
                            &prompt_requests,
                            &events,
                            provider,
                            &native_session_id,
                            grok_title_home.clone(),
                            title_refresh.clone(),
                            stream_state.clone(),
                        ) {
                            Ok(()) => {
                                let _ = events.send(DriverEvent::SteerAccepted { message: text });
                            }
                            Err(error) => {
                                let _ = events.send(DriverEvent::SteerRejected {
                                    message: text,
                                    reason: error.to_string(),
                                });
                            }
                        }
                    }
                    CommandMessage::Cancel => {
                        let _ = connection
                            .send_notification(CancelNotification::new(session_id.clone()));
                        cancel_pending_permissions(&pending_permissions);
                        cancel_pending_user_inputs(&pending_user_inputs);
                    }
                    CommandMessage::Respond {
                        request_id,
                        option_id,
                    } => {
                        if let Some(responder) = pending_permissions.lock().remove(&request_id) {
                            let _ = responder.respond(RequestPermissionResponse::new(
                                RequestPermissionOutcome::Selected(SelectedPermissionOutcome::new(
                                    option_id,
                                )),
                            ));
                        }
                    }
                    CommandMessage::RespondUserInput {
                        request_id,
                        answers,
                    } => {
                        if let Some(pending) = pending_user_inputs.lock().remove(&request_id) {
                            let response = match pending.kind {
                                AcpUserInputKind::Cursor => {
                                    cursor_user_input_response(&pending.params, &answers)
                                }
                                AcpUserInputKind::Xai => {
                                    xai_user_input_response(&pending.params, &answers)
                                }
                            };
                            let _ = pending.responder.respond(response);
                        }
                    }
                    CommandMessage::Options(options) => {
                        if provider == ProviderKind::Renoa {
                            let model_changed = options.model != current_model;
                            let reasoning_changed =
                                options.reasoning_effort != current_reasoning_effort;
                            if model_changed || reasoning_changed {
                                let model_update = if model_changed {
                                    options.model.as_deref()
                                } else {
                                    None
                                };
                                let reasoning_update = if model_changed || reasoning_changed {
                                    options.reasoning_effort.as_deref()
                                } else {
                                    None
                                };
                                current_config_options = renoa_config::apply(
                                    &connection,
                                    &session_id,
                                    current_config_options,
                                    model_update,
                                    reasoning_update,
                                )
                                .await
                                .map_err(renoa_config_error)?;
                                current_model = options.model;
                                current_reasoning_effort = options.reasoning_effort;
                            }
                        } else if options.model != current_model {
                            current_model = options.model;
                            apply_model(
                                &connection,
                                provider,
                                &session_id,
                                Some(&current_config_options),
                                current_model.as_deref(),
                                options.reasoning_effort.as_deref(),
                                &events,
                            )
                            .await;
                        }
                    }
                    CommandMessage::Shutdown => break,
                }
            }
            cancel_pending_permissions(&pending_permissions);
            cancel_pending_user_inputs(&pending_user_inputs);
            Ok(())
        })
        .await
}

#[derive(Debug)]
enum ReplayTransportError {
    Codec(waku_protocol::replay::ReplayCodecError),
    WireEncode(anyhow::Error),
    WireSerialization(serde_json::Error),
    Oversize { size: usize, maximum: usize },
    ConsumerClosed,
    AcknowledgementClosed,
}

impl std::fmt::Display for ReplayTransportError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Codec(error) => write!(formatter, "{error}"),
            Self::WireSerialization(error) => {
                write!(formatter, "could not serialize replay wire event: {error}")
            }
            Self::WireEncode(error) => write!(formatter, "could not encode replay event: {error}"),
            Self::Oversize { size, maximum } => write!(
                formatter,
                "serialized replay event is {size} bytes, exceeding the {maximum}-byte wire bound"
            ),
            Self::ConsumerClosed => write!(formatter, "replay consumer closed before commit"),
            Self::AcknowledgementClosed => write!(
                formatter,
                "replay consumer closed before acknowledging durable persistence"
            ),
        }
    }
}

impl std::error::Error for ReplayTransportError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Codec(error) => Some(error),
            Self::WireEncode(error) => Some(error.as_ref()),
            Self::WireSerialization(error) => Some(error),
            Self::Oversize { .. } | Self::ConsumerClosed | Self::AcknowledgementClosed => None,
        }
    }
}

fn prepare_replay_events(
    replay: &ProviderReplay,
) -> Result<Vec<DriverEvent>, ReplayTransportError> {
    let maximum = waku_protocol::MAX_WIRE_MESSAGE_BYTES;
    let fragments = replay
        .encode_fragments(
            uuid::Uuid::new_v4(),
            waku_protocol::SESSION_REPLAY_FRAGMENT_BYTES,
        )
        .map_err(ReplayTransportError::Codec)?;
    let mut events = Vec::with_capacity(fragments.len());
    for fragment in fragments {
        let event = DriverEvent::SessionReplayFragment {
            replay_id: fragment.replay_id,
            index: fragment.index,
            total: fragment.total,
            json: fragment.json,
        };
        let wire = waku_protocol::event_to_wire(event.clone())
            .map_err(ReplayTransportError::WireEncode)?;
        // Measure the complete daemon-to-desktop envelope with maximum-width
        // sequence metadata, not merely its nested driver event.
        let message = waku_protocol::ServerMessage::Event(waku_protocol::SequencedEvent {
            session_id: uuid::Uuid::nil(),
            runtime_id: uuid::Uuid::nil(),
            epoch: uuid::Uuid::nil(),
            sequence: u64::MAX,
            event: wire,
        });
        let size = serde_json::to_vec(&message)
            .map_err(ReplayTransportError::WireSerialization)?
            .len();
        if size > maximum {
            return Err(ReplayTransportError::Oversize { size, maximum });
        }
        events.push(event);
    }
    Ok(events)
}

fn replay_transport_error(error: ReplayTransportError) -> agent_client_protocol::Error {
    let mut acp_error = agent_client_protocol::Error::internal_error();
    acp_error.message = format!("could not commit Renoa replay: {error}");
    acp_error
}

fn renoa_config_error(error: renoa_config::ConfigError) -> agent_client_protocol::Error {
    let mut acp_error = agent_client_protocol::Error::internal_error();
    acp_error.message = format!("could not configure Renoa session: {error}");
    acp_error
}

async fn establish_session(
    connection: &ConnectionTo<Agent>,
    provider: ProviderKind,
    initialize: &InitializeResponse,
    resume_session_id: Option<&str>,
    cwd: &Path,
    suppress_session_updates: &AtomicBool,
    renoa_load_capture: &Mutex<Option<RenoaReplayCapture>>,
) -> agent_client_protocol::Result<(
    SessionId,
    Option<SessionModeState>,
    Option<Vec<SessionConfigOption>>,
    Option<RenoaLoadedReplay>,
)> {
    if let Some(existing) = resume_session_id {
        if provider != ProviderKind::Renoa
            && initialize
                .agent_capabilities
                .session_capabilities
                .resume
                .is_some()
            && let Ok(response) = connection
                .send_request(ResumeSessionRequest::new(existing.to_owned(), cwd))
                .block_task()
                .await
        {
            return Ok((
                SessionId::new(existing.to_owned()),
                response.modes,
                response.config_options,
                None,
            ));
        }

        if initialize.agent_capabilities.load_session {
            // Renoa replays its complete durable history through ordinary
            // session updates before the load response. Capture instead of
            // suppressing, and commit only on success; every other provider
            // keeps discarding load-time updates.
            let capture_renoa = provider == ProviderKind::Renoa;
            if capture_renoa {
                renoa_load_capture
                    .lock()
                    .replace(RenoaReplayCapture::default());
            } else {
                suppress_session_updates.store(true, Ordering::Release);
            }
            let response = connection
                .send_request(LoadSessionRequest::new(existing.to_owned(), cwd))
                .block_task()
                .await;
            let captured = if capture_renoa {
                renoa_load_capture.lock().take()
            } else {
                suppress_session_updates.store(false, Ordering::Release);
                None
            };
            match response {
                Ok(response) => {
                    let replay = match captured {
                        Some(capture) => Some(capture.finalize().map_err(|error| {
                            let reason = error.to_string();
                            let mut load_error = agent_client_protocol::Error::internal_error();
                            load_error.message =
                                format!("failed to load Renoa session {existing}: {reason}");
                            load_error
                        })?),
                        None => None,
                    };
                    return Ok((
                        SessionId::new(existing.to_owned()),
                        response.modes,
                        response.config_options,
                        replay,
                    ));
                }
                Err(error) if acp_load_fails_closed(provider) => {
                    return Err(renoa_load_error(existing, Some(error)));
                }
                Err(_) => {}
            }
        } else if acp_load_fails_closed(provider) {
            return Err(renoa_load_error(existing, None));
        }
    }

    let response = connection
        .send_request(NewSessionRequest::new(cwd))
        .block_task()
        .await?;
    Ok((
        response.session_id,
        response.modes,
        response.config_options,
        None,
    ))
}

fn acp_load_fails_closed(provider: ProviderKind) -> bool {
    provider == ProviderKind::Renoa
}

fn renoa_load_error(
    session_id: &str,
    error: Option<agent_client_protocol::Error>,
) -> agent_client_protocol::Error {
    match error {
        Some(mut error) => {
            error.message = format!(
                "failed to load Renoa session {session_id}: {}",
                error.message
            );
            error
        }
        None => {
            let mut error = agent_client_protocol::Error::internal_error();
            error.message = format!(
                "failed to load Renoa session {session_id}: the agent does not advertise session/load"
            );
            error
        }
    }
}

fn prompt_extension_id(provider: ProviderKind, turn_id: Option<uuid::Uuid>) -> Option<String> {
    match provider {
        ProviderKind::Grok => Some(format!("waku-{}", uuid::Uuid::new_v4())),
        ProviderKind::Renoa => turn_id.map(|id| id.to_string()),
        _ => None,
    }
}

fn prompt_request_meta(identity: Option<&str>) -> Option<serde_json::Map<String, Value>> {
    let identity = identity?;
    let mut meta = serde_json::Map::new();
    meta.insert("promptId".into(), Value::String(identity.to_owned()));
    meta.insert("requestId".into(), Value::String(identity.to_owned()));
    Some(meta)
}

fn acp_prompt_request(
    session_id: SessionId,
    text: String,
    extension_id: Option<&str>,
) -> PromptRequest {
    let mut request =
        PromptRequest::new(session_id, vec![ContentBlock::Text(TextContent::new(text))]);
    if let Some(meta) = prompt_request_meta(extension_id) {
        request = request.meta(meta);
    }
    request
}

fn desired_mode(
    provider: ProviderKind,
    modes: Option<&SessionModeState>,
    mode: RuntimeMode,
    interaction_mode: InteractionMode,
) -> Option<SessionModeId> {
    let modes = modes?;
    let desired = if provider == ProviderKind::Fx {
        if mode == RuntimeMode::Ask {
            "ask"
        } else {
            "code"
        }
    } else {
        if interaction_mode != InteractionMode::Plan && mode != RuntimeMode::Plan {
            return None;
        }
        "plan"
    };
    let desired = modes
        .available_modes
        .iter()
        .find(|mode| mode.id.to_string().eq_ignore_ascii_case(desired))?
        .id
        .clone();
    (modes.current_mode_id != desired).then_some(desired)
}

/// Which session config option carries reasoning effort. ACP leaves the id to
/// the agent: Kimi Code exposes it as its `thinking` level, while the other
/// agents Waku drives keep it on `mode`.
fn reasoning_effort_config_id(provider: ProviderKind) -> &'static str {
    match provider {
        ProviderKind::Kimi => "thinking",
        _ => "mode",
    }
}

#[derive(Debug, Eq, PartialEq)]
struct CursorModelSelection {
    value: String,
    suffix: String,
}

fn session_config_select_values(option: &SessionConfigOption) -> Vec<&str> {
    let SessionConfigKind::Select(select) = &option.kind else {
        return Vec::new();
    };
    match &select.options {
        SessionConfigSelectOptions::Ungrouped(options) => options
            .iter()
            .map(|option| option.value.0.as_ref())
            .collect(),
        SessionConfigSelectOptions::Grouped(groups) => groups
            .iter()
            .flat_map(|group| group.options.iter())
            .map(|option| option.value.0.as_ref())
            .collect(),
        _ => Vec::new(),
    }
}

fn cursor_model_aliases(requested: &str) -> Vec<String> {
    let mut aliases = vec![requested.to_owned()];
    if let Some(alias) = requested.strip_prefix("cursor-") {
        aliases.push(alias.to_owned());
    }

    // Cursor's CLI spells a few aliases as `claude-4.6-sonnet-*`, while ACP
    // advertises the same family as `claude-sonnet-4-6`.
    if let Some(rest) = requested.strip_prefix("claude-")
        && let Some((version, family_and_suffix)) = rest.split_once('-')
    {
        let (family, suffix) = family_and_suffix
            .split_once('-')
            .map_or((family_and_suffix, ""), |(family, suffix)| (family, suffix));
        if matches!(family, "haiku" | "opus" | "sonnet") {
            let mut alias = format!("claude-{family}-{}", version.replace('.', "-"));
            if !suffix.is_empty() {
                alias.push('-');
                alias.push_str(suffix);
            }
            if !aliases.contains(&alias) {
                aliases.push(alias);
            }
        }
    }
    aliases
}

/// Resolves Cursor's CLI-facing model aliases against the base values its
/// parameterized ACP picker advertises. The unconsumed suffix carries values
/// such as `thinking`, `xhigh`, and `fast` for the dynamic options returned
/// after the base model changes.
fn cursor_model_selection(
    option: &SessionConfigOption,
    requested: &str,
) -> Option<CursorModelSelection> {
    let values = session_config_select_values(option);
    let aliases = cursor_model_aliases(requested);

    for alias in &aliases {
        if let Some(value) = values.iter().find(|value| **value == alias) {
            return Some(CursorModelSelection {
                value: (*value).to_owned(),
                suffix: String::new(),
            });
        }
    }
    if requested == "auto"
        && let Some(value) = values.iter().find(|value| **value == "default")
    {
        return Some(CursorModelSelection {
            value: (*value).to_owned(),
            suffix: String::new(),
        });
    }

    aliases
        .iter()
        .flat_map(|alias| {
            values.iter().filter_map(move |value| {
                alias
                    .strip_prefix(*value)
                    .and_then(|suffix| suffix.strip_prefix('-'))
                    .map(|suffix| CursorModelSelection {
                        value: (*value).to_owned(),
                        suffix: suffix.to_owned(),
                    })
            })
        })
        .max_by_key(|selection| selection.value.len())
}

fn cursor_suffix_has(suffix: &str, value: &str) -> bool {
    suffix.split('-').any(|part| part == value)
}

fn cursor_desired_select_value(
    option: &SessionConfigOption,
    selection: &CursorModelSelection,
    reasoning_effort: Option<&str>,
) -> Option<String> {
    let values = session_config_select_values(option);
    match option.category.as_ref()? {
        SessionConfigOptionCategory::ThoughtLevel => {
            if let Some(effort) = reasoning_effort
                && values.contains(&effort)
            {
                return Some(effort.to_owned());
            }
            if selection.suffix.contains("extra-high") && values.contains(&"xhigh") {
                return Some("xhigh".to_owned());
            }
            values
                .iter()
                .find(|value| cursor_suffix_has(&selection.suffix, value))
                .map(|value| (*value).to_owned())
        }
        SessionConfigOptionCategory::ModelConfig => {
            let id = option.id.to_string().to_ascii_lowercase();
            let enabled = match id.as_str() {
                "fast" => cursor_suffix_has(&selection.suffix, "fast"),
                "thinking" => cursor_suffix_has(&selection.suffix, "thinking"),
                _ => return None,
            };
            let value = if enabled { "true" } else { "false" };
            values.contains(&value).then(|| value.to_owned())
        }
        _ => None,
    }
}

fn session_config_current_value(option: &SessionConfigOption) -> Option<&str> {
    let SessionConfigKind::Select(select) = &option.kind else {
        return None;
    };
    Some(select.current_value.0.as_ref())
}

async fn apply_cursor_variant_configs(
    connection: &ConnectionTo<Agent>,
    session_id: &SessionId,
    mut options: Vec<SessionConfigOption>,
    selection: &CursorModelSelection,
    reasoning_effort: Option<&str>,
) -> agent_client_protocol::Result<()> {
    // Thinking can reveal a thought-level option, so apply it first and use
    // each response's refreshed option set for the next selection.
    for target in ["thinking", "thought_level", "fast"] {
        let Some(option) = options.iter().find(|option| match target {
            "thinking" => {
                option.category == Some(SessionConfigOptionCategory::ModelConfig)
                    && option.id.to_string().eq_ignore_ascii_case("thinking")
            }
            "thought_level" => option.category == Some(SessionConfigOptionCategory::ThoughtLevel),
            "fast" => {
                option.category == Some(SessionConfigOptionCategory::ModelConfig)
                    && option.id.to_string().eq_ignore_ascii_case("fast")
            }
            _ => false,
        }) else {
            continue;
        };
        let Some(value) = cursor_desired_select_value(option, selection, reasoning_effort) else {
            continue;
        };
        if session_config_current_value(option) == Some(value.as_str()) {
            continue;
        }
        let config_id = option.id.clone();
        options = connection
            .send_request(SetSessionConfigOptionRequest::new(
                session_id.clone(),
                config_id,
                value.as_str(),
            ))
            .block_task()
            .await?
            .config_options;
    }
    Ok(())
}

fn find_config_option(
    config_options: &[SessionConfigOption],
    category: SessionConfigOptionCategory,
) -> Option<&SessionConfigOption> {
    config_options
        .iter()
        .find(|option| option.category.as_ref() == Some(&category))
}

fn fx_model_option(config_options: &[SessionConfigOption]) -> Option<&SessionConfigOption> {
    config_options.iter().find(|option| {
        option.category == Some(SessionConfigOptionCategory::Model)
            && option.id.to_string().eq_ignore_ascii_case("model")
    })
}

fn fx_model_provider_switch<'a>(
    config_options: &'a [SessionConfigOption],
    model: &str,
) -> Option<(&'a SessionConfigOption, &'static str)> {
    if fx_model_option(config_options)
        .is_some_and(|option| session_config_select_values(option).contains(&model))
    {
        return None;
    }
    // Fx scopes model options to the selected account route. AI Gateway IDs
    // are provider/model pairs, while subscription IDs are flat. Selecting the
    // Gateway route returns a refreshed model option that contains these IDs.
    if !model.contains('/') {
        return None;
    }
    let provider = config_options.iter().find(|option| {
        option.category == Some(SessionConfigOptionCategory::Model)
            && option.id.to_string().eq_ignore_ascii_case("provider")
    })?;
    (session_config_current_value(provider) != Some("gateway")
        && session_config_select_values(provider).contains(&"gateway"))
    .then_some((provider, "gateway"))
}

async fn apply_model(
    connection: &ConnectionTo<Agent>,
    provider: ProviderKind,
    session_id: &SessionId,
    config_options: Option<&[SessionConfigOption]>,
    model: Option<&str>,
    reasoning_effort: Option<&str>,
    events: &DriverEventSender,
) {
    let Some(model) = model else {
        return;
    };
    let cursor_model_option = (provider == ProviderKind::Cursor)
        .then_some(config_options)
        .flatten()
        .and_then(|options| find_config_option(options, SessionConfigOptionCategory::Model));
    if let Some(option) = cursor_model_option
        && let Some(selection) = cursor_model_selection(option, model)
    {
        match connection
            .send_request(SetSessionConfigOptionRequest::new(
                session_id.clone(),
                option.id.clone(),
                selection.value.as_str(),
            ))
            .block_task()
            .await
        {
            Ok(response) => {
                if let Err(error) = apply_cursor_variant_configs(
                    connection,
                    session_id,
                    response.config_options,
                    &selection,
                    reasoning_effort,
                )
                .await
                {
                    let _ = events.send(DriverEvent::Error(tr!(
                        "errors.select_model",
                        error = error
                    )));
                }
            }
            Err(error) => {
                let _ = events.send(DriverEvent::Error(tr!(
                    "errors.select_model",
                    error = error
                )));
            }
        }
        return;
    }

    if provider == ProviderKind::Fx {
        let mut options = config_options.unwrap_or_default().to_vec();
        if let Some((provider_option, value)) = fx_model_provider_switch(&options, model) {
            let config_id = provider_option.id.clone();
            match connection
                .send_request(SetSessionConfigOptionRequest::new(
                    session_id.clone(),
                    config_id,
                    value,
                ))
                .block_task()
                .await
            {
                Ok(response) => options = response.config_options,
                Err(error) => {
                    let _ = events.send(DriverEvent::Error(tr!(
                        "errors.select_model",
                        error = error
                    )));
                    return;
                }
            }
        }
        let Some(option) = fx_model_option(&options) else {
            let _ = events.send(DriverEvent::Error(tr!(
                "errors.select_model",
                error = "Fx did not advertise its model configuration"
            )));
            return;
        };
        if !session_config_select_values(option).contains(&model) {
            let _ = events.send(DriverEvent::Error(tr!(
                "errors.select_model",
                error = format!("Fx did not advertise model {model}")
            )));
            return;
        }
        if let Err(error) = connection
            .send_request(SetSessionConfigOptionRequest::new(
                session_id.clone(),
                option.id.clone(),
                model,
            ))
            .block_task()
            .await
        {
            let _ = events.send(DriverEvent::Error(tr!(
                "errors.select_model",
                error = error
            )));
        }
        return;
    }

    // Grok, Kimi, OpenCode, and Cursor agents that do not advertise a model
    // config option retain the legacy request unchanged. Fx intentionally
    // stays on session/set_config_option, its documented model API.
    let request = match UntypedMessage::new(
        "session/set_model",
        json!({"sessionId": session_id, "modelId": model}),
    ) {
        Ok(request) => request,
        Err(error) => {
            let _ = events.send(DriverEvent::Error(tr!(
                "errors.select_model",
                error = error
            )));
            return;
        }
    };
    if let Err(error) = connection.send_request(request).block_task().await {
        let _ = events.send(DriverEvent::Error(tr!(
            "errors.select_model",
            error = error
        )));
        return;
    }
    if let Some(effort) = reasoning_effort {
        // Reasoning effort is an optional config extension and is deliberately
        // non-fatal when an agent does not expose it.
        let _ = connection
            .send_request(SetSessionConfigOptionRequest::new(
                session_id.clone(),
                reasoning_effort_config_id(provider),
                effort,
            ))
            .block_task()
            .await;
    }
}

#[allow(clippy::too_many_arguments)]
fn send_prompt(
    connection: &ConnectionTo<Agent>,
    session_id: &SessionId,
    text: String,
    extension_id: Option<String>,
    prompt_requests: &PendingPromptRequests,
    events: &DriverEventSender,
    provider: ProviderKind,
    native_session_id: &str,
    grok_title_home: Option<std::path::PathBuf>,
    title_refresh: super::title_refresh::NativeTitleRefresh,
    stream_state: Arc<Mutex<AcpStreamState>>,
) -> agent_client_protocol::Result<()> {
    stream_state.lock().produced_content = false;
    // Read before the turn runs, so the failure lookup cannot mistake an
    // earlier turn's record for this one's.
    let wire_offset = (provider == ProviderKind::Kimi)
        .then(|| crate::kimi_session::wire_offset(native_session_id));
    let request = acp_prompt_request(session_id.clone(), text, extension_id.as_deref());
    let sent = connection.send_request(request);
    let request_id = sent.id().clone();
    prompt_requests.lock().insert(
        request_id.clone(),
        extension_id,
        native_session_id.to_owned(),
    );
    let callback_request_id = request_id.clone();
    let callback_requests = prompt_requests.clone();
    let callback_events = events.clone();
    let native_session_id = native_session_id.to_owned();
    let registered = sent.on_receiving_result(async move |result| {
        if settle_prompt_request(&callback_requests, &callback_request_id) {
            // Only an empty turn pays for this lookup, so a healthy turn never
            // waits on Kimi's records.
            let native_failure = wire_offset
                .filter(|_| !stream_state.lock().produced_content)
                .and_then(|offset| crate::kimi_session::turn_failure(&native_session_id, offset));
            let success = finish_prompt(result, native_failure, &callback_events);
            if provider == ProviderKind::Grok && success {
                start_grok_title_refresh(
                    grok_title_home.as_deref(),
                    &native_session_id,
                    &title_refresh,
                    callback_events,
                );
            }
        }
        Ok(())
    });
    if registered.is_err() {
        prompt_requests.lock().settle_request(&request_id);
    }
    registered
}

fn settle_prompt_request(prompt_requests: &Mutex<PendingPrompts>, request_id: &RequestId) -> bool {
    prompt_requests.lock().settle_request(request_id)
}

fn finish_xai_prompt_complete(
    params: &Value,
    prompt_requests: &Mutex<PendingPrompts>,
    events: &DriverEventSender,
) -> Option<String> {
    let Some(session_id) = params.get("sessionId").and_then(Value::as_str) else {
        return None;
    };
    let prompt_id = params.get("promptId").and_then(Value::as_str);
    if !prompt_requests
        .lock()
        .settle_extension(session_id, prompt_id)
    {
        return None;
    }

    let stop_reason = match params.get("stopReason").and_then(Value::as_str) {
        Some("cancelled") => StopReason::Cancelled,
        Some("max_tokens") => StopReason::MaxTokens,
        Some("max_turn_requests") => StopReason::MaxTurnRequests,
        Some("refusal") => StopReason::Refusal,
        _ => StopReason::EndTurn,
    };
    finish_prompt(Ok(PromptResponse::new(stop_reason)), None, events).then(|| session_id.to_owned())
}

fn start_grok_title_refresh(
    grok_title_home: Option<&Path>,
    native_session_id: &str,
    title_refresh: &super::title_refresh::NativeTitleRefresh,
    events: DriverEventSender,
) {
    let grok_title_home = grok_title_home.map(ToOwned::to_owned);
    let native_session_id = native_session_id.to_owned();
    title_refresh.start(
        "waku-grok-title",
        vec![
            Duration::ZERO,
            Duration::from_millis(250),
            Duration::from_millis(750),
            Duration::from_millis(1_500),
            Duration::from_secs(3),
            Duration::from_secs(5),
            Duration::from_millis(7_500),
            Duration::from_secs(10),
        ],
        events,
        move || match grok_title_home.as_deref() {
            Some(home) => crate::grok_session::generated_title_in(home, &native_session_id),
            None => crate::grok_session::generated_title(&native_session_id),
        },
    );
}

fn finish_prompt(
    result: agent_client_protocol::Result<PromptResponse>,
    native_failure: Option<String>,
    events: &impl DriverEventSink,
) -> bool {
    let response = match result {
        Ok(response) => response,
        Err(error) => {
            let _ = events.send(DriverEvent::Error(error.to_string()));
            let _ = events.send(DriverEvent::TurnFinished {
                success: false,
                summary: None,
            });
            return false;
        }
    };
    // An agent can end a turn cleanly and still have failed upstream. Where
    // that failure is recoverable from the provider's own records, it outranks
    // the protocol's verdict: reporting success here would show the user an
    // empty answer and no reason for it.
    if let Some(failure) = native_failure {
        let _ = events.send(DriverEvent::Error(failure));
        let _ = events.send(DriverEvent::TurnFinished {
            success: false,
            summary: None,
        });
        return false;
    }
    let (success, summary) = match response.stop_reason {
        StopReason::EndTurn | StopReason::Cancelled => (true, None),
        StopReason::MaxTokens => (false, Some(tr!("session.agent_ran_out_of_context"))),
        StopReason::Refusal => (false, Some(tr!("session.agent_declined_turn"))),
        StopReason::MaxTurnRequests => (
            false,
            Some(tr!(
                "session.agent_stopped_reason",
                reason = "max_turn_requests"
            )),
        ),
        _ => (
            false,
            Some(tr!("session.agent_stopped_reason", reason = "unknown")),
        ),
    };
    let _ = events.send(DriverEvent::TurnFinished { success, summary });
    success
}

fn cancel_pending_permissions(pending: &PendingPermissions) {
    for (_, responder) in pending.lock().drain() {
        let _ = responder.respond(RequestPermissionResponse::new(
            RequestPermissionOutcome::Cancelled,
        ));
    }
}

fn cancel_pending_user_inputs(pending: &PendingAcpUserInputs) {
    for (_, pending) in pending.lock().drain() {
        let _ = pending
            .responder
            .respond(cancelled_user_input_response(pending.kind));
    }
}

fn cancelled_user_input_response(kind: AcpUserInputKind) -> Value {
    match kind {
        AcpUserInputKind::Cursor => json!({"answers": {}}),
        AcpUserInputKind::Xai => json!({"outcome": "cancelled"}),
    }
}

fn unwrap_xai_question_params(params: &Value) -> &Value {
    if matches!(
        params.get("method").and_then(Value::as_str),
        Some("x.ai/ask_user_question" | "_x.ai/ask_user_question")
    ) {
        params.get("params").unwrap_or(params)
    } else {
        params
    }
}

fn cursor_user_input_questions(params: &Value) -> Vec<UserInputQuestion> {
    params
        .get("questions")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
        .filter_map(|question| {
            let text = question.get("prompt").and_then(Value::as_str)?.trim();
            if text.is_empty() {
                return None;
            }
            let mut options = question
                .get("options")
                .and_then(Value::as_array)
                .into_iter()
                .flatten()
                .filter_map(|option| {
                    let label = option.get("label").and_then(Value::as_str)?.trim();
                    (!label.is_empty()).then(|| UserInputOption {
                        label: label.to_owned(),
                        description: Some(label.to_owned()),
                    })
                })
                .collect::<Vec<_>>();
            if options.is_empty() {
                options.push(UserInputOption {
                    label: "OK".into(),
                    description: Some("Continue".into()),
                });
            }
            Some(UserInputQuestion {
                id: question
                    .get("id")
                    .and_then(Value::as_str)
                    .filter(|id| !id.is_empty())
                    .unwrap_or(text)
                    .to_owned(),
                header: "Question".into(),
                question: text.to_owned(),
                options,
                multi_select: question
                    .get("allowMultiple")
                    .and_then(Value::as_bool)
                    .unwrap_or(false),
            })
        })
        .collect()
}

fn cursor_user_input_response(params: &Value, submitted: &[UserInputAnswer]) -> Value {
    let mut answers = serde_json::Map::new();
    for question in params
        .get("questions")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
    {
        let Some(id) = question.get("id").and_then(Value::as_str) else {
            continue;
        };
        let values = submitted
            .iter()
            .find(|answer| answer.question_id == id)
            .map(|answer| answer.answers.as_slice())
            .unwrap_or_default();
        let value = if question
            .get("allowMultiple")
            .and_then(Value::as_bool)
            .unwrap_or(false)
        {
            json!(values)
        } else {
            values
                .first()
                .map_or(Value::String(String::new()), |value| json!(value))
        };
        answers.insert(id.to_owned(), value);
    }
    json!({"answers": answers})
}

fn xai_user_input_questions(params: &Value) -> Vec<UserInputQuestion> {
    params
        .get("questions")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
        .enumerate()
        .filter_map(|(index, question)| {
            let text = question.get("question").and_then(Value::as_str)?.trim();
            if text.is_empty() {
                return None;
            }
            let mut options = question
                .get("options")
                .and_then(Value::as_array)
                .into_iter()
                .flatten()
                .filter_map(|option| {
                    let label = option.get("label").and_then(Value::as_str)?.trim();
                    (!label.is_empty()).then(|| UserInputOption {
                        label: label.to_owned(),
                        description: option
                            .get("description")
                            .and_then(Value::as_str)
                            .map(str::trim)
                            .filter(|description| !description.is_empty())
                            .map(str::to_owned),
                    })
                })
                .collect::<Vec<_>>();
            if options.is_empty() {
                options.push(UserInputOption {
                    label: "OK".into(),
                    description: Some("Continue".into()),
                });
            }
            Some(UserInputQuestion {
                id: question
                    .get("id")
                    .and_then(Value::as_str)
                    .filter(|id| !id.is_empty())
                    .unwrap_or(text)
                    .to_owned(),
                header: format!("Question {}", index + 1),
                question: text.to_owned(),
                options,
                multi_select: question
                    .get("multiSelect")
                    .and_then(Value::as_bool)
                    .unwrap_or(false),
            })
        })
        .collect()
}

fn xai_user_input_response(params: &Value, submitted: &[UserInputAnswer]) -> Value {
    let mut answers = serde_json::Map::new();
    let mut annotations = serde_json::Map::new();
    for question in params
        .get("questions")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
    {
        let Some(question_text) = question.get("question").and_then(Value::as_str) else {
            continue;
        };
        let id = question
            .get("id")
            .and_then(Value::as_str)
            .unwrap_or(question_text);
        let values = submitted
            .iter()
            .find(|answer| answer.question_id == id || answer.question_id == question_text)
            .map(|answer| answer.answers.as_slice())
            .unwrap_or_default();
        let options = question
            .get("options")
            .and_then(Value::as_array)
            .into_iter()
            .flatten()
            .collect::<Vec<_>>();
        let option_labels = options
            .iter()
            .filter_map(|option| option.get("label").and_then(Value::as_str))
            .collect::<Vec<_>>();
        let selected = values
            .iter()
            .filter(|value| option_labels.contains(&value.as_str()))
            .cloned()
            .collect::<Vec<_>>();
        let notes = values
            .iter()
            .filter(|value| !option_labels.contains(&value.as_str()))
            .cloned()
            .collect::<Vec<_>>()
            .join("\n");
        let preview = if question
            .get("multiSelect")
            .and_then(Value::as_bool)
            .unwrap_or(false)
        {
            None
        } else {
            selected.iter().find_map(|selected| {
                options.iter().find_map(|option| {
                    (option.get("label").and_then(Value::as_str) == Some(selected.as_str()))
                        .then(|| {
                            option
                                .get("preview")
                                .and_then(Value::as_str)
                                .map(str::trim)
                                .filter(|preview| !preview.is_empty())
                                .map(str::to_owned)
                        })
                        .flatten()
                })
            })
        };
        answers.insert(
            question_text.to_owned(),
            json!(if selected.is_empty() && !notes.is_empty() {
                vec!["Other".to_owned()]
            } else {
                selected
            }),
        );
        let mut annotation = serde_json::Map::new();
        if let Some(preview) = preview {
            annotation.insert("preview".into(), Value::String(preview));
        }
        if !notes.is_empty() {
            annotation.insert("notes".into(), Value::String(notes));
        }
        if !annotation.is_empty() {
            annotations.insert(question_text.to_owned(), Value::Object(annotation));
        }
    }
    let mut response = json!({"outcome": "accepted", "answers": answers});
    if !annotations.is_empty() {
        response["annotations"] = Value::Object(annotations);
    }
    response
}

fn handle_permission_request(
    request: RequestPermissionRequest,
    responder: PermissionResponder,
    auto_approve: bool,
    pending: &PendingPermissions,
    events: &impl DriverEventSink,
) -> agent_client_protocol::Result<()> {
    let request_id = responder.id().to_string();
    let params = serde_json::to_value(&request)?;
    let options = request
        .options
        .iter()
        .map(|option| PermissionOption {
            id: option.option_id.to_string(),
            label: option.name.clone(),
            allow: matches!(
                option.kind,
                PermissionOptionKind::AllowOnce | PermissionOptionKind::AllowAlways
            ),
        })
        .collect::<Vec<_>>();

    if auto_approve {
        let choice = request
            .options
            .iter()
            .find(|option| option.kind == PermissionOptionKind::AllowAlways)
            .or_else(|| {
                request
                    .options
                    .iter()
                    .find(|option| option.kind == PermissionOptionKind::AllowOnce)
            });
        return match choice {
            Some(choice) => responder.respond(RequestPermissionResponse::new(
                RequestPermissionOutcome::Selected(SelectedPermissionOutcome::new(
                    choice.option_id.clone(),
                )),
            )),
            None => responder.respond(RequestPermissionResponse::new(
                RequestPermissionOutcome::Cancelled,
            )),
        };
    }

    let title = params
        .pointer("/toolCall/title")
        .and_then(Value::as_str)
        .map(str::to_owned)
        .unwrap_or_else(|| tr!("permission.run_a_tool"));
    let detail = permission_reason(&params).unwrap_or_else(|| {
        params
            .pointer("/toolCall/kind")
            .and_then(Value::as_str)
            .map(|kind| tr!("permission.agent_wants_to", action = kind))
            .unwrap_or_else(|| tr!("permission.agent_asks_for_permission"))
    });
    pending.lock().insert(request_id.clone(), responder);
    if events
        .send(DriverEvent::Permission {
            request_id: request_id.clone(),
            title,
            detail,
            options,
        })
        .is_err()
        && let Some(responder) = pending.lock().remove(&request_id)
    {
        let _ = responder.respond(RequestPermissionResponse::new(
            RequestPermissionOutcome::Cancelled,
        ));
    }
    Ok(())
}

fn handle_session_update(
    provider: ProviderKind,
    notification: SessionNotification,
    events: &impl DriverEventSink,
    state: &mut AcpStreamState,
) -> agent_client_protocol::Result<()> {
    let update = serde_json::to_value(notification.update)?;
    let kind = update.get("sessionUpdate").and_then(Value::as_str);
    if provider == ProviderKind::Fx
        && !state.produced_content
        && kind == Some("agent_message_chunk")
        && update
            .pointer("/content/text")
            .and_then(Value::as_str)
            .is_some_and(fx_context_notice)
    {
        return Ok(());
    }
    if matches!(
        kind,
        Some(
            "agent_message_chunk"
                | "agent_thought_chunk"
                | "tool_call"
                | "tool_call_update"
                | "plan"
        )
    ) {
        state.produced_content = true;
    }
    match kind {
        Some("agent_message_chunk") => {
            if let Some(text) = update
                .pointer("/content/text")
                .and_then(Value::as_str)
                .filter(|text| !text.is_empty())
            {
                let _ = events.send(DriverEvent::TextDelta(text.to_owned()));
            }
        }
        Some("agent_thought_chunk") => {
            if let Some(text) = update
                .pointer("/content/text")
                .and_then(Value::as_str)
                .filter(|text| !text.is_empty())
            {
                let _ = events.send(DriverEvent::ReasoningDelta(text.to_owned()));
            }
        }
        Some("tool_call" | "tool_call_update") => tool_activity(&update, events, state),
        Some("plan") => {
            let _ = events.send(DriverEvent::Activity {
                id: Some("acp-plan".into()),
                kind: ActivityKind::Plan,
                title: tr!("activity.plan_updated"),
                detail: None,
                complete: false,
            });
        }
        Some("available_commands_update") => {
            let commands = update
                .get("availableCommands")
                .and_then(Value::as_array)
                .map(|list| {
                    list.iter()
                        .filter_map(|command| {
                            let name = command.get("name").and_then(Value::as_str)?;
                            Some(crate::model::ReportedCommand {
                                name: name.to_owned(),
                                description: command
                                    .get("description")
                                    .and_then(Value::as_str)
                                    .unwrap_or_default()
                                    .to_owned(),
                            })
                        })
                        .collect::<Vec<_>>()
                })
                .unwrap_or_default();
            if !commands.is_empty() {
                let _ = events.send(DriverEvent::AvailableCommands(commands));
            }
        }
        Some("session_info_update") => {
            if update.get("title").is_some() {
                let title = update
                    .get("title")
                    .and_then(Value::as_str)
                    .map(str::to_owned);
                let _ = events.send(DriverEvent::AutoTitleUpdated(title));
            }
        }
        Some("usage_update") => {
            let used = update
                .get("used")
                .and_then(Value::as_u64)
                .filter(|used| *used > 0);
            let window = ["max", "limit", "size", "contextWindow", "context_window"]
                .into_iter()
                .find_map(|key| update.get(key).and_then(Value::as_u64))
                .filter(|window| *window > 0);
            if used.is_some() || window.is_some() {
                let _ = events.send(DriverEvent::UsageUpdated {
                    context_tokens: used,
                    context_window: window,
                });
            }
        }
        // `user_message_chunk` is Waku's own prompt echoed back. Other typed
        // updates currently have no transcript representation.
        _ => {}
    }
    Ok(())
}

fn fx_context_notice(text: &str) -> bool {
    text.starts_with("[context] ") || text.starts_with("skill discovery warning: ")
}

/// Captures Renoa's durable-history replay while `session/load` is in flight.
///
/// Updates are validated and grouped as they arrive, so the buffer can only
/// finalize as a complete, well-identified transcript. Context usage is held
/// beside that transcript until the consumer acknowledges the replay. Known
/// configuration and status updates carry no transcript semantics and are
/// ignored; any other unrecognized kind fails the load instead of silently
/// truncating history.
#[derive(Default)]
struct RenoaReplayCapture {
    items: Vec<ReplayItem>,
    usage: Option<ReplayedContextUsage>,
    error: Option<String>,
    user_turns: HashMap<uuid::Uuid, uuid::Uuid>,
    turn_messages: HashMap<uuid::Uuid, uuid::Uuid>,
    assistant_positions: HashMap<uuid::Uuid, usize>,
    open_tools: HashMap<String, usize>,
    current_turn: Option<uuid::Uuid>,
}

impl RenoaReplayCapture {
    fn observe(&mut self, update: &SessionUpdate) {
        if self.error.is_some() {
            return;
        }
        match update {
            SessionUpdate::UserMessageChunk(chunk) => self.user_chunk(chunk),
            SessionUpdate::AgentMessageChunk(chunk) => {
                self.assistant_chunk(chunk, false);
            }
            SessionUpdate::AgentThoughtChunk(chunk) => {
                self.assistant_chunk(chunk, true);
            }
            SessionUpdate::ToolCall(call) => self.tool_call(call),
            SessionUpdate::ToolCallUpdate(update) => self.tool_update(update),
            SessionUpdate::UsageUpdate(usage) => {
                let usage = ReplayedContextUsage {
                    context_tokens: (usage.used > 0).then_some(usage.used),
                    context_window: (usage.size > 0).then_some(usage.size),
                };
                if usage.context_tokens.is_some() || usage.context_window.is_some() {
                    self.usage = Some(usage);
                }
            }
            SessionUpdate::Plan(_)
            | SessionUpdate::AvailableCommandsUpdate(_)
            | SessionUpdate::CurrentModeUpdate(_)
            | SessionUpdate::ConfigOptionUpdate(_)
            | SessionUpdate::SessionInfoUpdate(_) => {}
            _ => self.fail(
                "the load replay carried an unknown update kind; \
                 refusing to build a partial transcript",
            ),
        }
    }

    fn user_chunk(&mut self, chunk: &ContentChunk) {
        let Some(message_id) = self.message_uuid(&chunk.message_id, "user message") else {
            return;
        };
        let turn_id = chunk
            .meta
            .as_ref()
            .and_then(|meta| meta.get("requestId"))
            .and_then(Value::as_str)
            .and_then(|id| uuid::Uuid::parse_str(id).ok());
        let Some(turn_id) = turn_id else {
            self.fail("the replayed user message is missing its requestId UUID");
            return;
        };
        let Some(text) = self.chunk_text(chunk, "user") else {
            return;
        };
        if let Some(ReplayItem::UserMessage(user)) = self.items.last_mut()
            && user.message_id == message_id
            && user.turn_id == turn_id
        {
            user.text.push_str(&text);
            return;
        }
        if !self.open_tools.is_empty() {
            self.fail(
                "a new replayed user message arrived before the prior tool lifecycle settled",
            );
            return;
        }
        if self.assistant_positions.contains_key(&message_id) {
            self.fail(&format!(
                "replayed messageId {message_id} changed from assistant to user"
            ));
            return;
        }
        if let Some(previous_turn) = self.user_turns.insert(message_id, turn_id) {
            self.fail(&format!(
                "replayed user message {message_id} resumed after an event boundary (previous request {previous_turn})"
            ));
            return;
        }
        if let Some(previous_message) = self.turn_messages.insert(turn_id, message_id) {
            self.fail(&format!(
                "replayed request {turn_id} maps to user messages {previous_message} and {message_id}"
            ));
            return;
        }
        self.current_turn = Some(turn_id);
        self.items.push(ReplayItem::UserMessage(ReplayUserMessage {
            message_id,
            turn_id,
            text,
        }));
    }

    fn assistant_chunk(&mut self, chunk: &ContentChunk, reasoning: bool) {
        if self.current_turn.is_none() {
            self.fail("a replayed assistant message arrived before any user message");
            return;
        }
        let Some(message_id) = self.message_uuid(&chunk.message_id, "assistant message") else {
            return;
        };
        let Some(text) = self.chunk_text(chunk, "assistant") else {
            return;
        };
        let segment = if reasoning {
            ReplaySegment::Reasoning(text)
        } else {
            ReplaySegment::Text(text)
        };
        if let Some(ReplayItem::AssistantMessage(assistant)) = self.items.last_mut()
            && assistant.message_id == message_id
        {
            assistant.segments.push(segment);
            return;
        }
        if self.assistant_positions.contains_key(&message_id) {
            self.fail(&format!(
                "replayed assistant message {message_id} resumed after a tool or message boundary"
            ));
            return;
        }
        if self.user_turns.contains_key(&message_id) {
            self.fail(&format!(
                "replayed messageId {message_id} changed from user to assistant"
            ));
            return;
        }
        self.assistant_positions
            .insert(message_id, self.items.len());
        self.items
            .push(ReplayItem::AssistantMessage(ReplayAssistantMessage {
                message_id,
                segments: vec![segment],
            }));
    }

    fn tool_call(&mut self, call: &AcpToolCall) {
        if self.current_turn.is_none() {
            self.fail("a replayed tool call arrived before any user message");
            return;
        }
        let call_id = call.tool_call_id.0.trim();
        if call_id.is_empty() {
            self.fail("the replayed tool call is missing its toolCallId");
            return;
        }
        let complete = matches!(
            call.status,
            ToolCallStatus::Completed | ToolCallStatus::Failed
        );
        let activity = super::activity::replay_tool_activity(super::activity::ReplayToolActivity {
            source_id: Some(call_id.to_owned()),
            kind: renoa_tool_kind(call.kind, &call.title),
            title: call.title.clone(),
            arguments: call.raw_input.as_ref(),
            output: None,
            raw_output: None,
            image_source: None,
            failed: false,
            complete,
        });
        let tool = ReplayTool {
            call_id: call_id.to_owned(),
            activity: Box::new(activity),
        };
        // Repeating a start updates only the latest still-open lifecycle. Once
        // settled, the same ACP id in a later model round is a new lifecycle.
        if let Some(position) = self.open_tools.get(call_id).copied() {
            self.items[position] = ReplayItem::ToolCall(tool);
        } else {
            let position = self.items.len();
            if !complete {
                self.open_tools.insert(call_id.to_owned(), position);
            }
            self.items.push(ReplayItem::ToolCall(tool));
        }
    }

    fn tool_update(&mut self, update: &ToolCallUpdate) {
        if self.current_turn.is_none() {
            self.fail("a replayed tool result arrived before any user message");
            return;
        }
        let call_id = update.tool_call_id.0.trim();
        if call_id.is_empty() {
            self.fail("the replayed tool result is missing its toolCallId");
            return;
        }
        let failed = update.fields.status == Some(ToolCallStatus::Failed);
        let terminal = failed || update.fields.status == Some(ToolCallStatus::Completed);
        if !terminal {
            self.fail(&format!(
                "replayed tool update {call_id:?} is non-terminal and cannot represent durable history"
            ));
            return;
        }
        if let Some(items) = update.fields.content.as_ref()
            && !self.tool_result_content_is_representable(call_id, items)
        {
            return;
        }
        let Some(position) = self.open_tools.remove(call_id) else {
            self.fail(&format!(
                "replayed tool result {call_id:?} has no still-open tool lifecycle"
            ));
            return;
        };
        {
            let content_value = match &update.fields.content {
                Some(items) if !items.is_empty() => match serde_json::to_value(items) {
                    Ok(value) => Some(value),
                    Err(error) => {
                        self.fail(&format!(
                            "the replayed tool result {call_id:?} could not be serialized: {error}"
                        ));
                        return;
                    }
                },
                _ => None,
            };
            let raw_output = update
                .fields
                .raw_output
                .clone()
                .filter(|value| !value.is_null());
            let output = content_value.or_else(|| raw_output.clone());
            let ReplayItem::ToolCall(tool) = &mut self.items[position] else {
                unreachable!("rposition matched this variant");
            };
            if let Some(title) = update
                .fields
                .title
                .clone()
                .filter(|title| !title.is_empty())
            {
                tool.activity.title = title;
            }
            // Mirrors the live path for text and arguments. That walk
            // HashSet-dedups image URLs, so image-only typed content is
            // overwritten below to keep order and multiplicity.
            let mut rebuilt =
                super::activity::replay_tool_activity(super::activity::ReplayToolActivity {
                    source_id: Some(tool.call_id.clone()),
                    kind: tool.activity.kind,
                    title: std::mem::take(&mut tool.activity.title),
                    arguments: update.fields.raw_input.as_ref(),
                    output: output.as_ref(),
                    raw_output: raw_output.as_ref(),
                    image_source: raw_output.as_ref(),
                    failed,
                    complete: true,
                });
            // A settled result usually repeats only outputs; keep the input
            // presentation the start already derived.
            if update.fields.raw_input.is_none() {
                rebuilt.arguments = tool.activity.arguments.take();
                rebuilt.authoritative_arguments = tool.activity.authoritative_arguments.take();
                rebuilt.display_target = tool.activity.display_target.take();
                rebuilt.display_description = tool.activity.display_description.take();
                rebuilt.file_changes = std::mem::take(&mut tool.activity.file_changes);
            }
            if let Some(urls) = update
                .fields
                .content
                .as_deref()
                .and_then(typed_replay_image_urls)
            {
                rebuilt = rebuilt.with_image_urls(urls);
            }
            *tool.activity = rebuilt;
        }
    }

    /// Waku's activity row stores text and images separately. Mixed or
    /// otherwise unrepresentable ordered content is rejected before any
    /// replay fragment is committed so the cache and cursor stay untouched.
    fn tool_result_content_is_representable(
        &mut self,
        call_id: &str,
        items: &[ToolCallContent],
    ) -> bool {
        let mut saw_text = false;
        let mut saw_image = false;
        for item in items {
            match item {
                ToolCallContent::Content(content) => match &content.content {
                    ContentBlock::Text(_) => saw_text = true,
                    ContentBlock::Image(_) => saw_image = true,
                    ContentBlock::Audio(_) => {
                        self.fail(&unsupported_tool_result(call_id, "audio"));
                        return false;
                    }
                    ContentBlock::Resource(_) => {
                        self.fail(&unsupported_tool_result(call_id, "an embedded resource"));
                        return false;
                    }
                    ContentBlock::ResourceLink(_) => {
                        self.fail(&unsupported_tool_result(call_id, "a resource link"));
                        return false;
                    }
                    _ => {
                        self.fail(&unsupported_tool_result(
                            call_id,
                            "an unknown content block",
                        ));
                        return false;
                    }
                },
                ToolCallContent::Diff(_) => {
                    self.fail(&unsupported_tool_result(call_id, "a diff"));
                    return false;
                }
                ToolCallContent::Terminal(_) => {
                    self.fail(&unsupported_tool_result(call_id, "a terminal"));
                    return false;
                }
                _ => {
                    self.fail(&unsupported_tool_result(
                        call_id,
                        "an unknown tool-result kind",
                    ));
                    return false;
                }
            }
            if saw_text && saw_image {
                self.fail(&format!(
                    "the replayed tool result {call_id:?} mixes text and image blocks; \
                     Waku cannot represent that ordered content losslessly"
                ));
                return false;
            }
        }
        true
    }

    fn message_uuid(&mut self, id: &Option<MessageId>, what: &str) -> Option<uuid::Uuid> {
        let Some(raw) = id.as_ref().map(|id| id.0.to_string()) else {
            self.fail(&format!("the replayed {what} is missing its messageId"));
            return None;
        };
        match uuid::Uuid::parse_str(&raw) {
            Ok(uuid) => Some(uuid),
            Err(_) => {
                self.fail(&format!(
                    "the replayed {what} carries a non-UUID messageId {raw:?}"
                ));
                None
            }
        }
    }

    fn chunk_text(&mut self, chunk: &ContentChunk, what: &str) -> Option<String> {
        match &chunk.content {
            ContentBlock::Text(text) => (!text.text.is_empty()).then(|| text.text.clone()),
            _ => {
                self.fail(&format!(
                    "the replayed {what} chunk carries unsupported content; \
                     only text projects into Waku's cache"
                ));
                None
            }
        }
    }

    fn fail(&mut self, reason: &str) {
        if self.error.is_none() {
            self.error = Some(reason.to_owned());
        }
    }

    fn finalize(self) -> anyhow::Result<RenoaLoadedReplay> {
        if let Some(error) = self.error {
            anyhow::bail!("{error}");
        }
        let replay = ProviderReplay { items: self.items };
        replay.validate()?;
        Ok(RenoaLoadedReplay {
            replay,
            usage: self.usage,
        })
    }
}

#[derive(Debug)]
struct RenoaLoadedReplay {
    replay: ProviderReplay,
    usage: Option<ReplayedContextUsage>,
}

#[derive(Clone, Copy, Debug)]
struct ReplayedContextUsage {
    context_tokens: Option<u64>,
    context_window: Option<u64>,
}

fn unsupported_tool_result(call_id: &str, what: &str) -> String {
    format!(
        "the replayed tool result {call_id:?} carries unsupported content ({what}); \
         Waku cannot represent that ordered content losslessly"
    )
}

/// Ordered image URLs from an image-only typed content array. Duplicate
/// payloads stay as separate entries; HashSet collection of the JSON form
/// must not replace this list.
fn typed_replay_image_urls(items: &[ToolCallContent]) -> Option<Vec<String>> {
    let mut urls = Vec::new();
    for item in items {
        let ToolCallContent::Content(content) = item else {
            return None;
        };
        match &content.content {
            ContentBlock::Image(image) => urls.push(replay_image_url(image)),
            ContentBlock::Text(_) => return None,
            _ => return None,
        }
    }
    (!urls.is_empty()).then_some(urls)
}

fn replay_image_url(image: &ImageContent) -> String {
    image
        .uri
        .as_deref()
        .filter(|uri| !uri.is_empty())
        .map(str::to_owned)
        .unwrap_or_else(|| format!("data:{};base64,{}", image.mime_type, image.data))
}

fn renoa_tool_kind(kind: ToolKind, title: &str) -> ActivityKind {
    let mapped = match kind {
        ToolKind::Read => ActivityKind::FileRead,
        ToolKind::Edit | ToolKind::Delete | ToolKind::Move => ActivityKind::FileChange,
        ToolKind::Search | ToolKind::Fetch => ActivityKind::Search,
        ToolKind::Execute => ActivityKind::Command,
        ToolKind::Think => ActivityKind::Reasoning,
        ToolKind::SwitchMode | ToolKind::Other => ActivityKind::Tool,
        _ => ActivityKind::Tool,
    };
    if matches!(mapped, ActivityKind::Search | ActivityKind::Tool) {
        let named = ActivityKind::from_tool_name(title);
        if named != ActivityKind::Tool {
            return named;
        }
    }
    mapped
}

#[derive(Default)]
struct AcpStreamState {
    tools: HashMap<String, (ActivityKind, String)>,
    /// Whether the running turn has produced anything visible. A turn that
    /// ends having produced nothing is the shape a swallowed provider error
    /// takes, which is what makes a native failure worth looking up.
    produced_content: bool,
}

/// Pull the agent's explanation out of a permission request's tool call.
fn permission_reason(params: &Value) -> Option<String> {
    let content = params
        .pointer("/toolCall/content")
        .and_then(Value::as_array)?;
    let reason = content
        .iter()
        .filter_map(|entry| {
            entry
                .pointer("/content/text")
                .or_else(|| entry.get("text"))
                .and_then(Value::as_str)
                .map(str::trim)
                .filter(|text| !text.is_empty())
        })
        .collect::<Vec<_>>()
        .join("\n");
    (!reason.is_empty()).then(|| truncate(&reason, 400))
}

fn truncate(text: &str, max_chars: usize) -> String {
    if text.chars().count() <= max_chars {
        return text.to_owned();
    }
    text.chars()
        .take(max_chars)
        .chain(std::iter::once('…'))
        .collect()
}

fn tool_activity(update: &Value, events: &impl DriverEventSink, state: &mut AcpStreamState) {
    let id = update
        .get("toolCallId")
        .and_then(Value::as_str)
        .map(str::to_owned);
    let status = update
        .get("status")
        .and_then(Value::as_str)
        .unwrap_or("pending");
    let complete = matches!(status, "completed" | "failed");
    let failed = status == "failed";

    let wire_kind = update.get("kind").and_then(Value::as_str);
    let wire_title = update.get("title").and_then(Value::as_str);
    let stored = id.as_ref().and_then(|id| {
        if complete {
            state.tools.remove(id)
        } else {
            state.tools.get(id).cloned()
        }
    });
    let mut kind = wire_kind
        .map(classify)
        .or_else(|| stored.as_ref().map(|(kind, _)| *kind))
        .unwrap_or(ActivityKind::Tool);
    if matches!(kind, ActivityKind::Search | ActivityKind::Tool)
        && let Some(wire_title) = wire_title
    {
        let named_kind = ActivityKind::from_tool_name(wire_title);
        if named_kind != ActivityKind::Tool {
            kind = named_kind;
        }
    }
    let arguments = update.get("rawInput").filter(|value| !value.is_null());
    let title = activity::input_title(arguments)
        .or_else(|| {
            wire_title
                .filter(|title| !title.is_empty())
                .map(str::to_owned)
        })
        .or_else(|| stored.map(|(_, title)| title))
        .unwrap_or_else(|| "Tool".to_owned());
    if !complete && let Some(id) = id.as_ref() {
        state.tools.insert(id.clone(), (kind, title.clone()));
    }

    let output = update
        .get("content")
        .filter(|value| !value.is_null())
        .or_else(|| update.get("rawOutput").filter(|value| !value.is_null()));
    let item =
        activity::tool_activity(id, kind, title, arguments, output, output, failed, complete);
    let _ = events.send(DriverEvent::RichActivity(item));
}

fn classify(kind: &str) -> ActivityKind {
    match kind {
        "execute" => ActivityKind::Command,
        "edit" | "delete" | "move" => ActivityKind::FileChange,
        "read" => ActivityKind::FileRead,
        "search" | "fetch" => ActivityKind::Search,
        "think" => ActivityKind::Reasoning,
        _ => ActivityKind::Tool,
    }
}

impl DriverControl for AcpDriver {
    fn prompt(&self, turn: TurnPrompt) {
        let _ = self.commands.try_send(CommandMessage::Prompt(turn));
    }

    fn acknowledge_replay(&self) -> bool {
        if self.replay_cancelled.load(Ordering::Acquire) {
            return false;
        }
        if self.replay_acknowledged.swap(true, Ordering::AcqRel) {
            return true;
        }
        if self.replay_gate.try_send(ReplayGate::Committed).is_ok() {
            true
        } else {
            self.replay_acknowledged.store(false, Ordering::Release);
            false
        }
    }

    fn supports_steer(&self) -> bool {
        self.supports_steer
    }

    fn steer(&self, prompt: String) {
        let _ = self.commands.try_send(CommandMessage::Steer(prompt));
    }

    fn cancel(&self) {
        self.replay_cancelled.store(true, Ordering::Release);
        let _ = self.replay_gate.try_send(ReplayGate::Aborted);
        let _ = self.commands.try_send(CommandMessage::Cancel);
    }

    fn cancel_computer_use(&self) {
        if let Some(computer_use) = self.computer_use.as_ref() {
            computer_use.stop();
        }
    }

    fn respond(&self, request_id: String, option_id: String) {
        let _ = self.commands.try_send(CommandMessage::Respond {
            request_id,
            option_id,
        });
    }

    fn respond_user_input(&self, request_id: String, answers: Vec<UserInputAnswer>) {
        let _ = self.commands.try_send(CommandMessage::RespondUserInput {
            request_id,
            answers,
        });
    }

    fn apply_options(&self, options: SessionOptions) -> bool {
        if options.mode != self.mode || options.interaction_mode != self.interaction_mode {
            return false;
        }
        self.commands
            .try_send(CommandMessage::Options(options))
            .is_ok()
    }

    fn rollback(&self, _turns: usize) -> anyhow::Result<Option<ProviderResumeCursor>> {
        Err(anyhow!(
            "conversation rollback is not supported by this provider transport"
        ))
    }
}

impl Drop for AcpDriver {
    fn drop(&mut self) {
        self.cancel_computer_use();
        self.replay_cancelled.store(true, Ordering::Release);
        let _ = self.replay_gate.try_send(ReplayGate::Aborted);
        let _ = self.commands.try_send(CommandMessage::Shutdown);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use agent_client_protocol::schema::v1::{
        SessionConfigSelectOption, SessionMode, SessionModeState, ToolCallContent, ToolCallUpdate,
        ToolCallUpdateFields,
    };

    struct TempDir {
        path: std::path::PathBuf,
    }

    impl TempDir {
        fn new() -> Self {
            let path = std::env::temp_dir().join(format!(
                "waku-fake-acp-{}-{}",
                std::process::id(),
                uuid::Uuid::new_v4().simple()
            ));
            std::fs::create_dir_all(&path).expect("create fake ACP temp dir");
            Self { path }
        }

        fn path(&self) -> &std::path::Path {
            &self.path
        }
    }

    impl Drop for TempDir {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.path);
        }
    }

    const FAKE_ACP_AGENT_SOURCE: &str = r#"
use std::io::{self, BufRead, Write};

fn main() {
    let log_path = std::env::current_exe()
        .expect("current exe")
        .with_file_name("acp.jsonl");
    let stdin = io::stdin();
    let mut stdout = io::stdout();
    for line in stdin.lock().lines() {
        let line = line.expect("stdin");
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        if let Ok(mut log) = std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(&log_path)
        {
            let _ = writeln!(log, "{line}");
        }
        let Some(id) = json_raw_field(line, "id") else {
            continue;
        };
        let method = json_string_field(line, "method").unwrap_or_default();
        let reply = match method.as_str() {
            "initialize" => {
                let resume = std::env::current_exe()
                    .ok()
                    .and_then(|exe| exe.parent().map(|dir| dir.join("advertise-resume")))
                    .is_some_and(|marker| marker.exists());
                let resume_capability = if resume { "\"resume\":{}" } else { "" };
                format!(
                    "{{\"jsonrpc\":\"2.0\",\"id\":{id},\"result\":{{\"protocolVersion\":1,\"agentCapabilities\":{{\"loadSession\":true,\"sessionCapabilities\":{{{resume_capability}}}}},\"agentInfo\":{{\"name\":\"fake-acp\",\"version\":\"0\"}}}}}}"
                )
            }
            "session/new" => {
                let config_options = std::env::current_exe()
                    .ok()
                    .and_then(|exe| exe.parent().map(|dir| dir.join("advertise-renoa-config")))
                    .is_some_and(|marker| marker.exists())
                    .then_some(concat!(
                        ",\"configOptions\":[",
                        "{\"id\":\"model\",\"name\":\"Model\",\"category\":\"model\",\"type\":\"select\",\"currentValue\":\"grok-a\",\"options\":[{\"value\":\"grok-a\",\"name\":\"Grok A\"},{\"value\":\"grok-b\",\"name\":\"Grok B\"}]},",
                        "{\"id\":\"thought_level\",\"name\":\"Reasoning\",\"category\":\"thought_level\",\"type\":\"select\",\"currentValue\":\"low\",\"options\":[{\"value\":\"low\",\"name\":\"Low\"}]}]"
                    ))
                    .unwrap_or("");
                format!(
                    "{{\"jsonrpc\":\"2.0\",\"id\":{id},\"result\":{{\"sessionId\":\"new-session\"{config_options}}}}}"
                )
            }
            "session/load" => {
                // A scripted history is replayed as ordinary session/update
                // notifications before the response, mirroring Renoa's
                // durable replay. Without a success marker beside the agent
                // binary the load fails, which exercises partial replays.
                let mut history: Option<String> = None;
                let mut ok = false;
                if let Ok(exe) = std::env::current_exe() {
                    if let Some(dir) = exe.parent() {
                        history = std::fs::read_to_string(dir.join("acp-load-history.jsonl")).ok();
                        ok = dir.join("acp-load-ok").exists();
                    }
                }
                match history {
                    Some(history) => {
                        for line in history.lines() {
                            if line.trim().is_empty() {
                                continue;
                            }
                            stdout.write_all(line.as_bytes()).expect("stdout");
                            stdout.write_all(b"\n").expect("stdout");
                        }
                        stdout.flush().expect("stdout");
                        if ok {
                            format!("{{\"jsonrpc\":\"2.0\",\"id\":{id},\"result\":{{}}}}")
                        } else {
                            format!(
                                "{{\"jsonrpc\":\"2.0\",\"id\":{id},\"error\":{{\"code\":-32001,\"message\":\"session missing\"}}}}"
                            )
                        }
                    }
                    None => format!(
                        "{{\"jsonrpc\":\"2.0\",\"id\":{id},\"error\":{{\"code\":-32001,\"message\":\"session missing\"}}}}"
                    ),
                }
            }
            "session/resume" => format!(
                "{{\"jsonrpc\":\"2.0\",\"id\":{id},\"result\":{{}}}}"
            ),
            "session/set_config_option" => {
                let current_reasoning = if line.contains("\"configId\":\"thought_level\"")
                    && line.contains("\"value\":\"high\"")
                {
                    "high"
                } else {
                    "low"
                };
                format!(
                    concat!(
                        "{{\"jsonrpc\":\"2.0\",\"id\":{id},\"result\":{{\"configOptions\":[",
                        "{{\"id\":\"model\",\"name\":\"Model\",\"category\":\"model\",\"type\":\"select\",\"currentValue\":\"grok-b\",\"options\":[{{\"value\":\"grok-a\",\"name\":\"Grok A\"}},{{\"value\":\"grok-b\",\"name\":\"Grok B\"}}]}},",
                        "{{\"id\":\"thought_level\",\"name\":\"Reasoning\",\"category\":\"thought_level\",\"type\":\"select\",\"currentValue\":\"{current_reasoning}\",\"options\":[{{\"value\":\"low\",\"name\":\"Low\"}},{{\"value\":\"high\",\"name\":\"High\"}}]}}]}}}}"
                    ),
                    id = id,
                    current_reasoning = current_reasoning,
                )
            }
            "session/prompt" => format!(
                "{{\"jsonrpc\":\"2.0\",\"id\":{id},\"result\":{{\"stopReason\":\"end_turn\"}}}}"
            ),
            _ => continue,
        };
        stdout.write_all(reply.as_bytes()).expect("stdout");
        stdout.write_all(b"\n").expect("stdout");
        stdout.flush().expect("stdout");
    }
}

fn json_string_field(json: &str, key: &str) -> Option<String> {
    let raw = json_raw_field(json, key)?;
    let raw = raw.strip_prefix('"')?.strip_suffix('"')?;
    Some(raw.to_owned())
}

fn json_raw_field<'a>(json: &'a str, key: &str) -> Option<&'a str> {
    let needle = format!("\"{key}\"");
    let rest = json.split_once(&needle)?.1.trim_start().strip_prefix(':')?.trim_start();
    if rest.starts_with('"') {
        let mut escaped = false;
        for (index, ch) in rest[1..].char_indices() {
            if escaped {
                escaped = false;
                continue;
            }
            match ch {
                '\\' => escaped = true,
                '"' => return Some(&rest[..=index + 1]),
                _ => {}
            }
        }
        None
    } else {
        let end = rest
            .find(|ch: char| ch == ',' || ch == '}' || ch.is_whitespace())
            .unwrap_or(rest.len());
        Some(&rest[..end])
    }
}
"#;

    struct CompiledFakeAgent {
        binary: std::path::PathBuf,
        _root: TempDir,
    }

    #[derive(Default)]
    struct CompiledFakeAgentCache {
        cached: std::sync::Mutex<std::sync::Weak<CompiledFakeAgent>>,
    }

    impl CompiledFakeAgentCache {
        fn acquire(&self) -> Arc<CompiledFakeAgent> {
            let mut cached = self.cached.lock().expect("lock fake ACP agent cache");
            if let Some(compiled) = cached.upgrade() {
                return compiled;
            }

            let build = TempDir::new();
            let source = build.path().join("fake_acp_agent.rs");
            let binary = build.path().join(if cfg!(windows) {
                "fake_acp_agent.exe"
            } else {
                "fake_acp_agent"
            });
            std::fs::write(&source, FAKE_ACP_AGENT_SOURCE).expect("write fake ACP source");
            let status = std::process::Command::new("rustc")
                .args(["--edition", "2021", "-o"])
                .arg(&binary)
                .arg(&source)
                .status()
                .expect("rustc is required to build the fake ACP agent");
            assert!(status.success(), "rustc failed to build the fake ACP agent");

            let compiled = Arc::new(CompiledFakeAgent {
                binary,
                _root: build,
            });
            *cached = Arc::downgrade(&compiled);
            compiled
        }
    }

    /// Shares one immutable binary only while fixtures are alive. The static
    /// cache owns a `Weak`, so the build directory is removed with the last
    /// fixture instead of leaking at test-process exit.
    fn fake_acp_agent_binary() -> Arc<CompiledFakeAgent> {
        use std::sync::OnceLock;
        static CACHE: OnceLock<CompiledFakeAgentCache> = OnceLock::new();
        CACHE.get_or_init(CompiledFakeAgentCache::default).acquire()
    }

    struct FakeAcpAgent {
        binary: std::path::PathBuf,
        cwd: std::path::PathBuf,
        events: DriverEventSender,
        event_rx: crossbeam_channel::Receiver<DriverEvent>,
        log_path: std::path::PathBuf,
        _compiled: Arc<CompiledFakeAgent>,
        _root: TempDir,
    }

    impl FakeAcpAgent {
        fn new() -> Self {
            let root = TempDir::new();
            let binary = root.path().join(if cfg!(windows) {
                "fake_acp_agent.exe"
            } else {
                "fake_acp_agent"
            });
            let compiled = fake_acp_agent_binary();
            std::fs::hard_link(&compiled.binary, &binary)
                .expect("link fake ACP agent into its isolated fixture");
            let cwd = root.path().to_path_buf();
            let log_path = binary.with_file_name("acp.jsonl");
            let (events, event_rx) = crate::driver::test_event_channel();
            Self {
                binary,
                cwd,
                events,
                event_rx,
                log_path,
                _compiled: compiled,
                _root: root,
            }
        }

        fn methods(&self) -> Vec<String> {
            self.messages()
                .into_iter()
                .filter_map(|message| {
                    message
                        .get("method")
                        .and_then(Value::as_str)
                        .map(str::to_owned)
                })
                .collect()
        }

        fn messages(&self) -> Vec<Value> {
            let Ok(contents) = std::fs::read_to_string(&self.log_path) else {
                return Vec::new();
            };
            contents
                .lines()
                .filter(|line| !line.is_empty())
                .map(|line| serde_json::from_str(line).expect("fake ACP log line"))
                .collect()
        }

        fn wait_for_method(&self, method: &str) -> Value {
            let deadline = std::time::Instant::now() + Duration::from_secs(5);
            while std::time::Instant::now() < deadline {
                if let Some(message) =
                    self.messages().into_iter().rev().find(|message| {
                        message.get("method").and_then(Value::as_str) == Some(method)
                    })
                {
                    return message;
                }
                std::thread::sleep(Duration::from_millis(10));
            }
            panic!("timed out waiting for {method}");
        }

        fn wait_for_method_count(&self, method: &str, count: usize) -> Vec<Value> {
            let deadline = std::time::Instant::now() + Duration::from_secs(5);
            while std::time::Instant::now() < deadline {
                let messages = self
                    .messages()
                    .into_iter()
                    .filter(|message| message.get("method").and_then(Value::as_str) == Some(method))
                    .collect::<Vec<_>>();
                if messages.len() >= count {
                    return messages;
                }
                std::thread::yield_now();
            }
            panic!("timed out waiting for {count} {method} requests");
        }

        /// Stages notifications the agent emits as its durable replay before
        /// answering `session/load`.
        fn stage_load_history(&self, notifications: &[Value]) {
            let body = notifications
                .iter()
                .map(Value::to_string)
                .collect::<Vec<_>>()
                .join("\n");
            std::fs::write(
                self._root.path().join("acp-load-history.jsonl"),
                body + "\n",
            )
            .expect("stage fake ACP load history");
        }

        /// Makes the staged `session/load` succeed after emitting the history.
        fn stage_load_success(&self) {
            std::fs::write(self._root.path().join("acp-load-ok"), b"")
                .expect("stage fake ACP load success");
        }

        fn advertise_resume(&self) {
            std::fs::write(self._root.path().join("advertise-resume"), b"")
                .expect("advertise resume capability");
        }

        fn advertise_renoa_config(&self) {
            std::fs::write(self._root.path().join("advertise-renoa-config"), b"")
                .expect("advertise Renoa config options");
        }
    }

    #[test]
    fn compiled_fake_agent_cache_releases_its_build_directory() {
        let cache = CompiledFakeAgentCache::default();
        let first = cache.acquire();
        let build_root = first._root.path().to_path_buf();
        let second = cache.acquire();

        assert!(Arc::ptr_eq(&first, &second));
        drop(first);
        assert!(
            build_root.exists(),
            "one fixture still owns the compiler output"
        );
        drop(second);

        assert!(
            !build_root.exists(),
            "the last fixture must remove the compiler output"
        );
        assert!(
            cache
                .cached
                .lock()
                .expect("lock fake ACP agent cache")
                .upgrade()
                .is_none(),
            "the cache must not own compiled fixture state"
        );
    }

    fn fake_acp_start_options(
        fixture: &FakeAcpAgent,
        provider_cursor: Option<ProviderResumeCursor>,
    ) -> DriverStartOptions {
        DriverStartOptions {
            binary: fixture.binary.clone(),
            cwd: fixture.cwd.clone(),
            mode: RuntimeMode::FullAccess,
            interaction_mode: InteractionMode::Build,
            model: None,
            reasoning_effort: None,
            service_tier: None,
            context_window: None,
            agent_preset: None,
            computer_use_enabled: false,
            provider_cursor,
        }
    }

    fn start_fake_acp(
        fixture: &FakeAcpAgent,
        provider: ProviderKind,
        provider_cursor: Option<ProviderResumeCursor>,
    ) -> AcpDriver {
        AcpDriver::start(
            provider,
            fake_acp_start_options(fixture, provider_cursor),
            fixture.events.clone(),
        )
        .expect("the fake ACP session should start")
    }

    fn start_fake_renoa(
        fixture: &FakeAcpAgent,
        provider_cursor: Option<ProviderResumeCursor>,
    ) -> AcpDriver {
        start_fake_acp(fixture, ProviderKind::Renoa, provider_cursor)
    }

    fn wait_for_acp_connected(
        events: &crossbeam_channel::Receiver<DriverEvent>,
        expected: ProviderKind,
    ) {
        loop {
            match events
                .recv_timeout(Duration::from_secs(5))
                .expect("the agent should report its session")
            {
                DriverEvent::Connected {
                    provider_cursor: Some(cursor),
                } if cursor.provider() == expected => return,
                DriverEvent::Error(error) => panic!("the agent reported: {error}"),
                _ => {}
            }
        }
    }

    fn wait_for_renoa_connected(events: &crossbeam_channel::Receiver<DriverEvent>) {
        wait_for_acp_connected(events, ProviderKind::Renoa);
    }

    fn wait_for_renoa_error(events: &crossbeam_channel::Receiver<DriverEvent>) -> String {
        loop {
            match events
                .recv_timeout(Duration::from_secs(5))
                .expect("Renoa should report the load failure")
            {
                DriverEvent::Error(error) => return error,
                DriverEvent::ProcessExited => {
                    panic!("Renoa exited before reporting the load failure")
                }
                _ => {}
            }
        }
    }

    fn select_config_option(
        id: &str,
        category: SessionConfigOptionCategory,
        current: &str,
        values: &[&str],
    ) -> SessionConfigOption {
        SessionConfigOption::select(
            id.to_owned(),
            id.to_owned(),
            current.to_owned(),
            values
                .iter()
                .map(|value| SessionConfigSelectOption::new((*value).to_owned(), *value))
                .collect::<Vec<_>>(),
        )
        .category(category)
    }

    #[test]
    fn cursor_question_response_uses_native_scalar_and_array_answers() {
        let params = json!({
            "toolCallId": "ask-1",
            "questions": [
                {
                    "id": "scope",
                    "prompt": "Which scope?",
                    "options": [{"id": "workspace", "label": "Workspace"}]
                },
                {
                    "id": "checks",
                    "prompt": "Which checks?",
                    "options": [
                        {"id": "tests", "label": "Tests"},
                        {"id": "lint", "label": "Lint"}
                    ],
                    "allowMultiple": true
                }
            ]
        });

        let questions = cursor_user_input_questions(&params);
        assert_eq!(questions.len(), 2);
        assert!(!questions[0].multi_select);
        assert!(questions[1].multi_select);

        let response = cursor_user_input_response(
            &params,
            &[
                UserInputAnswer {
                    question_id: "scope".into(),
                    answers: vec!["Workspace".into()],
                },
                UserInputAnswer {
                    question_id: "checks".into(),
                    answers: vec!["Tests".into(), "Lint".into()],
                },
            ],
        );
        assert_eq!(
            response.pointer("/answers/scope"),
            Some(&json!("Workspace"))
        );
        assert_eq!(
            response.pointer("/answers/checks"),
            Some(&json!(["Tests", "Lint"]))
        );
    }

    #[test]
    fn grok_question_response_keeps_native_labels_and_annotates_custom_text() {
        let params = json!({
            "sessionId": "session-1",
            "toolCallId": "tool-1",
            "mode": "default",
            "questions": [
                {
                    "id": "environment",
                    "question": "Where should this deploy?",
                    "options": [{"label": "Preview", "preview": "Deploy to preview"}],
                    "multiSelect": false
                },
                {
                    "id": "notes",
                    "question": "Anything else?",
                    "options": [{"label": "No"}],
                    "multiSelect": false
                }
            ]
        });
        let response = xai_user_input_response(
            &params,
            &[
                UserInputAnswer {
                    question_id: "environment".into(),
                    answers: vec!["Preview".into()],
                },
                UserInputAnswer {
                    question_id: "notes".into(),
                    answers: vec!["Use the EU region".into()],
                },
            ],
        );

        assert_eq!(response["outcome"], "accepted");
        assert_eq!(
            response.pointer("/answers/Where should this deploy?/0"),
            Some(&json!("Preview"))
        );
        assert_eq!(
            response.pointer("/answers/Anything else?/0"),
            Some(&json!("Other"))
        );
        assert_eq!(
            response.pointer("/annotations/Where should this deploy?/preview"),
            Some(&json!("Deploy to preview"))
        );
        assert_eq!(
            response.pointer("/annotations/Anything else?/notes"),
            Some(&json!("Use the EU region"))
        );
    }

    #[test]
    fn plan_mode_selects_the_advertised_plan_mode() {
        let modes = SessionModeState::new(
            "agent",
            vec![
                SessionMode::new("agent", "Agent"),
                SessionMode::new("plan", "Plan"),
            ],
        );
        assert_eq!(
            desired_mode(
                ProviderKind::Cursor,
                Some(&modes),
                RuntimeMode::FullAccess,
                InteractionMode::Plan
            )
            .map(|mode| mode.to_string()),
            Some("plan".to_owned())
        );
        assert!(
            desired_mode(
                ProviderKind::Cursor,
                Some(&modes),
                RuntimeMode::FullAccess,
                InteractionMode::Build
            )
            .is_none()
        );
    }

    #[test]
    fn fx_access_mode_selects_ask_or_code() {
        let modes = SessionModeState::new(
            "code",
            vec![
                SessionMode::new("ask", "Ask before sensitive actions"),
                SessionMode::new("code", "Review sensitive actions automatically"),
            ],
        );
        assert_eq!(
            desired_mode(
                ProviderKind::Fx,
                Some(&modes),
                RuntimeMode::Ask,
                InteractionMode::Build
            )
            .map(|mode| mode.to_string()),
            Some("ask".to_owned())
        );
        assert!(
            desired_mode(
                ProviderKind::Fx,
                Some(&modes),
                RuntimeMode::FullAccess,
                InteractionMode::Build
            )
            .is_none()
        );
    }

    #[test]
    fn fx_launches_its_documented_acp_subcommand() {
        let launch = launch_for(ProviderKind::Fx).unwrap();
        assert_eq!(launch.args, ["acp"]);
        assert!(launch.env.is_empty());
    }

    #[test]
    fn renoa_launches_its_documented_acp_subcommand() {
        let launch = launch_for(ProviderKind::Renoa).unwrap();
        assert_eq!(launch.args, ["acp"]);
        assert!(launch.env.is_empty());
    }

    #[test]
    fn renoa_prompt_metadata_reuses_the_stable_turn_id() {
        let turn_id = uuid::Uuid::parse_str("3b1c0e7a-2f64-4c91-9d2e-0a1b2c3d4e5f").unwrap();
        let identity = prompt_extension_id(ProviderKind::Renoa, Some(turn_id)).unwrap();
        let request =
            acp_prompt_request(SessionId::new("session"), "hello".into(), Some(&identity));
        let value = serde_json::to_value(&request).unwrap();
        assert_eq!(value["_meta"]["requestId"], turn_id.to_string());
        assert_eq!(value["_meta"]["promptId"], turn_id.to_string());
        assert_eq!(value["_meta"]["requestId"], value["_meta"]["promptId"]);
    }

    #[test]
    fn grok_prompt_metadata_stays_provider_generated() {
        let turn_id = uuid::Uuid::parse_str("3b1c0e7a-2f64-4c91-9d2e-0a1b2c3d4e5f").unwrap();
        let identity = prompt_extension_id(ProviderKind::Grok, Some(turn_id)).unwrap();
        assert_ne!(identity, turn_id.to_string());
        assert!(identity.starts_with("waku-"));
        uuid::Uuid::parse_str(identity.strip_prefix("waku-").unwrap()).unwrap();
        let request =
            acp_prompt_request(SessionId::new("session"), "hello".into(), Some(&identity));
        let value = serde_json::to_value(&request).unwrap();
        assert_eq!(value["_meta"]["requestId"], identity);
        assert_eq!(value["_meta"]["promptId"], identity);
        assert!(
            acp_prompt_request(SessionId::new("session"), "hello".into(), None)
                .meta
                .is_none()
        );
    }

    #[test]
    fn renoa_rejects_an_empty_resume_cursor() {
        let (events, _event_rx) = crate::driver::test_event_channel();
        let error = AcpDriver::start(
            ProviderKind::Renoa,
            DriverStartOptions {
                binary: std::path::PathBuf::from("/nonexistent/renoa-agent"),
                cwd: std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR")),
                mode: RuntimeMode::FullAccess,
                interaction_mode: InteractionMode::Build,
                model: None,
                reasoning_effort: None,
                service_tier: None,
                context_window: None,
                agent_preset: None,
                computer_use_enabled: false,
                provider_cursor: Some(ProviderResumeCursor::Renoa {
                    session_id: String::new(),
                }),
            },
            events,
        )
        .err()
        .expect("an empty Renoa cursor must fail before launch");
        assert!(
            error
                .to_string()
                .contains("cannot resume Renoa without a durable session id")
        );
    }

    #[test]
    fn renoa_does_not_advertise_steering() {
        let fixture = FakeAcpAgent::new();
        let driver = start_fake_renoa(&fixture, None);
        assert!(!driver.supports_steer());
        wait_for_renoa_connected(&fixture.event_rx);
        driver.prompt(TurnPrompt::new(uuid::Uuid::new_v4(), "hello"));
        let _ = fixture.wait_for_method("session/prompt");
        driver.steer("mid-turn instruction".into());
        let rejected = loop {
            match fixture
                .event_rx
                .recv_timeout(Duration::from_secs(5))
                .expect("Renoa should reject steering")
            {
                DriverEvent::SteerRejected { reason, .. } => break reason,
                DriverEvent::Error(error) => panic!("the agent reported: {error}"),
                _ => {}
            }
        };
        assert!(rejected.contains("does not support steering"));
        assert_eq!(
            fixture
                .methods()
                .into_iter()
                .filter(|method| method == "session/prompt")
                .count(),
            1
        );
    }

    #[test]
    fn a_selectable_renoa_provider_resolves_its_binary_and_starts() {
        let fixture = FakeAcpAgent::new();
        let probe = crate::model::provider_probe(
            ProviderKind::Renoa,
            Some(fixture.binary.to_str().expect("utf-8 fake agent path")),
        );
        assert!(probe.installed);
        assert_eq!(probe.path.as_deref(), Some(fixture.binary.as_path()));
        assert!(ProviderKind::SELECTABLE.contains(&probe.provider));

        let driver = crate::driver::start_local(
            ProviderKind::Renoa,
            fake_acp_start_options(&fixture, None),
            fixture.events.clone(),
        )
        .expect("a restored Renoa provider must reach driver startup");
        assert!(!driver.supports_steer());
        wait_for_renoa_connected(&fixture.event_rx);
        assert_eq!(fixture.methods(), ["initialize", "session/new"]);
    }

    #[test]
    fn renoa_applies_model_then_refreshed_reasoning_before_first_prompt() {
        let fixture = FakeAcpAgent::new();
        fixture.advertise_renoa_config();
        let mut options = fake_acp_start_options(&fixture, None);
        options.model = Some("grok-b".into());
        options.reasoning_effort = Some("high".into());
        let driver = AcpDriver::start(ProviderKind::Renoa, options, fixture.events.clone())
            .expect("the configured Renoa session should start");
        wait_for_renoa_connected(&fixture.event_rx);

        driver.prompt(TurnPrompt::new(uuid::Uuid::new_v4(), "hello"));
        let _ = fixture.wait_for_method("session/prompt");
        let requests = fixture.messages();
        let methods = requests
            .iter()
            .filter_map(|request| request.get("method").and_then(Value::as_str))
            .collect::<Vec<_>>();
        assert_eq!(
            methods,
            [
                "initialize",
                "session/new",
                "session/set_config_option",
                "session/set_config_option",
                "session/prompt"
            ]
        );
        assert_eq!(requests[2]["params"]["configId"], "model");
        assert_eq!(requests[2]["params"]["value"], "grok-b");
        assert_eq!(requests[3]["params"]["configId"], "thought_level");
        assert_eq!(requests[3]["params"]["value"], "high");
        assert!(!methods.contains(&"session/set_model"));
    }

    #[test]
    fn renoa_missing_standard_config_fails_before_prompt() {
        let fixture = FakeAcpAgent::new();
        let mut options = fake_acp_start_options(&fixture, None);
        options.model = Some("grok-b".into());
        options.reasoning_effort = Some("high".into());
        let driver = AcpDriver::start(ProviderKind::Renoa, options, fixture.events.clone())
            .expect("the Renoa transport should start before catalog validation");
        wait_for_renoa_connected(&fixture.event_rx);
        driver.prompt(TurnPrompt::new(uuid::Uuid::new_v4(), "must not run"));

        let error = wait_for_renoa_error(&fixture.event_rx);
        assert!(error.contains("did not advertise a model option"));
        let methods = fixture.methods();
        assert_eq!(methods, ["initialize", "session/new"]);
        assert!(!methods.contains(&"session/prompt".to_owned()));
        assert!(!methods.contains(&"session/set_model".to_owned()));
    }

    #[test]
    fn renoa_reasoning_only_option_change_does_not_resend_model() {
        let fixture = FakeAcpAgent::new();
        fixture.advertise_renoa_config();
        let mut options = fake_acp_start_options(&fixture, None);
        options.model = Some("grok-b".into());
        options.reasoning_effort = Some("low".into());
        let driver = AcpDriver::start(ProviderKind::Renoa, options, fixture.events.clone())
            .expect("the configured Renoa session should start");
        wait_for_renoa_connected(&fixture.event_rx);
        driver.prompt(TurnPrompt::new(uuid::Uuid::new_v4(), "first"));
        let _ = fixture.wait_for_method("session/prompt");

        assert!(driver.apply_options(SessionOptions {
            mode: RuntimeMode::FullAccess,
            interaction_mode: InteractionMode::Build,
            model: Some("grok-b".into()),
            reasoning_effort: Some("high".into()),
            service_tier: None,
            context_window: None,
        }));
        driver.prompt(TurnPrompt::new(uuid::Uuid::new_v4(), "second"));
        let prompts = fixture.wait_for_method_count("session/prompt", 2);
        let second_prompt = &prompts[1];
        assert_eq!(second_prompt["params"]["prompt"][0]["text"], "second");

        let config_requests = fixture
            .messages()
            .into_iter()
            .filter(|request| {
                request.get("method").and_then(Value::as_str) == Some("session/set_config_option")
            })
            .collect::<Vec<_>>();
        assert_eq!(config_requests.len(), 3);
        assert_eq!(config_requests[0]["params"]["configId"], "model");
        assert_eq!(config_requests[1]["params"]["configId"], "thought_level");
        assert_eq!(config_requests[2]["params"]["configId"], "thought_level");
        assert_eq!(config_requests[2]["params"]["value"], "high");
    }

    #[test]
    fn renoa_prompt_sends_stable_turn_metadata_on_the_wire() {
        let fixture = FakeAcpAgent::new();
        let driver = start_fake_renoa(&fixture, None);
        wait_for_renoa_connected(&fixture.event_rx);
        let turn_id = uuid::Uuid::parse_str("3b1c0e7a-2f64-4c91-9d2e-0a1b2c3d4e5f").unwrap();
        driver.prompt(TurnPrompt::new(turn_id, "hello"));
        let prompt = fixture.wait_for_method("session/prompt");
        assert_eq!(prompt["params"]["_meta"]["requestId"], turn_id.to_string());
        assert_eq!(prompt["params"]["_meta"]["promptId"], turn_id.to_string());
    }

    #[test]
    fn failed_renoa_session_load_never_sends_session_new() {
        let fixture = FakeAcpAgent::new();
        fixture.stage_load_history(&[renoa_notification(
            "3b1c0e7a-2f64-4c91-9d2e-0a1b2c3d4e5f",
            json!({
                "sessionUpdate": "usage_update",
                "used": 12_345,
                "size": 500_000,
            }),
        )]);
        let _driver = start_fake_renoa(
            &fixture,
            Some(ProviderResumeCursor::Renoa {
                session_id: "3b1c0e7a-2f64-4c91-9d2e-0a1b2c3d4e5f".into(),
            }),
        );
        let error = loop {
            match fixture
                .event_rx
                .recv_timeout(Duration::from_secs(5))
                .expect("Renoa should report the load failure")
            {
                DriverEvent::Error(error) => break error,
                DriverEvent::UsageUpdated { .. } => {
                    panic!("usage from a failed Renoa load reached the presentation cache")
                }
                DriverEvent::ProcessExited => {
                    panic!("Renoa exited before reporting the load failure")
                }
                _ => {}
            }
        };
        assert!(
            error.contains("failed to load Renoa session 3b1c0e7a-2f64-4c91-9d2e-0a1b2c3d4e5f")
        );
        assert!(error.contains("session missing"));
        let methods = fixture.methods();
        assert!(methods.contains(&"initialize".to_owned()));
        assert!(methods.contains(&"session/load".to_owned()));
        assert!(!methods.contains(&"session/new".to_owned()));
    }

    #[test]
    fn renoa_uses_load_even_when_resume_is_advertised() {
        let fixture = FakeAcpAgent::new();
        let durable_session = "3b1c0e7a-2f64-4c91-9d2e-0a1b2c3d4e5f";
        fixture.advertise_resume();
        fixture.stage_load_history(&[]);
        fixture.stage_load_success();
        let _driver = start_fake_renoa(
            &fixture,
            Some(ProviderResumeCursor::Renoa {
                session_id: durable_session.into(),
            }),
        );
        let items = wait_for_renoa_replay(&fixture);
        assert!(items.is_empty());
        wait_for_renoa_connected(&fixture.event_rx);
        assert_eq!(fixture.methods(), ["initialize", "session/load"]);
    }

    #[test]
    fn a_renoa_user_image_chunk_fails_the_load_atomically() {
        use agent_client_protocol::schema::v1::ImageContent;

        let mut capture = RenoaReplayCapture::default();
        let mut meta = Map::new();
        meta.insert(
            "requestId".into(),
            Value::String(uuid::Uuid::new_v4().to_string()),
        );
        capture.observe(&SessionUpdate::UserMessageChunk(
            ContentChunk::new(ContentBlock::Image(ImageContent::new("AAEC", "image/png")))
                .message_id(MessageId::from(uuid::Uuid::new_v4().to_string()))
                .meta(meta),
        ));

        let error = capture
            .finalize()
            .expect_err("replayed user images cannot project into the cache");
        assert!(error.to_string().contains("unsupported content"));
    }

    fn observe_completed_tool(
        capture: &mut RenoaReplayCapture,
        call_id: &'static str,
        content: Vec<ToolCallContent>,
    ) {
        let turn = uuid::Uuid::new_v4();
        let mut meta = Map::new();
        meta.insert("requestId".into(), Value::String(turn.to_string()));
        capture.observe(&SessionUpdate::UserMessageChunk(
            ContentChunk::new(ContentBlock::Text(TextContent::new("prompt")))
                .message_id(MessageId::from(uuid::Uuid::new_v4().to_string()))
                .meta(meta),
        ));
        capture.observe(&SessionUpdate::ToolCall(
            AcpToolCall::new(call_id, "run")
                .kind(ToolKind::Execute)
                .status(ToolCallStatus::InProgress),
        ));
        capture.observe(&SessionUpdate::ToolCallUpdate(ToolCallUpdate::new(
            call_id,
            ToolCallUpdateFields::new()
                .status(ToolCallStatus::Completed)
                .content(content),
        )));
    }

    #[test]
    fn mixed_text_and_image_tool_results_fail_before_any_replay_commit() {
        use agent_client_protocol::schema::v1::ImageContent;

        let mut capture = RenoaReplayCapture::default();
        observe_completed_tool(
            &mut capture,
            "mixed",
            vec![
                ToolCallContent::from(ContentBlock::Text(TextContent::new("A"))),
                ToolCallContent::from(ContentBlock::Image(ImageContent::new("AAEC", "image/png"))),
                ToolCallContent::from(ContentBlock::Text(TextContent::new("B"))),
            ],
        );

        let error = capture
            .finalize()
            .expect_err("mixed ordered tool content cannot be stored losslessly");
        assert!(
            error.to_string().contains("mixes text and image"),
            "the error must name the mixed shape: {error}"
        );
    }

    #[test]
    fn text_only_and_image_only_tool_results_still_project() {
        use agent_client_protocol::schema::v1::ImageContent;

        let mut text_capture = RenoaReplayCapture::default();
        observe_completed_tool(
            &mut text_capture,
            "text-only",
            vec![ToolCallContent::from(ContentBlock::Text(TextContent::new(
                "stdout",
            )))],
        );
        let text_replay = text_capture
            .finalize()
            .expect("text-only tool results work");
        let ReplayItem::ToolCall(text_tool) = &text_replay.replay.items[1] else {
            panic!("text tool is retained");
        };
        assert!(
            text_tool
                .activity
                .output
                .as_deref()
                .is_some_and(|output| output.contains("stdout"))
        );
        assert!(text_tool.activity.image_urls.is_empty());

        let mut image_capture = RenoaReplayCapture::default();
        observe_completed_tool(
            &mut image_capture,
            "image-only",
            vec![ToolCallContent::from(ContentBlock::Image(
                ImageContent::new("AAEC", "image/png"),
            ))],
        );
        let image_replay = image_capture
            .finalize()
            .expect("image-only tool results work");
        let ReplayItem::ToolCall(image_tool) = &image_replay.replay.items[1] else {
            panic!("image tool is retained");
        };
        assert_eq!(
            image_tool.activity.image_urls,
            vec!["data:image/png;base64,AAEC".to_owned()]
        );
    }

    #[test]
    fn image_only_tool_results_preserve_order_and_multiplicity() {
        let png = |data: &str| ImageContent::new(data, "image/png");
        let image = |content: ImageContent| ToolCallContent::from(ContentBlock::Image(content));

        let mut repeated = RenoaReplayCapture::default();
        observe_completed_tool(
            &mut repeated,
            "repeat",
            vec![image(png("AAEC")), image(png("AAEC"))],
        );
        let repeated_replay = repeated
            .finalize()
            .expect("repeated images are representable");
        let ReplayItem::ToolCall(repeated_tool) = &repeated_replay.replay.items[1] else {
            panic!("repeated image tool is retained");
        };
        assert_eq!(
            repeated_tool.activity.image_urls,
            vec![
                "data:image/png;base64,AAEC".to_owned(),
                "data:image/png;base64,AAEC".to_owned(),
            ],
            "duplicate payloads must not collapse"
        );

        let mut interleaved = RenoaReplayCapture::default();
        observe_completed_tool(
            &mut interleaved,
            "interleaved",
            vec![
                image(png("AAEC")),
                image(png("").uri("https://example.com/b.png")),
                image(png("AAEC")),
            ],
        );
        let interleaved_replay = interleaved
            .finalize()
            .expect("interleaved images are representable");
        let ReplayItem::ToolCall(interleaved_tool) = &interleaved_replay.replay.items[1] else {
            panic!("interleaved image tool is retained");
        };
        assert_eq!(
            interleaved_tool.activity.image_urls,
            vec![
                "data:image/png;base64,AAEC".to_owned(),
                "https://example.com/b.png".to_owned(),
                "data:image/png;base64,AAEC".to_owned(),
            ]
        );
    }

    #[test]
    fn a_renoa_message_id_cannot_change_semantic_roles() {
        let mut capture = RenoaReplayCapture::default();
        let message_id = uuid::Uuid::new_v4();
        let mut meta = Map::new();
        meta.insert(
            "requestId".into(),
            Value::String(uuid::Uuid::new_v4().to_string()),
        );
        capture.observe(&SessionUpdate::UserMessageChunk(
            ContentChunk::new(ContentBlock::Text(TextContent::new("prompt")))
                .message_id(MessageId::from(message_id.to_string()))
                .meta(meta),
        ));
        capture.observe(&SessionUpdate::AgentMessageChunk(
            ContentChunk::new(ContentBlock::Text(TextContent::new("answer")))
                .message_id(MessageId::from(message_id.to_string())),
        ));

        let error = capture
            .finalize()
            .expect_err("messageId role changes are ambiguous");
        assert!(error.to_string().contains("changed from user to assistant"));
    }

    #[test]
    fn renoa_tool_call_ids_may_be_reused_across_turns() {
        let mut capture = RenoaReplayCapture::default();
        for turn in [uuid::Uuid::new_v4(), uuid::Uuid::new_v4()] {
            let mut meta = Map::new();
            meta.insert("requestId".into(), Value::String(turn.to_string()));
            capture.observe(&SessionUpdate::UserMessageChunk(
                ContentChunk::new(ContentBlock::Text(TextContent::new("prompt")))
                    .message_id(MessageId::from(uuid::Uuid::new_v4().to_string()))
                    .meta(meta),
            ));
            capture.observe(&SessionUpdate::ToolCall(
                AcpToolCall::new("call-1", "run")
                    .kind(ToolKind::Execute)
                    .status(ToolCallStatus::InProgress),
            ));
            capture.observe(&SessionUpdate::ToolCallUpdate(ToolCallUpdate::new(
                "call-1",
                ToolCallUpdateFields::new()
                    .status(ToolCallStatus::Completed)
                    .content(vec![ToolCallContent::from(ContentBlock::Text(
                        TextContent::new(turn.to_string()),
                    ))]),
            )));
        }

        let replay = capture.finalize().expect("both turns are well formed");
        assert_eq!(replay.replay.items.len(), 4);
        let outputs = replay
            .replay
            .items
            .iter()
            .filter_map(|item| match item {
                ReplayItem::ToolCall(tool) => tool.activity.output.clone(),
                _ => None,
            })
            .collect::<Vec<_>>();
        assert_eq!(
            outputs.len(),
            2,
            "a reused call id must not overwrite the earlier turn's tool"
        );
    }

    #[test]
    fn renoa_tool_call_ids_may_be_reused_in_later_rounds_of_one_turn() {
        let mut capture = RenoaReplayCapture::default();
        let turn = uuid::Uuid::new_v4();
        let mut meta = Map::new();
        meta.insert("requestId".into(), Value::String(turn.to_string()));
        capture.observe(&SessionUpdate::UserMessageChunk(
            ContentChunk::new(ContentBlock::Text(TextContent::new("prompt")))
                .message_id(MessageId::from(uuid::Uuid::new_v4().to_string()))
                .meta(meta),
        ));

        for (round, output) in [(1, "first"), (2, "second")] {
            capture.observe(&SessionUpdate::ToolCall(
                AcpToolCall::new("call-reused", format!("run {round}"))
                    .kind(ToolKind::Execute)
                    .status(ToolCallStatus::InProgress),
            ));
            capture.observe(&SessionUpdate::ToolCallUpdate(ToolCallUpdate::new(
                "call-reused",
                ToolCallUpdateFields::new()
                    .status(ToolCallStatus::Completed)
                    .content(vec![ToolCallContent::from(ContentBlock::Text(
                        TextContent::new(output),
                    ))]),
            )));
            if round == 1 {
                capture.observe(&SessionUpdate::AgentMessageChunk(
                    ContentChunk::new(ContentBlock::Text(TextContent::new("continue")))
                        .message_id(MessageId::from(uuid::Uuid::new_v4().to_string())),
                ));
            }
        }

        let replay = capture.finalize().expect("both tool rounds are valid");
        let tools = replay
            .replay
            .items
            .iter()
            .filter_map(|item| match item {
                ReplayItem::ToolCall(tool) => Some(tool),
                _ => None,
            })
            .collect::<Vec<_>>();
        assert_eq!(tools.len(), 2);
        assert_eq!(tools[0].call_id, "call-reused");
        assert_eq!(tools[1].call_id, "call-reused");
        assert_ne!(tools[0].activity.output, tools[1].activity.output);
    }

    #[test]
    fn renoa_replay_keeps_complete_tool_output_beyond_the_live_preview_cap() {
        let mut capture = RenoaReplayCapture::default();
        let turn = uuid::Uuid::new_v4();
        let mut meta = Map::new();
        meta.insert("requestId".into(), Value::String(turn.to_string()));
        capture.observe(&SessionUpdate::UserMessageChunk(
            ContentChunk::new(ContentBlock::Text(TextContent::new("prompt")))
                .message_id(MessageId::from(uuid::Uuid::new_v4().to_string()))
                .meta(meta),
        ));
        capture.observe(&SessionUpdate::ToolCall(
            AcpToolCall::new("large-output", "run")
                .kind(ToolKind::Execute)
                .raw_input(json!({"payload": "y".repeat(40_000)}))
                .status(ToolCallStatus::InProgress),
        ));
        let output = format!("{}END", "x".repeat(40_000));
        let raw_output = json!({"diagnostic": format!("{}RAW-END", "z".repeat(40_000))});
        capture.observe(&SessionUpdate::ToolCallUpdate(ToolCallUpdate::new(
            "large-output",
            ToolCallUpdateFields::new()
                .status(ToolCallStatus::Completed)
                .content(vec![ToolCallContent::from(ContentBlock::Text(
                    TextContent::new(output),
                ))])
                .raw_output(raw_output),
        )));

        let replay = capture.finalize().expect("large output is supported");
        let ReplayItem::ToolCall(tool) = &replay.replay.items[1] else {
            panic!("tool result is retained");
        };
        let durable = tool.activity.durable_output().expect("durable output");
        assert!(durable.len() > 40_000);
        assert!(durable.contains("END"));
        assert!(!durable.contains("truncated"));
        let durable_raw = tool
            .activity
            .durable_raw_output()
            .expect("durable raw output");
        assert!(durable_raw.len() > 40_000);
        assert!(durable_raw.contains("RAW-END"));
        assert!(!durable_raw.contains("truncated"));
        assert!(
            tool.activity
                .output
                .as_ref()
                .is_some_and(|preview| preview.chars().count() < durable.chars().count()),
            "the disclosure preview stays bounded without truncating durable output"
        );
        let durable_arguments = tool
            .activity
            .durable_arguments()
            .expect("durable tool input");
        assert!(durable_arguments.len() > 40_000);
        assert!(
            tool.activity
                .arguments
                .as_ref()
                .is_some_and(|preview| preview.chars().count() < durable_arguments.chars().count())
        );
    }

    #[test]
    fn renoa_unsupported_assistant_order_fails_before_replay_is_emitted() {
        let mut capture = RenoaReplayCapture::default();
        let turn = uuid::Uuid::new_v4();
        let mut meta = Map::new();
        meta.insert("requestId".into(), Value::String(turn.to_string()));
        capture.observe(&SessionUpdate::UserMessageChunk(
            ContentChunk::new(ContentBlock::Text(TextContent::new("prompt")))
                .message_id(MessageId::from(uuid::Uuid::new_v4().to_string()))
                .meta(meta),
        ));
        let assistant = MessageId::from(uuid::Uuid::new_v4().to_string());
        for (reasoning, text) in [(true, "think"), (false, "answer"), (true, "more")] {
            let chunk = ContentChunk::new(ContentBlock::Text(TextContent::new(text)))
                .message_id(assistant.clone());
            if reasoning {
                capture.observe(&SessionUpdate::AgentThoughtChunk(chunk));
            } else {
                capture.observe(&SessionUpdate::AgentMessageChunk(chunk));
            }
        }

        let error = capture
            .finalize()
            .expect_err("alternating assistant content is not representable losslessly");
        assert!(error.to_string().contains("alternates text and reasoning"));
    }

    #[test]
    fn renoa_malformed_child_replay_emits_neither_replay_nor_connected() {
        let fixture = FakeAcpAgent::new();
        let durable_session = "3b1c0e7a-2f64-4c91-9d2e-0a1b2c3d4e5f";
        fixture.stage_load_history(&[renoa_notification(
            durable_session,
            json!({
                "sessionUpdate": "user_message_chunk",
                "content": {"type": "text", "text": "missing identity"},
                "messageId": uuid::Uuid::new_v4().to_string(),
            }),
        )]);
        fixture.stage_load_success();
        let _driver = start_fake_renoa(
            &fixture,
            Some(ProviderResumeCursor::Renoa {
                session_id: durable_session.into(),
            }),
        );

        let error = wait_for_renoa_error(&fixture.event_rx);
        assert!(error.contains("missing its requestId UUID"));
        assert!(fixture.event_rx.try_iter().all(|event| !matches!(
            event,
            DriverEvent::SessionReplayFragment { .. } | DriverEvent::Connected { .. }
        )));
    }

    #[test]
    fn mixed_tool_result_content_emits_neither_replay_nor_connected() {
        let fixture = FakeAcpAgent::new();
        let durable_session = "3b1c0e7a-2f64-4c91-9d2e-0a1b2c3d4e5f";
        fixture.stage_load_history(&[
            renoa_notification(
                durable_session,
                json!({
                    "sessionUpdate": "user_message_chunk",
                    "content": {"type": "text", "text": "prompt"},
                    "messageId": uuid::Uuid::new_v4().to_string(),
                    "_meta": {"requestId": uuid::Uuid::new_v4().to_string()},
                }),
            ),
            renoa_notification(
                durable_session,
                json!({
                    "sessionUpdate": "tool_call",
                    "toolCallId": "mixed",
                    "title": "run",
                    "kind": "execute",
                    "status": "in_progress",
                }),
            ),
            renoa_notification(
                durable_session,
                json!({
                    "sessionUpdate": "tool_call_update",
                    "toolCallId": "mixed",
                    "status": "completed",
                    "content": [
                        {"type": "content", "content": {"type": "text", "text": "A"}},
                        {
                            "type": "content",
                            "content": {"type": "image", "data": "AAEC", "mimeType": "image/png"}
                        },
                        {"type": "content", "content": {"type": "text", "text": "B"}}
                    ],
                }),
            ),
        ]);
        fixture.stage_load_success();
        let _driver = start_fake_renoa(
            &fixture,
            Some(ProviderResumeCursor::Renoa {
                session_id: durable_session.into(),
            }),
        );

        let error = wait_for_renoa_error(&fixture.event_rx);
        assert!(
            error.contains("mixes text and image"),
            "the load must fail closed: {error}"
        );
        assert!(fixture.event_rx.try_iter().all(|event| !matches!(
            event,
            DriverEvent::SessionReplayFragment { .. } | DriverEvent::Connected { .. }
        )));
    }

    fn renoa_notification(session_id: &str, update: Value) -> Value {
        json!({
            "jsonrpc": "2.0",
            "method": "session/update",
            "params": {
                "sessionId": session_id,
                "update": update,
            },
        })
    }

    /// Collects and decodes one committed replay transaction.
    fn wait_for_renoa_replay(fixture: &FakeAcpAgent) -> Vec<ReplayItem> {
        let mut assembler = waku_protocol::replay::ReplayFragmentAssembler::default();
        loop {
            match fixture
                .event_rx
                .recv_timeout(Duration::from_secs(10))
                .expect("Renoa should commit its replay")
            {
                DriverEvent::SessionReplayFragment {
                    replay_id,
                    index,
                    total,
                    json,
                } => {
                    let assembled = assembler
                        .accept(
                            waku_protocol::replay::ReplayFragment {
                                replay_id,
                                index,
                                total,
                                json,
                            },
                            waku_protocol::SESSION_REPLAY_FRAGMENT_BYTES,
                        )
                        .expect("replay fragments are contiguous and bounded");
                    if let Some(assembled) = assembled {
                        return assembled.decode().expect("replay JSON decodes").items;
                    }
                }
                other => panic!("unexpected event before the committed replay: {other:?}"),
            }
        }
    }

    #[test]
    fn renoa_replay_commits_the_durable_history_before_connected() {
        let fixture = FakeAcpAgent::new();
        let durable_session = "3b1c0e7a-2f64-4c91-9d2e-0a1b2c3d4e5f";
        let turn_id = uuid::Uuid::new_v4();
        let user_message_id = uuid::Uuid::new_v4();
        let assistant_id = uuid::Uuid::new_v4();
        let second_assistant_id = uuid::Uuid::new_v4();
        fixture.stage_load_history(&[
            renoa_notification(
                durable_session,
                json!({
                    "sessionUpdate": "user_message_chunk",
                    "content": {"type": "text", "text": "First"},
                    "messageId": user_message_id.to_string(),
                    "_meta": {"requestId": turn_id.to_string()},
                }),
            ),
            renoa_notification(
                durable_session,
                json!({
                    "sessionUpdate": "agent_thought_chunk",
                    "content": {"type": "text", "text": "thinking"},
                    "messageId": assistant_id.to_string(),
                }),
            ),
            renoa_notification(
                durable_session,
                json!({
                    "sessionUpdate": "agent_message_chunk",
                    "content": {"type": "text", "text": "Reply."},
                    "messageId": assistant_id.to_string(),
                }),
            ),
            renoa_notification(
                durable_session,
                json!({
                    "sessionUpdate": "tool_call",
                    "toolCallId": "call-1",
                    "title": "run bash",
                    "kind": "execute",
                    "status": "in_progress",
                    "rawInput": {"command": "ls"},
                }),
            ),
            renoa_notification(
                durable_session,
                json!({
                    "sessionUpdate": "tool_call_update",
                    "toolCallId": "call-1",
                    "status": "completed",
                    "content": [
                        {"type": "content", "content": {"type": "text", "text": "out"}}
                    ],
                    "rawOutput": {"exit": 0},
                }),
            ),
            renoa_notification(
                durable_session,
                json!({
                    "sessionUpdate": "tool_call",
                    "toolCallId": "call-2",
                    "title": "render chart",
                    "kind": "other",
                    "status": "in_progress",
                }),
            ),
            renoa_notification(
                durable_session,
                json!({
                    "sessionUpdate": "tool_call_update",
                    "toolCallId": "call-2",
                    "status": "completed",
                    "content": [
                        {
                            "type": "content",
                            "content": {"type": "image", "data": "AAEC", "mimeType": "image/png"}
                        }
                    ],
                }),
            ),
            renoa_notification(
                durable_session,
                json!({
                    "sessionUpdate": "agent_message_chunk",
                    "content": {"type": "text", "text": "Second."},
                    "messageId": second_assistant_id.to_string(),
                }),
            ),
        ]);
        fixture.stage_load_success();

        let _driver = start_fake_renoa(
            &fixture,
            Some(ProviderResumeCursor::Renoa {
                session_id: durable_session.into(),
            }),
        );

        // The very first events are the committed replay fragments; live deltas
        // never carry the load transcript.
        let items = wait_for_renoa_replay(&fixture);
        assert_eq!(items.len(), 5);
        let Some(ReplayItem::UserMessage(user)) = items.first() else {
            panic!("the replay starts with the durable user message");
        };
        assert_eq!(user.message_id, user_message_id);
        assert_eq!(user.turn_id, turn_id);
        assert_eq!(user.text, "First");
        let Some(ReplayItem::AssistantMessage(assistant)) = items.get(1) else {
            panic!("the assistant reply follows");
        };
        assert_eq!(assistant.message_id, assistant_id);
        assert_eq!(
            assistant.segments,
            vec![
                ReplaySegment::Reasoning("thinking".to_owned()),
                ReplaySegment::Text("Reply.".to_owned()),
            ]
        );
        let Some(ReplayItem::ToolCall(first_tool)) = items.get(2) else {
            panic!("the settled tool work follows");
        };
        assert_eq!(first_tool.call_id, "call-1");
        assert_eq!(first_tool.activity.kind, ActivityKind::Command);
        assert!(first_tool.activity.complete);
        assert!(
            first_tool
                .activity
                .arguments
                .as_deref()
                .is_some_and(|arguments| arguments.contains("ls")),
            "the raw input survives: {:?}",
            first_tool.activity.arguments
        );
        assert!(
            first_tool
                .activity
                .output
                .as_deref()
                .is_some_and(|output| output.contains("out")),
            "the tool output survives: {:?} and content was {:?}",
            first_tool.activity.output,
            first_tool.activity.arguments
        );
        let Some(ReplayItem::ToolCall(image_tool)) = items.get(3) else {
            panic!("the image tool follows");
        };
        assert_eq!(image_tool.call_id, "call-2");
        assert_eq!(
            image_tool.activity.image_urls,
            vec!["data:image/png;base64,AAEC".to_owned()],
            "replayed images round-trip losslessly"
        );
        let Some(ReplayItem::AssistantMessage(second)) = items.get(4) else {
            panic!("the final assistant message closes the replay");
        };
        assert_eq!(second.message_id, second_assistant_id);
        assert_eq!(
            second.segments,
            vec![ReplaySegment::Text("Second.".to_owned())]
        );

        // Connected arrives only after the committed replay.
        wait_for_renoa_connected(&fixture.event_rx);
        let methods = fixture.methods();
        assert!(methods.contains(&"initialize".to_owned()));
        assert!(methods.contains(&"session/load".to_owned()));
        assert!(
            !methods.contains(&"session/new".to_owned()),
            "a successful Renoa load must not create a replacement session"
        );
    }

    #[test]
    fn renoa_prompt_waits_for_the_durable_replay_acknowledgement() {
        let fixture = FakeAcpAgent::new();
        let durable_session = "3b1c0e7a-2f64-4c91-9d2e-0a1b2c3d4e5f";
        fixture.stage_load_history(&[]);
        fixture.stage_load_success();
        let driver = start_fake_renoa(
            &fixture,
            Some(ProviderResumeCursor::Renoa {
                session_id: durable_session.into(),
            }),
        );
        driver.prompt(TurnPrompt {
            id: uuid::Uuid::from_u128(90),
            prompt: "must wait".into(),
        });

        assert!(wait_for_renoa_replay(&fixture).is_empty());
        wait_for_renoa_connected(&fixture.event_rx);
        assert!(
            !fixture
                .methods()
                .iter()
                .any(|method| method == "session/prompt"),
            "the queued prompt crossed before durable replay acknowledgement"
        );

        assert!(driver.acknowledge_replay());
        loop {
            match fixture
                .event_rx
                .recv_timeout(Duration::from_secs(5))
                .expect("the acknowledged prompt should settle")
            {
                DriverEvent::TurnFinished { success: true, .. } => break,
                DriverEvent::Error(error) => panic!("the agent reported: {error}"),
                DriverEvent::ProcessExited => panic!("the agent exited before prompting"),
                _ => {}
            }
        }
        assert_eq!(
            fixture
                .methods()
                .iter()
                .filter(|method| *method == "session/prompt")
                .count(),
            1,
            "the acknowledgement must release the queued prompt exactly once"
        );
    }

    #[test]
    fn restored_renoa_usage_is_published_only_after_replay_acknowledgement() {
        let fixture = FakeAcpAgent::new();
        let durable_session = "3b1c0e7a-2f64-4c91-9d2e-0a1b2c3d4e5f";
        fixture.stage_load_history(&[renoa_notification(
            durable_session,
            json!({
                "sessionUpdate": "usage_update",
                "used": 12_345,
                "size": 500_000,
            }),
        )]);
        fixture.stage_load_success();
        let driver = start_fake_renoa(
            &fixture,
            Some(ProviderResumeCursor::Renoa {
                session_id: durable_session.into(),
            }),
        );

        assert!(wait_for_renoa_replay(&fixture).is_empty());
        wait_for_renoa_connected(&fixture.event_rx);
        assert!(
            fixture.event_rx.try_recv().is_err(),
            "restored usage crossed the durable replay barrier"
        );

        assert!(driver.acknowledge_replay());
        match fixture
            .event_rx
            .recv_timeout(Duration::from_secs(5))
            .expect("acknowledged replay should publish restored usage")
        {
            DriverEvent::UsageUpdated {
                context_tokens: Some(12_345),
                context_window: Some(500_000),
            } => {}
            other => panic!("unexpected event after replay acknowledgement: {other:?}"),
        }
    }

    #[test]
    fn cancelling_while_replay_is_uncommitted_never_releases_a_queued_prompt() {
        let fixture = FakeAcpAgent::new();
        let durable_session = "3b1c0e7a-2f64-4c91-9d2e-0a1b2c3d4e5f";
        fixture.stage_load_history(&[]);
        fixture.stage_load_success();
        let driver = start_fake_renoa(
            &fixture,
            Some(ProviderResumeCursor::Renoa {
                session_id: durable_session.into(),
            }),
        );
        driver.prompt(TurnPrompt {
            id: uuid::Uuid::from_u128(91),
            prompt: "must never run".into(),
        });

        assert!(wait_for_renoa_replay(&fixture).is_empty());
        wait_for_renoa_connected(&fixture.event_rx);
        driver.cancel();
        loop {
            match fixture
                .event_rx
                .recv_timeout(Duration::from_secs(5))
                .expect("cancelling the replay barrier should stop the runtime")
            {
                DriverEvent::ProcessExited => break,
                DriverEvent::TurnStarted | DriverEvent::TurnFinished { .. } => {
                    panic!("the cancelled prompt crossed the replay barrier")
                }
                _ => {}
            }
        }
        assert!(
            !fixture
                .methods()
                .iter()
                .any(|method| method == "session/prompt"),
            "the cancelled prompt reached Renoa"
        );
        assert!(!driver.acknowledge_replay());
    }

    #[test]
    fn renoa_load_collects_hundreds_of_ordered_chunks_before_responding() {
        let fixture = FakeAcpAgent::new();
        let durable_session = "3b1c0e7a-2f64-4c91-9d2e-0a1b2c3d4e5f";
        let turn_id = uuid::Uuid::new_v4();
        let user_id = uuid::Uuid::new_v4();
        let assistant_id = uuid::Uuid::new_v4();
        let mut notifications = vec![renoa_notification(
            durable_session,
            json!({
                "sessionUpdate": "user_message_chunk",
                "content": {"type": "text", "text": "prompt"},
                "messageId": user_id.to_string(),
                "_meta": {"requestId": turn_id.to_string()},
            }),
        )];
        for index in 0..200 {
            notifications.push(renoa_notification(
                durable_session,
                json!({
                    "sessionUpdate": "agent_message_chunk",
                    "content": {"type": "text", "text": format!("t{index},")},
                    "messageId": assistant_id.to_string(),
                }),
            ));
        }
        for index in 0..200 {
            notifications.push(renoa_notification(
                durable_session,
                json!({
                    "sessionUpdate": "agent_thought_chunk",
                    "content": {"type": "text", "text": format!("r{index},")},
                    "messageId": assistant_id.to_string(),
                }),
            ));
        }
        fixture.stage_load_history(&notifications);
        fixture.stage_load_success();
        let _driver = start_fake_renoa(
            &fixture,
            Some(ProviderResumeCursor::Renoa {
                session_id: durable_session.into(),
            }),
        );

        let items = wait_for_renoa_replay(&fixture);
        let ReplayItem::AssistantMessage(assistant) = &items[1] else {
            panic!("assistant chunks retain one durable message identity");
        };
        assert_eq!(assistant.segments.len(), 400);
        assert_eq!(assistant.segments[0], ReplaySegment::Text("t0,".into()));
        assert_eq!(assistant.segments[199], ReplaySegment::Text("t199,".into()));
        assert_eq!(
            assistant.segments[200],
            ReplaySegment::Reasoning("r0,".into())
        );
        assert_eq!(
            assistant.segments[399],
            ReplaySegment::Reasoning("r199,".into())
        );
        wait_for_renoa_connected(&fixture.event_rx);
    }

    #[test]
    fn renoa_large_replay_fragments_fit_the_complete_server_wire_envelope() {
        let replay = ProviderReplay {
            items: vec![ReplayItem::UserMessage(ReplayUserMessage {
                message_id: uuid::Uuid::new_v4(),
                turn_id: uuid::Uuid::new_v4(),
                text: "x".repeat(waku_protocol::SESSION_REPLAY_FRAGMENT_BYTES + 1),
            })],
        };
        let events = prepare_replay_events(&replay).expect("fragment large replay");
        assert!(events.len() > 1);
        for event in events {
            let wire = waku_protocol::event_to_wire(event).expect("encode fragment");
            let envelope = waku_protocol::ServerMessage::Event(waku_protocol::SequencedEvent {
                session_id: uuid::Uuid::new_v4(),
                runtime_id: uuid::Uuid::new_v4(),
                epoch: uuid::Uuid::new_v4(),
                sequence: u64::MAX,
                event: wire,
            });
            let size = serde_json::to_vec(&envelope)
                .expect("serialize complete replay envelope")
                .len();
            assert!(size <= waku_protocol::MAX_WIRE_MESSAGE_BYTES);
        }
    }

    #[test]
    fn a_partial_renoa_replay_with_a_failed_load_commits_nothing() {
        let fixture = FakeAcpAgent::new();
        fixture.stage_load_history(&[renoa_notification(
            "3b1c0e7a-2f64-4c91-9d2e-0a1b2c3d4e5f",
            json!({
                "sessionUpdate": "user_message_chunk",
                "content": {"type": "text", "text": "Half"},
                "messageId": uuid::Uuid::new_v4().to_string(),
                "_meta": {"requestId": uuid::Uuid::new_v4().to_string()},
            }),
        )]);
        let _driver = start_fake_renoa(
            &fixture,
            Some(ProviderResumeCursor::Renoa {
                session_id: "3b1c0e7a-2f64-4c91-9d2e-0a1b2c3d4e5f".into(),
            }),
        );

        let error = wait_for_renoa_error(&fixture.event_rx);
        assert!(error.contains("failed to load Renoa session"));
        // Drain everything the process still sends: the staged half-transcript
        // must produce neither a replay commit nor a connected session.
        loop {
            match fixture
                .event_rx
                .recv_timeout(Duration::from_secs(5))
                .expect("the process should exit after the failed load")
            {
                DriverEvent::ProcessExited => break,
                DriverEvent::SessionReplayFragment { .. } => {
                    panic!("a failed load committed a partial replay")
                }
                DriverEvent::Connected { .. } => {
                    panic!("a failed load produced a connected session")
                }
                _ => {}
            }
        }
        let methods = fixture.methods();
        assert!(!methods.contains(&"session/new".to_owned()));
    }

    #[test]
    fn a_malformed_renoa_replay_identity_fails_the_load() {
        let fixture = FakeAcpAgent::new();
        fixture.stage_load_history(&[
            // A user chunk without any messageId cannot anchor semantic
            // identity; the successful HTTP-shaped response changes nothing.
            renoa_notification(
                "3b1c0e7a-2f64-4c91-9d2e-0a1b2c3d4e5f",
                json!({
                    "sessionUpdate": "user_message_chunk",
                    "content": {"type": "text", "text": "No identity"},
                }),
            ),
        ]);
        fixture.stage_load_success();
        let _driver = start_fake_renoa(
            &fixture,
            Some(ProviderResumeCursor::Renoa {
                session_id: "3b1c0e7a-2f64-4c91-9d2e-0a1b2c3d4e5f".into(),
            }),
        );

        let error = wait_for_renoa_error(&fixture.event_rx);
        assert!(
            error.contains("failed to load Renoa session"),
            "actual: {error}"
        );
        assert!(error.contains("missing its messageId"), "actual: {error}");
        loop {
            match fixture
                .event_rx
                .recv_timeout(Duration::from_secs(5))
                .expect("the process should exit after the rejected replay")
            {
                DriverEvent::ProcessExited => break,
                DriverEvent::SessionReplayFragment { .. } => {
                    panic!("a malformed replay was committed")
                }
                DriverEvent::Connected { .. } => {
                    panic!("a malformed replay produced a connected session")
                }
                _ => {}
            }
        }
        let methods = fixture.methods();
        assert!(!methods.contains(&"session/new".to_owned()));
    }

    #[test]
    fn existing_acp_providers_still_fall_back_to_session_new() {
        for provider in [ProviderKind::Grok, ProviderKind::Cursor] {
            let fixture = FakeAcpAgent::new();
            let session_id = "3b1c0e7a-2f64-4c91-9d2e-0a1b2c3d4e5f";
            // Even when an agent replays updates during load, non-Renoa
            // providers keep discarding them and falling back.
            fixture.stage_load_history(&[renoa_notification(
                session_id,
                json!({
                    "sessionUpdate": "agent_message_chunk",
                    "content": {"type": "text", "text": "load-time leak"},
                    "messageId": uuid::Uuid::new_v4().to_string(),
                }),
            )]);
            let _driver = start_fake_acp(
                &fixture,
                provider,
                Some(ProviderResumeCursor::from_session_id(
                    provider,
                    session_id.into(),
                )),
            );
            let mut replay_seen = false;
            loop {
                match fixture
                    .event_rx
                    .recv_timeout(Duration::from_secs(5))
                    .expect("the agent should report its session")
                {
                    DriverEvent::Connected {
                        provider_cursor: Some(cursor),
                    } if cursor.provider() == provider => break,
                    DriverEvent::SessionReplayFragment { .. } => replay_seen = true,
                    DriverEvent::Error(error) => panic!("the agent reported: {error}"),
                    _ => {}
                }
            }
            assert!(
                !replay_seen,
                "{provider:?} must not emit authoritative replays"
            );
            let methods = fixture.methods();
            assert!(
                methods.contains(&"initialize".to_owned()),
                "{provider:?} never initialized"
            );
            assert!(
                methods.contains(&"session/load".to_owned()),
                "{provider:?} never attempted session/load"
            );
            assert!(
                methods.contains(&"session/new".to_owned()),
                "{provider:?} did not fall back to session/new"
            );
        }
    }

    #[test]
    fn fx_model_option_ignores_provider_selector_in_same_category() {
        let provider = select_config_option(
            "provider",
            SessionConfigOptionCategory::Model,
            "gateway",
            &["gateway", "codex", "grok"],
        );
        let model = select_config_option(
            "model",
            SessionConfigOptionCategory::Model,
            "openai/gpt-5.6-sol",
            &["openai/gpt-5.6-sol", "anthropic/claude-sonnet-5"],
        );

        assert_eq!(
            fx_model_option(&[provider, model]).map(|option| option.id.to_string()),
            Some("model".to_owned())
        );
    }

    #[test]
    fn fx_gateway_model_selects_the_gateway_route_first() {
        let provider = select_config_option(
            "provider",
            SessionConfigOptionCategory::Model,
            "codex",
            &["gateway", "codex", "grok"],
        );
        let model = select_config_option(
            "model",
            SessionConfigOptionCategory::Model,
            "gpt-5.6-luna",
            &["gpt-5.6-sol", "gpt-5.6-luna"],
        );
        let options = [provider, model];

        let (option, value) =
            fx_model_provider_switch(&options, "openai/gpt-5.6-luna-fast").unwrap();
        assert_eq!(option.id.to_string(), "provider");
        assert_eq!(value, "gateway");
    }

    #[test]
    fn cursor_model_aliases_resolve_to_advertised_parameterized_picker_values() {
        let option = select_config_option(
            "model",
            SessionConfigOptionCategory::Model,
            "default",
            &["default", "grok-4.6", "composer-2.5", "claude-sonnet-4-6"],
        );

        assert_eq!(
            cursor_model_selection(&option, "auto"),
            Some(CursorModelSelection {
                value: "default".into(),
                suffix: String::new(),
            })
        );
        assert_eq!(
            cursor_model_selection(&option, "composer-2.5"),
            Some(CursorModelSelection {
                value: "composer-2.5".into(),
                suffix: String::new(),
            })
        );
        assert_eq!(
            cursor_model_selection(&option, "cursor-grok-4.6-xhigh-fast"),
            Some(CursorModelSelection {
                value: "grok-4.6".into(),
                suffix: "xhigh-fast".into(),
            })
        );
        assert_eq!(
            cursor_model_selection(&option, "claude-4.6-sonnet-medium-thinking"),
            Some(CursorModelSelection {
                value: "claude-sonnet-4-6".into(),
                suffix: "medium-thinking".into(),
            })
        );
    }

    #[test]
    fn cursor_model_suffix_selects_dynamic_effort_thinking_and_fast_options() {
        let selection = CursorModelSelection {
            value: "claude-opus-5".into(),
            suffix: "thinking-extra-high-fast".into(),
        };
        let effort = select_config_option(
            "effort",
            SessionConfigOptionCategory::ThoughtLevel,
            "high",
            &["low", "medium", "high", "xhigh"],
        );
        let thinking = select_config_option(
            "thinking",
            SessionConfigOptionCategory::ModelConfig,
            "false",
            &["false", "true"],
        );
        let fast = select_config_option(
            "fast",
            SessionConfigOptionCategory::ModelConfig,
            "false",
            &["false", "true"],
        );

        assert_eq!(
            cursor_desired_select_value(&effort, &selection, None).as_deref(),
            Some("xhigh")
        );
        assert_eq!(
            cursor_desired_select_value(&thinking, &selection, None).as_deref(),
            Some("true")
        );
        assert_eq!(
            cursor_desired_select_value(&fast, &selection, None).as_deref(),
            Some("true")
        );
        assert_eq!(
            cursor_desired_select_value(&effort, &selection, Some("low")).as_deref(),
            Some("low")
        );
    }

    #[test]
    fn a_steer_only_settles_when_the_last_sdk_request_finishes() {
        let requests = Mutex::new(PendingPrompts::default());
        requests
            .lock()
            .insert(RequestId::Str("first".into()), None, "session".into());
        requests
            .lock()
            .insert(RequestId::Str("steer".into()), None, "session".into());
        assert!(!settle_prompt_request(
            &requests,
            &RequestId::Str("first".into())
        ));
        assert!(settle_prompt_request(
            &requests,
            &RequestId::Str("steer".into())
        ));
        assert!(!settle_prompt_request(
            &requests,
            &RequestId::Str("steer".into())
        ));
    }

    #[test]
    fn xai_prompt_complete_settles_a_missing_standard_response_once() {
        let requests = Mutex::new(PendingPrompts::default());
        let request_id = RequestId::Str("sdk-request".into());
        requests.lock().insert(
            request_id.clone(),
            Some("waku-prompt".into()),
            "grok-session".into(),
        );
        let (events, event_rx) = crate::driver::test_event_channel();

        assert_eq!(
            finish_xai_prompt_complete(
                &json!({
                    "sessionId": "grok-session",
                    "promptId": "waku-prompt",
                    "stopReason": "end_turn"
                }),
                &requests,
                &events,
            ),
            Some("grok-session".into())
        );
        assert!(matches!(
            event_rx.try_recv().unwrap(),
            DriverEvent::TurnFinished {
                success: true,
                summary: None
            }
        ));
        assert!(!settle_prompt_request(&requests, &request_id));
        assert!(event_rx.try_recv().is_err());
    }

    /// Kimi ends a failed turn with `end_turn` and no content at all, so the
    /// provider's own record is the only thing that can name the cause.
    #[test]
    fn a_recovered_provider_failure_overrides_a_clean_stop_reason() {
        let (events, event_rx) = crossbeam_channel::unbounded();

        assert!(!finish_prompt(
            Ok(PromptResponse::new(StopReason::EndTurn)),
            Some("402 membership inactive".to_owned()),
            &events
        ));

        assert!(matches!(
            event_rx.try_recv().unwrap(),
            DriverEvent::Error(message) if message == "402 membership inactive"
        ));
        assert!(matches!(
            event_rx.try_recv().unwrap(),
            DriverEvent::TurnFinished {
                success: false,
                summary: None
            }
        ));
    }

    #[test]
    fn typed_prompt_response_settles_the_turn() {
        let (events, event_rx) = crossbeam_channel::unbounded();
        assert!(finish_prompt(
            Ok(PromptResponse::new(StopReason::EndTurn)),
            None,
            &events
        ));
        assert!(matches!(
            event_rx.try_recv().unwrap(),
            DriverEvent::TurnFinished {
                success: true,
                summary: None
            }
        ));
    }

    #[test]
    fn typed_updates_preserve_text_reasoning_and_correlated_tools() {
        let (events, event_rx) = crossbeam_channel::unbounded();
        let mut state = AcpStreamState::default();
        let updates = [
            json!({"sessionUpdate":"agent_thought_chunk","content":{"type":"text","text":"thinking"}}),
            json!({"sessionUpdate":"tool_call","toolCallId":"call_1","title":"read","kind":"read","status":"pending","rawInput":{}}),
            json!({"sessionUpdate":"tool_call_update","toolCallId":"call_1","status":"completed","title":"fixture.txt","content":[{"type":"content","content":{"type":"text","text":"waku probe fixture"}}]}),
            json!({"sessionUpdate":"agent_message_chunk","content":{"type":"text","text":"OK"}}),
            json!({"sessionUpdate":"usage_update","used":9677,"size":500000}),
        ];
        for update in updates {
            let update = serde_json::from_value(update).unwrap();
            handle_session_update(
                ProviderKind::Cursor,
                SessionNotification::new("s", update),
                &events,
                &mut state,
            )
            .unwrap();
        }

        let seen = event_rx.try_iter().collect::<Vec<_>>();
        assert!(matches!(&seen[0], DriverEvent::ReasoningDelta(text) if text == "thinking"));
        assert!(matches!(&seen[1], DriverEvent::RichActivity(item)
                if item.kind == ActivityKind::FileRead && !item.complete));
        assert!(matches!(&seen[2], DriverEvent::RichActivity(item)
                if item.complete
                    && item.title == "fixture.txt"
                    && item.output.as_deref().is_some_and(|output| output.contains("waku probe fixture"))));
        assert!(matches!(&seen[3], DriverEvent::TextDelta(text) if text == "OK"));
        assert!(matches!(
            &seen[4],
            DriverEvent::UsageUpdated {
                context_tokens: Some(9677),
                context_window: Some(500000),
            }
        ));
    }

    #[test]
    fn fx_context_notices_do_not_become_assistant_text() {
        let (events, event_rx) = crossbeam_channel::unbounded();
        let mut state = AcpStreamState::default();
        for text in [
            "[context] skill catalog omitted 19 entries",
            "skill discovery warning: candidate was skipped",
            "Hi! How can I help?",
            "[context] is ordinary text after the answer starts",
        ] {
            let update = serde_json::from_value(json!({
                "sessionUpdate": "agent_message_chunk",
                "content": {"type": "text", "text": text}
            }))
            .unwrap();
            handle_session_update(
                ProviderKind::Fx,
                SessionNotification::new("s", update),
                &events,
                &mut state,
            )
            .unwrap();
        }

        let seen = event_rx.try_iter().collect::<Vec<_>>();
        assert_eq!(seen.len(), 2);
        assert!(matches!(&seen[0], DriverEvent::TextDelta(text) if text == "Hi! How can I help?"));
        assert!(matches!(&seen[1], DriverEvent::TextDelta(text) if text.starts_with("[context]")));
        assert!(state.produced_content);
    }

    #[test]
    fn permission_reason_preserves_the_agents_explanation() {
        let tool_call = ToolCallUpdate::new(
            "tool-1",
            serde_json::from_value::<ToolCallUpdateFields>(json!({
                "title": "rm -rf build",
                "kind": "execute",
                "content": [
                    {"type":"content","content":{"type":"text","text":"Not in allowlist: rm"}}
                ]
            }))
            .unwrap(),
        );
        let request = RequestPermissionRequest::new("s", tool_call, Vec::new());
        let params = serde_json::to_value(request).unwrap();
        assert_eq!(
            permission_reason(&params).as_deref(),
            Some("Not in allowlist: rm")
        );
    }

    /// Drives a real agent through the SDK-backed driver. Ignored by default:
    /// it needs the CLI installed, credentials, and the network.
    #[test]
    #[ignore = "requires an installed, authenticated grok"]
    fn grok_prompt_response_from_the_sdk_finishes_the_turn() {
        let binary = crate::command_env::find_executable("grok").expect("grok is not installed");
        let (events, event_rx) = crate::driver::test_event_channel();
        let driver = AcpDriver::start(
            ProviderKind::Grok,
            DriverStartOptions {
                binary,
                cwd: std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR")),
                mode: RuntimeMode::FullAccess,
                interaction_mode: InteractionMode::Build,
                model: Some("grok-4.5".into()),
                reasoning_effort: None,
                service_tier: None,
                context_window: None,
                agent_preset: None,
                computer_use_enabled: false,
                provider_cursor: None,
            },
            events,
        )
        .expect("the ACP session should open");

        loop {
            let event = event_rx
                .recv_timeout(Duration::from_secs(60))
                .expect("the agent should report its session");
            match event {
                DriverEvent::Connected {
                    provider_cursor: Some(ProviderResumeCursor::Grok { .. }),
                } => break,
                DriverEvent::Error(error) => panic!("the agent reported: {error}"),
                _ => {}
            }
        }
        driver.prompt(TurnPrompt::new(uuid::Uuid::new_v4(), "hi"));
        let mut finished = None;
        while let Ok(event) = event_rx.recv_timeout(Duration::from_secs(120)) {
            match event {
                DriverEvent::TurnFinished { success, .. } => {
                    finished = Some(success);
                    break;
                }
                DriverEvent::Error(error) => panic!("the agent reported: {error}"),
                _ => {}
            }
        }
        assert_eq!(finished, Some(true));
    }

    /// Covers Cursor's provider-private parameterized picker with a model id
    /// whose CLI alias carries both effort and fast-mode values.
    #[test]
    #[ignore = "requires an installed, authenticated cursor-agent"]
    fn cursor_parameterized_model_selection_finishes_a_real_turn() {
        let binary = crate::command_env::find_executable("cursor-agent")
            .expect("cursor-agent is not installed");
        let (events, event_rx) = crate::driver::test_event_channel();
        let driver = AcpDriver::start(
            ProviderKind::Cursor,
            DriverStartOptions {
                binary,
                cwd: std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR")),
                mode: RuntimeMode::FullAccess,
                interaction_mode: InteractionMode::Build,
                model: Some("cursor-grok-4.6-xhigh".into()),
                reasoning_effort: None,
                service_tier: None,
                context_window: None,
                agent_preset: None,
                computer_use_enabled: false,
                provider_cursor: None,
            },
            events,
        )
        .expect("the ACP session should open");

        loop {
            let event = event_rx
                .recv_timeout(Duration::from_secs(60))
                .expect("the agent should report its session");
            match event {
                DriverEvent::Connected {
                    provider_cursor: Some(ProviderResumeCursor::Cursor { .. }),
                } => break,
                DriverEvent::Error(error) => panic!("the agent reported: {error}"),
                _ => {}
            }
        }
        driver.prompt(TurnPrompt::new(uuid::Uuid::new_v4(), "Reply exactly OK."));

        let mut produced_text = false;
        let mut finished = None;
        while let Ok(event) = event_rx.recv_timeout(Duration::from_secs(120)) {
            match event {
                DriverEvent::TextDelta(text) => produced_text |= !text.is_empty(),
                DriverEvent::TurnFinished { success, .. } => {
                    finished = Some(success);
                    break;
                }
                DriverEvent::Error(error) => panic!("the agent reported: {error}"),
                _ => {}
            }
        }
        assert!(produced_text, "the Cursor turn produced no text");
        assert_eq!(finished, Some(true));
    }

    /// The invariant Kimi's silent failures break: a turn may finish
    /// successfully or report why it did not, but it must never claim success
    /// having produced nothing at all. Holds whether or not the account is
    /// currently able to serve the request.
    #[test]
    #[ignore = "requires an installed, authenticated kimi"]
    fn kimi_never_reports_an_empty_turn_as_a_success() {
        let binary = crate::command_env::find_executable("kimi").expect("kimi is not installed");
        let (events, event_rx) = crate::driver::test_event_channel();
        let driver = AcpDriver::start(
            ProviderKind::Kimi,
            DriverStartOptions {
                binary,
                cwd: std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR")),
                mode: RuntimeMode::FullAccess,
                interaction_mode: InteractionMode::Build,
                model: None,
                reasoning_effort: None,
                service_tier: None,
                context_window: None,
                agent_preset: None,
                computer_use_enabled: false,
                provider_cursor: None,
            },
            events,
        )
        .expect("the ACP session should open");

        loop {
            let event = event_rx
                .recv_timeout(Duration::from_secs(60))
                .expect("the agent should report its session");
            match event {
                DriverEvent::Connected {
                    provider_cursor: Some(ProviderResumeCursor::Kimi { .. }),
                } => break,
                DriverEvent::Error(error) => panic!("the agent reported: {error}"),
                _ => {}
            }
        }
        driver.prompt(TurnPrompt::new(
            uuid::Uuid::new_v4(),
            "Say hi in three words.",
        ));

        let mut produced_content = false;
        let mut reported_error = None;
        let mut finished = None;
        while let Ok(event) = event_rx.recv_timeout(Duration::from_secs(120)) {
            match event {
                DriverEvent::TextDelta(_) | DriverEvent::ReasoningDelta(_) => {
                    produced_content = true;
                }
                DriverEvent::Error(error) => reported_error = Some(error),
                DriverEvent::TurnFinished { success, .. } => {
                    finished = Some(success);
                    break;
                }
                _ => {}
            }
        }

        match finished.expect("the turn should settle") {
            true => assert!(
                produced_content,
                "the turn was reported successful without producing anything"
            ),
            false => assert!(
                reported_error.is_some_and(|error| !error.trim().is_empty()),
                "the turn failed without naming a reason"
            ),
        }
    }
}
