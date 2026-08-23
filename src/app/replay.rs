use super::*;

use std::fmt;

use waku_protocol::replay::{
    AssembledReplay, RenoaReplayCommit, ReplayFragment, ReplayFragmentAssembler,
};

mod persist;
use persist::prepare_and_persist_replay;
mod snapshot;
use snapshot::{ReplayPreparationSnapshot, ReplaySnapshotGuard};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) enum ReplayPhase {
    Receiving,
    AwaitingCursor,
    WaitingForHydration,
    Applying { generation: u64 },
    Committed,
    Failed,
}

pub(super) struct SessionReplayState {
    required: bool,
    phase: ReplayPhase,
    assembler: ReplayFragmentAssembler,
    assembled: Option<AssembledReplay>,
    accepted_cursor: Option<RuntimeEventCursor>,
    next_generation: u64,
}

impl SessionReplayState {
    pub(super) fn new(required: bool) -> Self {
        Self {
            required,
            phase: ReplayPhase::Receiving,
            assembler: ReplayFragmentAssembler::default(),
            assembled: None,
            accepted_cursor: None,
            next_generation: 0,
        }
    }

    pub(super) fn accept_fragment(
        &mut self,
        fragment: ReplayFragment,
    ) -> Result<(), ReplayApplyError> {
        if !self.required {
            return Err(ReplayApplyError::UnexpectedReplay);
        }
        if self.phase != ReplayPhase::Receiving {
            return Err(ReplayApplyError::UnexpectedFragmentPhase(self.phase));
        }
        if let Some(assembled) = self
            .assembler
            .accept(fragment, waku_protocol::SESSION_REPLAY_FRAGMENT_BYTES)
            .map_err(ReplayApplyError::Assembly)?
        {
            self.assembled = Some(assembled);
            self.phase = ReplayPhase::AwaitingCursor;
        }
        Ok(())
    }

    pub(super) fn intercept_cursor(&mut self, cursor: RuntimeEventCursor) -> bool {
        if self.phase != ReplayPhase::AwaitingCursor {
            return false;
        }
        self.accepted_cursor = Some(cursor);
        self.phase = ReplayPhase::WaitingForHydration;
        true
    }

    pub(super) fn blocks_event_drain(&self) -> bool {
        matches!(
            self.phase,
            ReplayPhase::WaitingForHydration | ReplayPhase::Applying { .. }
        )
    }

    pub(super) fn waiting_for_hydration(&self) -> bool {
        self.phase == ReplayPhase::WaitingForHydration
    }

    pub(super) fn is_control_cursor(&self) -> bool {
        self.phase == ReplayPhase::AwaitingCursor
    }

    pub(super) fn connected_is_allowed(&self) -> bool {
        !self.required || self.phase == ReplayPhase::Committed
    }

    pub(super) fn failed(&self) -> bool {
        self.phase == ReplayPhase::Failed
    }

    fn begin(&mut self) -> Option<ReplayWork> {
        if self.phase != ReplayPhase::WaitingForHydration {
            return None;
        }
        let assembled = self.assembled.take()?;
        let accepted_cursor = self.accepted_cursor?;
        self.next_generation = self.next_generation.saturating_add(1);
        let generation = self.next_generation;
        self.phase = ReplayPhase::Applying { generation };
        Some(ReplayWork {
            assembled,
            accepted_cursor,
            generation,
        })
    }

    fn applying(&self, generation: u64) -> bool {
        self.phase == ReplayPhase::Applying { generation }
    }

    fn commit(&mut self, generation: u64) -> bool {
        if !self.applying(generation) {
            return false;
        }
        self.phase = ReplayPhase::Committed;
        self.assembled = None;
        self.accepted_cursor = None;
        true
    }

    pub(super) fn fail(&mut self) {
        self.phase = ReplayPhase::Failed;
    }
}

struct ReplayWork {
    assembled: AssembledReplay,
    accepted_cursor: RuntimeEventCursor,
    generation: u64,
}

struct AppliedReplay {
    session: AgentSession,
    snapshot_guard: ReplaySnapshotGuard,
}

#[derive(Debug)]
pub(super) enum ReplayApplyError {
    UnexpectedReplay,
    UnexpectedFragmentPhase(ReplayPhase),
    Assembly(waku_protocol::replay::ReplayAssemblyError),
    Codec(waku_protocol::replay::ReplayCodecError),
    Reconcile(waku_protocol::replay::ReplayReconcileError),
    Snapshot(serde_json::Error),
    CommitCodec(waku_protocol::replay::ReplayCodecError),
    CommitWireSerialization(serde_json::Error),
    CommitWireOversize { size: usize, maximum: usize },
    Daemon(anyhow::Error),
    InvalidDaemonResponse,
    PersistedProjectionMismatch,
    PersistedImageMismatch,
    ReplayBaseCursorMismatch,
    ConflictingDaemonTail,
    InvalidLocalTail,
    Hydration(String),
    StaleSession,
    ConnectedBeforeCommit,
}

impl fmt::Display for ReplayApplyError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::UnexpectedReplay => {
                write!(formatter, "unexpected replay for a new Renoa session")
            }
            Self::UnexpectedFragmentPhase(phase) => {
                write!(formatter, "replay fragment arrived during {phase:?}")
            }
            Self::Assembly(error) => write!(formatter, "invalid replay transaction: {error}"),
            Self::Codec(error) => write!(formatter, "invalid replay encoding: {error}"),
            Self::Reconcile(error) => write!(formatter, "unsupported replay transcript: {error}"),
            Self::Snapshot(error) => write!(formatter, "could not snapshot replay state: {error}"),
            Self::CommitCodec(error) => {
                write!(formatter, "could not encode replay commit: {error}")
            }
            Self::CommitWireSerialization(error) => {
                write!(
                    formatter,
                    "could not serialize replay commit request: {error}"
                )
            }
            Self::CommitWireOversize { size, maximum } => write!(
                formatter,
                "serialized replay commit request is {size} bytes, exceeding the {maximum}-byte wire bound"
            ),
            Self::Daemon(error) => write!(formatter, "could not persist Renoa replay: {error}"),
            Self::InvalidDaemonResponse => {
                write!(
                    formatter,
                    "daemon returned an invalid replay-commit response"
                )
            }
            Self::PersistedProjectionMismatch => write!(
                formatter,
                "daemon persisted transcript content different from the validated replay"
            ),
            Self::PersistedImageMismatch => write!(
                formatter,
                "daemon persisted image content different from the validated replay"
            ),
            Self::ReplayBaseCursorMismatch => write!(
                formatter,
                "daemon and desktop disagree about the Renoa replay base cursor"
            ),
            Self::ConflictingDaemonTail => write!(
                formatter,
                "daemon holds a different unresolved Renoa turn than the desktop snapshot"
            ),
            Self::InvalidLocalTail => write!(
                formatter,
                "the unresolved local Renoa turn is not a contiguous transcript tail"
            ),
            Self::Hydration(error) => {
                write!(formatter, "could not hydrate the replay target: {error}")
            }
            Self::StaleSession => write!(
                formatter,
                "session changed while its replay was being prepared; refusing a stale overwrite"
            ),
            Self::ConnectedBeforeCommit => write!(
                formatter,
                "Renoa reported Connected before its authoritative replay was committed"
            ),
        }
    }
}

impl std::error::Error for ReplayApplyError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Assembly(error) => Some(error),
            Self::Codec(error) | Self::CommitCodec(error) => Some(error),
            Self::Reconcile(error) => Some(error),
            Self::Snapshot(error) | Self::CommitWireSerialization(error) => Some(error),
            Self::Daemon(error) => Some(error.as_ref()),
            Self::UnexpectedReplay
            | Self::UnexpectedFragmentPhase(_)
            | Self::CommitWireOversize { .. }
            | Self::InvalidDaemonResponse
            | Self::PersistedProjectionMismatch
            | Self::PersistedImageMismatch
            | Self::ReplayBaseCursorMismatch
            | Self::ConflictingDaemonTail
            | Self::InvalidLocalTail
            | Self::Hydration(_)
            | Self::StaleSession
            | Self::ConnectedBeforeCommit => None,
        }
    }
}

impl Waku {
    pub(super) fn receive_session_replay_fragment(
        &mut self,
        runtime: &mut SessionRuntime,
        fragment: ReplayFragment,
    ) -> Result<(), ReplayApplyError> {
        runtime.replay.accept_fragment(fragment)
    }

    pub(super) fn accept_replay_cursor(
        &mut self,
        session_id: Uuid,
        runtime: &mut SessionRuntime,
        cursor: RuntimeEventCursor,
        cx: &mut Context<Self>,
    ) -> bool {
        if !runtime.replay.intercept_cursor(cursor) {
            return false;
        }
        self.start_session_replay_if_ready(session_id, runtime, cx);
        true
    }

    pub(super) fn apply_pending_session_replay(
        &mut self,
        session_id: Uuid,
        cx: &mut Context<Self>,
    ) {
        let Some(mut runtime) = self.runtimes.remove(&session_id) else {
            return;
        };
        self.start_session_replay_if_ready(session_id, &mut runtime, cx);
        self.runtimes.insert(session_id, runtime);
    }

    fn start_session_replay_if_ready(
        &mut self,
        session_id: Uuid,
        runtime: &mut SessionRuntime,
        cx: &mut Context<Self>,
    ) {
        let Some(session) = self
            .state
            .sessions
            .iter()
            .find(|session| session.id == session_id)
            .filter(|session| can_apply_session_replay(session.provider, session.detail_loaded))
        else {
            return;
        };
        let snapshot = match ReplayPreparationSnapshot::capture(session) {
            Ok(snapshot) => snapshot,
            Err(error) => {
                self.fail_session_replay(session_id, runtime, error);
                return;
            }
        };
        let Some(work) = runtime.replay.begin() else {
            return;
        };
        let runtime_instance = runtime.instance_id;
        let daemon = self.daemon.client();
        let reconciled_at = unix_time();
        let generation = work.generation;

        cx.spawn(async move |waku, cx| {
            let result =
                cx.background_executor()
                    .spawn(async move {
                        prepare_and_persist_replay(daemon, snapshot, work, reconciled_at)
                    })
                    .await;
            let _ = waku.update(cx, |waku, cx| {
                waku.finish_session_replay(session_id, runtime_instance, generation, result, cx);
            });
        })
        .detach();
    }

    fn finish_session_replay(
        &mut self,
        session_id: Uuid,
        runtime_instance: Uuid,
        generation: u64,
        result: Result<AppliedReplay, ReplayApplyError>,
        cx: &mut Context<Self>,
    ) {
        let Some(mut runtime) = self.runtimes.remove(&session_id) else {
            return;
        };
        if runtime.instance_id != runtime_instance || !runtime.replay.applying(generation) {
            self.runtimes.insert(session_id, runtime);
            return;
        }
        let applied = match result {
            Ok(applied) => applied,
            Err(error) => {
                runtime.replay.fail();
                runtime.last_driver_error = Some(error.to_string());
                runtime.driver.close();
                self.runtimes.insert(session_id, runtime);
                self.surface_session_replay_error(session_id, error);
                signal_event_pump(&self.event_wake_tx);
                cx.notify();
                return;
            }
        };
        let Some(session) = self
            .state
            .sessions
            .iter_mut()
            .find(|session| session.id == session_id)
        else {
            runtime.replay.fail();
            runtime.last_driver_error = Some(ReplayApplyError::StaleSession.to_string());
            runtime.driver.close();
            self.runtimes.insert(session_id, runtime);
            signal_event_pump(&self.event_wake_tx);
            cx.notify();
            return;
        };
        // While Applying, the event drain is closed and Connecting prevents
        // every transcript-writing send/edit path. Stop replaces this runtime,
        // hydration has already completed, and remote catalog sync only merges
        // list metadata. The runtime instance + replay generation above and
        // this cheap snapshot guard therefore cover every legal local writer;
        // the daemon separately compares the exact serialized base before its
        // SQLite commit.
        if !applied.snapshot_guard.still_current(session) {
            runtime.replay.fail();
            runtime.last_driver_error = Some(ReplayApplyError::StaleSession.to_string());
            runtime.driver.close();
            self.runtimes.insert(session_id, runtime);
            self.surface_session_replay_error(session_id, ReplayApplyError::StaleSession);
            signal_event_pump(&self.event_wake_tx);
            cx.notify();
            return;
        }
        session.messages = applied.session.messages;
        session.transcript_blocks = applied.session.transcript_blocks;
        session.turns = applied.session.turns;
        session.runtime_event_cursor = applied.session.runtime_event_cursor;
        session.renoa_replay_cursor = applied.session.renoa_replay_cursor;
        session.updated_at = session.updated_at.max(applied.session.updated_at);
        if !runtime.replay.commit(generation) {
            self.runtimes.insert(session_id, runtime);
            return;
        }
        runtime.last_driver_error = None;
        self.runtimes.insert(session_id, runtime);
        if self.state.selected_session == Some(session_id) {
            self.reset_visible_state();
            self.reset_transcript_rows(self.transcript_row_count());
        }
        signal_event_pump(&self.event_wake_tx);
        cx.notify();
    }

    pub(super) fn fail_session_replay(
        &mut self,
        session_id: Uuid,
        runtime: &mut SessionRuntime,
        error: ReplayApplyError,
    ) {
        runtime.replay.fail();
        runtime.last_driver_error = Some(error.to_string());
        runtime.driver.close();
        self.surface_session_replay_error(session_id, error);
    }

    pub(super) fn fail_replay_hydration(
        &mut self,
        session_id: Uuid,
        hydration_error: &anyhow::Error,
    ) -> bool {
        let Some(mut runtime) = self.runtimes.remove(&session_id) else {
            return false;
        };
        if !runtime.replay.waiting_for_hydration() {
            self.runtimes.insert(session_id, runtime);
            return false;
        }
        self.fail_session_replay(
            session_id,
            &mut runtime,
            ReplayApplyError::Hydration(hydration_error.to_string()),
        );
        self.runtimes.insert(session_id, runtime);
        true
    }

    fn surface_session_replay_error(&mut self, session_id: Uuid, error: ReplayApplyError) {
        if self.state.selected_session == Some(session_id) {
            self.show_toast(error.to_string());
        }
    }
}

#[cfg(test)]
mod tests;
