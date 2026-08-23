use super::*;

use base64::Engine as _;

pub(super) fn prepare_and_persist_replay(
    daemon: waku_client::DaemonClient,
    snapshot: ReplayPreparationSnapshot,
    work: ReplayWork,
    reconciled_at: u64,
) -> Result<AppliedReplay, ReplayApplyError> {
    let ReplayWork {
        assembled,
        accepted_cursor,
        generation: _,
    } = work;
    let commit_id = assembled.replay_id;
    let local_active_tail_fingerprint = snapshot.active_tail_fingerprint()?;
    let response = daemon
        .request(
            snapshot.session_id,
            accepted_cursor.runtime_id,
            waku_client::Command::ReadRenoaReplayBase,
        )
        .map_err(ReplayApplyError::Daemon)?;
    let waku_client::ResponsePayload::RenoaReplayBase {
        base: daemon_base,
        session: mut reconciliation_session,
    } = response
    else {
        return Err(ReplayApplyError::InvalidDaemonResponse);
    };
    if reconciliation_session.id != snapshot.session_id {
        return Err(ReplayApplyError::InvalidDaemonResponse);
    }
    if daemon_base.cursor != snapshot.runtime_event_cursor {
        return Err(ReplayApplyError::ReplayBaseCursorMismatch);
    }
    if daemon_base.active_tail_fingerprint.is_some()
        && daemon_base.active_tail_fingerprint != local_active_tail_fingerprint
    {
        return Err(ReplayApplyError::ConflictingDaemonTail);
    }
    if daemon_base.active_tail_fingerprint.is_none() {
        snapshot.append_active_tail(&mut reconciliation_session)?;
    }
    let replay = assembled.decode().map_err(ReplayApplyError::Codec)?;
    let reconciled = reconcile_replay(
        &reconciliation_session.messages,
        &reconciliation_session.transcript_blocks,
        &reconciliation_session.turns,
        &replay,
        reconciled_at,
    )
    .map_err(ReplayApplyError::Reconcile)?;
    let reconciled_active_tail_fingerprint = waku_protocol::replay::replay_active_tail_fingerprint(
        &reconciled.messages,
        &reconciled.transcript_blocks,
        &reconciled.turns,
    )
    .map_err(ReplayApplyError::Snapshot)?;
    let commit = RenoaReplayCommit {
        expected_cursor: daemon_base.cursor,
        expected_projection_fingerprint: daemon_base.projection_fingerprint,
        expected_settled_projection_fingerprint: daemon_base.settled_projection_fingerprint,
        local_active_tail_fingerprint,
        reconciled_active_tail_fingerprint,
        accepted_cursor,
        messages: reconciled.messages,
        transcript_blocks: reconciled.transcript_blocks,
        turns: reconciled.turns,
        reconciled_at,
    };
    let expected_projection = commit_projection_fingerprint(&commit)?;
    let fragments = commit
        .encode_fragments(commit_id, waku_protocol::SESSION_REPLAY_FRAGMENT_BYTES)
        .map_err(ReplayApplyError::CommitCodec)?;
    let commands = fragments
        .into_iter()
        .map(|fragment| waku_client::Command::CommitRenoaReplay { fragment })
        .collect::<Vec<_>>();
    // Preflight every complete client envelope before the first fragment can
    // mutate the daemon's transient assembler.
    for command in &commands {
        let message = waku_client::ClientMessage::Request(waku_client::Request {
            request_id: Uuid::nil(),
            session_id: snapshot.session_id,
            runtime_id: accepted_cursor.runtime_id,
            command: command.clone(),
        });
        let size = serde_json::to_vec(&message)
            .map_err(ReplayApplyError::CommitWireSerialization)?
            .len();
        if size > waku_protocol::MAX_WIRE_MESSAGE_BYTES {
            return Err(ReplayApplyError::CommitWireOversize {
                size,
                maximum: waku_protocol::MAX_WIRE_MESSAGE_BYTES,
            });
        }
    }
    let mut committed_session = None;
    let command_count = commands.len();
    for (index, command) in commands.into_iter().enumerate() {
        let response = daemon
            .request(snapshot.session_id, accepted_cursor.runtime_id, command)
            .map_err(ReplayApplyError::Daemon)?;
        if index + 1 == command_count {
            let waku_client::ResponsePayload::RenoaReplayCommitted { session } = response else {
                return Err(ReplayApplyError::InvalidDaemonResponse);
            };
            committed_session = Some(session);
        } else if !matches!(response, waku_client::ResponsePayload::Ack) {
            return Err(ReplayApplyError::InvalidDaemonResponse);
        }
    }
    let Some(session) = committed_session else {
        return Err(ReplayApplyError::InvalidDaemonResponse);
    };
    if persisted_projection_fingerprint(&session)? != expected_projection {
        return Err(ReplayApplyError::PersistedProjectionMismatch);
    }
    validate_persisted_images(&daemon, &commit, &session)?;
    Ok(AppliedReplay {
        session,
        snapshot_guard: snapshot.snapshot_guard,
    })
}

fn commit_projection_fingerprint(commit: &RenoaReplayCommit) -> Result<Vec<u8>, ReplayApplyError> {
    let blocks = blocks_without_image_payloads(&commit.transcript_blocks);
    serde_json::to_vec(&(
        &commit.messages,
        blocks,
        &commit.turns,
        Some(commit.accepted_cursor),
        Some(commit.accepted_cursor),
    ))
    .map_err(ReplayApplyError::Snapshot)
}

fn persisted_projection_fingerprint(session: &AgentSession) -> Result<Vec<u8>, ReplayApplyError> {
    let blocks = blocks_without_image_payloads(&session.transcript_blocks);
    serde_json::to_vec(&(
        &session.messages,
        blocks,
        &session.turns,
        session.runtime_event_cursor,
        session.renoa_replay_cursor,
    ))
    .map_err(ReplayApplyError::Snapshot)
}

fn blocks_without_image_payloads(blocks: &[TranscriptBlock]) -> Vec<TranscriptBlock> {
    let mut blocks = blocks.to_vec();
    for block in &mut blocks {
        for activity in &mut block.activities {
            for image in &mut activity.image_urls {
                image.clear();
            }
        }
    }
    blocks
}

fn validate_persisted_images(
    daemon: &waku_client::DaemonClient,
    commit: &RenoaReplayCommit,
    session: &AgentSession,
) -> Result<(), ReplayApplyError> {
    let expected = commit
        .transcript_blocks
        .iter()
        .flat_map(|block| block.activities.iter())
        .flat_map(|activity| activity.image_urls.iter());
    let persisted = session
        .transcript_blocks
        .iter()
        .flat_map(|block| block.activities.iter())
        .flat_map(|activity| activity.image_urls.iter());
    for (expected, persisted) in expected.zip(persisted) {
        if expected == persisted {
            continue;
        }
        let Some(expected_bytes) = decode_data_url(expected) else {
            return Err(ReplayApplyError::PersistedImageMismatch);
        };
        if !waku_protocol::blob::is_reference(persisted) {
            return Err(ReplayApplyError::PersistedImageMismatch);
        }
        let response = daemon
            .request(
                session.id,
                commit.accepted_cursor.runtime_id,
                waku_client::Command::ReadBlob {
                    reference: persisted.clone(),
                },
            )
            .map_err(ReplayApplyError::Daemon)?;
        let waku_client::ResponsePayload::BlobData { bytes } = response else {
            return Err(ReplayApplyError::InvalidDaemonResponse);
        };
        if bytes != expected_bytes {
            return Err(ReplayApplyError::PersistedImageMismatch);
        }
    }
    Ok(())
}

fn decode_data_url(value: &str) -> Option<Vec<u8>> {
    let (header, encoded) = value.split_once(',')?;
    let header = header.strip_prefix("data:")?;
    header.contains(";base64").then_some(())?;
    base64::engine::general_purpose::STANDARD
        .decode(encoded)
        .ok()
}
