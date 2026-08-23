use crate::model::{
    AgentSession, AgentTurn, Message, RuntimeEventCursor, TranscriptBlock, TurnStatus,
};

#[derive(Clone, Copy, Debug, serde::Deserialize, serde::Serialize, ts_rs::TS)]
#[serde(rename_all = "camelCase")]
pub struct RenoaReplayBase {
    pub cursor: Option<RuntimeEventCursor>,
    pub projection_fingerprint: [u64; 2],
    pub settled_projection_fingerprint: [u64; 2],
    pub active_tail_fingerprint: Option<[u64; 2]>,
}

/// Fingerprints the daemon-owned replay base and any unresolved local tail.
/// The full transcript stays daemon-side; the desktop needs only these exact
/// compare-and-swap facts before preparing a replacement off-thread.
pub fn renoa_replay_base(session: &AgentSession) -> Result<RenoaReplayBase, serde_json::Error> {
    let active_turn_id = session.active_turn_id();
    let projection = serde_json::to_vec(&(
        &session.messages,
        &session.transcript_blocks,
        &session.turns,
        session.runtime_event_cursor,
        session.renoa_replay_cursor,
    ))?;
    let settled_projection = serde_json::to_vec(&(
        session
            .messages
            .iter()
            .filter(|message| message.turn_id != active_turn_id)
            .collect::<Vec<_>>(),
        session
            .transcript_blocks
            .iter()
            .filter(|block| block.turn_id != active_turn_id)
            .collect::<Vec<_>>(),
        session
            .turns
            .iter()
            .filter(|turn| Some(turn.id) != active_turn_id)
            .collect::<Vec<_>>(),
        session.runtime_event_cursor,
        session.renoa_replay_cursor,
    ))?;
    let active_tail_fingerprint = replay_active_tail_fingerprint(
        &session.messages,
        &session.transcript_blocks,
        &session.turns,
    )?;
    Ok(RenoaReplayBase {
        cursor: session.runtime_event_cursor,
        projection_fingerprint: replay_projection_fingerprint(&projection),
        settled_projection_fingerprint: replay_projection_fingerprint(&settled_projection),
        active_tail_fingerprint,
    })
}

/// Fingerprints the one unresolved local turn in a transcript projection.
/// This is kept separate from the pre-reconciliation base because replay may
/// adopt that turn under Renoa's authoritative message identities.
pub fn replay_active_tail_fingerprint(
    messages: &[Message],
    transcript_blocks: &[TranscriptBlock],
    turns: &[AgentTurn],
) -> Result<Option<[u64; 2]>, serde_json::Error> {
    let active_turn_id = turns
        .last()
        .filter(|turn| turn.status == TurnStatus::Running)
        .map(|turn| turn.id);
    active_turn_id
        .map(|turn_id| {
            let turn = turns.iter().find(|turn| turn.id == turn_id);
            let messages = messages
                .iter()
                .filter(|message| message.turn_id == Some(turn_id))
                .collect::<Vec<_>>();
            let blocks = transcript_blocks
                .iter()
                .filter(|block| block.turn_id == Some(turn_id))
                .collect::<Vec<_>>();
            serde_json::to_vec(&(turn, messages, blocks))
                .map(|bytes| replay_projection_fingerprint(&bytes))
        })
        .transpose()
}

/// The replayed replacement for a transcript's covered region.
#[derive(Clone, Debug)]
pub struct ReconciledTranscript {
    pub messages: Vec<Message>,
    pub transcript_blocks: Vec<TranscriptBlock>,
    pub turns: Vec<AgentTurn>,
}

/// Exact transcript-only persistence request for one committed Renoa replay.
/// The daemon validates the live runtime, base cursor, and exact base
/// projection before replacing these fields in one SQLite transaction; all
/// chat metadata stays untouched.
#[derive(Clone, Debug, serde::Deserialize, serde::Serialize)]
#[serde(rename_all = "camelCase")]
pub struct RenoaReplayCommit {
    pub expected_cursor: Option<RuntimeEventCursor>,
    /// Deterministic digest of the exact transcript snapshot reconciliation
    /// read. The daemon checks it while holding the task-state lock, closing
    /// the interval between off-thread preparation and durable replacement.
    pub expected_projection_fingerprint: [u64; 2],
    /// The same base with its one unresolved local turn removed. This lets a
    /// delayed ordinary save publish that exact tail between base-read and
    /// commit without making correctness depend on timing.
    pub expected_settled_projection_fingerprint: [u64; 2],
    /// Exact unresolved desktop tail retained by reconciliation. A daemon tail
    /// may be absent (the ordinary save has not landed yet) or must match this.
    pub local_active_tail_fingerprint: Option<[u64; 2]>,
    /// Active tail after authoritative reconciliation. This can differ from
    /// the local tail when replay adopts it under agent-owned message IDs.
    pub reconciled_active_tail_fingerprint: Option<[u64; 2]>,
    pub accepted_cursor: RuntimeEventCursor,
    pub messages: Vec<Message>,
    pub transcript_blocks: Vec<TranscriptBlock>,
    pub turns: Vec<AgentTurn>,
    pub reconciled_at: u64,
}

/// Dependency-free deterministic digest for an internal compare-and-swap.
/// Two independently seeded directions make accidental collisions negligible;
/// this is concurrency detection, not an authentication boundary.
pub fn replay_projection_fingerprint(bytes: &[u8]) -> [u64; 2] {
    fn fold(bytes: impl Iterator<Item = u8>, seed: u64, prime: u64) -> u64 {
        bytes.fold(seed, |hash, byte| {
            (hash ^ u64::from(byte)).wrapping_mul(prime)
        })
    }

    [
        fold(
            bytes.iter().copied(),
            0xcbf2_9ce4_8422_2325,
            0x0000_0100_0000_01b3,
        ),
        fold(
            bytes.iter().rev().copied(),
            0x8422_2325_cbf2_9ce4,
            0x9e37_79b1_85eb_ca87,
        ),
    ]
}
