use super::*;

use crate::model::AgentTurn;

/// The only local transcript state that can legitimately be newer than the
/// daemon while a Renoa load is gated: one accepted running turn. Settled
/// presentation history comes from the daemon's background-fetched snapshot,
/// so starting reconciliation never clones a long transcript on GPUI.
pub(super) struct ReplayPreparationSnapshot {
    pub(super) session_id: Uuid,
    pub(super) runtime_event_cursor: Option<RuntimeEventCursor>,
    pub(super) active_tail: Option<ReplayLocalTail>,
    pub(super) snapshot_guard: ReplaySnapshotGuard,
}

pub(super) struct ReplayLocalTail {
    first_message_index: usize,
    pub(super) messages: Vec<Message>,
    pub(super) transcript_blocks: Vec<TranscriptBlock>,
    turn: AgentTurn,
}

impl ReplayPreparationSnapshot {
    pub(super) fn capture(session: &AgentSession) -> Result<Self, ReplayApplyError> {
        let snapshot_guard = ReplaySnapshotGuard::capture(session);
        let active_tail = session
            .active_turn_id()
            .map(|turn_id| {
                let turn = session
                    .turns
                    .last()
                    .filter(|turn| turn.id == turn_id)
                    .cloned()
                    .ok_or(ReplayApplyError::InvalidLocalTail)?;
                let first_message_index = session
                    .messages
                    .iter()
                    .rposition(|message| message.turn_id != Some(turn_id))
                    .map_or(0, |index| index + 1);
                let messages = session.messages[first_message_index..].to_vec();
                if messages.is_empty()
                    || messages
                        .iter()
                        .any(|message| message.turn_id != Some(turn_id))
                {
                    return Err(ReplayApplyError::InvalidLocalTail);
                }
                let first_block_index = session
                    .transcript_blocks
                    .iter()
                    .rposition(|block| block.turn_id != Some(turn_id))
                    .map_or(0, |index| index + 1);
                let transcript_blocks = session.transcript_blocks[first_block_index..].to_vec();
                if transcript_blocks
                    .iter()
                    .any(|block| block.turn_id != Some(turn_id))
                {
                    return Err(ReplayApplyError::InvalidLocalTail);
                }
                Ok(ReplayLocalTail {
                    first_message_index,
                    messages,
                    transcript_blocks,
                    turn,
                })
            })
            .transpose()?;
        Ok(Self {
            session_id: session.id,
            runtime_event_cursor: session.runtime_event_cursor,
            active_tail,
            snapshot_guard,
        })
    }

    pub(super) fn active_tail_fingerprint(&self) -> Result<Option<[u64; 2]>, ReplayApplyError> {
        let Some(tail) = &self.active_tail else {
            return Ok(None);
        };
        waku_protocol::replay::replay_active_tail_fingerprint(
            &tail.messages,
            &tail.transcript_blocks,
            std::slice::from_ref(&tail.turn),
        )
        .map_err(ReplayApplyError::Snapshot)
    }

    pub(super) fn append_active_tail(
        &self,
        session: &mut AgentSession,
    ) -> Result<(), ReplayApplyError> {
        let Some(tail) = &self.active_tail else {
            return Ok(());
        };
        let settled_message_count = session.messages.len();
        let mut blocks = tail.transcript_blocks.clone();
        for block in &mut blocks {
            let relative = block
                .after_message
                .checked_sub(tail.first_message_index)
                .filter(|relative| *relative <= tail.messages.len())
                .ok_or(ReplayApplyError::InvalidLocalTail)?;
            block.after_message = settled_message_count + relative;
        }
        session.messages.extend(tail.messages.iter().cloned());
        session.transcript_blocks.extend(blocks);
        session.turns.push(tail.turn.clone());
        Ok(())
    }
}

#[derive(Clone)]
pub(super) struct ReplaySnapshotGuard {
    provider: ProviderKind,
    provider_cursor: Option<ProviderResumeCursor>,
    runtime_event_cursor: Option<RuntimeEventCursor>,
    active_turn_id: Option<Uuid>,
    message_count: usize,
    block_count: usize,
    turn_count: usize,
}

impl ReplaySnapshotGuard {
    pub(super) fn capture(session: &AgentSession) -> Self {
        Self {
            provider: session.provider,
            provider_cursor: session.provider_cursor.clone(),
            runtime_event_cursor: session.runtime_event_cursor,
            active_turn_id: session.active_turn_id(),
            message_count: session.messages.len(),
            block_count: session.transcript_blocks.len(),
            turn_count: session.turns.len(),
        }
    }

    pub(super) fn still_current(&self, session: &AgentSession) -> bool {
        session.detail_loaded
            && session.provider == self.provider
            && session.provider_cursor == self.provider_cursor
            && session.runtime_event_cursor == self.runtime_event_cursor
            && session.active_turn_id() == self.active_turn_id
            && session.messages.len() == self.message_count
            && session.transcript_blocks.len() == self.block_count
            && session.turns.len() == self.turn_count
    }
}
