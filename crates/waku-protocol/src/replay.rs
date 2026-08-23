//! Authoritative provider replay and transcript reconciliation.
//!
//! Renoa owns semantic conversation truth. After either side restarts,
//! `session/load` replays the complete durable history, and Waku must rebuild
//! its presentation cache from that replay instead of treating the cache as a
//! second execution history. This module defines the typed replay produced at
//! the ACP boundary and the pure reconciliation from a replay onto Waku's
//! transcript rows.
//!
//! Reconciliation rules:
//!
//! - Replay wins for semantic content, semantic order, and durable identity.
//!   Cached rows describing settled turns that the replay does not mention are
//!   stale history and are removed; an empty replay therefore clears the
//!   transcript.
//! - The only rows that survive behind the replay belong to the session's one
//!   actively running turn — explicitly unresolved local optimistic work such
//!   as the turn being submitted while the load ran.
//! - A replayed user row adopts its durable `messageId` while correlating the
//!   cached optimistic row through the originating turn UUID, so presentation
//!   (display content, attachments, timestamps) survives.
//! - Assistant rows correlate by their adopted durable message id and reasoning
//!   activities by their carried source id, so a second application of the same
//!   replay finds every row, keeps every timestamp, and changes nothing.
//! - Rows Waku has no cached timestamp for receive the caller-supplied
//!   reconciliation instant; matched rows keep their cached timestamps. This
//!   slice does not extend the ACP contract with timestamps.
//! - Every reconciled turn, cached or rebuilt, is numbered by its position in
//!   the rebuilt transcript (`1..n`). Fork, rewind, and the next `begin_turn`
//!   all index by that contiguous count. A checkpoint is kept only when it was
//!   captured at that same position; a moved turn discards position-dependent
//!   checkpoint metadata because this layer cannot rename Git refs.

use std::collections::{HashMap, HashSet, VecDeque};

use uuid::Uuid;

use crate::model::{
    ActivityItem, AgentTurn, Checkpoint, Message, MessageRole, ReasoningBlock, TranscriptBlock,
    TurnStatus, TurnStatus::Running,
};
/// One settled semantic item of a provider's durable replay, in replay order.
#[derive(Clone, Debug, serde::Deserialize, serde::Serialize)]
pub enum ReplayItem {
    UserMessage(ReplayUserMessage),
    AssistantMessage(ReplayAssistantMessage),
    ToolCall(ReplayTool),
}

#[derive(Clone, Debug, serde::Deserialize, serde::Serialize)]
pub struct ReplayUserMessage {
    /// Durable semantic-event identity from the provider.
    pub message_id: Uuid,
    /// Originating command/turn UUID (`_meta.requestId`). Matches the Waku
    /// [`crate::model::TurnPrompt`] id that was sent with the prompt.
    pub turn_id: Uuid,
    pub text: String,
}

#[derive(Clone, Debug, serde::Deserialize, serde::Serialize)]
pub struct ReplayAssistantMessage {
    pub message_id: Uuid,
    /// Ordered text and reasoning segments as replayed. Chunks sharing one
    /// `messageId` belong to this one semantic message.
    pub segments: Vec<ReplaySegment>,
}

#[derive(Clone, Debug, PartialEq, serde::Deserialize, serde::Serialize)]
pub enum ReplaySegment {
    Text(String),
    Reasoning(String),
}

/// One replayed tool call. `activity` is the presentation-complete row built
/// once at the ACP boundary with the same helpers the live stream uses;
/// `call_id` is the stable ACP tool-call identity the activity carries as its
/// source id.
#[derive(Clone, Debug, serde::Deserialize, serde::Serialize)]
pub struct ReplayTool {
    pub call_id: String,
    pub activity: Box<ActivityItem>,
}

/// The complete ordered replay of one provider session.
#[derive(Clone, Debug, Default, serde::Deserialize, serde::Serialize)]
pub struct ProviderReplay {
    pub items: Vec<ReplayItem>,
}
#[path = "replay/validate.rs"]
mod validate;
pub use validate::ReplayReconcileError;

#[path = "replay/codec.rs"]
mod codec;
pub use codec::{
    AssembledReplay, ReplayAssemblyError, ReplayCodecError, ReplayFragment, ReplayFragmentAssembler,
};

#[path = "replay/commit.rs"]
mod commit;
pub use commit::{
    ReconciledTranscript, RenoaReplayBase, RenoaReplayCommit, renoa_replay_base,
    replay_active_tail_fingerprint, replay_projection_fingerprint,
};

/// Rebuilds the transcript from the authoritative replay.
///
/// Every settled row comes from the replay; cached rows that agree by identity
/// donate their presentation, and everything else in the cache is stale history
/// that is removed. The one exception is the session's actively running turn —
/// explicitly unresolved local work such as the submission that triggered the
/// load — whose rows survive verbatim behind the rebuilt prefix.
///
/// `reconciled_at_seconds` is the deterministic presentation-time policy for
/// rows Waku has no cached timestamp for; matched rows keep their cached
/// timestamps. Runs in roughly linear time over the cached transcript and the
/// replay combined.
pub fn reconcile_replay(
    cached_messages: &[Message],
    cached_blocks: &[TranscriptBlock],
    cached_turns: &[AgentTurn],
    replay: &ProviderReplay,
    reconciled_at_seconds: u64,
) -> Result<ReconciledTranscript, ReplayReconcileError> {
    replay.validate()?;
    let replayed_turn_ids = replay_turn_ids(replay);
    let replayed_turn_set = replayed_turn_ids.iter().copied().collect::<HashSet<_>>();
    // Explicitly unresolved local optimistic work: the one running turn.
    let active_turn_id = cached_turns
        .iter()
        .rev()
        .find(|turn| turn.status == Running)
        .map(|turn| turn.id);
    let surviving_turn_id = active_turn_id.filter(|turn_id| !replayed_turn_set.contains(turn_id));

    // Index the cache once so correlation never rescans the transcript.
    let mut assistant_by_id: HashMap<Uuid, usize> = HashMap::new();
    let mut users_by_turn: HashMap<Uuid, VecDeque<usize>> = HashMap::new();
    for (index, message) in cached_messages.iter().enumerate() {
        match message.role {
            MessageRole::Assistant => {
                assistant_by_id.entry(message.id).or_insert(index);
            }
            MessageRole::User => {
                if let Some(turn_id) = message.turn_id {
                    users_by_turn.entry(turn_id).or_default().push_back(index);
                }
            }
            MessageRole::System => {}
        }
    }
    let mut activity_presentations: HashMap<
        Option<Uuid>,
        HashMap<ActivitySourceKey, VecDeque<ActivityPresentation>>,
    > = HashMap::new();
    for block in cached_blocks {
        for activity in &block.activities {
            if let Some(source_id) = &activity.source_id {
                let source = if activity.reasoning.is_some() {
                    ActivitySourceKey::Reasoning(source_id.clone())
                } else {
                    ActivitySourceKey::Tool(source_id.clone())
                };
                activity_presentations
                    .entry(block.turn_id)
                    .or_default()
                    .entry(source)
                    .or_default()
                    .push_back(ActivityPresentation {
                        id: activity.id,
                        reasoning_window: activity
                            .reasoning
                            .as_ref()
                            .map(|reasoning| (reasoning.started_at_ms, reasoning.finished_at_ms)),
                    });
            }
        }
    }

    let mut projection = ReplayProjection {
        cached_messages,
        consumed: vec![false; cached_messages.len()],
        assistant_by_id,
        users_by_turn,
        activity_presentations,
        reconciled_at_seconds,
        messages: Vec::new(),
        blocks: Vec::new(),
        pending_activities: Vec::new(),
        current_turn_id: None,
    };
    for item in &replay.items {
        match item {
            ReplayItem::UserMessage(user) => projection.push_user(user),
            ReplayItem::AssistantMessage(assistant) => projection.push_assistant(assistant),
            ReplayItem::ToolCall(tool) => projection.push_tool(tool),
        }
    }
    projection.flush_activities();
    let ReplayProjection {
        mut messages,
        blocks: mut transcript_blocks,
        ..
    } = projection;

    // Surviving optimistic rows keep their relative order behind the rebuilt
    // prefix; their block positions follow the surviving message mapping so a
    // shrinking or growing prefix can never underflow or misplace a row.
    let surviving_indices = cached_messages
        .iter()
        .enumerate()
        .filter(|(_, message)| {
            surviving_turn_id.is_some_and(|turn_id| message.turn_id == Some(turn_id))
        })
        .map(|(index, _)| index)
        .collect::<Vec<_>>();
    let projected_prefix = messages.len();
    messages.extend(
        surviving_indices
            .iter()
            .filter_map(|index| cached_messages.get(*index))
            .cloned(),
    );
    for mut block in cached_blocks
        .iter()
        .filter(|block| {
            block
                .turn_id
                .is_some_and(|turn_id| Some(turn_id) == surviving_turn_id)
        })
        .cloned()
    {
        // Activities of the surviving turn attach at the closest surviving row
        // at or before their old position, never above the rebuilt prefix.
        let surviving_before =
            surviving_indices.partition_point(|index| *index < block.after_message);
        block.after_message = projected_prefix + surviving_before;
        transcript_blocks.push(block);
    }

    Ok(ReconciledTranscript {
        messages,
        transcript_blocks,
        turns: reconciled_turns(
            cached_turns,
            &replayed_turn_ids,
            &replayed_turn_set,
            surviving_turn_id,
            reconciled_at_seconds,
        ),
    })
}

fn replay_turn_ids(replay: &ProviderReplay) -> Vec<Uuid> {
    let mut turn_ids = Vec::new();
    let mut seen = HashSet::new();
    for item in &replay.items {
        if let ReplayItem::UserMessage(user) = item
            && seen.insert(user.turn_id)
        {
            turn_ids.push(user.turn_id);
        }
    }
    turn_ids
}

fn reconciled_turns(
    cached_turns: &[AgentTurn],
    replayed_turn_ids: &[Uuid],
    replayed_turn_set: &HashSet<Uuid>,
    active_turn_id: Option<Uuid>,
    reconciled_at_seconds: u64,
) -> Vec<AgentTurn> {
    let mut turns = Vec::new();
    let cached_by_id = cached_turns
        .iter()
        .map(|turn| (turn.id, turn))
        .collect::<HashMap<_, _>>();
    for (index, turn_id) in replayed_turn_ids.iter().enumerate() {
        let turn_count = index + 1;
        match cached_by_id.get(turn_id) {
            Some(cached) => turns.push(renumbered_cached_turn(cached, turn_count)),
            None => turns.push(AgentTurn {
                id: *turn_id,
                turn_count,
                status: TurnStatus::Completed,
                provider_turn_started: true,
                provider_resume_at: None,
                started_at: reconciled_at_seconds,
                completed_at: Some(reconciled_at_seconds),
                checkpoint: None,
            }),
        }
    }
    if let Some(active) = active_turn_id
        && !replayed_turn_set.contains(&active)
        && let Some(cached) = cached_by_id.get(&active)
    {
        turns.push(renumbered_cached_turn(cached, turns.len() + 1));
    }
    turns
}

fn renumbered_cached_turn(cached: &AgentTurn, turn_count: usize) -> AgentTurn {
    let mut turn = cached.clone();
    turn.turn_count = turn_count;
    turn.checkpoint = checkpoint_for_position(turn.checkpoint.take(), turn_count);
    turn
}

/// Git checkpoint refs are named by turn position. Reconciliation cannot
/// rename those refs, so a checkpoint is valid only when it was captured at
/// the same contiguous count the rebuilt transcript now assigns.
fn checkpoint_for_position(
    checkpoint: Option<Checkpoint>,
    turn_count: usize,
) -> Option<Checkpoint> {
    checkpoint.filter(|checkpoint| checkpoint.turn_count == turn_count)
}

struct ReplayProjection<'a> {
    cached_messages: &'a [Message],
    consumed: Vec<bool>,
    assistant_by_id: HashMap<Uuid, usize>,
    users_by_turn: HashMap<Uuid, VecDeque<usize>>,
    activity_presentations:
        HashMap<Option<Uuid>, HashMap<ActivitySourceKey, VecDeque<ActivityPresentation>>>,
    reconciled_at_seconds: u64,
    messages: Vec<Message>,
    blocks: Vec<TranscriptBlock>,
    /// Activities accumulating for the current message position; flushed as a
    /// block when the next message is appended or the projection ends.
    pending_activities: Vec<ActivityItem>,
    current_turn_id: Option<Uuid>,
}

#[derive(Clone, Copy)]
struct ActivityPresentation {
    id: Uuid,
    reasoning_window: Option<(u64, u64)>,
}

#[derive(Eq, Hash, PartialEq)]
enum ActivitySourceKey {
    Reasoning(String),
    Tool(String),
}

impl ReplayProjection<'_> {
    fn push_user(&mut self, user: &ReplayUserMessage) {
        self.flush_activities();
        self.current_turn_id = Some(user.turn_id);
        let correlated = self
            .users_by_turn
            .get_mut(&user.turn_id)
            .and_then(|candidates| {
                // First unconsumed cached row for this turn, in cache order.
                while let Some(index) = candidates.pop_front() {
                    if !self.consumed[index] {
                        self.consumed[index] = true;
                        return Some(index);
                    }
                }
                None
            });
        let mut message = Message {
            // The durable semantic-event id replaces Waku's locally generated
            // id; the turn correlation stays on `turn_id`.
            id: user.message_id,
            turn_id: Some(user.turn_id),
            role: MessageRole::User,
            content: user.text.clone(),
            display_content: None,
            attachments: Vec::new(),
            created_at: self.reconciled_at_seconds,
            streaming: false,
        };
        if let Some(index) = correlated
            && let Some(cached) = self.cached_messages.get(index)
        {
            message.display_content = cached.display_content.clone();
            message.attachments = cached.attachments.clone();
            message.created_at = cached.created_at;
        }
        self.messages.push(message);
    }

    fn push_assistant(&mut self, assistant: &ReplayAssistantMessage) {
        self.flush_activities();
        let text = assistant
            .segments
            .iter()
            .filter_map(|segment| match segment {
                ReplaySegment::Text(text) => Some(text.as_str()),
                ReplaySegment::Reasoning(_) => None,
            })
            .collect::<String>();
        let reasoning = assistant
            .segments
            .iter()
            .filter_map(|segment| match segment {
                ReplaySegment::Reasoning(text) => Some(text.as_str()),
                ReplaySegment::Text(_) => None,
            })
            .collect::<Vec<_>>()
            .join("");
        let correlated = self
            .assistant_by_id
            .get(&assistant.message_id)
            .copied()
            .filter(|index| !self.consumed[*index]);
        if let Some(index) = correlated {
            self.consumed[index] = true;
        }
        let created_at = correlated
            .and_then(|index| self.cached_messages.get(index))
            .map(|cached| cached.created_at)
            .unwrap_or(self.reconciled_at_seconds);
        // Reasoning first renders above the message body, mirroring where the
        // live stream would have placed it; otherwise it follows the body. The
        // leading block must be flushed before the body lands so its position
        // points at the answer instead of after it.
        let reasoning_leads = reasoning_first(&assistant.segments);
        if reasoning_leads && !reasoning.is_empty() {
            self.push_reasoning_activity(assistant.message_id, reasoning.clone());
            self.flush_activities();
        }
        if !text.is_empty() {
            self.messages.push(Message {
                id: assistant.message_id,
                turn_id: self.current_turn_id,
                role: MessageRole::Assistant,
                content: text,
                display_content: None,
                attachments: Vec::new(),
                created_at,
                streaming: false,
            });
        }
        if !reasoning_leads && !reasoning.is_empty() {
            self.push_reasoning_activity(assistant.message_id, reasoning);
        }
    }

    fn push_reasoning_activity(&mut self, message_id: Uuid, content: String) {
        // A previously reconciled reasoning activity keeps its original
        // presentation window; only genuinely new work gets local times.
        let source_id = message_id.to_string();
        let presentation =
            self.take_activity_presentation(&ActivitySourceKey::Reasoning(source_id.clone()));
        let (started_at_ms, finished_at_ms) = presentation
            .and_then(|presentation| presentation.reasoning_window)
            .unwrap_or_else(|| {
                let at = self.reconciled_at_seconds.saturating_mul(1_000);
                (at, at)
            });
        let mut activity = ActivityItem::from_reasoning(
            ReasoningBlock {
                content,
                started_at_ms,
                finished_at_ms,
            },
            true,
        );
        if let Some(presentation) = presentation {
            activity.id = presentation.id;
        }
        // Carry the durable identity on the activity so a later reconciliation
        // can correlate reasoning work without confusing it with a message id
        // lookup.
        activity.source_id = Some(source_id);
        self.pending_activities.push(activity);
    }

    fn push_tool(&mut self, tool: &ReplayTool) {
        let mut activity = (*tool.activity).clone();
        // The ACP tool-call id is this work's stable identity.
        activity.source_id = Some(tool.call_id.clone());
        if let Some(presentation) =
            self.take_activity_presentation(&ActivitySourceKey::Tool(tool.call_id.clone()))
        {
            activity.id = presentation.id;
        }
        self.pending_activities.push(activity);
    }

    fn take_activity_presentation(
        &mut self,
        source: &ActivitySourceKey,
    ) -> Option<ActivityPresentation> {
        self.activity_presentations
            .get_mut(&self.current_turn_id)
            .and_then(|by_source| by_source.get_mut(source))
            .and_then(VecDeque::pop_front)
    }

    fn flush_activities(&mut self) {
        if self.pending_activities.is_empty() {
            return;
        }
        let activities = std::mem::take(&mut self.pending_activities);
        self.blocks.push(TranscriptBlock {
            after_message: self.messages.len(),
            turn_id: self.current_turn_id,
            activities,
        });
    }
}

fn reasoning_first(segments: &[ReplaySegment]) -> bool {
    matches!(segments.first(), Some(ReplaySegment::Reasoning(_)))
}

#[cfg(test)]
mod tests;
