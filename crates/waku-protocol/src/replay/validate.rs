use std::collections::{HashMap, HashSet};
use std::fmt;

use uuid::Uuid;

use super::{ProviderReplay, ReplayAssistantMessage, ReplayItem, ReplaySegment};

impl ProviderReplay {
    /// Validates the complete semantic shape before reconciliation mutates any
    /// presentation state.
    pub fn validate(&self) -> Result<(), ReplayReconcileError> {
        let mut current_turn = None;
        let mut user_messages = HashMap::new();
        let mut turns = HashMap::new();
        let mut assistant_messages = HashSet::new();
        let mut semantic_messages = HashSet::new();
        for (index, item) in self.items.iter().enumerate() {
            match item {
                ReplayItem::UserMessage(user) => {
                    if user.text.is_empty() {
                        return Err(ReplayReconcileError::EmptyUserMessage {
                            message_id: user.message_id,
                        });
                    }
                    if user_messages
                        .insert(user.message_id, user.turn_id)
                        .is_some()
                    {
                        return Err(ReplayReconcileError::RepeatedUserMessage {
                            message_id: user.message_id,
                        });
                    }
                    if !semantic_messages.insert(user.message_id) {
                        return Err(ReplayReconcileError::RepeatedMessageIdentity {
                            message_id: user.message_id,
                        });
                    }
                    if turns.insert(user.turn_id, user.message_id).is_some() {
                        return Err(ReplayReconcileError::RepeatedTurnIdentity {
                            turn_id: user.turn_id,
                        });
                    }
                    current_turn = Some(user.turn_id);
                }
                ReplayItem::AssistantMessage(assistant) => {
                    require_turn(current_turn, index, "assistant message")?;
                    if !assistant_messages.insert(assistant.message_id) {
                        return Err(ReplayReconcileError::RepeatedAssistantMessage {
                            message_id: assistant.message_id,
                        });
                    }
                    if !semantic_messages.insert(assistant.message_id) {
                        return Err(ReplayReconcileError::RepeatedMessageIdentity {
                            message_id: assistant.message_id,
                        });
                    }
                    validate_segments(assistant)?;
                }
                ReplayItem::ToolCall(tool) => {
                    require_turn(current_turn, index, "tool call")?;
                    if tool.call_id.trim().is_empty() {
                        return Err(ReplayReconcileError::EmptyToolCallId { index });
                    }
                    if tool.activity.source_id.as_deref() != Some(tool.call_id.as_str()) {
                        return Err(ReplayReconcileError::ConflictingToolIdentity {
                            call_id: tool.call_id.clone(),
                        });
                    }
                }
            }
        }
        Ok(())
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum ReplayReconcileError {
    ItemBeforeUser { index: usize, item: &'static str },
    EmptyUserMessage { message_id: Uuid },
    RepeatedUserMessage { message_id: Uuid },
    RepeatedTurnIdentity { turn_id: Uuid },
    RepeatedAssistantMessage { message_id: Uuid },
    RepeatedMessageIdentity { message_id: Uuid },
    EmptyAssistantMessage { message_id: Uuid },
    UnsupportedAssistantOrder { message_id: Uuid },
    EmptyToolCallId { index: usize },
    ConflictingToolIdentity { call_id: String },
}

impl fmt::Display for ReplayReconcileError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ItemBeforeUser { index, item } => {
                write!(
                    formatter,
                    "replay item {index} is a {item} before any user message"
                )
            }
            Self::EmptyUserMessage { message_id } => {
                write!(formatter, "replayed user message {message_id} has no text")
            }
            Self::RepeatedUserMessage { message_id } => write!(
                formatter,
                "replayed user message {message_id} resumes after an event boundary"
            ),
            Self::RepeatedTurnIdentity { turn_id } => write!(
                formatter,
                "replayed request {turn_id} maps to more than one user message"
            ),
            Self::RepeatedAssistantMessage { message_id } => write!(
                formatter,
                "replayed assistant message {message_id} resumes after an event boundary"
            ),
            Self::RepeatedMessageIdentity { message_id } => write!(
                formatter,
                "replayed messageId {message_id} is reused across semantic message roles"
            ),
            Self::EmptyAssistantMessage { message_id } => write!(
                formatter,
                "replayed assistant message {message_id} has no supported content"
            ),
            Self::UnsupportedAssistantOrder { message_id } => write!(
                formatter,
                "replayed assistant message {message_id} alternates text and reasoning in a shape Waku cannot project losslessly"
            ),
            Self::EmptyToolCallId { index } => {
                write!(
                    formatter,
                    "replayed tool call at item {index} has no toolCallId"
                )
            }
            Self::ConflictingToolIdentity { call_id } => write!(
                formatter,
                "replayed tool call {call_id:?} does not carry the same activity source identity"
            ),
        }
    }
}

impl std::error::Error for ReplayReconcileError {}

fn require_turn(
    current_turn: Option<Uuid>,
    index: usize,
    item: &'static str,
) -> Result<Uuid, ReplayReconcileError> {
    current_turn.ok_or(ReplayReconcileError::ItemBeforeUser { index, item })
}

fn validate_segments(assistant: &ReplayAssistantMessage) -> Result<(), ReplayReconcileError> {
    let Some(first) = assistant.segments.first() else {
        return Err(ReplayReconcileError::EmptyAssistantMessage {
            message_id: assistant.message_id,
        });
    };
    let first_is_reasoning = matches!(first, ReplaySegment::Reasoning(_));
    let mut crossed = false;
    for segment in &assistant.segments {
        let is_reasoning = matches!(segment, ReplaySegment::Reasoning(_));
        if is_reasoning != first_is_reasoning {
            crossed = true;
        } else if crossed {
            return Err(ReplayReconcileError::UnsupportedAssistantOrder {
                message_id: assistant.message_id,
            });
        }
    }
    Ok(())
}
