use std::fmt;

use uuid::Uuid;

use super::{ProviderReplay, RenoaReplayCommit};

/// Bounds one replay transaction within the daemon's retained event journal.
pub const MAX_REPLAY_FRAGMENTS: usize = 2_048;

#[derive(Clone, Debug, Eq, PartialEq, serde::Deserialize, serde::Serialize, ts_rs::TS)]
#[serde(rename_all = "camelCase")]
pub struct ReplayFragment {
    pub replay_id: Uuid,
    pub index: usize,
    pub total: usize,
    pub json: String,
}

#[derive(Debug, Eq, PartialEq)]
pub struct AssembledReplay {
    pub replay_id: Uuid,
    fragments: Vec<String>,
}

impl AssembledReplay {
    pub fn decode(self) -> Result<ProviderReplay, ReplayCodecError> {
        serde_json::from_str(&self.fragments.concat()).map_err(ReplayCodecError::Deserialize)
    }

    pub fn decode_commit(self) -> Result<RenoaReplayCommit, ReplayCodecError> {
        serde_json::from_str(&self.fragments.concat()).map_err(ReplayCodecError::Deserialize)
    }
}

#[derive(Debug)]
pub enum ReplayCodecError {
    ZeroFragmentLimit,
    TransactionTooLarge { size: usize, maximum: usize },
    TooManyFragments { count: usize, maximum: usize },
    Serialize(serde_json::Error),
    Deserialize(serde_json::Error),
}

impl fmt::Display for ReplayCodecError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ZeroFragmentLimit => write!(formatter, "replay fragment limit must be non-zero"),
            Self::TransactionTooLarge { size, maximum } => write!(
                formatter,
                "serialized replay is {size} bytes, exceeding the {maximum}-byte transaction bound"
            ),
            Self::TooManyFragments { count, maximum } => write!(
                formatter,
                "replay needs {count} wire fragments, exceeding the transaction limit {maximum}"
            ),
            Self::Serialize(error) => write!(formatter, "could not serialize replay: {error}"),
            Self::Deserialize(error) => write!(formatter, "could not deserialize replay: {error}"),
        }
    }
}

impl std::error::Error for ReplayCodecError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Serialize(error) | Self::Deserialize(error) => Some(error),
            Self::ZeroFragmentLimit
            | Self::TransactionTooLarge { .. }
            | Self::TooManyFragments { .. } => None,
        }
    }
}

impl ProviderReplay {
    /// Serializes the complete typed replay before fragmenting it. Empty
    /// history therefore still produces an explicit transaction, and large
    /// supported fields are split without changing semantic item boundaries.
    pub fn encode_fragments(
        &self,
        replay_id: Uuid,
        max_fragment_bytes: usize,
    ) -> Result<Vec<ReplayFragment>, ReplayCodecError> {
        encode_fragments_bounded(
            self,
            replay_id,
            max_fragment_bytes,
            crate::SESSION_REPLAY_MAX_BYTES,
        )
    }

    #[cfg(test)]
    fn encode_fragments_bounded(
        &self,
        replay_id: Uuid,
        max_fragment_bytes: usize,
        max_transaction_bytes: usize,
    ) -> Result<Vec<ReplayFragment>, ReplayCodecError> {
        encode_fragments_bounded(self, replay_id, max_fragment_bytes, max_transaction_bytes)
    }
}

impl RenoaReplayCommit {
    pub fn encode_fragments(
        &self,
        commit_id: Uuid,
        max_fragment_bytes: usize,
    ) -> Result<Vec<ReplayFragment>, ReplayCodecError> {
        encode_fragments_bounded(
            self,
            commit_id,
            max_fragment_bytes,
            crate::SESSION_REPLAY_MAX_BYTES,
        )
    }
}

fn encode_fragments_bounded<T: serde::Serialize>(
    value: &T,
    replay_id: Uuid,
    max_fragment_bytes: usize,
    max_transaction_bytes: usize,
) -> Result<Vec<ReplayFragment>, ReplayCodecError> {
    if max_fragment_bytes == 0 {
        return Err(ReplayCodecError::ZeroFragmentLimit);
    }
    let json = serde_json::to_string(value).map_err(ReplayCodecError::Serialize)?;
    if json.len() > max_transaction_bytes {
        return Err(ReplayCodecError::TransactionTooLarge {
            size: json.len(),
            maximum: max_transaction_bytes,
        });
    }
    let estimated_count = json.len().div_ceil(max_fragment_bytes).max(1);
    if estimated_count > MAX_REPLAY_FRAGMENTS {
        return Err(ReplayCodecError::TooManyFragments {
            count: estimated_count,
            maximum: MAX_REPLAY_FRAGMENTS,
        });
    }

    let mut chunks = Vec::with_capacity(estimated_count);
    let mut start = 0;
    while start < json.len() {
        let mut end = (start + max_fragment_bytes).min(json.len());
        while !json.is_char_boundary(end) {
            end -= 1;
        }
        if end == start {
            end = json[start..]
                .char_indices()
                .nth(1)
                .map_or(json.len(), |(offset, _)| start + offset);
        }
        chunks.push(json[start..end].to_owned());
        start = end;
    }
    if chunks.is_empty() {
        chunks.push(String::new());
    }
    if chunks.len() > MAX_REPLAY_FRAGMENTS {
        return Err(ReplayCodecError::TooManyFragments {
            count: chunks.len(),
            maximum: MAX_REPLAY_FRAGMENTS,
        });
    }
    let total = chunks.len();
    Ok(chunks
        .into_iter()
        .enumerate()
        .map(|(index, json)| ReplayFragment {
            replay_id,
            index,
            total,
            json,
        })
        .collect())
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum ReplayAssemblyError {
    InvalidTotal { total: usize },
    FragmentTooLarge { size: usize, maximum: usize },
    TransactionTooLarge { size: usize, maximum: usize },
    UnexpectedReplay { expected: Uuid, received: Uuid },
    InconsistentTotal { expected: usize, received: usize },
    OutOfOrder { expected: usize, received: usize },
}

impl fmt::Display for ReplayAssemblyError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidTotal { total } => write!(
                formatter,
                "replay advertises invalid fragment count {total}; expected 1..={MAX_REPLAY_FRAGMENTS}"
            ),
            Self::FragmentTooLarge { size, maximum } => write!(
                formatter,
                "replay fragment is {size} bytes, exceeding the {maximum}-byte bound"
            ),
            Self::TransactionTooLarge { size, maximum } => write!(
                formatter,
                "replay transaction reached {size} bytes, exceeding the {maximum}-byte bound"
            ),
            Self::UnexpectedReplay { expected, received } => write!(
                formatter,
                "replay {received} arrived before incomplete replay {expected} finished"
            ),
            Self::InconsistentTotal { expected, received } => write!(
                formatter,
                "replay fragment count changed from {expected} to {received}"
            ),
            Self::OutOfOrder { expected, received } => write!(
                formatter,
                "replay fragment {received} arrived when fragment {expected} was required"
            ),
        }
    }
}

impl std::error::Error for ReplayAssemblyError {}

#[derive(Default)]
pub struct ReplayFragmentAssembler {
    replay_id: Option<Uuid>,
    total: usize,
    next_index: usize,
    assembled_size: usize,
    fragments: Vec<String>,
}

impl ReplayFragmentAssembler {
    pub fn accept(
        &mut self,
        fragment: ReplayFragment,
        max_fragment_bytes: usize,
    ) -> Result<Option<AssembledReplay>, ReplayAssemblyError> {
        self.accept_bounded(
            fragment,
            max_fragment_bytes,
            crate::SESSION_REPLAY_MAX_BYTES,
        )
    }

    fn accept_bounded(
        &mut self,
        fragment: ReplayFragment,
        max_fragment_bytes: usize,
        max_transaction_bytes: usize,
    ) -> Result<Option<AssembledReplay>, ReplayAssemblyError> {
        if fragment.total == 0 || fragment.total > MAX_REPLAY_FRAGMENTS {
            return Err(ReplayAssemblyError::InvalidTotal {
                total: fragment.total,
            });
        }
        if fragment.json.len() > max_fragment_bytes {
            return Err(ReplayAssemblyError::FragmentTooLarge {
                size: fragment.json.len(),
                maximum: max_fragment_bytes,
            });
        }
        let assembled_size = self.assembled_size.saturating_add(fragment.json.len());
        if assembled_size > max_transaction_bytes {
            return Err(ReplayAssemblyError::TransactionTooLarge {
                size: assembled_size,
                maximum: max_transaction_bytes,
            });
        }
        match self.replay_id {
            None => {
                self.replay_id = Some(fragment.replay_id);
                self.total = fragment.total;
            }
            Some(expected) if expected != fragment.replay_id => {
                return Err(ReplayAssemblyError::UnexpectedReplay {
                    expected,
                    received: fragment.replay_id,
                });
            }
            Some(_) if self.total != fragment.total => {
                return Err(ReplayAssemblyError::InconsistentTotal {
                    expected: self.total,
                    received: fragment.total,
                });
            }
            Some(_) => {}
        }
        if fragment.index != self.next_index {
            return Err(ReplayAssemblyError::OutOfOrder {
                expected: self.next_index,
                received: fragment.index,
            });
        }
        self.assembled_size = assembled_size;
        self.fragments.push(fragment.json);
        self.next_index += 1;
        if self.next_index != self.total {
            return Ok(None);
        }
        let Some(replay_id) = self.replay_id else {
            return Err(ReplayAssemblyError::OutOfOrder {
                expected: 0,
                received: fragment.index,
            });
        };
        let fragments = std::mem::take(&mut self.fragments);
        *self = Self::default();
        Ok(Some(AssembledReplay {
            replay_id,
            fragments,
        }))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::replay::{ReplayItem, ReplayUserMessage};

    #[test]
    fn empty_and_unicode_replays_fragment_and_reassemble_losslessly() {
        for replay in [
            ProviderReplay::default(),
            ProviderReplay {
                items: vec![ReplayItem::UserMessage(ReplayUserMessage {
                    message_id: Uuid::from_u128(1),
                    turn_id: Uuid::from_u128(2),
                    text: "नमस्ते🙂".repeat(20),
                })],
            },
        ] {
            let id = Uuid::new_v4();
            let fragments = replay.encode_fragments(id, 17).expect("encode replay");
            assert!(!fragments.is_empty());
            let mut assembler = ReplayFragmentAssembler::default();
            let mut assembled = None;
            for fragment in fragments {
                assembled = assembler.accept(fragment, 17).expect("accept fragment");
            }
            let decoded = assembled
                .expect("transaction completes")
                .decode()
                .expect("decode replay");
            assert_eq!(
                serde_json::to_vec(&decoded).expect("serialize decoded"),
                serde_json::to_vec(&replay).expect("serialize original")
            );
        }
    }

    #[test]
    fn partial_and_interleaved_transactions_fail_closed() {
        let first = Uuid::new_v4();
        let second = Uuid::new_v4();
        let mut assembler = ReplayFragmentAssembler::default();
        assert!(
            assembler
                .accept(
                    ReplayFragment {
                        replay_id: first,
                        index: 0,
                        total: 2,
                        json: "{".into(),
                    },
                    16,
                )
                .expect("first fragment")
                .is_none()
        );
        assert!(matches!(
            assembler.accept(
                ReplayFragment {
                    replay_id: second,
                    index: 0,
                    total: 1,
                    json: "{}".into(),
                },
                16,
            ),
            Err(ReplayAssemblyError::UnexpectedReplay { .. })
        ));
        assert!(matches!(
            assembler.accept(
                ReplayFragment {
                    replay_id: first,
                    index: 0,
                    total: 2,
                    json: "}".into(),
                },
                16,
            ),
            Err(ReplayAssemblyError::OutOfOrder { .. })
        ));
    }

    #[test]
    fn oversized_transactions_fail_before_any_fragment_can_be_emitted_or_applied() {
        let replay = ProviderReplay {
            items: vec![ReplayItem::UserMessage(ReplayUserMessage {
                message_id: Uuid::from_u128(1),
                turn_id: Uuid::from_u128(2),
                text: "too large".into(),
            })],
        };
        let error = replay
            .encode_fragments_bounded(Uuid::from_u128(3), 16, 8)
            .expect_err("serialized transaction exceeds the configured bound");
        assert!(matches!(
            error,
            ReplayCodecError::TransactionTooLarge { maximum: 8, .. }
        ));

        let mut assembler = ReplayFragmentAssembler::default();
        let error = assembler
            .accept_bounded(
                ReplayFragment {
                    replay_id: Uuid::from_u128(4),
                    index: 0,
                    total: 2,
                    json: "123456".into(),
                },
                8,
                8,
            )
            .expect("first fragment remains below the transaction bound");
        assert!(error.is_none());
        let error = assembler
            .accept_bounded(
                ReplayFragment {
                    replay_id: Uuid::from_u128(4),
                    index: 1,
                    total: 2,
                    json: "789".into(),
                },
                8,
                8,
            )
            .expect_err("assembly must reject the complete oversized transaction");
        assert!(matches!(
            error,
            ReplayAssemblyError::TransactionTooLarge {
                size: 9,
                maximum: 8
            }
        ));
    }
}
