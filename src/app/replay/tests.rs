use super::*;

use waku_protocol::replay::{ProviderReplay, ReplayItem, ReplayUserMessage};

fn replay() -> ProviderReplay {
    ProviderReplay {
        items: vec![ReplayItem::UserMessage(ReplayUserMessage {
            message_id: Uuid::from_u128(1),
            turn_id: Uuid::from_u128(2),
            text: "durable".into(),
        })],
    }
}

fn cursor(sequence: u64) -> RuntimeEventCursor {
    RuntimeEventCursor {
        runtime_id: Uuid::from_u128(3),
        epoch: Uuid::from_u128(4),
        sequence,
    }
}

#[test]
fn complete_transaction_blocks_connected_until_commit() {
    let fragments = replay()
        .encode_fragments(Uuid::from_u128(5), 8)
        .expect("encode replay");
    let mut state = SessionReplayState::new(true);
    for fragment in fragments {
        state.accept_fragment(fragment).expect("accept fragment");
    }
    assert_eq!(state.phase, ReplayPhase::AwaitingCursor);
    assert!(!state.connected_is_allowed());
    assert!(state.intercept_cursor(cursor(9)));
    assert!(state.blocks_event_drain());

    let work = state.begin().expect("hydration releases preparation");
    assert_eq!(work.accepted_cursor, cursor(9));
    assert!(state.blocks_event_drain());
    assert!(
        state.begin().is_none(),
        "hydration may release replay only once"
    );
    assert!(state.commit(work.generation));
    assert!(state.begin().is_none(), "a committed replay cannot restart");
    assert!(state.connected_is_allowed());
    assert!(!state.blocks_event_drain());
}

#[test]
fn partial_delivery_does_not_commit_and_restart_accepts_the_full_transaction() {
    let fragments = replay()
        .encode_fragments(Uuid::from_u128(6), 8)
        .expect("encode replay");
    assert!(fragments.len() > 1);

    let mut interrupted = SessionReplayState::new(true);
    interrupted
        .accept_fragment(fragments[0].clone())
        .expect("accept first fragment");
    assert_eq!(interrupted.phase, ReplayPhase::Receiving);
    assert!(!interrupted.connected_is_allowed());

    let mut restarted = SessionReplayState::new(true);
    for fragment in fragments {
        restarted
            .accept_fragment(fragment)
            .expect("redeliver complete transaction");
    }
    assert_eq!(restarted.phase, ReplayPhase::AwaitingCursor);
}

#[test]
fn stale_generation_cannot_commit_over_newer_state() {
    let fragments = replay()
        .encode_fragments(Uuid::from_u128(7), 128)
        .expect("encode replay");
    let mut state = SessionReplayState::new(true);
    for fragment in fragments {
        state.accept_fragment(fragment).expect("accept fragment");
    }
    assert!(state.intercept_cursor(cursor(10)));
    let work = state.begin().expect("begin replay");
    state.fail();
    assert!(!state.commit(work.generation));
    assert!(!state.connected_is_allowed());
}

#[test]
fn new_session_rejects_authoritative_replay() {
    let fragment = replay()
        .encode_fragments(Uuid::from_u128(8), 1024)
        .expect("encode replay")
        .remove(0);
    let error = SessionReplayState::new(false)
        .accept_fragment(fragment)
        .expect_err("new session must not apply replay");
    assert!(matches!(error, ReplayApplyError::UnexpectedReplay));
}

#[test]
fn snapshot_guard_rejects_a_changed_session_boundary() {
    let mut session = AgentSession::new(Uuid::from_u128(20), ProviderKind::Renoa);
    session.provider_cursor = Some(ProviderResumeCursor::Renoa {
        session_id: Uuid::from_u128(21).to_string(),
    });
    session.begin_turn_with_id_and_presentation(Uuid::from_u128(22), "pending", None, Vec::new());
    let guard = ReplaySnapshotGuard::capture(&session);
    assert!(guard.still_current(&session));

    session.push_message(MessageRole::Assistant, "newer local content");
    assert!(!guard.still_current(&session));
}

#[test]
fn preparation_snapshot_carries_only_the_active_tail_and_rebases_its_block() {
    let mut daemon_base = AgentSession::new(Uuid::from_u128(30), ProviderKind::Renoa);
    daemon_base.provider_cursor = Some(ProviderResumeCursor::Renoa {
        session_id: Uuid::from_u128(31).to_string(),
    });
    let settled = Uuid::from_u128(32);
    daemon_base.begin_turn_with_id_and_presentation(settled, "settled", None, Vec::new());
    daemon_base.push_message(MessageRole::Assistant, "answer");
    daemon_base.finish_active_turn(TurnStatus::Completed);

    let active = Uuid::from_u128(33);
    let mut desktop = daemon_base.clone();
    desktop.begin_turn_with_id_and_presentation(active, "pending", None, Vec::new());
    desktop.transcript_blocks.push(TranscriptBlock {
        after_message: desktop.messages.len(),
        turn_id: Some(active),
        activities: Vec::new(),
    });
    let snapshot = ReplayPreparationSnapshot::capture(&desktop).expect("capture active tail");
    let tail = snapshot.active_tail.as_ref().expect("one active tail");
    assert_eq!(tail.messages.len(), 1);
    assert_eq!(tail.messages[0].turn_id, Some(active));
    assert_eq!(tail.transcript_blocks.len(), 1);

    snapshot
        .append_active_tail(&mut daemon_base)
        .expect("append tail to background-fetched base");
    assert_eq!(daemon_base.messages.len(), 3);
    assert_eq!(daemon_base.messages[2].turn_id, Some(active));
    assert_eq!(daemon_base.transcript_blocks[0].after_message, 3);
    assert_eq!(daemon_base.turns.last().map(|turn| turn.id), Some(active));
}
