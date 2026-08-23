use super::*;
use crate::model::{
    ActivityKind, AgentSession, Checkpoint, CheckpointStatus, MessageAttachment, ProviderKind,
    ProviderResumeCursor,
};
use std::path::PathBuf;

const RECONCILED_AT: u64 = 5_000_000;

fn user_message(turn_id: Uuid) -> ReplayUserMessage {
    ReplayUserMessage {
        message_id: Uuid::new_v4(),
        turn_id,
        text: "Fix the login bug".to_owned(),
    }
}

fn assistant_message(segments: Vec<ReplaySegment>) -> ReplayAssistantMessage {
    ReplayAssistantMessage {
        message_id: Uuid::new_v4(),
        segments,
    }
}

fn replay_of(items: Vec<ReplayItem>) -> ProviderReplay {
    ProviderReplay { items }
}

fn cached_turn(
    id: Uuid,
    turn_count: usize,
    status: TurnStatus,
    checkpoint: Option<Checkpoint>,
) -> AgentTurn {
    AgentTurn {
        id,
        turn_count,
        status,
        provider_turn_started: status != Running,
        provider_resume_at: None,
        started_at: 100,
        completed_at: (status != Running).then_some(200),
        checkpoint,
    }
}

fn position_checkpoint(turn_count: usize) -> Checkpoint {
    Checkpoint {
        turn_count,
        git_ref: format!("refs/waku/session-x-turn-{turn_count}"),
        status: CheckpointStatus::Ready,
        files: Vec::new(),
        additions: 0,
        deletions: 0,
        created_at: 150,
    }
}

fn tool_item(call_id: &str, kind: ActivityKind) -> ReplayItem {
    ReplayItem::ToolCall(ReplayTool {
        call_id: call_id.to_owned(),
        activity: Box::new(ActivityItem::new(
            Some(call_id.to_owned()),
            kind,
            call_id.replace('-', " "),
            None,
            true,
        )),
    })
}

/// Semantic fingerprint including presentation timestamps and reasoning
/// content: a second application that drifts anywhere fails this.
fn fingerprint(transcript: &ReconciledTranscript) -> String {
    let messages = transcript
        .messages
        .iter()
        .map(|message| {
            format!(
                "{:?}/{}/{}/{}/{}/{:?}",
                message.role,
                message.id,
                message.turn_id.map(|id| id.to_string()).unwrap_or_default(),
                message.created_at,
                message.content,
                message.display_content.clone().unwrap_or_default(),
            )
        })
        .collect::<Vec<_>>()
        .join("|");
    let blocks = transcript
        .transcript_blocks
        .iter()
        .flat_map(|block| block.activities.iter())
        .map(|activity| {
            let reasoning = activity
                .reasoning
                .as_ref()
                .map(|reasoning| {
                    format!(
                        "{}/{}-{}",
                        reasoning.content, reasoning.started_at_ms, reasoning.finished_at_ms
                    )
                })
                .unwrap_or_default();
            format!(
                "{:?}/{}/{}/{}/{}/{:?}",
                activity.kind,
                activity.source_id.clone().unwrap_or_default(),
                activity.complete,
                activity.output.clone().unwrap_or_default(),
                reasoning,
                activity.image_urls,
            )
        })
        .collect::<Vec<_>>()
        .join("|");
    let turns = transcript
        .turns
        .iter()
        .map(|turn| {
            format!(
                "{}/{}/{}",
                turn.id,
                turn.turn_count,
                turn.status == TurnStatus::Completed
            )
        })
        .collect::<Vec<_>>()
        .join("|");
    format!("{messages}#{blocks}#{turns}")
}

#[test]
fn an_exact_cache_and_replay_produce_no_duplicates_or_semantic_change() {
    let turn = Uuid::new_v4();
    let user_id = Uuid::new_v4();
    let assistant_id = Uuid::new_v4();
    let call_id = "call-1";
    let replay = replay_of(vec![
        ReplayItem::UserMessage(ReplayUserMessage {
            message_id: user_id,
            turn_id: turn,
            text: "Fix the login bug".to_owned(),
        }),
        tool_item(call_id, ActivityKind::Command),
        ReplayItem::AssistantMessage(assistant_message(vec![ReplaySegment::Text(
            "Fixed.".to_owned(),
        )])),
    ]);

    // The cache already carries the adopted durable identities.
    let mut cached_user = Message::new_for_turn(MessageRole::User, "Fix the login bug", turn);
    cached_user.id = user_id;
    let mut cached_assistant = Message::new_for_turn(MessageRole::Assistant, "Fixed.", turn);
    cached_assistant.id = assistant_id;
    let cache_blocks = vec![TranscriptBlock {
        after_message: 2,
        turn_id: Some(turn),
        activities: vec![match tool_item(call_id, ActivityKind::Command) {
            ReplayItem::ToolCall(tool) => *tool.activity,
            _ => unreachable!(),
        }],
    }];

    let once = reconcile_replay(
        &[cached_user, cached_assistant],
        &cache_blocks,
        &[],
        &replay,
        RECONCILED_AT,
    )
    .expect("valid replay");
    assert_eq!(once.messages.len(), 2);
    assert_eq!(once.messages[0].id, user_id);
    assert_eq!(once.messages[0].content, "Fix the login bug");
    assert_eq!(once.messages[1].content, "Fixed.");
    // The replayed copy replaces the previous reconciliation's block; the
    // stale one is not duplicated.
    assert_eq!(once.transcript_blocks.len(), 1);
    let activities = &once.transcript_blocks[0].activities;
    assert_eq!(activities.len(), 1);
    assert_eq!(activities[0].source_id.as_deref(), Some(call_id));
    assert!(activities[0].complete);
}

#[test]
fn an_optimistic_user_row_is_adopted_and_unmatched_rows_do_not_leak() {
    let turn = Uuid::new_v4();
    let replay = replay_of(vec![ReplayItem::UserMessage(user_message(turn))]);
    let mut optimistic = Message::new_for_turn(MessageRole::User, "Fix the login bug", turn);
    optimistic.created_at = 172_000;
    optimistic.display_content = Some("Fix the **login** bug".to_owned());
    optimistic.attachments = vec![MessageAttachment {
        path: PathBuf::from("/tmp/screenshot.png"),
        mention: "screenshot.png".into(),
        name: "screenshot.png".into(),
        is_dir: false,
        is_image: true,
        blob_reference: None,
    }];
    // A settled row Renoa does not mention is stale history, and its
    // presentation must never move onto another semantic row.
    let other_turn = Uuid::new_v4();
    let mut stale = Message::new_for_turn(MessageRole::User, "Fix the login bug", other_turn);
    stale.created_at = 171_000;
    stale.attachments = optimistic.attachments.clone();

    let reconciled = reconcile_replay(&[optimistic, stale], &[], &[], &replay, RECONCILED_AT)
        .expect("valid replay");

    assert_eq!(reconciled.messages.len(), 1);
    let projected = &reconciled.messages[0];
    assert_eq!(projected.turn_id, Some(turn));
    assert_eq!(
        projected.display_content.as_deref(),
        Some("Fix the **login** bug")
    );
    assert_eq!(projected.attachments.len(), 1);
    assert_eq!(projected.created_at, 172_000);
}

#[test]
fn a_shrinking_projection_is_safe_and_positions_stay_valid() {
    let turn = Uuid::new_v4();
    let assistant_id = Uuid::new_v4();
    let replay = replay_of(vec![
        ReplayItem::UserMessage(user_message(turn)),
        ReplayItem::AssistantMessage(ReplayAssistantMessage {
            message_id: assistant_id,
            segments: vec![ReplaySegment::Text("Done.".to_owned())],
        }),
    ]);
    // The cache holds more covered rows than the replay projects.
    let cached = vec![
        Message::new_for_turn(MessageRole::User, "Fix the login bug", turn),
        Message::new_for_turn(MessageRole::Assistant, "Partial", turn),
        Message::new_for_turn(MessageRole::Assistant, "Partial continued", turn),
    ];

    let reconciled =
        reconcile_replay(&cached, &[], &[], &replay, RECONCILED_AT).expect("valid replay");

    assert_eq!(reconciled.messages.len(), 2);
    assert_eq!(reconciled.messages[1].id, assistant_id);
    for block in &reconciled.transcript_blocks {
        assert!(block.after_message <= reconciled.messages.len());
    }
}

#[test]
fn an_empty_authoritative_replay_clears_stale_history() {
    let stale_turn = Uuid::new_v4();
    let cached = vec![
        Message::new_for_turn(MessageRole::User, "Old question", stale_turn),
        Message::new_for_turn(MessageRole::Assistant, "Old answer", stale_turn),
    ];
    let cached_turns = vec![AgentTurn {
        id: stale_turn,
        turn_count: 1,
        status: TurnStatus::Completed,
        provider_turn_started: true,
        provider_resume_at: None,
        started_at: 10,
        completed_at: Some(20),
        checkpoint: None,
    }];

    let reconciled = reconcile_replay(
        &cached,
        &[],
        &cached_turns,
        &ProviderReplay::default(),
        RECONCILED_AT,
    )
    .expect("valid replay");

    assert!(reconciled.messages.is_empty());
    assert!(reconciled.transcript_blocks.is_empty());
    assert!(reconciled.turns.is_empty());
}

#[test]
fn only_an_active_running_turn_survives_behind_the_replay() {
    let settled_turn = Uuid::new_v4();
    let running_turn = Uuid::new_v4();
    let replay = replay_of(vec![ReplayItem::UserMessage(user_message(settled_turn))]);
    let cached = vec![
        Message::new_for_turn(MessageRole::User, "Settled", settled_turn),
        Message::new_for_turn(MessageRole::Assistant, "Answer", settled_turn),
        // Optimistic rows of the submission that triggered the load.
        Message::new_for_turn(MessageRole::User, "In flight", running_turn),
    ];
    let as_activity = |call_id: &str, kind| match tool_item(call_id, kind) {
        ReplayItem::ToolCall(tool) => *tool.activity,
        _ => unreachable!(),
    };
    let blocks = vec![
        TranscriptBlock {
            after_message: 2,
            turn_id: Some(settled_turn),
            activities: vec![as_activity("stale-tool", ActivityKind::Tool)],
        },
        TranscriptBlock {
            after_message: 3,
            turn_id: Some(running_turn),
            activities: vec![as_activity("live-tool", ActivityKind::Command)],
        },
    ];
    let cached_turns = vec![
        AgentTurn {
            id: settled_turn,
            turn_count: 1,
            status: TurnStatus::Completed,
            provider_turn_started: true,
            provider_resume_at: None,
            started_at: 10,
            completed_at: Some(20),
            checkpoint: None,
        },
        AgentTurn {
            id: running_turn,
            turn_count: 2,
            status: Running,
            provider_turn_started: false,
            provider_resume_at: None,
            started_at: 30,
            completed_at: None,
            checkpoint: None,
        },
    ];

    let reconciled = reconcile_replay(&cached, &blocks, &cached_turns, &replay, RECONCILED_AT)
        .expect("valid replay");

    // Settled projection (one replayed row) plus exactly the optimistic
    // user row behind it.
    assert_eq!(reconciled.messages.len(), 2);
    assert_eq!(reconciled.messages[0].turn_id, Some(settled_turn));
    assert_eq!(reconciled.messages[1].content, "In flight");
    assert_eq!(reconciled.messages[1].turn_id, Some(running_turn));
    // The stale covered block is rebuilt away; only the running turn's
    // block survives, attached at its own row inside the rebuilt list.
    assert_eq!(reconciled.transcript_blocks.len(), 1);
    assert_eq!(reconciled.transcript_blocks[0].turn_id, Some(running_turn));
    assert_eq!(
        reconciled.transcript_blocks[0].after_message, 2,
        "after_message is an insertion count after the surviving user row"
    );
    // Turns: the replayed one plus the still-running one; nothing else.
    assert_eq!(reconciled.turns.len(), 2);
    assert_eq!(reconciled.turns[0].id, settled_turn);
    assert_eq!(reconciled.turns[1].id, running_turn);
    assert_eq!(reconciled.turns[1].status, Running);
}

#[test]
fn a_running_tail_already_in_replay_is_not_duplicated() {
    let running_turn = Uuid::new_v4();
    let replay = replay_of(vec![ReplayItem::UserMessage(user_message(running_turn))]);
    let cached = vec![Message::new_for_turn(
        MessageRole::User,
        "Fix the login bug",
        running_turn,
    )];
    let cached_turn = AgentTurn {
        id: running_turn,
        turn_count: 1,
        status: Running,
        provider_turn_started: false,
        provider_resume_at: None,
        started_at: 30,
        completed_at: None,
        checkpoint: None,
    };

    let reconciled = reconcile_replay(&cached, &[], &[cached_turn], &replay, RECONCILED_AT)
        .expect("valid replay");

    assert_eq!(reconciled.messages.len(), 1);
    assert_eq!(reconciled.messages[0].turn_id, Some(running_turn));
    assert_eq!(reconciled.turns.len(), 1);
}

#[test]
fn leading_reasoning_renders_before_its_answer() {
    let turn = Uuid::new_v4();
    let assistant_id = Uuid::new_v4();
    let replay = replay_of(vec![
        ReplayItem::UserMessage(user_message(turn)),
        // Renoa's ordinary shape: thought, then text, then tool work.
        ReplayItem::AssistantMessage(ReplayAssistantMessage {
            message_id: assistant_id,
            segments: vec![
                ReplaySegment::Reasoning("thinking".to_owned()),
                ReplaySegment::Text("Done.".to_owned()),
            ],
        }),
        tool_item("call-1", ActivityKind::Command),
    ]);

    let reconciled = reconcile_replay(&[], &[], &[], &replay, RECONCILED_AT).expect("valid replay");

    assert_eq!(reconciled.messages.len(), 2);
    let reasoning_block = &reconciled.transcript_blocks[0];
    assert!(reasoning_block.activities[0].reasoning.is_some());
    // The thought block points at the answer's index, so it renders above.
    assert_eq!(reasoning_block.after_message, 1);
    let tool_block = &reconciled.transcript_blocks[1];
    assert!(tool_block.activities[0].reasoning.is_none());
    assert_eq!(tool_block.after_message, 2);
}

#[test]
fn applying_the_same_replay_twice_is_fully_idempotent() {
    let turn = Uuid::new_v4();
    let replay = replay_of(vec![
        ReplayItem::UserMessage(user_message(turn)),
        ReplayItem::AssistantMessage(assistant_message(vec![
            ReplaySegment::Reasoning("step".to_owned()),
            ReplaySegment::Text("Answer".to_owned()),
        ])),
        tool_item("call-9", ActivityKind::Search),
    ]);

    let first = reconcile_replay(&[], &[], &[], &replay, RECONCILED_AT).expect("valid replay");
    let second = reconcile_replay(
        &first.messages,
        &first.transcript_blocks,
        &first.turns,
        &replay,
        RECONCILED_AT + 100,
    )
    .expect("valid replay");

    assert_eq!(fingerprint(&first), fingerprint(&second));
}

mod ordering;
#[test]
fn a_covered_cached_turn_keeps_a_checkpoint_only_at_the_same_position() {
    let turn = Uuid::new_v4();
    let replay = replay_of(vec![ReplayItem::UserMessage(user_message(turn))]);
    let kept = cached_turn(turn, 1, TurnStatus::Completed, Some(position_checkpoint(1)));

    let reconciled =
        reconcile_replay(&[], &[], &[kept], &replay, RECONCILED_AT).expect("valid replay");

    assert_eq!(reconciled.turns.len(), 1);
    assert_eq!(reconciled.turns[0].id, turn);
    assert_eq!(reconciled.turns[0].turn_count, 1);
    assert_eq!(
        reconciled.turns[0]
            .checkpoint
            .as_ref()
            .map(|checkpoint| checkpoint.turn_count),
        Some(1)
    );
    assert_eq!(reconciled.turns[0].started_at, 100);
}

#[test]
fn a_moved_turn_discards_its_position_dependent_checkpoint() {
    let turn = Uuid::new_v4();
    let replay = replay_of(vec![ReplayItem::UserMessage(user_message(turn))]);
    let moved = cached_turn(turn, 7, TurnStatus::Completed, Some(position_checkpoint(7)));

    let reconciled =
        reconcile_replay(&[], &[], &[moved], &replay, RECONCILED_AT).expect("valid replay");

    assert_eq!(reconciled.turns[0].turn_count, 1);
    assert!(
        reconciled.turns[0].checkpoint.is_none(),
        "a checkpoint captured at turn 7 cannot silently attach to position 1"
    );
}

#[test]
fn abc_reconciled_against_ac_is_contiguous_and_fork_rewind_index_by_position() {
    let turn_a = Uuid::from_u128(10);
    let turn_b = Uuid::from_u128(11);
    let turn_c = Uuid::from_u128(12);
    let replay = replay_of(vec![
        ReplayItem::UserMessage(user_message(turn_a)),
        ReplayItem::UserMessage(user_message(turn_c)),
    ]);
    let cached_turns = vec![
        cached_turn(
            turn_a,
            1,
            TurnStatus::Completed,
            Some(position_checkpoint(1)),
        ),
        cached_turn(
            turn_b,
            2,
            TurnStatus::Completed,
            Some(position_checkpoint(2)),
        ),
        cached_turn(
            turn_c,
            3,
            TurnStatus::Completed,
            Some(position_checkpoint(3)),
        ),
    ];

    let reconciled =
        reconcile_replay(&[], &[], &cached_turns, &replay, RECONCILED_AT).expect("valid replay");

    assert_eq!(
        reconciled
            .turns
            .iter()
            .map(|turn| (turn.id, turn.turn_count, turn.checkpoint.is_some()))
            .collect::<Vec<_>>(),
        [(turn_a, 1, true), (turn_c, 2, false)]
    );

    let mut session = AgentSession::new(Uuid::from_u128(1), ProviderKind::Renoa);
    session.messages = reconciled.messages;
    session.transcript_blocks = reconciled.transcript_blocks;
    session.turns = reconciled.turns;

    let forked = session
        .fork_through_turn(
            2,
            ProviderResumeCursor::Renoa {
                session_id: Uuid::from_u128(99).to_string(),
            },
            "fork",
        )
        .expect("fork through the rebuilt second turn");
    assert_eq!(forked.turns.len(), 2);
    assert_eq!(
        forked
            .turns
            .iter()
            .map(|turn| turn.turn_count)
            .collect::<Vec<_>>(),
        [1, 2]
    );

    session.truncate_after_turn(1);
    assert_eq!(session.turns.len(), 1);
    assert_eq!(session.turns[0].id, turn_a);
    assert_eq!(session.turns[0].turn_count, 1);
}

#[test]
fn next_turn_after_ac_reconciliation_receives_count_three() {
    let turn_a = Uuid::from_u128(20);
    let turn_c = Uuid::from_u128(21);
    let replay = replay_of(vec![
        ReplayItem::UserMessage(user_message(turn_a)),
        ReplayItem::UserMessage(user_message(turn_c)),
    ]);
    let cached_turns = vec![
        cached_turn(turn_a, 1, TurnStatus::Completed, None),
        cached_turn(Uuid::from_u128(2), 2, TurnStatus::Completed, None),
        cached_turn(turn_c, 3, TurnStatus::Completed, None),
    ];

    let reconciled =
        reconcile_replay(&[], &[], &cached_turns, &replay, RECONCILED_AT).expect("valid replay");
    assert_eq!(
        reconciled
            .turns
            .iter()
            .map(|turn| (turn.id, turn.turn_count))
            .collect::<Vec<_>>(),
        [(turn_a, 1), (turn_c, 2)]
    );

    let mut session = AgentSession::new(Uuid::from_u128(1), ProviderKind::Renoa);
    session.turns = reconciled.turns;
    session.begin_turn("next");
    assert_eq!(session.turns.last().map(|turn| turn.turn_count), Some(3));
    assert_eq!(
        session
            .turns
            .iter()
            .map(|turn| turn.turn_count)
            .collect::<Vec<_>>(),
        [1, 2, 3]
    );
}

#[test]
fn a_missing_replayed_turn_is_rebuilt_as_completed() {
    let turn = Uuid::new_v4();
    let replay = replay_of(vec![ReplayItem::UserMessage(user_message(turn))]);

    let reconciled = reconcile_replay(&[], &[], &[], &replay, RECONCILED_AT).expect("valid replay");

    assert_eq!(reconciled.turns.len(), 1);
    assert_eq!(reconciled.turns[0].id, turn);
    assert_eq!(reconciled.turns[0].status, TurnStatus::Completed);
    assert_eq!(reconciled.turns[0].turn_count, 1);
    assert!(reconciled.turns[0].provider_turn_started);
}
