use super::*;

#[test]
fn repeated_tool_identity_projects_two_idempotent_lifecycles() {
    let turn = Uuid::new_v4();
    let mut first = match tool_item("shared", ActivityKind::Command) {
        ReplayItem::ToolCall(tool) => tool,
        _ => unreachable!(),
    };
    first.activity.output = Some("first".into());
    let mut second = first.clone();
    second.activity.output = Some("second".into());
    let replay = replay_of(vec![
        ReplayItem::UserMessage(user_message(turn)),
        ReplayItem::ToolCall(first),
        ReplayItem::AssistantMessage(assistant_message(vec![ReplaySegment::Text(
            "continue".into(),
        )])),
        ReplayItem::ToolCall(second),
    ]);

    let once = reconcile_replay(&[], &[], &[], &replay, RECONCILED_AT)
        .expect("reused tool identity is scoped to each lifecycle");
    let outputs = once
        .transcript_blocks
        .iter()
        .flat_map(|block| block.activities.iter())
        .filter_map(|activity| activity.output.as_deref())
        .collect::<Vec<_>>();
    assert_eq!(outputs, ["first", "second"]);
    let twice = reconcile_replay(
        &once.messages,
        &once.transcript_blocks,
        &once.turns,
        &replay,
        RECONCILED_AT + 1,
    )
    .expect("repeat reconciliation");
    assert_eq!(fingerprint(&once), fingerprint(&twice));
}

#[test]
fn trailing_reasoning_keeps_text_before_reasoning_and_alternation_is_rejected() {
    let turn = Uuid::new_v4();
    let supported = replay_of(vec![
        ReplayItem::UserMessage(user_message(turn)),
        ReplayItem::AssistantMessage(assistant_message(vec![
            ReplaySegment::Text("answer".into()),
            ReplaySegment::Reasoning("postscript".into()),
        ])),
    ]);
    let projected = reconcile_replay(&[], &[], &[], &supported, RECONCILED_AT)
        .expect("one text/reasoning transition is lossless");
    assert_eq!(projected.messages[1].content, "answer");
    assert_eq!(projected.transcript_blocks[0].after_message, 2);
    assert_eq!(
        projected.transcript_blocks[0].activities[0]
            .reasoning
            .as_ref()
            .map(|reasoning| reasoning.content.as_str()),
        Some("postscript")
    );

    let unsupported = replay_of(vec![
        ReplayItem::UserMessage(user_message(Uuid::new_v4())),
        ReplayItem::AssistantMessage(assistant_message(vec![
            ReplaySegment::Text("one".into()),
            ReplaySegment::Reasoning("two".into()),
            ReplaySegment::Text("three".into()),
        ])),
    ]);
    assert!(matches!(
        reconcile_replay(&[], &[], &[], &unsupported, RECONCILED_AT),
        Err(ReplayReconcileError::UnsupportedAssistantOrder { .. })
    ));
}

#[test]
fn distinct_assistant_messages_preserve_tool_interleaving_exactly() {
    let turn = Uuid::new_v4();
    let first_id = Uuid::new_v4();
    let second_id = Uuid::new_v4();
    let replay = replay_of(vec![
        ReplayItem::UserMessage(user_message(turn)),
        ReplayItem::AssistantMessage(ReplayAssistantMessage {
            message_id: first_id,
            segments: vec![ReplaySegment::Text("before tool".into())],
        }),
        tool_item("between", ActivityKind::Command),
        ReplayItem::AssistantMessage(ReplayAssistantMessage {
            message_id: second_id,
            segments: vec![
                ReplaySegment::Reasoning("after tool thought".into()),
                ReplaySegment::Text("after tool answer".into()),
            ],
        }),
    ]);

    let projected = reconcile_replay(&[], &[], &[], &replay, RECONCILED_AT)
        .expect("distinct assistant identities make the interleaving lossless");
    assert_eq!(
        projected
            .messages
            .iter()
            .map(|message| (message.id, message.content.as_str()))
            .collect::<Vec<_>>(),
        [
            (projected.messages[0].id, "Fix the login bug"),
            (first_id, "before tool"),
            (second_id, "after tool answer"),
        ]
    );
    assert_eq!(projected.transcript_blocks.len(), 2);
    assert_eq!(projected.transcript_blocks[0].after_message, 2);
    assert_eq!(
        projected.transcript_blocks[0].activities[0]
            .source_id
            .as_deref(),
        Some("between")
    );
    assert_eq!(projected.transcript_blocks[1].after_message, 2);
    assert_eq!(
        projected.transcript_blocks[1].activities[0]
            .reasoning
            .as_ref()
            .map(|reasoning| reasoning.content.as_str()),
        Some("after tool thought")
    );

    let unsupported = replay_of(vec![
        ReplayItem::UserMessage(user_message(Uuid::new_v4())),
        ReplayItem::AssistantMessage(ReplayAssistantMessage {
            message_id: first_id,
            segments: vec![ReplaySegment::Text("before".into())],
        }),
        tool_item("middle", ActivityKind::Command),
        ReplayItem::AssistantMessage(ReplayAssistantMessage {
            message_id: first_id,
            segments: vec![ReplaySegment::Text("same message resumed".into())],
        }),
    ]);
    assert!(matches!(
        reconcile_replay(&[], &[], &[], &unsupported, RECONCILED_AT),
        Err(ReplayReconcileError::RepeatedAssistantMessage { message_id })
            if message_id == first_id
    ));
}

#[test]
fn active_tail_block_uses_after_message_as_an_insertion_count() {
    let settled = Uuid::new_v4();
    let active = Uuid::new_v4();
    let replay = replay_of(vec![ReplayItem::UserMessage(user_message(settled))]);
    let cached = vec![
        Message::new_for_turn(MessageRole::User, "stale", settled),
        Message::new_for_turn(MessageRole::User, "pending", active),
        Message::new_for_turn(MessageRole::Assistant, "streaming", active),
    ];
    let block = TranscriptBlock {
        after_message: 3,
        turn_id: Some(active),
        activities: vec![match tool_item("active-tool", ActivityKind::Command) {
            ReplayItem::ToolCall(tool) => *tool.activity,
            _ => unreachable!(),
        }],
    };
    let turns = vec![AgentTurn {
        id: active,
        turn_count: 2,
        status: TurnStatus::Running,
        provider_turn_started: true,
        provider_resume_at: None,
        started_at: 1,
        completed_at: None,
        checkpoint: None,
    }];

    let projected = reconcile_replay(&cached, &[block], &turns, &replay, RECONCILED_AT)
        .expect("active tail remains representable");
    assert_eq!(projected.messages.len(), 3);
    assert_eq!(projected.transcript_blocks[0].after_message, 3);
}

#[test]
fn one_message_id_cannot_name_both_user_and_assistant_semantics() {
    let turn = Uuid::new_v4();
    let message_id = Uuid::new_v4();
    let replay = replay_of(vec![
        ReplayItem::UserMessage(ReplayUserMessage {
            message_id,
            turn_id: turn,
            text: "prompt".into(),
        }),
        ReplayItem::AssistantMessage(ReplayAssistantMessage {
            message_id,
            segments: vec![ReplaySegment::Text("answer".into())],
        }),
    ]);

    assert!(matches!(
        reconcile_replay(&[], &[], &[], &replay, RECONCILED_AT),
        Err(ReplayReconcileError::RepeatedMessageIdentity { message_id: repeated })
            if repeated == message_id
    ));
}
