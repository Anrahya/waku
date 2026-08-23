use super::*;

#[test]
fn unsaved_desktop_tail_commits_against_the_exact_daemon_base() {
    let fixture = ReplayFixture::new();
    let daemon_session = fixture.persisted_session();
    let response = fixture
        .backend
        .read_renoa_replay_base(fixture.session_id, fixture.runtime_id)
        .expect("read replay base");
    let ResponsePayload::RenoaReplayBase { base, session } = response else {
        panic!("replay base query returned the wrong response");
    };
    assert_eq!(session.id, fixture.session_id);
    assert_eq!(
        base.projection_fingerprint,
        session_base_fingerprint(&daemon_session)
    );
    assert!(base.active_tail_fingerprint.is_none());

    // This accepted turn has not reached the delayed ordinary task-state save.
    // Replay commit must still preserve it while comparing against the actual
    // daemon base, not a timing-dependent desktop fingerprint.
    let pending_turn = Uuid::from_u128(80);
    let mut desktop_session = daemon_session.clone();
    desktop_session.begin_turn_with_id_and_presentation(
        pending_turn,
        "accepted but not ordinarily saved",
        None,
        Vec::new(),
    );
    let projection = reconcile_replay(
        &desktop_session.messages,
        &desktop_session.transcript_blocks,
        &desktop_session.turns,
        &ProviderReplay::default(),
        300,
    )
    .expect("empty Renoa history preserves only the active local tail");
    let cursor = RuntimeEventCursor {
        runtime_id: fixture.runtime_id,
        epoch: Uuid::from_u128(81),
        sequence: 1,
    };
    let commit = RenoaReplayCommit {
        expected_cursor: base.cursor,
        expected_projection_fingerprint: base.projection_fingerprint,
        expected_settled_projection_fingerprint: base.settled_projection_fingerprint,
        local_active_tail_fingerprint: waku_protocol::replay::renoa_replay_base(&desktop_session)
            .expect("fingerprint retained desktop tail")
            .active_tail_fingerprint,
        reconciled_active_tail_fingerprint: waku_protocol::replay::replay_active_tail_fingerprint(
            &projection.messages,
            &projection.transcript_blocks,
            &projection.turns,
        )
        .expect("fingerprint reconciled active tail"),
        accepted_cursor: cursor,
        messages: projection.messages,
        transcript_blocks: projection.transcript_blocks,
        turns: projection.turns,
        reconciled_at: 300,
    };

    fixture
        .send_commit(Uuid::from_u128(82), &commit, 256)
        .expect("commit replay with unsaved desktop tail");
    let persisted = fixture.persisted_session();
    assert_eq!(persisted.messages.len(), 1);
    assert_eq!(persisted.messages[0].turn_id, Some(pending_turn));
    assert_eq!(persisted.turns.len(), 1);
    assert_eq!(persisted.turns[0].id, pending_turn);
    assert_eq!(persisted.runtime_event_cursor, Some(cursor));
}

#[test]
fn delayed_save_of_the_exact_desktop_tail_does_not_invalidate_replay() {
    let fixture = ReplayFixture::new();
    let daemon_session = fixture.persisted_session();
    let ResponsePayload::RenoaReplayBase { base, session } = fixture
        .backend
        .read_renoa_replay_base(fixture.session_id, fixture.runtime_id)
        .expect("read replay base")
    else {
        panic!("replay base query returned the wrong response");
    };
    assert_eq!(session.id, fixture.session_id);

    let pending_turn = Uuid::from_u128(83);
    let mut desktop_session = daemon_session;
    desktop_session.begin_turn_with_id_and_presentation(
        pending_turn,
        "accepted before the delayed save",
        None,
        Vec::new(),
    );
    let desktop_base = waku_protocol::replay::renoa_replay_base(&desktop_session)
        .expect("fingerprint retained desktop tail");
    let projection = reconcile_replay(
        &desktop_session.messages,
        &desktop_session.transcript_blocks,
        &desktop_session.turns,
        &ProviderReplay::default(),
        301,
    )
    .expect("empty Renoa history preserves the active local tail");

    // Model the ordinary delayed task-state save landing after the base read
    // but before the authoritative commit. Only this exact active tail may be
    // tolerated; settled transcript or cursor changes still invalidate CAS.
    {
        let mut state = fixture.backend.task_state.lock();
        let stored = state
            .sessions
            .iter_mut()
            .find(|session| session.id == fixture.session_id)
            .expect("live daemon session");
        *stored = desktop_session;
        state.mark_session_dirty(fixture.session_id);
        fixture
            .backend
            .task_store
            .save(&mut state)
            .expect("persist delayed ordinary save");
    }

    let cursor = RuntimeEventCursor {
        runtime_id: fixture.runtime_id,
        epoch: Uuid::from_u128(84),
        sequence: 1,
    };
    let commit = RenoaReplayCommit {
        expected_cursor: base.cursor,
        expected_projection_fingerprint: base.projection_fingerprint,
        expected_settled_projection_fingerprint: base.settled_projection_fingerprint,
        local_active_tail_fingerprint: desktop_base.active_tail_fingerprint,
        reconciled_active_tail_fingerprint: waku_protocol::replay::replay_active_tail_fingerprint(
            &projection.messages,
            &projection.transcript_blocks,
            &projection.turns,
        )
        .expect("fingerprint reconciled active tail"),
        accepted_cursor: cursor,
        messages: projection.messages,
        transcript_blocks: projection.transcript_blocks,
        turns: projection.turns,
        reconciled_at: 301,
    };

    fixture
        .send_commit(Uuid::from_u128(85), &commit, 256)
        .expect("commit replay after delayed ordinary save");
    let persisted = fixture.persisted_session();
    assert_eq!(persisted.messages.len(), 1);
    assert_eq!(persisted.messages[0].turn_id, Some(pending_turn));
    assert_eq!(persisted.runtime_event_cursor, Some(cursor));
}

#[test]
fn authoritative_identity_can_adopt_the_active_tail_without_duplication() {
    let fixture = ReplayFixture::new();
    let daemon_session = fixture.persisted_session();
    let ResponsePayload::RenoaReplayBase { base, session } = fixture
        .backend
        .read_renoa_replay_base(fixture.session_id, fixture.runtime_id)
        .expect("read replay base")
    else {
        panic!("replay base query returned the wrong response");
    };
    assert_eq!(session.id, fixture.session_id);
    let pending_turn = Uuid::from_u128(86);
    let authoritative_message = Uuid::from_u128(87);
    let mut desktop_session = daemon_session;
    desktop_session.begin_turn_with_id_and_presentation(
        pending_turn,
        "already accepted by Renoa",
        None,
        Vec::new(),
    );
    let local_tail = waku_protocol::replay::renoa_replay_base(&desktop_session)
        .expect("fingerprint local tail")
        .active_tail_fingerprint;
    let replay = ProviderReplay {
        items: vec![ReplayItem::UserMessage(ReplayUserMessage {
            message_id: authoritative_message,
            turn_id: pending_turn,
            text: "already accepted by Renoa".into(),
        })],
    };
    let projection = reconcile_replay(
        &desktop_session.messages,
        &desktop_session.transcript_blocks,
        &desktop_session.turns,
        &replay,
        302,
    )
    .expect("adopt active turn from authoritative replay");
    let reconciled_tail = waku_protocol::replay::replay_active_tail_fingerprint(
        &projection.messages,
        &projection.transcript_blocks,
        &projection.turns,
    )
    .expect("fingerprint adopted tail");
    assert_ne!(local_tail, reconciled_tail);

    let cursor = RuntimeEventCursor {
        runtime_id: fixture.runtime_id,
        epoch: Uuid::from_u128(88),
        sequence: 1,
    };
    let commit = RenoaReplayCommit {
        expected_cursor: base.cursor,
        expected_projection_fingerprint: base.projection_fingerprint,
        expected_settled_projection_fingerprint: base.settled_projection_fingerprint,
        local_active_tail_fingerprint: local_tail,
        reconciled_active_tail_fingerprint: reconciled_tail,
        accepted_cursor: cursor,
        messages: projection.messages,
        transcript_blocks: projection.transcript_blocks,
        turns: projection.turns,
        reconciled_at: 302,
    };

    fixture
        .send_commit(Uuid::from_u128(89), &commit, 256)
        .expect("commit adopted active tail");
    let persisted = fixture.persisted_session();
    assert_eq!(persisted.messages.len(), 1);
    assert_eq!(persisted.messages[0].id, authoritative_message);
    assert_eq!(persisted.messages[0].turn_id, Some(pending_turn));
    assert_eq!(persisted.turns.len(), 1);
}

#[test]
fn active_tail_fingerprint_detects_a_conflicting_daemon_turn() {
    let fixture = ReplayFixture::new();
    let mut daemon_session = fixture.persisted_session();
    daemon_session.begin_turn_with_id_and_presentation(
        Uuid::from_u128(90),
        "daemon pending",
        None,
        Vec::new(),
    );
    let daemon_base =
        waku_protocol::replay::renoa_replay_base(&daemon_session).expect("fingerprint daemon tail");

    let mut desktop_session = fixture.persisted_session();
    desktop_session.begin_turn_with_id_and_presentation(
        Uuid::from_u128(91),
        "desktop pending",
        None,
        Vec::new(),
    );
    let desktop_base = waku_protocol::replay::renoa_replay_base(&desktop_session)
        .expect("fingerprint desktop tail");

    assert!(daemon_base.active_tail_fingerprint.is_some());
    assert_ne!(
        daemon_base.active_tail_fingerprint,
        desktop_base.active_tail_fingerprint
    );
}
