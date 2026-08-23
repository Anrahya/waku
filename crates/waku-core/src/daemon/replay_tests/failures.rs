use super::*;

#[test]
fn ordinary_client_projection_cannot_replace_a_committed_replay() {
    let mut durable = AgentSession::new(Uuid::from_u128(60), ProviderKind::Renoa);
    let cursor = RuntimeEventCursor {
        runtime_id: Uuid::from_u128(61),
        epoch: Uuid::from_u128(62),
        sequence: 7,
    };
    durable.runtime_event_cursor = Some(cursor);
    durable.renoa_replay_cursor = Some(cursor);
    durable.push_message(MessageRole::User, "authoritative");
    let mut incoming = durable.clone();
    incoming.renoa_replay_cursor = None;
    incoming.messages.clear();
    incoming.push_message(MessageRole::User, "stale client cache");
    incoming.updated_at = durable.updated_at.saturating_add(1);

    assert!(session_projection_precedes(&durable, &incoming, None));
    let authoritative_messages = messages_fingerprint(&durable.messages);
    merge_stale_session_metadata(&mut durable, incoming.clone());
    assert_eq!(
        messages_fingerprint(&durable.messages),
        authoritative_messages
    );

    preserve_daemon_replay_marker(&durable, &mut incoming);
    assert_eq!(incoming.renoa_replay_cursor, durable.renoa_replay_cursor);

    let mut later_live_projection = durable.clone();
    later_live_projection.runtime_event_cursor = Some(RuntimeEventCursor {
        sequence: cursor.sequence + 1,
        ..cursor
    });
    assert!(
        !session_projection_precedes(&durable, &later_live_projection, None),
        "a later event cursor from the same runtime may extend the replay"
    );
}

#[test]
fn stale_runtime_or_base_cursor_cannot_overwrite_newer_state() {
    let fixture = ReplayFixture::new();
    let before = fixture.persisted_session();
    let cursor = RuntimeEventCursor {
        runtime_id: fixture.runtime_id,
        epoch: Uuid::from_u128(60),
        sequence: 1,
    };
    let mut commit = authoritative_commit(&before, cursor);
    commit.expected_cursor = Some(RuntimeEventCursor {
        runtime_id: Uuid::from_u128(99),
        epoch: Uuid::from_u128(99),
        sequence: 99,
    });
    assert!(
        fixture
            .send_commit(Uuid::from_u128(61), &commit, 512)
            .is_err()
    );
    assert_eq!(
        messages_fingerprint(&fixture.persisted_session().messages),
        messages_fingerprint(&before.messages)
    );
    assert_eq!(fixture.persisted_session().runtime_event_cursor, None);

    let fragments = commit
        .encode_fragments(Uuid::from_u128(62), 512)
        .expect("fragment stale runtime commit");
    assert!(
        fixture
            .backend
            .commit_renoa_replay_fragment(
                fixture.session_id,
                Uuid::from_u128(404),
                fragments[0].clone(),
            )
            .is_err()
    );
    assert_eq!(
        messages_fingerprint(&fixture.persisted_session().messages),
        messages_fingerprint(&before.messages)
    );
}

#[test]
fn stale_off_thread_projection_cannot_overwrite_a_newer_transcript() {
    let fixture = ReplayFixture::new();
    let before = fixture.persisted_session();
    let cursor = RuntimeEventCursor {
        runtime_id: fixture.runtime_id,
        epoch: Uuid::from_u128(65),
        sequence: 1,
    };
    let commit = authoritative_commit(&before, cursor);
    let newer = {
        let mut state = fixture.backend.task_state.lock();
        let index = state
            .sessions
            .iter()
            .position(|session| session.id == fixture.session_id)
            .expect("in-memory Renoa session");
        fixture
            .backend
            .task_store
            .hydrate(&mut state.sessions[index])
            .expect("hydrate newer projection");
        state.sessions[index].messages[0].content = "newer local projection".into();
        state.mark_session_dirty(fixture.session_id);
        fixture
            .backend
            .task_store
            .save(&mut state)
            .expect("persist newer projection");
        state.sessions[index].clone()
    };

    let error = fixture
        .send_commit(Uuid::from_u128(66), &commit, 512)
        .expect_err("stale snapshot must fail its compare-and-swap");
    assert!(error.to_string().contains("base transcript changed"));
    let persisted = fixture.persisted_session();
    assert_eq!(
        session_projection_fingerprint(&persisted),
        session_projection_fingerprint(&newer)
    );
    assert_eq!(persisted.runtime_event_cursor, None);
    assert_eq!(fixture.replay_acks.load(Ordering::Acquire), 0);
}

#[test]
fn sqlite_failure_leaves_memory_and_disk_at_the_previous_projection() {
    let fixture = ReplayFixture::new();
    let before = fixture.persisted_session();
    let cursor = RuntimeEventCursor {
        runtime_id: fixture.runtime_id,
        epoch: Uuid::from_u128(70),
        sequence: 1,
    };
    let commit = authoritative_commit(&before, cursor);
    let connection = rusqlite::Connection::open(&fixture.database).expect("open test database");
    connection
        .execute_batch(
            "CREATE TRIGGER reject_authoritative_replay
             BEFORE DELETE ON messages
             BEGIN
                 SELECT RAISE(ABORT, 'forced replay persistence failure');
             END;",
        )
        .expect("install deterministic persistence failure");

    let error = fixture
        .send_commit(Uuid::from_u128(71), &commit, 256)
        .expect_err("SQLite failure must reject the complete transaction");
    assert!(
        error
            .to_string()
            .contains("could not persist authoritative Renoa replay")
    );

    let persisted = fixture.persisted_session();
    assert_eq!(
        session_projection_fingerprint(&persisted),
        session_projection_fingerprint(&before)
    );
    assert_eq!(persisted.runtime_event_cursor, None);

    let state = fixture.backend.task_state.lock();
    let in_memory = state
        .sessions
        .iter()
        .find(|session| session.id == fixture.session_id)
        .expect("in-memory Renoa session");
    assert_eq!(
        session_projection_fingerprint(in_memory),
        session_projection_fingerprint(&before)
    );
    assert_eq!(in_memory.runtime_event_cursor, None);
    assert_eq!(fixture.replay_acks.load(Ordering::Acquire), 0);
}

#[test]
fn oversized_acknowledgement_fails_before_persistence_or_prompt_release() {
    let fixture = ReplayFixture::new();
    let before = fixture.persisted_session();
    let cursor = RuntimeEventCursor {
        runtime_id: fixture.runtime_id,
        epoch: Uuid::from_u128(75),
        sequence: 1,
    };
    let mut commit = authoritative_commit(&before, cursor);
    commit.messages[1].content = "x".repeat(waku_protocol::MAX_WIRE_MESSAGE_BYTES + 1_024);

    let error = fixture
        .send_commit(
            Uuid::from_u128(76),
            &commit,
            waku_protocol::SESSION_REPLAY_FRAGMENT_BYTES,
        )
        .expect_err("an oversized acknowledgement must fail before persistence");
    assert!(error.to_string().contains("replay acknowledgement"));
    assert_eq!(
        session_projection_fingerprint(&fixture.persisted_session()),
        session_projection_fingerprint(&before)
    );
    assert_eq!(fixture.replay_acks.load(Ordering::Acquire), 0);
}
