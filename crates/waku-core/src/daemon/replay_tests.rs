use super::*;

use crate::driver::DriverControl;
use crate::model::{ActivityItem, MessageRole, RuntimeEventCursor, TurnStatus};
use base64::Engine as _;
use std::fs;
use std::sync::atomic::{AtomicUsize, Ordering};
use waku_protocol::replay::{
    ProviderReplay, RenoaReplayCommit, ReplayAssistantMessage, ReplayItem, ReplaySegment,
    ReplayTool, ReplayUserMessage, reconcile_replay,
};

struct TestRoot(PathBuf);

impl TestRoot {
    fn new() -> Self {
        let path = std::env::temp_dir().join(format!("waku-renoa-replay-{}", Uuid::new_v4()));
        fs::create_dir_all(&path).expect("create test root");
        Self(path)
    }
}

impl Drop for TestRoot {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.0);
    }
}

#[derive(Default)]
struct InertDriver {
    replay_acks: Arc<AtomicUsize>,
}

impl DriverControl for InertDriver {
    fn prompt(&self, _turn: waku_protocol::TurnPrompt) {}
    fn acknowledge_replay(&self) -> bool {
        self.replay_acks.fetch_add(1, Ordering::AcqRel);
        true
    }
    fn cancel(&self) {}
    fn respond(&self, _request_id: String, _option_id: String) {}
    fn rollback(&self, _turns: usize) -> anyhow::Result<Option<ProviderResumeCursor>> {
        Ok(None)
    }
}

struct ReplayFixture {
    database: PathBuf,
    backend: WakuBackend,
    session_id: Uuid,
    runtime_id: Uuid,
    replay_acks: Arc<AtomicUsize>,
    _root: TestRoot,
}

impl ReplayFixture {
    fn new() -> Self {
        let root = TestRoot::new();
        let database = root.0.join("app.db");
        let store = StateStore::daemon(database.clone());
        let mut state = PersistedState::empty();
        let mut session = AgentSession::new(Uuid::new_v4(), ProviderKind::Renoa);
        session.provider_cursor = Some(ProviderResumeCursor::Renoa {
            session_id: Uuid::from_u128(10).to_string(),
        });
        session.begin_turn("stale local question");
        session.push_message(MessageRole::Assistant, "stale local answer");
        session.finish_active_turn(TurnStatus::Completed);
        let session_id = session.id;
        state.push_session(session);
        store.save(&mut state).expect("seed task database");

        let backend = WakuBackend::new(
            DaemonSettingsStore::open(root.0.join("settings.json")).expect("open test settings"),
            store,
        )
        .expect("open test backend");
        let runtime_id = Uuid::from_u128(20);
        let replay_acks = Arc::new(AtomicUsize::new(0));
        backend.sessions.lock().insert(
            session_id,
            (
                runtime_id,
                DriverHandle::from_control(Arc::new(InertDriver {
                    replay_acks: replay_acks.clone(),
                })),
            ),
        );
        Self {
            database,
            backend,
            session_id,
            runtime_id,
            replay_acks,
            _root: root,
        }
    }

    fn persisted_session(&self) -> AgentSession {
        let store = StateStore::daemon(self.database.clone());
        let mut state = store.load().expect("reload task database");
        let session = state
            .sessions
            .iter_mut()
            .find(|session| session.id == self.session_id)
            .expect("stored Renoa session");
        store.hydrate(session).expect("hydrate stored transcript");
        session.clone()
    }

    fn send_commit(
        &self,
        commit_id: Uuid,
        commit: &RenoaReplayCommit,
        fragment_bytes: usize,
    ) -> anyhow::Result<ResponsePayload> {
        let fragments = commit.encode_fragments(commit_id, fragment_bytes)?;
        let count = fragments.len();
        let mut final_response = None;
        for (index, fragment) in fragments.into_iter().enumerate() {
            let response = self.backend.commit_renoa_replay_fragment(
                self.session_id,
                self.runtime_id,
                fragment,
            )?;
            if index + 1 == count {
                final_response = Some(response);
            } else {
                assert!(matches!(response, ResponsePayload::Ack));
            }
        }
        final_response.ok_or_else(|| anyhow!("commit produced no fragments"))
    }

    fn process_reload(self) -> (TestRoot, AgentSession) {
        let Self {
            database,
            backend,
            session_id,
            _root,
            ..
        } = self;
        drop(backend);
        let store = StateStore::daemon(database);
        let mut state = store.load().expect("reload after daemon shutdown");
        let session = state
            .sessions
            .iter_mut()
            .find(|session| session.id == session_id)
            .expect("reloaded Renoa session");
        store
            .hydrate(session)
            .expect("hydrate after daemon shutdown");
        (_root, session.clone())
    }
}

fn authoritative_commit(session: &AgentSession, cursor: RuntimeEventCursor) -> RenoaReplayCommit {
    let turn_id = Uuid::from_u128(30);
    let mut activity = ActivityItem::new(
        Some("shared-tool".into()),
        ActivityKind::Command,
        "run build",
        None,
        true,
    );
    activity.output = Some("bounded preview".to_owned());
    activity.authoritative_output = Some(format!("{}END", "x".repeat(40_000)));
    activity.authoritative_raw_output = Some(format!(
        r#"{{"diagnostic":"{}RAW-END"}}"#,
        "z".repeat(40_000)
    ));
    activity.image_urls = vec![format!(
        "data:image/png;base64,{}",
        base64::engine::general_purpose::STANDARD.encode(large_image_bytes())
    )];
    let replay = ProviderReplay {
        items: vec![
            ReplayItem::UserMessage(ReplayUserMessage {
                message_id: Uuid::from_u128(31),
                turn_id,
                text: "durable question".into(),
            }),
            ReplayItem::AssistantMessage(ReplayAssistantMessage {
                message_id: Uuid::from_u128(32),
                segments: vec![
                    ReplaySegment::Reasoning("reason".into()),
                    ReplaySegment::Text("durable answer".into()),
                ],
            }),
            ReplayItem::ToolCall(ReplayTool {
                call_id: "shared-tool".into(),
                activity: Box::new(activity),
            }),
        ],
    };
    let projection = reconcile_replay(
        &session.messages,
        &session.transcript_blocks,
        &session.turns,
        &replay,
        100,
    )
    .expect("reconcile authoritative replay");
    let base = waku_protocol::replay::renoa_replay_base(session).expect("fingerprint replay base");
    let reconciled_active_tail_fingerprint = waku_protocol::replay::replay_active_tail_fingerprint(
        &projection.messages,
        &projection.transcript_blocks,
        &projection.turns,
    )
    .expect("fingerprint reconciled active tail");
    RenoaReplayCommit {
        expected_cursor: session.runtime_event_cursor,
        expected_projection_fingerprint: base.projection_fingerprint,
        expected_settled_projection_fingerprint: base.settled_projection_fingerprint,
        local_active_tail_fingerprint: base.active_tail_fingerprint,
        reconciled_active_tail_fingerprint,
        accepted_cursor: cursor,
        messages: projection.messages,
        transcript_blocks: projection.transcript_blocks,
        turns: projection.turns,
        reconciled_at: 100,
    }
}

fn session_base_fingerprint(session: &AgentSession) -> [u64; 2] {
    let bytes = serde_json::to_vec(&(
        &session.messages,
        &session.transcript_blocks,
        &session.turns,
        session.runtime_event_cursor,
        session.renoa_replay_cursor,
    ))
    .expect("serialize replay base");
    waku_protocol::replay::replay_projection_fingerprint(&bytes)
}

fn large_image_bytes() -> Vec<u8> {
    vec![0x5a; 9 * 1024]
}

fn messages_fingerprint(messages: &[crate::model::Message]) -> Vec<u8> {
    serde_json::to_vec(messages).expect("serialize messages")
}

fn session_projection_fingerprint(session: &AgentSession) -> Vec<u8> {
    let blocks = blocks_without_image_payloads(&session.transcript_blocks);
    serde_json::to_vec(&(&session.messages, blocks, &session.turns))
        .expect("serialize session projection")
}

fn commit_projection_fingerprint(commit: &RenoaReplayCommit) -> Vec<u8> {
    let blocks = blocks_without_image_payloads(&commit.transcript_blocks);
    serde_json::to_vec(&(&commit.messages, blocks, &commit.turns))
        .expect("serialize commit projection")
}

fn blocks_without_image_payloads(
    blocks: &[crate::model::TranscriptBlock],
) -> Vec<crate::model::TranscriptBlock> {
    let mut blocks = blocks.to_vec();
    for block in &mut blocks {
        for activity in &mut block.activities {
            for image in &mut activity.image_urls {
                image.clear();
            }
        }
    }
    blocks
}

#[test]
fn fragmented_commit_is_atomic_restartable_idempotent_and_reloadable() {
    let fixture = ReplayFixture::new();
    let before = fixture.persisted_session();
    let cursor = RuntimeEventCursor {
        runtime_id: fixture.runtime_id,
        epoch: Uuid::from_u128(21),
        sequence: 50,
    };
    let commit = authoritative_commit(&before, cursor);
    let first_transaction = commit
        .encode_fragments(Uuid::from_u128(40), 256)
        .expect("fragment replay commit");
    assert!(first_transaction.len() > 1);

    let response = fixture
        .backend
        .commit_renoa_replay_fragment(
            fixture.session_id,
            fixture.runtime_id,
            first_transaction[0].clone(),
        )
        .expect("accept partial transaction");
    assert!(matches!(response, ResponsePayload::Ack));
    assert_eq!(fixture.replay_acks.load(Ordering::Acquire), 0);
    assert_eq!(fixture.persisted_session().runtime_event_cursor, None);
    assert_eq!(
        messages_fingerprint(&fixture.persisted_session().messages),
        messages_fingerprint(&before.messages)
    );

    let mut invalid_projection = authoritative_commit(
        &before,
        RuntimeEventCursor {
            runtime_id: fixture.runtime_id,
            epoch: Uuid::from_u128(49),
            sequence: 1,
        },
    );
    invalid_projection.messages[0].streaming = true;
    assert!(
        fixture
            .send_commit(Uuid::from_u128(53), &invalid_projection, 256)
            .is_err()
    );
    assert_eq!(
        messages_fingerprint(&fixture.persisted_session().messages),
        messages_fingerprint(&before.messages)
    );

    // A restarted desktop starts at fragment zero with a new transaction id;
    // the abandoned prefix cannot poison or partially advance persistence.
    let response = fixture
        .send_commit(Uuid::from_u128(41), &commit, 256)
        .expect("redeliver complete transaction");
    let ResponsePayload::RenoaReplayCommitted { session } = response else {
        panic!("final fragment must acknowledge the durable session");
    };
    assert_eq!(session.runtime_event_cursor, Some(cursor));
    assert_eq!(session.renoa_replay_cursor, Some(cursor));
    assert_eq!(fixture.replay_acks.load(Ordering::Acquire), 1);
    assert_eq!(
        session_projection_fingerprint(&session),
        commit_projection_fingerprint(&commit)
    );
    let persisted_activity = &session.transcript_blocks[1].activities[0];
    assert_eq!(
        persisted_activity.output.as_deref(),
        Some("bounded preview")
    );
    assert!(
        persisted_activity
            .durable_output()
            .is_some_and(|output| output.len() > 40_000 && output.ends_with("END"))
    );
    assert!(
        persisted_activity
            .durable_raw_output()
            .is_some_and(|output| output.len() > 40_000 && output.contains("RAW-END"))
    );
    let image_reference = persisted_activity
        .image_urls
        .first()
        .expect("persisted replay image");
    assert!(waku_protocol::blob::is_reference(image_reference));
    let image_path = fixture
        .backend
        .task_store
        .blobs()
        .path_for(image_reference)
        .expect("persisted replay blob path");
    assert_eq!(
        fs::read(image_path).expect("read replay blob"),
        large_image_bytes()
    );

    let persisted = fixture.persisted_session();
    assert_eq!(
        session_projection_fingerprint(&persisted),
        commit_projection_fingerprint(&commit)
    );
    assert_eq!(persisted.runtime_event_cursor, Some(cursor));
    assert_eq!(persisted.renoa_replay_cursor, Some(cursor));

    // Lost acknowledgement: exact semantic redelivery is a no-op success.
    let response = fixture
        .send_commit(Uuid::from_u128(42), &commit, 512)
        .expect("idempotent redelivery");
    assert!(matches!(
        response,
        ResponsePayload::RenoaReplayCommitted { .. }
    ));
    assert_eq!(fixture.replay_acks.load(Ordering::Acquire), 2);
    assert_eq!(
        session_projection_fingerprint(&fixture.persisted_session()),
        commit_projection_fingerprint(&commit)
    );

    let (_root, reloaded) = fixture.process_reload();
    assert_eq!(
        session_projection_fingerprint(&reloaded),
        commit_projection_fingerprint(&commit)
    );
    assert_eq!(reloaded.runtime_event_cursor, Some(cursor));
    assert_eq!(reloaded.renoa_replay_cursor, Some(cursor));
}

#[test]
fn empty_commit_clears_stale_history_and_malformed_input_changes_nothing() {
    let fixture = ReplayFixture::new();
    let before = fixture.persisted_session();
    let malformed = waku_protocol::replay::ReplayFragment {
        replay_id: Uuid::from_u128(50),
        index: 0,
        total: 0,
        json: String::new(),
    };
    assert!(
        fixture
            .backend
            .commit_renoa_replay_fragment(fixture.session_id, fixture.runtime_id, malformed)
            .is_err()
    );
    assert_eq!(
        messages_fingerprint(&fixture.persisted_session().messages),
        messages_fingerprint(&before.messages)
    );

    let projection = reconcile_replay(
        &before.messages,
        &before.transcript_blocks,
        &before.turns,
        &ProviderReplay::default(),
        200,
    )
    .expect("empty replay is valid");
    let cursor = RuntimeEventCursor {
        runtime_id: fixture.runtime_id,
        epoch: Uuid::from_u128(51),
        sequence: 7,
    };
    let base =
        waku_protocol::replay::renoa_replay_base(&before).expect("fingerprint empty-replay base");
    let reconciled_active_tail_fingerprint = waku_protocol::replay::replay_active_tail_fingerprint(
        &projection.messages,
        &projection.transcript_blocks,
        &projection.turns,
    )
    .expect("fingerprint empty replay's active tail");
    let commit = RenoaReplayCommit {
        expected_cursor: None,
        expected_projection_fingerprint: base.projection_fingerprint,
        expected_settled_projection_fingerprint: base.settled_projection_fingerprint,
        local_active_tail_fingerprint: base.active_tail_fingerprint,
        reconciled_active_tail_fingerprint,
        accepted_cursor: cursor,
        messages: projection.messages,
        transcript_blocks: projection.transcript_blocks,
        turns: projection.turns,
        reconciled_at: 200,
    };
    fixture
        .send_commit(Uuid::from_u128(52), &commit, 128)
        .expect("commit empty authoritative replay");
    let persisted = fixture.persisted_session();
    assert!(persisted.messages.is_empty());
    assert!(persisted.transcript_blocks.is_empty());
    assert!(persisted.turns.is_empty());
    assert_eq!(persisted.runtime_event_cursor, Some(cursor));
    assert_eq!(persisted.renoa_replay_cursor, Some(cursor));
}

#[path = "replay_tests/failures.rs"]
mod failures;

#[path = "replay_tests/base.rs"]
mod base;
