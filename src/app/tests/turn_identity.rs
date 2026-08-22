use uuid::Uuid;

use super::super::ComposerSubmission;
use super::super::runtime::{prepared_submission_can_start, unwind_failed_submission};
use crate::model::{AgentSession, ProviderKind};

#[test]
fn accepted_submission_uses_one_id_for_turn_and_user_message() {
    let id = Uuid::from_u128(7);
    let submission = ComposerSubmission {
        id,
        prompt: "Build the feature".into(),
        display_content: None,
        attachments: Vec::new(),
    };
    let mut session = AgentSession::new(Uuid::new_v4(), ProviderKind::Codex);
    let turn_id = session.begin_turn_with_id_and_presentation(
        submission.id,
        &submission.prompt,
        submission.display_content.clone(),
        submission.attachments.clone(),
    );
    assert_eq!(turn_id, id);
    assert_eq!(session.turns.last().map(|turn| turn.id), Some(id));
    assert_eq!(
        session.messages.last().and_then(|message| message.turn_id),
        Some(id)
    );
}

#[test]
fn queued_and_delayed_submission_reuses_its_accepted_id() {
    let submission = ComposerSubmission::plain("Follow up".into());
    let accepted_id = submission.id;
    let queued = submission.into_queued_message();
    assert_eq!(queued.id, accepted_id);

    let restored = ComposerSubmission::from_queued_message(queued);
    let (arrived, wait_for_release) = crossbeam_channel::bounded(1);
    let (release, released) = crossbeam_channel::bounded(1);
    let preparation = std::thread::spawn(move || {
        arrived.send(restored.id).unwrap();
        released.recv().unwrap();
        restored
    });
    assert_eq!(wait_for_release.recv().unwrap(), accepted_id);
    release.send(()).unwrap();
    let prepared = preparation.join().unwrap();

    let mut session = AgentSession::new(Uuid::new_v4(), ProviderKind::Codex);
    let turn_id = session.begin_turn_with_id_and_presentation(
        prepared.id,
        &prepared.prompt,
        prepared.display_content.clone(),
        prepared.attachments.clone(),
    );
    session.status = crate::model::SessionStatus::Connecting;
    assert_eq!(turn_id, accepted_id);
    assert_eq!(prepared.turn_prompt("Follow up".into()).id, accepted_id);
    assert!(prepared_submission_can_start(&session, accepted_id));
    assert!(!prepared_submission_can_start(&session, Uuid::new_v4()));
}

#[test]
fn stale_preparation_failure_does_not_unwind_replacement_turn() {
    let stale_id = Uuid::from_u128(11);
    let replacement_id = Uuid::from_u128(12);
    let mut session = AgentSession::new(Uuid::new_v4(), ProviderKind::Codex);
    session.begin_turn_with_id_and_presentation(stale_id, "Old", None, Vec::new());
    session.unwind_unstarted_turn(stale_id);
    session.begin_turn_with_id_and_presentation(replacement_id, "Current", None, Vec::new());
    session.status = crate::model::SessionStatus::Connecting;

    assert!(!unwind_failed_submission(&mut session, stale_id));
    assert_eq!(session.active_turn_id(), Some(replacement_id));
    assert_eq!(session.messages.len(), 1);
    assert_eq!(
        session.messages.last().and_then(|message| message.turn_id),
        Some(replacement_id)
    );
    assert_eq!(session.status, crate::model::SessionStatus::Connecting);
}
