use serde::{Deserialize, Serialize};
use ts_rs::TS;
use uuid::Uuid;

/// One logical user turn crossing into a provider runtime.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize, TS)]
#[serde(rename_all = "camelCase")]
pub struct TurnPrompt {
    pub id: Uuid,
    pub prompt: String,
}

impl TurnPrompt {
    pub fn new(id: Uuid, prompt: impl Into<String>) -> Self {
        Self {
            id,
            prompt: prompt.into(),
        }
    }
}
