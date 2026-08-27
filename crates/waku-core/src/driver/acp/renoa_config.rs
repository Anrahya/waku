use agent_client_protocol::schema::v1::{
    SessionConfigKind, SessionConfigOption, SessionConfigOptionCategory,
    SessionConfigSelectOptions, SessionId, SetSessionConfigOptionRequest,
};
use agent_client_protocol::{Agent, ConnectionTo};

#[derive(Clone, Copy, Debug)]
pub(super) enum ConfigKind {
    Model,
    Reasoning,
}

impl ConfigKind {
    fn category(self) -> SessionConfigOptionCategory {
        match self {
            Self::Model => SessionConfigOptionCategory::Model,
            Self::Reasoning => SessionConfigOptionCategory::ThoughtLevel,
        }
    }
}

impl std::fmt::Display for ConfigKind {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Model => formatter.write_str("model"),
            Self::Reasoning => formatter.write_str("reasoning"),
        }
    }
}

#[derive(Debug)]
pub(super) enum ConfigError {
    Missing(ConfigKind),
    NotSelectable(ConfigKind),
    ValueNotAdvertised {
        kind: ConfigKind,
        value: String,
    },
    Request {
        kind: ConfigKind,
        source: agent_client_protocol::Error,
    },
}

impl std::fmt::Display for ConfigError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Missing(kind) => write!(formatter, "agent did not advertise a {kind} option"),
            Self::NotSelectable(kind) => {
                write!(formatter, "agent advertised a non-selectable {kind} option")
            }
            Self::ValueNotAdvertised { kind, value } => {
                write!(formatter, "agent did not advertise {kind} value {value}")
            }
            Self::Request { kind, source } => {
                write!(formatter, "agent rejected the {kind} selection: {source}")
            }
        }
    }
}

impl std::error::Error for ConfigError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Request { source, .. } => Some(source),
            _ => None,
        }
    }
}

/// Applies only the requested Renoa selections. When both are present, model
/// is applied first and the response's complete option set is the sole input
/// used to resolve reasoning.
pub(super) async fn apply(
    connection: &ConnectionTo<Agent>,
    session_id: &SessionId,
    mut options: Vec<SessionConfigOption>,
    model: Option<&str>,
    reasoning: Option<&str>,
) -> Result<Vec<SessionConfigOption>, ConfigError> {
    if let Some(model) = model {
        options = apply_one(connection, session_id, &options, ConfigKind::Model, model).await?;
    }
    if let Some(reasoning) = reasoning {
        options = apply_one(
            connection,
            session_id,
            &options,
            ConfigKind::Reasoning,
            reasoning,
        )
        .await?;
    }
    Ok(options)
}

async fn apply_one(
    connection: &ConnectionTo<Agent>,
    session_id: &SessionId,
    options: &[SessionConfigOption],
    kind: ConfigKind,
    value: &str,
) -> Result<Vec<SessionConfigOption>, ConfigError> {
    let option = options
        .iter()
        .find(|option| option.category.as_ref() == Some(&kind.category()))
        .ok_or(ConfigError::Missing(kind))?;
    let SessionConfigKind::Select(select) = &option.kind else {
        return Err(ConfigError::NotSelectable(kind));
    };
    let advertised = match &select.options {
        SessionConfigSelectOptions::Ungrouped(options) => options
            .iter()
            .any(|option| option.value.0.as_ref() == value),
        SessionConfigSelectOptions::Grouped(groups) => groups
            .iter()
            .flat_map(|group| group.options.iter())
            .any(|option| option.value.0.as_ref() == value),
        _ => false,
    };
    // Renoa's provider catalog can refresh while this ACP session stays open.
    // A model selected from Waku's fresh provider probe may therefore be newer
    // than these session options. Forward it to Renoa, which refreshes and
    // validates the authoritative catalog before accepting the change.
    if !advertised && !matches!(kind, ConfigKind::Model) {
        return Err(ConfigError::ValueNotAdvertised {
            kind,
            value: value.to_owned(),
        });
    }

    connection
        .send_request(SetSessionConfigOptionRequest::new(
            session_id.clone(),
            option.id.clone(),
            value,
        ))
        .block_task()
        .await
        .map(|response| response.config_options)
        .map_err(|source| ConfigError::Request { kind, source })
}
