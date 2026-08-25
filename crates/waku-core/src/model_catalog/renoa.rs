use std::collections::HashSet;
use std::path::Path;

use serde::Deserialize;

use crate::model::{ProviderModel, ProviderModelOption};

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct Catalog {
    models: Vec<CatalogModel>,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
struct CatalogModel {
    id: String,
    name: String,
    is_default: bool,
    reasoning_levels: Vec<CatalogReasoningLevel>,
    default_reasoning: String,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct CatalogReasoningLevel {
    id: String,
    name: String,
}

#[derive(Debug, Eq, PartialEq)]
enum CatalogError {
    Json(String),
    EmptyModelId,
    DuplicateModelId(String),
    DefaultModelCount(usize),
    EmptyReasoningLevels(String),
    EmptyReasoningId(String),
    DuplicateReasoningId { model: String, reasoning: String },
    UnknownDefaultReasoning { model: String, reasoning: String },
}

impl std::fmt::Display for CatalogError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Json(error) => write!(formatter, "invalid Renoa model catalog JSON: {error}"),
            Self::EmptyModelId => write!(formatter, "Renoa model catalog contains an empty id"),
            Self::DuplicateModelId(id) => {
                write!(formatter, "Renoa model catalog repeats model id {id}")
            }
            Self::DefaultModelCount(count) => write!(
                formatter,
                "Renoa model catalog must contain exactly one default model, found {count}"
            ),
            Self::EmptyReasoningLevels(model) => {
                write!(formatter, "Renoa model {model} has no reasoning levels")
            }
            Self::EmptyReasoningId(model) => {
                write!(formatter, "Renoa model {model} has an empty reasoning id")
            }
            Self::DuplicateReasoningId { model, reasoning } => write!(
                formatter,
                "Renoa model {model} repeats reasoning id {reasoning}"
            ),
            Self::UnknownDefaultReasoning { model, reasoning } => write!(
                formatter,
                "Renoa model {model} defaults to unadvertised reasoning level {reasoning}"
            ),
        }
    }
}

pub(super) fn discover_models(binary: &Path) -> Vec<ProviderModel> {
    let mut command = crate::command_env::command(binary);
    let command = command.args(["models", "--json"]);
    let Ok(output) = crate::command_env::output(command) else {
        return Vec::new();
    };
    if !output.status.success() {
        return Vec::new();
    }
    parse_catalog(&output.stdout).unwrap_or_default()
}

fn parse_catalog(output: &[u8]) -> Result<Vec<ProviderModel>, CatalogError> {
    let catalog = serde_json::from_slice::<Catalog>(output)
        .map_err(|error| CatalogError::Json(error.to_string()))?;
    validate_catalog(catalog)
}

fn validate_catalog(catalog: Catalog) -> Result<Vec<ProviderModel>, CatalogError> {
    let default_count = catalog
        .models
        .iter()
        .filter(|model| model.is_default)
        .count();
    if default_count != 1 {
        return Err(CatalogError::DefaultModelCount(default_count));
    }

    let mut model_ids = HashSet::with_capacity(catalog.models.len());
    let mut models = Vec::with_capacity(catalog.models.len());
    for model in catalog.models {
        if model.id.trim().is_empty() {
            return Err(CatalogError::EmptyModelId);
        }
        if !model_ids.insert(model.id.clone()) {
            return Err(CatalogError::DuplicateModelId(model.id));
        }
        if model.reasoning_levels.is_empty() {
            return Err(CatalogError::EmptyReasoningLevels(model.id));
        }

        let mut reasoning_ids = HashSet::with_capacity(model.reasoning_levels.len());
        let mut reasoning = Vec::with_capacity(model.reasoning_levels.len());
        for level in model.reasoning_levels {
            if level.id.trim().is_empty() {
                return Err(CatalogError::EmptyReasoningId(model.id));
            }
            if !reasoning_ids.insert(level.id.clone()) {
                return Err(CatalogError::DuplicateReasoningId {
                    model: model.id,
                    reasoning: level.id,
                });
            }
            reasoning.push(ProviderModelOption::new(level.id, level.name));
        }
        if !reasoning_ids.contains(&model.default_reasoning) {
            return Err(CatalogError::UnknownDefaultReasoning {
                model: model.id,
                reasoning: model.default_reasoning,
            });
        }

        let mut provider_model =
            ProviderModel::new(model.id, model.name).reasoning(reasoning, model.default_reasoning);
        provider_model.is_default = model.is_default;
        models.push(provider_model);
    }
    Ok(models)
}

#[cfg(test)]
mod tests {
    use super::*;

    const VALID: &str = r#"{
        "models": [
            {
                "id": "grok-code",
                "name": "Grok Code",
                "isDefault": true,
                "reasoningLevels": [
                    {"id": "off", "name": "Off"},
                    {"id": "high", "name": "High"}
                ],
                "defaultReasoning": "high"
            },
            {
                "id": "grok-fast",
                "name": "Grok Fast",
                "isDefault": false,
                "reasoningLevels": [{"id": "off", "name": "Off"}],
                "defaultReasoning": "off"
            }
        ]
    }"#;

    #[test]
    fn valid_catalog_maps_models_and_reasoning_defaults() {
        let models = parse_catalog(VALID.as_bytes()).expect("valid Renoa catalog");

        assert_eq!(models.len(), 2);
        assert_eq!(models[0].id, "grok-code");
        assert!(models[0].is_default);
        assert_eq!(models[0].reasoning_efforts[0].id, "off");
        assert_eq!(models[0].reasoning_efforts[1].label, "High");
        assert_eq!(models[0].default_reasoning_effort.as_deref(), Some("high"));
        assert_eq!(models[1].default_reasoning_effort.as_deref(), Some("off"));
    }

    #[test]
    fn malformed_catalog_is_rejected_as_one_unit() {
        assert!(matches!(
            parse_catalog(br#"{"models": [}"#),
            Err(CatalogError::Json(_))
        ));
        assert!(matches!(
            parse_catalog(br#"{"models": [], "extra": true}"#),
            Err(CatalogError::Json(_))
        ));
    }

    #[test]
    fn duplicate_model_or_reasoning_ids_are_rejected() {
        let duplicate_model = VALID.replace("grok-fast", "grok-code");
        assert_eq!(
            parse_catalog(duplicate_model.as_bytes()),
            Err(CatalogError::DuplicateModelId("grok-code".into()))
        );

        let duplicate_reasoning = VALID.replace(
            r#"{"id": "high", "name": "High"}"#,
            r#"{"id": "off", "name": "High"}"#,
        );
        assert_eq!(
            parse_catalog(duplicate_reasoning.as_bytes()),
            Err(CatalogError::DuplicateReasoningId {
                model: "grok-code".into(),
                reasoning: "off".into(),
            })
        );
    }

    #[test]
    fn empty_model_or_reasoning_ids_are_rejected() {
        let empty_model = VALID.replacen(r#""id": "grok-code""#, r#""id": " ""#, 1);
        assert_eq!(
            parse_catalog(empty_model.as_bytes()),
            Err(CatalogError::EmptyModelId)
        );

        let empty_reasoning = VALID.replacen(r#""id": "off""#, r#""id": " ""#, 1);
        assert_eq!(
            parse_catalog(empty_reasoning.as_bytes()),
            Err(CatalogError::EmptyReasoningId("grok-code".into()))
        );
    }

    #[test]
    fn missing_or_multiple_default_models_are_rejected() {
        let missing = VALID.replace("\"isDefault\": true", "\"isDefault\": false");
        assert_eq!(
            parse_catalog(missing.as_bytes()),
            Err(CatalogError::DefaultModelCount(0))
        );
        let multiple = VALID.replace("\"isDefault\": false", "\"isDefault\": true");
        assert_eq!(
            parse_catalog(multiple.as_bytes()),
            Err(CatalogError::DefaultModelCount(2))
        );
    }

    #[test]
    fn empty_reasoning_or_unknown_default_reasoning_is_rejected() {
        let empty = VALID.replace(r#"[{"id": "off", "name": "Off"}]"#, "[]");
        assert_eq!(
            parse_catalog(empty.as_bytes()),
            Err(CatalogError::EmptyReasoningLevels("grok-fast".into()))
        );

        let mismatched = VALID.replace(
            r#""defaultReasoning": "high""#,
            r#""defaultReasoning": "xhigh""#,
        );
        assert_eq!(
            parse_catalog(mismatched.as_bytes()),
            Err(CatalogError::UnknownDefaultReasoning {
                model: "grok-code".into(),
                reasoning: "xhigh".into(),
            })
        );
    }
}
