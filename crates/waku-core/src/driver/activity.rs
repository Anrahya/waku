//! Provider-neutral activity normalization.

use std::collections::HashSet;

use serde_json::Value;

use crate::model::{ActivityItem, ActivityKind};

const MAX_ACTIVITY_CHARS: usize = 16_000;

pub(super) fn tool_activity(
    source_id: Option<String>,
    kind: ActivityKind,
    title: String,
    arguments: Option<&Value>,
    output: Option<&Value>,
    image_source: Option<&Value>,
    failed: bool,
    complete: bool,
) -> ActivityItem {
    tool_activity_with_limit(
        ToolActivityInput {
            source_id,
            kind,
            title,
            arguments,
            output,
            image_source,
            failed,
            complete,
        },
        Some(MAX_ACTIVITY_CHARS),
    )
}

pub(super) struct ReplayToolActivity<'a> {
    pub source_id: Option<String>,
    pub kind: ActivityKind,
    pub title: String,
    pub arguments: Option<&'a Value>,
    pub output: Option<&'a Value>,
    pub raw_output: Option<&'a Value>,
    pub image_source: Option<&'a Value>,
    pub failed: bool,
    pub complete: bool,
}

/// Builds the durable authoritative-replay form. Presentation may derive a
/// bounded preview later, but persistence must not truncate provider history.
pub(super) fn replay_tool_activity(input: ReplayToolActivity<'_>) -> ActivityItem {
    let complete_arguments = input
        .arguments
        .filter(|value| !value.is_null())
        .and_then(|value| format_json_with_limit(value, None));
    let complete_output = input
        .output
        .filter(|value| !value.is_null())
        .and_then(|value| format_output(value, None));
    let complete_raw_output = input
        .raw_output
        .filter(|value| !value.is_null())
        .and_then(|value| format_json_with_limit(value, None));
    let mut activity = tool_activity_with_limit(
        ToolActivityInput {
            source_id: input.source_id,
            kind: input.kind,
            title: input.title,
            arguments: input.arguments,
            output: input.output,
            image_source: input.image_source,
            failed: input.failed,
            complete: input.complete,
        },
        Some(MAX_ACTIVITY_CHARS),
    );
    if complete_output != activity.output {
        activity = activity.with_authoritative_output(complete_output);
    }
    if complete_arguments != activity.arguments {
        activity = activity.with_authoritative_arguments(complete_arguments);
    }
    activity.with_authoritative_raw_output(complete_raw_output)
}

struct ToolActivityInput<'a> {
    source_id: Option<String>,
    kind: ActivityKind,
    title: String,
    arguments: Option<&'a Value>,
    output: Option<&'a Value>,
    image_source: Option<&'a Value>,
    failed: bool,
    complete: bool,
}

fn tool_activity_with_limit(
    input: ToolActivityInput<'_>,
    character_limit: Option<usize>,
) -> ActivityItem {
    let raw_arguments = input.arguments;
    let arguments = input
        .arguments
        .filter(|value| !value.is_null())
        .and_then(|value| format_json_with_limit(value, character_limit));
    let formatted_output = input
        .output
        .filter(|value| !value.is_null())
        .and_then(|value| format_output(value, character_limit));
    let mut image_urls = Vec::new();
    if let Some(value) = input.output {
        collect_image_urls(value, &mut image_urls);
    }
    if let Some(value) = input.image_source {
        collect_image_urls(value, &mut image_urls);
    }
    let mut seen = HashSet::new();
    image_urls.retain(|url| seen.insert(url.clone()));
    let detail = input
        .failed
        .then(|| {
            formatted_output.as_deref()?.lines().find_map(|line| {
                let line = line.trim();
                (!line.is_empty()).then(|| line.to_owned())
            })
        })
        .flatten();

    ActivityItem::new(
        input.source_id,
        input.kind,
        input.title,
        detail,
        input.complete,
    )
    .with_arguments(arguments)
    .with_activity_source(raw_arguments)
    .with_output(formatted_output)
    .with_image_urls(image_urls)
    .with_failed(input.failed)
}

pub(super) fn input_title(value: Option<&Value>) -> Option<String> {
    let value = value?;
    value
        .get("title")
        .or_else(|| value.pointer("/arguments/title"))
        .or_else(|| value.pointer("/input/title"))
        .or_else(|| value.pointer("/tool_input/title"))
        .and_then(Value::as_str)
        .map(str::trim)
        .filter(|title| !title.is_empty())
        .map(str::to_owned)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn title_supports_direct_and_provider_wrapped_arguments() {
        assert_eq!(
            input_title(Some(&serde_json::json!({"title": "Inspect app"}))).as_deref(),
            Some("Inspect app")
        );
        assert_eq!(
            input_title(Some(&serde_json::json!({
                "tool_name": "waku_js_repl__js",
                "arguments": {"code": "1", "title": "Inspect wrapped app"}
            })))
            .as_deref(),
            Some("Inspect wrapped app")
        );
        assert_eq!(
            input_title(Some(&serde_json::json!({
                "tool_name": "waku_js_repl__js",
                "tool_input": {"code": "1", "title": "Verify Grok bridge"}
            })))
            .as_deref(),
            Some("Verify Grok bridge")
        );
    }
}

fn format_json_with_limit(value: &Value, character_limit: Option<usize>) -> Option<String> {
    serde_json::to_string_pretty(value)
        .ok()
        .and_then(|text| non_empty_text(text, character_limit))
}

fn format_output(value: &Value, character_limit: Option<usize>) -> Option<String> {
    if let Some(text) = value.as_str() {
        return non_empty_text(text.to_owned(), character_limit);
    }
    if let Some(structured) = value
        .get("structuredContent")
        .filter(|value| !value.is_null())
    {
        return format_json_with_limit(structured, character_limit);
    }
    if let Some(content) = value.get("content").filter(|value| !value.is_null()) {
        return format_output(content, character_limit);
    }
    if let Some(items) = value.as_array() {
        let text = items
            .iter()
            .filter(|item| !is_image_item(item))
            .filter_map(|item| {
                item.as_str().map(str::to_owned).or_else(|| {
                    (item.get("type").and_then(Value::as_str) == Some("text"))
                        .then(|| item.get("text").and_then(Value::as_str).map(str::to_owned))
                        .flatten()
                        .or_else(|| format_json_with_limit(item, character_limit))
                })
            })
            .collect::<Vec<_>>()
            .join("\n\n");
        return non_empty_text(text, character_limit);
    }
    format_json_with_limit(value, character_limit)
}

fn collect_image_urls(value: &Value, urls: &mut Vec<String>) {
    match value {
        Value::Array(items) => {
            for item in items {
                collect_image_urls(item, urls);
            }
        }
        Value::Object(object) => {
            if is_image_item(value) {
                if let Some(url) = object
                    .get("imageUrl")
                    .or_else(|| object.get("image_url"))
                    .or_else(|| object.get("url"))
                    .and_then(Value::as_str)
                {
                    urls.push(url.to_owned());
                } else if let Some(data) = object.get("data").and_then(Value::as_str) {
                    let mime = object
                        .get("mime")
                        .or_else(|| object.get("mimeType"))
                        .or_else(|| object.get("mime_type"))
                        .or_else(|| {
                            object
                                .get("source")
                                .and_then(|source| source.get("media_type"))
                        })
                        .and_then(Value::as_str)
                        .unwrap_or("image/png");
                    urls.push(format!("data:{mime};base64,{data}"));
                } else if let Some(data) = object
                    .get("source")
                    .and_then(|source| source.get("data"))
                    .and_then(Value::as_str)
                {
                    let mime = object
                        .get("source")
                        .and_then(|source| source.get("media_type"))
                        .and_then(Value::as_str)
                        .unwrap_or("image/png");
                    urls.push(format!("data:{mime};base64,{data}"));
                }
            }
            for key in ["content", "attachments", "files", "result"] {
                if let Some(nested) = object.get(key) {
                    collect_image_urls(nested, urls);
                }
            }
        }
        _ => {}
    }
}

fn is_image_item(value: &Value) -> bool {
    let item_type = value.get("type").and_then(Value::as_str);
    let mime = value
        .get("mime")
        .or_else(|| value.get("mimeType"))
        .or_else(|| value.get("mime_type"))
        .or_else(|| value.pointer("/source/media_type"))
        .and_then(Value::as_str);
    matches!(item_type, Some("image" | "inputImage"))
        || (item_type == Some("file") && mime.is_some_and(|mime| mime.starts_with("image/")))
}

fn non_empty_text(value: String, character_limit: Option<usize>) -> Option<String> {
    if character_limit.is_none() {
        return (!value.is_empty()).then_some(value);
    }
    let value = value.trim().to_owned();
    if value.is_empty() {
        return None;
    }
    let Some(character_limit) = character_limit else {
        return Some(value);
    };
    if value.chars().count() <= character_limit {
        return Some(value);
    }
    let mut truncated = value.chars().take(character_limit).collect::<String>();
    truncated.push_str(&tr!("activity.output_truncated"));
    Some(truncated)
}
