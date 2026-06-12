//! Skill attribution grading, shared by the load-time enforcement gate
//! ([`crate::skills::client`]) and the UI status surface (the slash-commands
//! route). A SKILL.md body is run through the `mcp-ext-interceptors`
//! attribution chain; this module extracts the compliance grade plus a
//! human-readable credit summary, and memoizes the result per skill URI so the
//! status surface need not re-read and re-grade on every poll.

use std::collections::HashMap;
use std::sync::{Arc, Mutex, OnceLock};

use mcp_ext_interceptors::{
    attribution::attribution_validator,
    chain::Chain,
    events::RESOURCES_READ,
    invocation::{InvocationContext, Principal, SystemClock},
};
use serde_json::Value;

/// Config key gating load-time enforcement. Off by default: skills load as
/// before and are only audited; when set true, a `non-compliant` skill is
/// withheld from the model.
pub const ATTRIBUTION_REQUIRE_KEY: &str = "GOOSE_SKILLS_ATTRIBUTION_REQUIRE";

/// Whether load-time enforcement (withholding) is enabled.
pub fn enforcement_enabled() -> bool {
    crate::config::Config::global()
        .get_param::<bool>(ATTRIBUTION_REQUIRE_KEY)
        .unwrap_or(false)
}

/// A skill's attribution status, as surfaced to the gate and the UI.
#[derive(Debug, Clone)]
pub struct SkillAttribution {
    /// `compliant_with_upstream_attribution` | `compliant` | `partial` | `non-compliant`.
    pub compliance: String,
    /// One-line credit summary, e.g. `by Ola Hungerford · CC-BY-4.0 · 2 sources`.
    /// Empty when nothing is declared.
    pub summary: String,
    /// The validator's per-field notes joined into one line (what is missing).
    pub detail: String,
}

impl SkillAttribution {
    pub fn is_non_compliant(&self) -> bool {
        self.compliance == "non-compliant"
    }
}

/// Grade a SKILL.md body through the attribution chain. `principal_id` stamps
/// the audit event (the requesting session). Emits the chain's
/// `[skill-attribution]` audit event as a side effect.
pub async fn grade(uri: &str, body: &str, principal_id: &str) -> SkillAttribution {
    let chain = Chain::new().with(attribution_validator(Arc::new(SystemClock)));
    let ctx = InvocationContext {
        principal: Some(Principal {
            principal_type: "user".to_string(),
            id: Some(principal_id.to_string()),
            claims: None,
        }),
        trace_id: Some(principal_id.to_string()),
        ..Default::default()
    };
    let payload = serde_json::json!({ "contents": [{ "uri": uri, "text": body }] });
    let outcome = chain
        .execute_response(RESOURCES_READ, payload, Some(ctx))
        .await;

    let record = outcome.results.first();
    let info = record.and_then(|r| r.info.as_ref());
    let compliance = info
        .and_then(|i| i.get("complianceLevel"))
        .and_then(|c| c.as_str())
        .unwrap_or("non-compliant")
        .to_string();
    let summary = info
        .and_then(|i| i.get("attribution"))
        .map(summarize)
        .unwrap_or_default();
    let detail = record
        .and_then(|r| r.validation.as_ref())
        .map(|v| {
            v.messages
                .iter()
                .map(|m| m.message.as_str())
                .collect::<Vec<_>>()
                .join(" ")
        })
        .unwrap_or_default();

    SkillAttribution {
        compliance,
        summary,
        detail,
    }
}

/// Build a one-line credit summary from the validator's `attribution` tuple.
fn summarize(attribution: &Value) -> String {
    let mut parts: Vec<String> = Vec::new();
    if let Some(name) = attribution
        .get("author")
        .and_then(|a| a.get("name").or(Some(a)))
        .and_then(|n| n.as_str())
    {
        parts.push(format!("by {name}"));
    }
    if let Some(license) = attribution.get("license").and_then(|l| l.as_str()) {
        parts.push(license.to_string());
    }
    if let Some(n) = attribution
        .get("sources")
        .and_then(|s| s.as_array())
        .map(|a| a.len())
        .filter(|n| *n > 0)
    {
        parts.push(format!("{n} source{}", if n == 1 { "" } else { "s" }));
    }
    parts.join(" · ")
}

/// Process-wide memo keyed by skill URI. The status surface may be polled; a
/// hit returns the grade without re-reading the body. Content is assumed stable
/// per URI for a session — a server that swaps a skill fires
/// `resources/list_changed`, and the next fresh read regrades.
fn cache() -> &'static Mutex<HashMap<String, SkillAttribution>> {
    static CACHE: OnceLock<Mutex<HashMap<String, SkillAttribution>>> = OnceLock::new();
    CACHE.get_or_init(|| Mutex::new(HashMap::new()))
}

/// Returns the cached grade for `uri`, if one has been computed this session.
pub fn cached(uri: &str) -> Option<SkillAttribution> {
    cache().lock().unwrap().get(uri).cloned()
}

/// Records a grade for `uri` so later status reads reuse it.
pub fn cache_put(uri: &str, attribution: SkillAttribution) {
    cache().lock().unwrap().insert(uri.to_string(), attribution);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn grades_compliant_and_summarizes() {
        let body = "---\nname: demo\nlicense: CC-BY-4.0\nmetadata:\n  skill_author: Vault-Tec\n  sources:\n    - https://example.com\n  attribution: \"Derived from the example SRD.\"\n---\n# Demo\nbody\n";
        let a = grade("skill://x/SKILL.md", body, "s1").await;
        assert_eq!(a.compliance, "compliant_with_upstream_attribution");
        assert!(a.summary.contains("by Vault-Tec"), "summary: {}", a.summary);
        assert!(a.summary.contains("CC-BY-4.0"), "summary: {}", a.summary);
        assert!(!a.is_non_compliant());
    }

    #[tokio::test]
    async fn grades_valid_but_uncredited_as_non_compliant() {
        let body = "---\nname: wasteland\ndescription: encounters\n---\n# Wasteland\nbody\n";
        let a = grade("skill://x/SKILL.md", body, "s1").await;
        assert_eq!(a.compliance, "non-compliant");
        assert!(a.is_non_compliant());
        assert!(a.summary.is_empty(), "summary: {}", a.summary);
    }

    #[tokio::test]
    async fn grades_author_only_as_partial() {
        let body = "---\nname: noted\ndescription: notes\nmetadata:\n  skill_author: J. Doe\n---\n# Noted\nbody\n";
        let a = grade("skill://x/SKILL.md", body, "s1").await;
        assert_eq!(a.compliance, "partial");
        assert!(a.summary.contains("by J. Doe"), "summary: {}", a.summary);
    }
}
