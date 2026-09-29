//! Skill attribution grading, shared by the load-time enforcement gate
//! ([`crate::skills::client`]) and the UI status surface (the ACP sources
//! route). Grading runs on an Interceptor Server (SEP-2624): a separate MCP
//! server, configured as an ordinary extension, that declares
//! `io.modelcontextprotocol/interceptors`. goose sends it two
//! `interceptor/invoke` requests per skill, `seal` with the entry as a
//! `skills/get` result and `attribution` with the SKILL.md as a
//! `resources/read` result, and reads the verdicts. Without such an extension
//! skills load ungraded. Grades are memoized per skill URI while the entry's
//! digests and credential are unchanged. The load path appends one activation
//! line per grade to the file named by `GOOSE_SKILLS_ACTIVATION_LOG`.

use std::collections::HashMap;
use std::io::Write;
use std::sync::{Arc, Mutex, OnceLock};

use serde_json::{json, Value};
use tokio_util::sync::CancellationToken;
use tracing::{debug, warn};

use super::mcp_client::{sha256_digest, McpSkillEntry};
use crate::agents::extension_manager::ExtensionManager;
use crate::agents::mcp_client::McpClientTrait;

/// Config key gating load-time enforcement. Off by default: skills load as
/// before and are only audited; when set true, a skill the validators reject
/// is withheld from the model.
pub const ATTRIBUTION_REQUIRE_KEY: &str = "GOOSE_SKILLS_ATTRIBUTION_REQUIRE";

/// Config key naming the activation log file. Unset: no file is written.
pub const ACTIVATION_LOG_KEY: &str = "GOOSE_SKILLS_ACTIVATION_LOG";

/// Interceptor names as the Interceptor Server lists them.
pub const SEAL_INTERCEPTOR: &str = "seal";
pub const ATTRIBUTION_INTERCEPTOR: &str = "attribution";

/// Grade and seal state when no Interceptor Server is configured.
pub const UNCHECKED: &str = "unchecked";
/// Grade and seal state when the Interceptor Server could not be asked.
pub const ERROR: &str = "error";

/// Whether load-time enforcement (withholding) is enabled.
pub fn enforcement_enabled() -> bool {
    crate::config::Config::global()
        .get_param::<bool>(ATTRIBUTION_REQUIRE_KEY)
        .unwrap_or(false)
}

/// `active` when enforcement withholds, `audit` when it only records.
pub fn mode() -> &'static str {
    if enforcement_enabled() {
        "active"
    } else {
        "audit"
    }
}

/// A skill's attribution status, as surfaced to the gate and the UI.
#[derive(Debug, Clone)]
pub struct SkillAttribution {
    /// `compliant_with_upstream_attribution` | `compliant` | `partial` |
    /// `non-compliant`, or [`UNCHECKED`] / [`ERROR`].
    pub compliance: String,
    /// One-line credit summary, e.g. `by Ola Hungerford · CC-BY-4.0 · 2 sources`.
    /// Empty when nothing is declared.
    pub summary: String,
    /// The validator's messages joined into one line (what is missing), or the
    /// invocation error.
    pub detail: String,
    /// `verified` | `mismatch` | `absent`, or [`UNCHECKED`] / [`ERROR`].
    pub seal: String,
    /// One-line seal summary: the signer and any non-passing status codes,
    /// or the mismatch reason. Empty when no credential is carried.
    pub seal_summary: String,
    /// The attribution validator's resolved credit tuple (`author`,
    /// `license`, `sources`, ...), kept for the activation log.
    pub credit: Value,
    /// The seal validator's record for this entry (`state`, `signer`,
    /// `failures`, `resources`, `credit`), verbatim from its `info`.
    pub seal_info: Value,
    /// The attribution validator's `info`, verbatim.
    pub attribution_info: Value,
    /// Extension name of the Interceptor Server that graded this skill.
    pub checked_by: Option<String>,
    /// Set when an Interceptor Server is configured but an invocation failed.
    pub error: Option<String>,
    /// Identity of the entry data this grade was computed from, so a cached
    /// grade is reused only while the digests and credential are unchanged.
    fingerprint: String,
}

impl SkillAttribution {
    pub fn is_non_compliant(&self) -> bool {
        self.compliance == "non-compliant"
    }

    pub fn is_seal_mismatch(&self) -> bool {
        self.seal == "mismatch"
    }

    /// Whether active mode withholds this skill: no credit, a broken seal, or
    /// a configured Interceptor Server that could not grade it. The
    /// interceptors declare `failOpen: false`.
    pub fn blocks(&self) -> bool {
        self.is_non_compliant() || self.is_seal_mismatch() || self.error.is_some()
    }

    fn unchecked(fingerprint: String) -> Self {
        SkillAttribution {
            compliance: UNCHECKED.to_string(),
            summary: String::new(),
            detail: "no Interceptor Server extension configured".to_string(),
            seal: UNCHECKED.to_string(),
            seal_summary: String::new(),
            credit: Value::Null,
            seal_info: Value::Null,
            attribution_info: Value::Null,
            checked_by: None,
            error: None,
            fingerprint,
        }
    }

    fn failed(server: String, error: String, fingerprint: String) -> Self {
        SkillAttribution {
            compliance: ERROR.to_string(),
            summary: String::new(),
            detail: error.clone(),
            seal: ERROR.to_string(),
            seal_summary: String::new(),
            credit: Value::Null,
            seal_info: Value::Null,
            attribution_info: Value::Null,
            checked_by: Some(server),
            error: Some(error),
            fingerprint,
        }
    }
}

/// One validation result as returned by `interceptor/invoke`.
struct Verdict {
    valid: bool,
    messages: String,
    info: Value,
}

fn parse_verdict(raw: Value) -> Result<Verdict, String> {
    let result_type = raw.get("type").and_then(|t| t.as_str());
    if result_type != Some("validation") {
        return Err(format!(
            "unexpected interceptor result type {result_type:?}"
        ));
    }
    let valid = raw
        .get("valid")
        .and_then(|v| v.as_bool())
        .ok_or_else(|| "interceptor result has no `valid`".to_string())?;
    let messages = raw
        .get("messages")
        .and_then(|m| m.as_array())
        .map(|ms| {
            ms.iter()
                .filter_map(|m| m.get("message").and_then(|s| s.as_str()))
                .collect::<Vec<_>>()
                .join(" ")
        })
        .unwrap_or_default();
    let info = raw.get("info").cloned().unwrap_or(Value::Null);
    Ok(Verdict {
        valid,
        messages,
        info,
    })
}

async fn invoke(
    client: &dyn McpClientTrait,
    session_id: &str,
    name: &str,
    event: &str,
    payload: Value,
    cancel: CancellationToken,
) -> Result<Verdict, String> {
    let params = json!({
        "name": name,
        "event": event,
        "phase": "response",
        "payload": payload,
        "context": {
            "principal": { "type": "user", "id": session_id },
            "traceId": session_id,
            "sessionId": session_id,
            "timestamp": chrono::Utc::now().to_rfc3339(),
        },
    });
    let raw = client
        .interceptor_invoke(session_id, params, cancel)
        .await
        .map_err(|e| format!("{name}: {e}"))?;
    parse_verdict(raw).map_err(|e| format!("{name}: {e}"))
}

/// Extensions already probed with `interceptors/list`, by extension name, and
/// whether they answered. goose negotiates 2025-11-25 with every extension,
/// and on that revision an Interceptor Server built on the Python mcp v2 SDK
/// cannot carry `capabilities.extensions`, so an absent capability means
/// unknown, not unsupported. Each extension is probed once per process.
fn probed() -> &'static Mutex<HashMap<String, bool>> {
    static PROBED: OnceLock<Mutex<HashMap<String, bool>>> = OnceLock::new();
    PROBED.get_or_init(|| Mutex::new(HashMap::new()))
}

/// The Interceptor Server among the connected extensions: the first that
/// declares `io.modelcontextprotocol/interceptors`, else the first that
/// answers `interceptors/list`.
async fn find_interceptor_server(
    mgr: &ExtensionManager,
    session_id: &str,
    cancel: CancellationToken,
) -> Option<(String, Arc<dyn McpClientTrait>)> {
    let clients = mgr.extension_clients().await;
    if let Some(found) = clients.iter().find(|(_, client)| {
        client
            .get_info()
            .is_some_and(super::mcp_client::server_declares_interceptors_capability)
    }) {
        return Some(found.clone());
    }
    for (name, client) in clients {
        let known = probed().lock().unwrap().get(&name).copied();
        let answers = match known {
            Some(answers) => answers,
            None => {
                let answers = client
                    .interceptors_list(session_id, None, cancel.clone())
                    .await
                    .is_ok_and(|v| v.get("interceptors").is_some_and(Value::is_array));
                probed().lock().unwrap().insert(name.clone(), answers);
                if answers {
                    debug!(extension = %name, "Interceptor Server found by probing interceptors/list");
                }
                answers
            }
        };
        if answers {
            return Some((name, client));
        }
    }
    None
}

/// Grade a verified SKILL.md by invoking the `seal` and `attribution`
/// validators on the configured Interceptor Server. `text` is the SKILL.md
/// as served; the entry supplies the digests and credential for the seal.
pub async fn grade(
    mgr: &ExtensionManager,
    session_id: &str,
    entry: &McpSkillEntry,
    text: &str,
    cancel: CancellationToken,
) -> SkillAttribution {
    let fingerprint = fingerprint(entry);
    let Some((server, client)) = find_interceptor_server(mgr, session_id, cancel.clone()).await
    else {
        debug!(uri = %entry.uri, "no Interceptor Server extension configured; skill loads ungraded");
        return SkillAttribution::unchecked(fingerprint);
    };

    let seal_payload = json!({ "skill": entry.wire_json() });
    let attribution_payload = json!({ "contents": [{ "uri": entry.uri, "text": text }] });
    let (seal, attribution) = tokio::join!(
        invoke(
            client.as_ref(),
            session_id,
            SEAL_INTERCEPTOR,
            "skills/get",
            seal_payload,
            cancel.clone(),
        ),
        invoke(
            client.as_ref(),
            session_id,
            ATTRIBUTION_INTERCEPTOR,
            "resources/read",
            attribution_payload,
            cancel,
        ),
    );

    match (seal, attribution) {
        (Ok(seal), Ok(attribution)) => from_verdicts(server, seal, attribution, fingerprint),
        (seal, attribution) => {
            let error = [seal.err(), attribution.err()]
                .into_iter()
                .flatten()
                .collect::<Vec<_>>()
                .join("; ");
            warn!(uri = %entry.uri, server, error, "interceptor invocation failed");
            SkillAttribution::failed(server, error, fingerprint)
        }
    }
}

fn from_verdicts(
    server: String,
    seal: Verdict,
    attribution: Verdict,
    fingerprint: String,
) -> SkillAttribution {
    let compliance = attribution
        .info
        .get("complianceLevel")
        .and_then(|c| c.as_str())
        .unwrap_or("non-compliant")
        .to_string();
    let credit = attribution
        .info
        .get("attribution")
        .cloned()
        .unwrap_or(Value::Null);
    let summary = summarize(&credit);

    let seal_info = seal
        .info
        .get("entries")
        .and_then(|e| e.as_array())
        .and_then(|a| a.first())
        .cloned()
        .unwrap_or(Value::Null);
    let sealed = seal_info
        .get("sealed")
        .and_then(|s| s.as_bool())
        .unwrap_or(false);
    let ok = seal_info.get("ok").and_then(|o| o.as_bool()) == Some(true);
    let seal_state = if !sealed {
        "absent"
    } else if ok && seal.valid {
        "verified"
    } else {
        "mismatch"
    };
    let seal_summary = match seal_state {
        "verified" => summarize_seal(&seal_info),
        "mismatch" => format!("seal mismatch: {}", seal.messages),
        _ => String::new(),
    };

    SkillAttribution {
        compliance,
        summary,
        detail: attribution.messages,
        seal: seal_state.to_string(),
        seal_summary,
        credit,
        seal_info,
        attribution_info: attribution.info,
        checked_by: Some(server),
        error: None,
        fingerprint,
    }
}

/// Grades an MCP skill from its entry's verbatim frontmatter, so the status
/// surface needs no resource read. SEP verification holds the served SKILL.md
/// to the same frontmatter at load time.
pub async fn grade_entry(
    mgr: &ExtensionManager,
    entry: &McpSkillEntry,
    session_id: &str,
) -> SkillAttribution {
    if let Some(hit) = cached(&entry.uri).filter(|a| a.fingerprint == fingerprint(entry)) {
        return hit;
    }
    let frontmatter = serde_yaml::to_string(&entry.frontmatter).unwrap_or_default();
    let text = format!(
        "---
{frontmatter}---
"
    );
    let attribution = grade(mgr, session_id, entry, &text, CancellationToken::new()).await;
    cache_put(&entry.uri, attribution.clone());
    attribution
}

fn fingerprint(entry: &McpSkillEntry) -> String {
    let digests = entry
        .manifest()
        .map(|m| {
            m.iter()
                .map(|r| format!("{} {}", r.uri, r.digest))
                .collect::<Vec<_>>()
                .join("\n")
        })
        .unwrap_or_default();
    let credential = entry.credential().unwrap_or_default();
    format!(
        "{}|{}",
        sha256_digest(digests.as_bytes()),
        sha256_digest(credential.as_bytes())
    )
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

fn summarize_seal(info: &Value) -> String {
    let signer = info
        .get("signer")
        .and_then(|s| s.as_str())
        .unwrap_or("unknown signer");
    let failures: Vec<&str> = info
        .get("failures")
        .and_then(|c| c.as_array())
        .map(|a| a.iter().filter_map(|c| c.as_str()).collect())
        .unwrap_or_default();
    if failures.is_empty() {
        format!("sealed by {signer}")
    } else {
        format!("sealed by {signer} ({})", failures.join(", "))
    }
}

/// Appends one JSON line for a skill activation to the file named by
/// `GOOSE_SKILLS_ACTIVATION_LOG`. `outcome` is `loaded` or `withheld`.
pub fn log_activation(entry: &McpSkillEntry, attribution: &SkillAttribution, outcome: &str) {
    let path = crate::config::Config::global()
        .get_param::<String>(ACTIVATION_LOG_KEY)
        .unwrap_or_default();
    if path.is_empty() {
        return;
    }
    log_activation_to(&path, entry, attribution, outcome);
}

fn log_activation_to(
    path: &str,
    entry: &McpSkillEntry,
    attribution: &SkillAttribution,
    outcome: &str,
) {
    let digests: Option<Vec<Value>> = entry.manifest().map(|m| {
        m.iter()
            .map(|r| json!({ "uri": r.uri, "digest": r.digest }))
            .collect()
    });
    let line = json!({
        "ts": chrono::Utc::now().to_rfc3339(),
        "mode": mode(),
        "outcome": outcome,
        "server": entry.server,
        "skill": entry.name,
        "uri": entry.uri,
        "digests": digests,
        "author": attribution.credit.get("author"),
        "sources": attribution.credit.get("sources"),
        "license": attribution.credit.get("license"),
        "compliance": attribution.compliance,
        "seal": attribution.seal,
        "signer": attribution.seal_info.get("signer"),
        "sealCodes": attribution.seal_info.get("failures"),
        "checkedBy": attribution.checked_by,
        "error": attribution.error,
        "interceptor": {
            "seal": attribution.seal_info,
            "attribution": attribution.attribution_info,
        },
    });
    let written = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(path)
        .and_then(|mut f| writeln!(f, "{line}"));
    if let Err(e) = written {
        warn!(path, error = %e, "could not append to the skills activation log");
    }
}

/// Process-wide memo keyed by skill URI. The status surface may be polled; a
/// hit returns the grade without another round trip. A hit is only reused
/// while the entry's digests and credential match the grade's fingerprint.
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

/// A stand-in Interceptor Server for tests: answers `interceptor/invoke` the
/// way `attribution-interceptors` does, from switches instead of C2PA.
#[cfg(test)]
pub(crate) mod test_support {
    use std::sync::atomic::{AtomicBool, Ordering};
    use std::sync::Arc;

    use async_trait::async_trait;
    use rmcp::model::{
        CallToolResult, ExtensionCapabilities, InitializeResult, JsonObject, ListToolsResult,
        ServerCapabilities,
    };
    use serde_json::{json, Value};
    use tokio_util::sync::CancellationToken;

    use crate::agents::extension::ExtensionConfig;
    use crate::agents::extension_manager::ExtensionManager;
    use crate::agents::mcp_client::{Error, McpClientTrait};
    use crate::agents::ToolCallContext;
    use crate::skills::mcp_client::INTERCEPTORS_EXTENSION_ID;

    pub(crate) struct FakeInterceptor {
        info: InitializeResult,
        /// Whether a sealed entry's credential verifies.
        pub seal_ok: AtomicBool,
        /// Whether every invocation fails at the transport.
        pub fail: AtomicBool,
    }

    impl FakeInterceptor {
        pub fn new() -> Self {
            let mut caps = ExtensionCapabilities::new();
            caps.insert(INTERCEPTORS_EXTENSION_ID.to_string(), JsonObject::new());
            Self::with_capabilities(
                ServerCapabilities::builder()
                    .enable_extensions_with(caps)
                    .build(),
            )
        }

        /// A server that answers `interceptors/list` but, like a Python mcp v2
        /// server on the 2025-11-25 handshake, advertises no `extensions`.
        pub fn undeclared() -> Self {
            Self::with_capabilities(ServerCapabilities::builder().build())
        }

        fn with_capabilities(capabilities: ServerCapabilities) -> Self {
            FakeInterceptor {
                info: InitializeResult::new(capabilities),
                seal_ok: AtomicBool::new(true),
                fail: AtomicBool::new(false),
            }
        }

        fn seal(&self, payload: &Value) -> Value {
            let entry = &payload["skill"];
            let uri = entry.get("uri").cloned().unwrap_or(Value::Null);
            let sealed = entry
                .get("_meta")
                .and_then(|m| m.get("org.c2pa/credential"))
                .is_some();
            let ok = self.seal_ok.load(Ordering::SeqCst);
            let record = if !sealed {
                json!({ "uri": uri, "sealed": false, "ok": null })
            } else {
                json!({
                    "uri": uri,
                    "sealed": true,
                    "ok": ok,
                    "state": if ok { "Valid" } else { "Invalid" },
                    "signer": "C2PA Test Signing Cert",
                    "failures": if ok {
                        json!(["signingCredential.untrusted"])
                    } else {
                        json!(["signingCredential.untrusted", "assertion.dataHash.mismatch"])
                    },
                    "resources": entry.get("resources"),
                })
            };
            let valid = !sealed || ok;
            json!({
                "type": "validation",
                "interceptor": "seal",
                "phase": "response",
                "valid": valid,
                "severity": if valid { Value::Null } else { json!("error") },
                "messages": if valid { json!([]) } else {
                    json!([{ "message": format!("seal does not match the resources of {uri}"), "severity": "error" }])
                },
                "info": { "event": "skills/get", "entries": [record] },
            })
        }

        fn attribution(&self, payload: &Value) -> Value {
            let text = payload["contents"][0]["text"].as_str().unwrap_or_default();
            let field = |key: &str| {
                text.lines()
                    .find_map(|l| l.trim().strip_prefix(&format!("{key}: ")))
                    .map(|v| v.trim_matches('"').to_string())
            };
            let author = field("skill_author");
            let license = field("license");
            let sources: Vec<String> = text
                .lines()
                .filter_map(|l| l.trim().strip_prefix("- "))
                .map(|s| s.to_string())
                .collect();
            let compliance = match (&author, &license, sources.is_empty()) {
                (Some(_), Some(_), false) => "compliant_with_upstream_attribution",
                (Some(_), Some(_), true) => "compliant",
                (None, None, _) => "non-compliant",
                _ => "partial",
            };
            let mut credit = serde_json::Map::new();
            if let Some(a) = author {
                credit.insert("author".into(), json!(a));
            }
            if let Some(l) = license {
                credit.insert("license".into(), json!(l));
            }
            if !sources.is_empty() {
                credit.insert("sources".into(), json!(sources));
            }
            let valid = compliance != "non-compliant";
            json!({
                "type": "validation",
                "interceptor": "attribution",
                "phase": "response",
                "valid": valid,
                "severity": if valid { json!("info") } else { json!("error") },
                "messages": if valid { json!([]) } else {
                    json!([{ "message": "Skill declares neither skill_author nor license; attribution grade is non-compliant.", "severity": "error" }])
                },
                "info": {
                    "skill": { "uri": payload["contents"][0]["uri"] },
                    "attribution": credit,
                    "complianceLevel": compliance,
                },
            })
        }
    }

    #[async_trait]
    impl McpClientTrait for FakeInterceptor {
        async fn list_tools(
            &self,
            _session_id: &str,
            _next_cursor: Option<String>,
            _cancel_token: CancellationToken,
        ) -> Result<ListToolsResult, Error> {
            Ok(ListToolsResult::default())
        }

        async fn call_tool(
            &self,
            _ctx: &ToolCallContext,
            _name: &str,
            _arguments: Option<JsonObject>,
            _cancel_token: CancellationToken,
        ) -> Result<CallToolResult, Error> {
            unreachable!("an Interceptor Server has no tools")
        }

        fn get_info(&self) -> Option<&InitializeResult> {
            Some(&self.info)
        }

        async fn interceptors_list(
            &self,
            _session_id: &str,
            _event: Option<String>,
            _cancel_token: CancellationToken,
        ) -> Result<Value, Error> {
            Ok(json!({ "interceptors": [
                { "name": "seal", "type": "validation", "hooks": [{ "events": ["skills/list", "skills/get"], "phase": "response" }] },
                { "name": "attribution", "type": "validation", "hooks": [{ "events": ["resources/read"], "phase": "response" }] },
            ]}))
        }

        async fn interceptor_invoke(
            &self,
            _session_id: &str,
            params: Value,
            _cancel_token: CancellationToken,
        ) -> Result<Value, Error> {
            if self.fail.load(Ordering::SeqCst) {
                return Err(Error::TransportClosed);
            }
            let payload = &params["payload"];
            match params["name"].as_str() {
                Some("seal") => Ok(self.seal(payload)),
                Some("attribution") => Ok(self.attribution(payload)),
                other => Ok(
                    json!({ "type": "validation", "valid": false, "messages": [{ "message": format!("unknown interceptor {other:?}"), "severity": "error" }] }),
                ),
            }
        }
    }

    /// An extension manager with `fake` registered as the extension
    /// `attribution_interceptors`.
    pub(crate) async fn manager_with(
        fake: Arc<FakeInterceptor>,
    ) -> (Arc<ExtensionManager>, tempfile::TempDir) {
        manager_with_named(fake, "attribution_interceptors").await
    }

    pub(crate) async fn manager_with_named(
        fake: Arc<FakeInterceptor>,
        name: &str,
    ) -> (Arc<ExtensionManager>, tempfile::TempDir) {
        let tmp = tempfile::tempdir().unwrap();
        let mgr = Arc::new(ExtensionManager::new_without_provider(
            tmp.path().to_path_buf(),
        ));
        mgr.add_client(
            name.to_string(),
            ExtensionConfig::Builtin {
                name: name.to_string(),
                display_name: Some(name.to_string()),
                description: "fake Interceptor Server".to_string(),
                timeout: None,
                bundled: None,
                available_tools: vec![],
            },
            fake,
            None,
            None,
            Some("s"),
        )
        .await;
        (mgr, tmp)
    }
}

#[cfg(test)]
mod tests {
    use std::sync::atomic::Ordering;
    use std::sync::Arc;

    use super::test_support::{manager_with, manager_with_named, FakeInterceptor};
    use super::*;
    use crate::skills::mcp_client::{SkillResourceRef, SkillResources};

    const COMPLIANT: &str = "---\nname: demo\nlicense: CC-BY-4.0\nmetadata:\n  skill_author: Vault-Tec\n  sources:\n    - https://example.com\n  attribution: \"Derived from the example SRD.\"\n---\n# Demo\nbody\n";
    const UNCREDITED: &str =
        "---\nname: wasteland\ndescription: encounters\n---\n# Wasteland\nbody\n";

    fn entry(uri: &str, sealed: bool) -> McpSkillEntry {
        McpSkillEntry {
            server: "spaceship-server".to_string(),
            name: uri
                .trim_end_matches("/SKILL.md")
                .rsplit('/')
                .next()
                .unwrap()
                .to_string(),
            description: "d".to_string(),
            uri: uri.to_string(),
            frontmatter: serde_json::json!({
                "name": "demo",
                "license": "CC-BY-4.0",
                "metadata": {"skill_author": "Vault-Tec"},
            }),
            resources: SkillResources::Manifest(vec![SkillResourceRef {
                uri: uri.to_string(),
                digest: sha256_digest(COMPLIANT.as_bytes()),
                size: COMPLIANT.len() as u64,
            }]),
            meta: sealed.then(|| {
                serde_json::json!({ "org.c2pa/credential": "data:application/c2pa;base64,AAAA" })
            }),
        }
    }

    #[tokio::test]
    async fn unchecked_without_an_interceptor_server() {
        let tmp = tempfile::tempdir().unwrap();
        let mgr = ExtensionManager::new_without_provider(tmp.path().to_path_buf());
        let e = entry("skill://unchecked/SKILL.md", true);
        let a = grade(&mgr, "s1", &e, UNCREDITED, CancellationToken::new()).await;
        assert_eq!(a.compliance, UNCHECKED);
        assert_eq!(a.seal, UNCHECKED);
        assert!(!a.blocks());
        assert!(a.checked_by.is_none());
    }

    #[tokio::test]
    async fn found_by_probe_when_capability_is_undeclared() {
        let fake = Arc::new(FakeInterceptor::undeclared());
        let (mgr, _tmp) = manager_with_named(fake, "icp_legacy_handshake").await;
        let e = entry("skill://probed/SKILL.md", true);
        let a = grade(&mgr, "s1", &e, COMPLIANT, CancellationToken::new()).await;
        assert_eq!(a.checked_by.as_deref(), Some("icp_legacy_handshake"));
        assert_eq!(a.seal, "verified");
        assert_eq!(a.compliance, "compliant_with_upstream_attribution");
        assert_eq!(
            probed().lock().unwrap().get("icp_legacy_handshake"),
            Some(&true)
        );
    }

    #[tokio::test]
    async fn grades_through_the_interceptor_server() {
        let fake = Arc::new(FakeInterceptor::new());
        let (mgr, _tmp) = manager_with(fake.clone()).await;

        let unsealed = entry("skill://unsealed/SKILL.md", false);
        let a = grade(&mgr, "s1", &unsealed, COMPLIANT, CancellationToken::new()).await;
        assert_eq!(a.compliance, "compliant_with_upstream_attribution");
        assert!(a.summary.contains("by Vault-Tec"), "summary: {}", a.summary);
        assert!(a.summary.contains("CC-BY-4.0"), "summary: {}", a.summary);
        assert_eq!(a.seal, "absent");
        assert_eq!(a.checked_by.as_deref(), Some("attribution_interceptors"));
        assert!(!a.blocks());

        let a = grade(&mgr, "s1", &unsealed, UNCREDITED, CancellationToken::new()).await;
        assert_eq!(a.compliance, "non-compliant");
        assert!(a.is_non_compliant() && a.blocks());
        assert!(a.summary.is_empty(), "summary: {}", a.summary);

        let sealed = entry("skill://build-a-rocket/SKILL.md", true);
        let a = grade(&mgr, "s1", &sealed, COMPLIANT, CancellationToken::new()).await;
        assert_eq!(a.seal, "verified", "{}", a.seal_summary);
        assert!(
            a.seal_summary.contains("C2PA Test Signing Cert"),
            "{}",
            a.seal_summary
        );
        assert!(!a.blocks());

        fake.seal_ok.store(false, Ordering::SeqCst);
        let a = grade(&mgr, "s1", &sealed, COMPLIANT, CancellationToken::new()).await;
        assert_eq!(a.seal, "mismatch");
        assert!(a.is_seal_mismatch() && a.blocks());
        assert!(
            a.seal_summary.starts_with("seal mismatch:"),
            "{}",
            a.seal_summary
        );
        assert_eq!(a.compliance, "compliant_with_upstream_attribution");
    }

    #[tokio::test]
    async fn invocation_failure_is_an_error_grade_that_blocks() {
        let fake = Arc::new(FakeInterceptor::new());
        fake.fail.store(true, Ordering::SeqCst);
        let (mgr, _tmp) = manager_with(fake).await;
        let e = entry("skill://down/SKILL.md", true);
        let a = grade(&mgr, "s1", &e, COMPLIANT, CancellationToken::new()).await;
        assert_eq!(a.compliance, ERROR);
        assert_eq!(a.seal, ERROR);
        assert!(a.error.is_some());
        assert!(a.blocks());
    }

    #[tokio::test]
    async fn grade_entry_caches_until_the_entry_changes() {
        let fake = Arc::new(FakeInterceptor::new());
        let (mgr, _tmp) = manager_with(fake.clone()).await;
        let mut e = entry("skill://cached/SKILL.md", true);
        let first = grade_entry(&mgr, &e, "s1").await;
        assert_eq!(first.seal, "verified");
        assert!(
            first.summary.contains("by Vault-Tec"),
            "summary: {}",
            first.summary
        );

        fake.seal_ok.store(false, Ordering::SeqCst);
        let hit = grade_entry(&mgr, &e, "s1").await;
        assert_eq!(
            hit.seal, "verified",
            "same digests and credential reuse the grade"
        );

        match &mut e.resources {
            SkillResources::Manifest(refs) => refs[0].digest = sha256_digest(b"changed"),
            _ => unreachable!(),
        }
        let second = grade_entry(&mgr, &e, "s1").await;
        assert_ne!(first.fingerprint, second.fingerprint);
        assert_eq!(second.seal, "mismatch");
        assert_eq!(cached(&e.uri).unwrap().fingerprint, second.fingerprint);
    }

    #[tokio::test]
    async fn activation_log_writes_one_json_line() {
        let fake = Arc::new(FakeInterceptor::new());
        let (mgr, _tmp) = manager_with(fake).await;
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("activations.jsonl");
        let e = entry("skill://build-a-rocket/SKILL.md", true);
        let a = grade(&mgr, "s1", &e, COMPLIANT, CancellationToken::new()).await;
        log_activation_to(path.to_str().unwrap(), &e, &a, "loaded");
        let text = std::fs::read_to_string(&path).unwrap();
        let lines: Vec<&str> = text.lines().collect();
        assert_eq!(lines.len(), 1, "{text}");
        let line: Value = serde_json::from_str(lines[0]).unwrap();
        assert_eq!(line["server"], "spaceship-server");
        assert_eq!(line["outcome"], "loaded");
        assert_eq!(line["seal"], "verified");
        assert_eq!(line["license"], "CC-BY-4.0");
        assert_eq!(line["author"], "Vault-Tec");
        assert_eq!(line["checkedBy"], "attribution_interceptors");
        assert_eq!(line["sealCodes"][0], "signingCredential.untrusted");
        assert_eq!(line["interceptor"]["seal"]["state"], "Valid");
        assert_eq!(line["digests"].as_array().unwrap().len(), 1);
        assert!(line["mode"] == "audit" || line["mode"] == "active");
    }
}
