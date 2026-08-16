//! MCP-served Agent Skills discovery, per SEP `io.modelcontextprotocol/skills`.
//!
//! Bridges skills served over MCP (via `skill://` or any scheme) into Goose's
//! existing skills pipeline. This module is the discovery layer: it enumerates
//! a server's skills via the extension's `skills/list` method, retrieves single
//! entries via `skills/get`, and returns [`McpSkillEntry`] values that the
//! skills platform extension caches and surfaces in the system prompt.
//!
//! Scheme-agnostic: the SEP permits servers to list skills under a
//! domain-native URI scheme (e.g. `github://owner/repo/.../SKILL.md`) so long
//! as the URI's final skill-path segment equals the frontmatter `name` and the
//! URI ends in `/SKILL.md`.
//!
//! Security: per the SEP, skill content from MCP servers is UNTRUSTED model
//! input. This module extracts only the skill's frontmatter and URI/digest
//! locators from entries — never execution-capable fields. Loaded SKILL.md and
//! supporting-file content is verified against the entry's `resources` digests
//! before use, and the fetched SKILL.md frontmatter must match the entry's
//! verbatim `frontmatter` field-by-field.

use rmcp::model::InitializeResult;
use serde::Deserialize;
use std::time::Duration;
use tokio_util::sync::CancellationToken;
use tracing::{debug, warn};

use crate::agents::mcp_client::McpClientTrait;
use crate::skills::SkillFrontmatter;

/// Extension identifier per the SEP.
pub(crate) const SKILLS_EXTENSION_ID: &str = "io.modelcontextprotocol/skills";

/// How long to wait for a server's full `skills/list` enumeration before
/// giving up. Applied at extension-registration time so a misbehaving server
/// cannot stall session startup indefinitely. An empty cache on timeout is
/// acceptable — a later refresh repopulates.
pub(crate) const LIST_FETCH_TIMEOUT: Duration = Duration::from_secs(5);

/// Upper bound on `skills/list` pages followed per enumeration, so a server
/// emitting endless cursors cannot spin the fetch forever.
const MAX_LIST_PAGES: usize = 64;

/// One `{uri, digest}` pair from a skill entry's `resources` enumeration.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SkillResourceRef {
    pub uri: String,
    /// `sha256:<hex>` digest of the file at `uri`.
    pub digest: String,
}

/// A single skill served over MCP, as surfaced to the skills platform
/// extension.
///
/// `name`/`description` are taken from the entry's verbatim `frontmatter`
/// block (per SEP, entries carry the full `SKILL.md` frontmatter).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct McpSkillEntry {
    pub server: String,
    pub name: String,
    pub description: String,
    /// Resource URI of the skill's `SKILL.md` (any scheme). Always ends in
    /// `/SKILL.md`; enforced at parse time.
    pub uri: String,
    /// Verbatim frontmatter from the entry, kept for the SEP-mandated
    /// field-by-field identity check against the fetched `SKILL.md`.
    pub frontmatter: serde_json::Value,
    /// Complete `{uri, digest}` enumeration of the skill's files, including
    /// `SKILL.md` itself. `None` when the server omitted `resources`
    /// (dynamically generated skill): no integrity verification is possible
    /// and reads within the skill are unrestricted by a listing.
    pub resources: Option<Vec<SkillResourceRef>>,
}

impl McpSkillEntry {
    /// Skill root URI: the entry `uri` with the `/SKILL.md` suffix removed,
    /// no trailing slash (SEP §Resource Mapping). Relative refs inside the
    /// skill resolve as `<root>/<relative-path>`.
    pub fn skill_root_uri(&self) -> &str {
        self.uri.strip_suffix("/SKILL.md").unwrap_or(&self.uri)
    }

    /// Resolve a skill-relative path (`references/GUIDE.md`) against the
    /// skill root.
    pub fn resolve_relative(&self, relative: &str) -> String {
        format!("{}/{}", self.skill_root_uri(), relative)
    }

    /// The digest recorded for `uri` in this entry's `resources`, if the
    /// entry carries one.
    pub fn digest_for(&self, uri: &str) -> Option<&str> {
        self.resources
            .as_ref()?
            .iter()
            .find(|r| r.uri == uri)
            .map(|r| r.digest.as_str())
    }

    /// Pre-read gate, per SEP §Integrity and verification: while acting on
    /// a skill for which the host holds an entry with `resources`, reads
    /// within the skill MUST resolve only to listed URIs. Entries without
    /// `resources` (dynamic skills) impose no restriction.
    pub fn verify_read_uri_listed(&self, uri: &str) -> Result<(), String> {
        let Some(resources) = &self.resources else {
            return Ok(());
        };
        if resources.iter().any(|r| r.uri == uri) {
            Ok(())
        } else {
            Err(format!(
                "'{}' is not listed in the skill's resources; refusing unverifiable read \
                 (the skill may have changed — refresh via skills/get)",
                uri
            ))
        }
    }

    /// Verify content fetched for `uri` against this entry, per SEP
    /// §Integrity and verification:
    /// - entry has `resources` and lists `uri` → digest must match;
    /// - entry has `resources` and does NOT list `uri` → verification
    ///   failure (an unlisted file is a change to the skill);
    /// - entry has no `resources` (dynamic skill) → nothing to verify.
    pub fn verify_read(&self, uri: &str, bytes: &[u8]) -> Result<(), String> {
        let Some(resources) = &self.resources else {
            return Ok(());
        };
        match resources.iter().find(|r| r.uri == uri) {
            Some(r) => verify_digest(&r.digest, bytes),
            None => Err(format!(
                "'{}' is not listed in the skill's resources; refusing unverifiable read \
                 (the skill may have changed — refresh via skills/get)",
                uri
            )),
        }
    }

    /// Field-by-field identity check between this entry's `frontmatter` and
    /// the frontmatter parsed from a fetched `SKILL.md`, per SEP
    /// §Integrity and verification. Any discrepancy is a verification
    /// failure equivalent to a digest mismatch.
    pub fn verify_frontmatter(&self, fetched: &serde_json::Value) -> Result<(), String> {
        if &self.frontmatter == fetched {
            Ok(())
        } else {
            Err(
                "SKILL.md frontmatter does not match the entry's frontmatter; \
                 the skill changed since it was listed"
                    .to_string(),
            )
        }
    }
}

/// All MCP-served skills discovered from a single server's `skills/list`.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct ServerSkills {
    pub skills: Vec<McpSkillEntry>,
    /// SEP-2549 freshness hint for the listing, when the server sent one.
    pub ttl_ms: Option<u64>,
    pub cache_scope: Option<String>,
}

impl ServerSkills {
    pub fn is_empty(&self) -> bool {
        self.skills.is_empty()
    }
}

/// Returns true if the server's initialize response declares the skills
/// extension capability. Per the SEP, `skills/list`/`skills/get` are only
/// issued against servers that declared the extension.
pub fn server_declares_skills_capability(info: &InitializeResult) -> bool {
    info.capabilities
        .extensions
        .as_ref()
        .is_some_and(|m| m.contains_key(SKILLS_EXTENSION_ID))
}

/// Returns true if the server declares the skills extension with
/// `directoryRead: true`. Per the SEP, clients MUST NOT call
/// `resources/directory/read` against a server that has not declared it.
pub fn server_declares_directory_read(info: &InitializeResult) -> bool {
    info.capabilities
        .extensions
        .as_ref()
        .and_then(|m| m.get(SKILLS_EXTENSION_ID))
        .and_then(|cfg| cfg.get("directoryRead"))
        .and_then(|v| v.as_bool())
        .unwrap_or(false)
}

/// `skills/list` result shape per the SEP. Unknown fields are ignored.
#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SkillsListResult {
    #[serde(default)]
    pub skills: Vec<WireSkillEntry>,
    #[serde(default)]
    pub next_cursor: Option<String>,
    #[serde(default)]
    pub ttl_ms: Option<u64>,
    #[serde(default)]
    pub cache_scope: Option<String>,
}

/// `skills/get` result shape per the SEP: one entry under `skill`.
#[derive(Debug, Deserialize)]
pub struct SkillsGetResult {
    pub skill: WireSkillEntry,
}

/// One wire-format skill entry. Fields default so one malformed entry is
/// skipped in `parse_entry` rather than failing the whole listing parse and
/// dropping every skill the server advertises.
#[derive(Debug, Deserialize)]
pub struct WireSkillEntry {
    #[serde(default)]
    pub uri: Option<String>,
    #[serde(default)]
    pub frontmatter: Option<serde_json::Value>,
    #[serde(default)]
    pub resources: Option<Vec<WireResourceRef>>,
}

#[derive(Debug, Deserialize)]
pub struct WireResourceRef {
    #[serde(default)]
    pub uri: String,
    #[serde(default)]
    pub digest: String,
}

/// Verify `bytes` against a `sha256:<hex>` digest from a skill entry. Per the
/// SEP, hosts MUST verify retrieved listed content against the entry digest
/// and MUST NOT use content that fails to match.
pub fn verify_digest(expected: &str, bytes: &[u8]) -> Result<(), String> {
    let Some(hex_expected) = expected.strip_prefix("sha256:") else {
        return Err(format!(
            "unsupported digest format '{}': expected 'sha256:<hex>'",
            expected
        ));
    };
    use sha2::{Digest, Sha256};
    let mut hasher = Sha256::new();
    hasher.update(bytes);
    let actual = crate::utils::bytes_to_hex(hasher.finalize());
    if actual.eq_ignore_ascii_case(hex_expected.trim()) {
        Ok(())
    } else {
        Err(format!(
            "digest mismatch: entry advertised {} but content hashes to sha256:{}",
            expected, actual
        ))
    }
}

/// Enumerates a single server's skills via `skills/list`, following
/// pagination. Returns an empty [`ServerSkills`] (with a log) on any failure —
/// this function MUST NOT propagate errors because it runs during extension
/// registration and must not block the agent from starting.
///
/// Issues no request unless the server declared the skills extension
/// capability (SEP §Capability Declaration).
///
/// Caller supplies the server name (extension key) because it's stamped into
/// each returned entry's `server` field for later routing.
pub async fn fetch_server_skills(
    server: &str,
    client: &dyn McpClientTrait,
    session_id: &str,
    cancel: CancellationToken,
) -> ServerSkills {
    match client.get_info() {
        Some(info) if server_declares_skills_capability(info) => {}
        _ => {
            debug!(
                server,
                "server does not declare the skills extension; skipping skills/list"
            );
            return ServerSkills::default();
        }
    }

    let fetch = async {
        let mut entries = Vec::new();
        let mut ttl_ms = None;
        let mut cache_scope = None;
        let mut cursor: Option<String> = None;
        for page in 0.. {
            if page == MAX_LIST_PAGES {
                warn!(
                    server,
                    pages = MAX_LIST_PAGES,
                    "skills/list pagination did not terminate; truncating enumeration"
                );
                break;
            }
            let result = client
                .skills_list(session_id, cursor.take(), cancel.clone())
                .await?;
            entries.extend(result.skills);
            ttl_ms = ttl_ms.or(result.ttl_ms);
            cache_scope = cache_scope.or(result.cache_scope);
            match result.next_cursor {
                Some(next) => cursor = Some(next),
                None => break,
            }
        }
        Ok::<_, crate::agents::mcp_client::Error>((entries, ttl_ms, cache_scope))
    };

    let (entries, ttl_ms, cache_scope) = match tokio::time::timeout(LIST_FETCH_TIMEOUT, fetch).await
    {
        Ok(Ok(page)) => page,
        Ok(Err(e)) => {
            debug!(server, error = %e, "skills/list fetch failed");
            return ServerSkills::default();
        }
        Err(_) => {
            warn!(
                server,
                timeout_secs = LIST_FETCH_TIMEOUT.as_secs(),
                "skills/list fetch timed out"
            );
            return ServerSkills::default();
        }
    };

    let mut out = ServerSkills {
        ttl_ms,
        cache_scope,
        ..Default::default()
    };
    for raw in entries {
        if let Some(entry) = parse_entry(server, raw) {
            out.skills.push(entry);
        }
    }
    out
}

/// Retrieves a single skill entry by URI via `skills/get` (SEP §Retrieval).
/// Used for URI-driven loading of skills that never appeared in a listing,
/// and to refresh one skill's digests after a verification failure. Errors
/// propagate — the caller is a user- or model-initiated load with somewhere
/// to report failure, unlike registration-time enumeration.
pub async fn fetch_skill_entry(
    server: &str,
    client: &dyn McpClientTrait,
    session_id: &str,
    uri: &str,
    cancel: CancellationToken,
) -> Result<McpSkillEntry, String> {
    match client.get_info() {
        Some(info) if server_declares_skills_capability(info) => {}
        _ => {
            return Err(format!(
                "server '{}' does not declare the skills extension",
                server
            ))
        }
    }

    let result = client
        .skills_get(session_id, uri, cancel)
        .await
        .map_err(|e| format!("skills/get failed for '{}': {}", uri, e))?;

    parse_entry(server, result.skill)
        .ok_or_else(|| format!("skills/get returned an unusable entry for '{}'", uri))
}

fn parse_entry(server: &str, raw: WireSkillEntry) -> Option<McpSkillEntry> {
    // `name`/`description` come from the verbatim frontmatter block.
    let Some(frontmatter) = raw.frontmatter else {
        warn!(server, "skipping skill entry with no `frontmatter`");
        return None;
    };
    let fm: SkillFrontmatter = match serde_json::from_value(frontmatter.clone()) {
        Ok(fm) => fm,
        Err(e) => {
            warn!(server, error = %e, "skipping skill entry with unparseable `frontmatter`");
            return None;
        }
    };
    let Some(name) = fm.name.filter(|s| !s.is_empty()) else {
        warn!(
            server,
            "skipping skill entry whose frontmatter has no `name`"
        );
        return None;
    };

    let Some(uri) = raw.uri.filter(|s| !s.is_empty()) else {
        warn!(server, name, "skipping skill entry with no `uri`");
        return None;
    };
    let Some(root) = uri.strip_suffix("/SKILL.md") else {
        warn!(server, name, uri = %uri, "skipping skill entry whose `uri` does not end in /SKILL.md");
        return None;
    };
    // SEP §Resource Mapping: the final skill-path segment MUST equal
    // `frontmatter.name`, so the name is recoverable from the URI alone.
    let final_segment = root.rsplit('/').next().unwrap_or(root);
    if final_segment != name {
        warn!(
            server,
            name,
            uri = %uri,
            "skipping skill entry whose URI's final skill-path segment does not equal frontmatter.name"
        );
        return None;
    }

    let resources = match raw.resources {
        None => None,
        Some(refs) => {
            let mut parsed = Vec::with_capacity(refs.len());
            for r in refs {
                if r.uri.is_empty() || r.digest.is_empty() {
                    // An incomplete `resources` set cannot honor the SEP's
                    // completeness guarantee, so the whole entry is dropped
                    // rather than silently degrading to unverified reads.
                    warn!(
                        server,
                        name,
                        "skipping skill entry with a malformed resources element (missing uri or digest)"
                    );
                    return None;
                }
                parsed.push(SkillResourceRef {
                    uri: r.uri,
                    digest: r.digest,
                });
            }
            if !parsed.iter().any(|r| r.uri == uri) {
                warn!(
                    server,
                    name,
                    uri = %uri,
                    "skipping skill entry whose resources omit the SKILL.md entry itself"
                );
                return None;
            }
            Some(parsed)
        }
    };

    Some(McpSkillEntry {
        server: server.to_string(),
        name,
        description: fm.description,
        uri,
        frontmatter,
        resources,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use async_trait::async_trait;
    use rmcp::model::{
        CallToolResult, ExtensionCapabilities, InitializeResult, JsonObject, ServerCapabilities,
    };
    use std::collections::HashMap;

    use crate::agents::mcp_client::Error;
    use crate::agents::ToolCallContext;

    /// Test double — answers `skills/list` from canned pages and `skills/get`
    /// from a URI-keyed map.
    struct FakeSkillsServer {
        info: InitializeResult,
        /// Successive `skills/list` result documents, keyed by cursor
        /// (`None` for the first page).
        list_pages: HashMap<Option<String>, serde_json::Value>,
        get_entries: HashMap<String, serde_json::Value>,
        delay: Option<Duration>,
    }

    impl FakeSkillsServer {
        fn with_capability() -> InitializeResult {
            let mut caps = ExtensionCapabilities::new();
            caps.insert(SKILLS_EXTENSION_ID.to_string(), JsonObject::new());
            InitializeResult::new(
                ServerCapabilities::builder()
                    .enable_resources()
                    .enable_extensions_with(caps)
                    .build(),
            )
        }

        fn without_capability() -> InitializeResult {
            InitializeResult::new(ServerCapabilities::builder().enable_resources().build())
        }

        fn single_page(doc: serde_json::Value) -> Self {
            FakeSkillsServer {
                info: Self::with_capability(),
                list_pages: HashMap::from([(None, doc)]),
                get_entries: HashMap::new(),
                delay: None,
            }
        }
    }

    #[async_trait]
    impl McpClientTrait for FakeSkillsServer {
        async fn list_tools(
            &self,
            _session_id: &str,
            _next_cursor: Option<String>,
            _cancel_token: CancellationToken,
        ) -> Result<rmcp::model::ListToolsResult, Error> {
            Ok(rmcp::model::ListToolsResult::default())
        }

        async fn call_tool(
            &self,
            _ctx: &ToolCallContext,
            _name: &str,
            _arguments: Option<JsonObject>,
            _cancel_token: CancellationToken,
        ) -> Result<CallToolResult, Error> {
            unreachable!("FakeSkillsServer has no tools")
        }

        fn get_info(&self) -> Option<&InitializeResult> {
            Some(&self.info)
        }

        async fn skills_list(
            &self,
            _session_id: &str,
            cursor: Option<String>,
            _cancel_token: CancellationToken,
        ) -> Result<SkillsListResult, Error> {
            if let Some(delay) = self.delay {
                tokio::time::sleep(delay).await;
            }
            match self.list_pages.get(&cursor) {
                Some(doc) => {
                    serde_json::from_value(doc.clone()).map_err(|_| Error::UnexpectedResponse)
                }
                None => Err(Error::TransportClosed),
            }
        }

        async fn skills_get(
            &self,
            _session_id: &str,
            uri: &str,
            _cancel_token: CancellationToken,
        ) -> Result<SkillsGetResult, Error> {
            match self.get_entries.get(uri) {
                Some(doc) => Ok(SkillsGetResult {
                    skill: serde_json::from_value(doc.clone())
                        .map_err(|_| Error::UnexpectedResponse)?,
                }),
                None => Err(Error::TransportClosed),
            }
        }
    }

    fn entry_json(name: &str, uri: &str) -> serde_json::Value {
        serde_json::json!({
            "uri": uri,
            "frontmatter": {"name": name, "description": format!("{} description", name)},
            "resources": [{"uri": uri, "digest": "sha256:0000000000000000000000000000000000000000000000000000000000000000"}],
        })
    }

    #[test]
    fn test_server_declares_capability() {
        assert!(server_declares_skills_capability(
            &FakeSkillsServer::with_capability()
        ));
        assert!(!server_declares_skills_capability(
            &FakeSkillsServer::without_capability()
        ));
    }

    #[test]
    fn test_server_declares_directory_read() {
        let mut caps = ExtensionCapabilities::new();
        let mut cfg = JsonObject::new();
        cfg.insert("directoryRead".to_string(), serde_json::json!(true));
        caps.insert(SKILLS_EXTENSION_ID.to_string(), cfg);
        let info = InitializeResult::new(
            ServerCapabilities::builder()
                .enable_resources()
                .enable_extensions_with(caps)
                .build(),
        );
        assert!(server_declares_directory_read(&info));

        // Declared extension but no directoryRead flag → false.
        assert!(!server_declares_directory_read(
            &FakeSkillsServer::with_capability()
        ));
        // No extension at all → false.
        assert!(!server_declares_directory_read(
            &FakeSkillsServer::without_capability()
        ));
    }

    #[test]
    fn test_verify_digest_match_and_mismatch() {
        // sha256("hello") = 2cf24dba5fb0a30e26e83b2ac5b9e29e1b161e5c1fa7425e73043362938b9824
        let bytes = b"hello";
        assert!(verify_digest(
            "sha256:2cf24dba5fb0a30e26e83b2ac5b9e29e1b161e5c1fa7425e73043362938b9824",
            bytes
        )
        .is_ok());
        assert!(verify_digest("sha256:deadbeef", bytes).is_err());
        assert!(verify_digest("md5:whatever", bytes).is_err());
    }

    #[test]
    fn test_verify_read_listed_unlisted_and_dynamic() {
        let uri = "skill://s/SKILL.md";
        let mut entry = McpSkillEntry {
            server: "srv".into(),
            name: "s".into(),
            description: String::new(),
            uri: uri.into(),
            frontmatter: serde_json::json!({"name": "s", "description": ""}),
            resources: Some(vec![SkillResourceRef {
                uri: uri.into(),
                digest: "sha256:2cf24dba5fb0a30e26e83b2ac5b9e29e1b161e5c1fa7425e73043362938b9824"
                    .into(),
            }]),
        };

        assert!(entry.verify_read(uri, b"hello").is_ok());
        assert!(entry.verify_read(uri, b"tampered").is_err());
        // Unlisted file within a held skill = verification failure.
        assert!(entry.verify_read("skill://s/extra.md", b"x").is_err());
        // Dynamic skill (no resources): nothing to verify.
        entry.resources = None;
        assert!(entry.verify_read("skill://s/extra.md", b"x").is_ok());
    }

    #[test]
    fn test_verify_frontmatter_identity() {
        let entry = McpSkillEntry {
            server: "srv".into(),
            name: "s".into(),
            description: String::new(),
            uri: "skill://s/SKILL.md".into(),
            frontmatter: serde_json::json!({"name": "s", "description": "d", "license": "MIT"}),
            resources: None,
        };
        assert!(entry
            .verify_frontmatter(
                &serde_json::json!({"license": "MIT", "description": "d", "name": "s"})
            )
            .is_ok());
        // A dropped or altered field is a verification failure.
        assert!(entry
            .verify_frontmatter(&serde_json::json!({"name": "s", "description": "d"}))
            .is_err());
        assert!(entry
            .verify_frontmatter(
                &serde_json::json!({"name": "s", "description": "d", "license": "GPL"})
            )
            .is_err());
    }

    #[tokio::test]
    async fn test_discover_via_skills_list() {
        let server = FakeSkillsServer::single_page(serde_json::json!({
            "skills": [
                entry_json("git-workflow", "skill://git-workflow/SKILL.md"),
                entry_json("refunds", "skill://acme/billing/refunds/SKILL.md"),
            ]
        }));

        let skills = fetch_server_skills(
            "gh",
            &server as &dyn McpClientTrait,
            "s",
            CancellationToken::new(),
        )
        .await;

        let entries = &skills.skills;
        assert_eq!(entries.len(), 2);
        assert_eq!(entries[0].name, "git-workflow");
        assert_eq!(entries[0].uri, "skill://git-workflow/SKILL.md");
        assert_eq!(entries[0].skill_root_uri(), "skill://git-workflow");
        assert_eq!(entries[0].server, "gh");
        assert_eq!(entries[1].name, "refunds");
        assert_eq!(entries[1].skill_root_uri(), "skill://acme/billing/refunds");
        assert_eq!(
            entries[1].resolve_relative("examples/email.md"),
            "skill://acme/billing/refunds/examples/email.md"
        );
    }

    #[tokio::test]
    async fn test_discover_follows_pagination() {
        let server = FakeSkillsServer {
            info: FakeSkillsServer::with_capability(),
            list_pages: HashMap::from([
                (
                    None,
                    serde_json::json!({
                        "skills": [entry_json("one", "skill://one/SKILL.md")],
                        "nextCursor": "p2",
                        "ttlMs": 60000,
                    }),
                ),
                (
                    Some("p2".to_string()),
                    serde_json::json!({
                        "skills": [entry_json("two", "skill://two/SKILL.md")],
                    }),
                ),
            ]),
            get_entries: HashMap::new(),
            delay: None,
        };

        let skills = fetch_server_skills(
            "srv",
            &server as &dyn McpClientTrait,
            "s",
            CancellationToken::new(),
        )
        .await;
        assert_eq!(skills.skills.len(), 2);
        assert_eq!(skills.skills[0].name, "one");
        assert_eq!(skills.skills[1].name, "two");
        assert_eq!(skills.ttl_ms, Some(60000));
    }

    #[tokio::test]
    async fn test_discover_skips_server_without_capability() {
        // Per the SEP, skills/list is only issued after the server declares
        // the extension.
        let server = FakeSkillsServer {
            info: FakeSkillsServer::without_capability(),
            list_pages: HashMap::from([(
                None,
                serde_json::json!({"skills": [entry_json("x", "skill://x/SKILL.md")]}),
            )]),
            get_entries: HashMap::new(),
            delay: None,
        };
        let skills = fetch_server_skills(
            "srv",
            &server as &dyn McpClientTrait,
            "s",
            CancellationToken::new(),
        )
        .await;
        assert!(skills.is_empty());
    }

    #[tokio::test]
    async fn test_discover_skips_name_uri_mismatch() {
        let server = FakeSkillsServer::single_page(serde_json::json!({
            "skills": [
                // Final skill-path segment "impostor" != frontmatter name "refunds".
                entry_json("refunds", "skill://acme/impostor/SKILL.md"),
                entry_json("clean", "skill://clean/SKILL.md"),
            ]
        }));
        let skills = fetch_server_skills(
            "srv",
            &server as &dyn McpClientTrait,
            "s",
            CancellationToken::new(),
        )
        .await;
        assert_eq!(skills.skills.len(), 1);
        assert_eq!(skills.skills[0].name, "clean");
    }

    #[tokio::test]
    async fn test_discover_skips_malformed_resources_keeps_rest() {
        let server = FakeSkillsServer::single_page(serde_json::json!({
            "skills": [
                {
                    "uri": "skill://broken/SKILL.md",
                    "frontmatter": {"name": "broken", "description": ""},
                    // Missing digest → incomplete resources → entry dropped.
                    "resources": [{"uri": "skill://broken/SKILL.md"}],
                },
                {
                    "uri": "skill://no-self/SKILL.md",
                    "frontmatter": {"name": "no-self", "description": ""},
                    // Omits the SKILL.md entry itself → dropped.
                    "resources": [{"uri": "skill://no-self/other.md", "digest": "sha256:aa"}],
                },
                entry_json("clean", "skill://clean/SKILL.md"),
            ]
        }));
        let skills = fetch_server_skills(
            "srv",
            &server as &dyn McpClientTrait,
            "s",
            CancellationToken::new(),
        )
        .await;
        assert_eq!(skills.skills.len(), 1);
        assert_eq!(skills.skills[0].name, "clean");
    }

    #[tokio::test]
    async fn test_discover_accepts_dynamic_skill_without_resources() {
        let server = FakeSkillsServer::single_page(serde_json::json!({
            "skills": [{
                "uri": "skill://dynamic/SKILL.md",
                "frontmatter": {"name": "dynamic", "description": "generated"},
            }]
        }));
        let skills = fetch_server_skills(
            "srv",
            &server as &dyn McpClientTrait,
            "s",
            CancellationToken::new(),
        )
        .await;
        assert_eq!(skills.skills.len(), 1);
        assert!(skills.skills[0].resources.is_none());
    }

    #[tokio::test]
    async fn test_discover_skips_entry_without_frontmatter_name() {
        let server = FakeSkillsServer::single_page(serde_json::json!({
            "skills": [{
                "uri": "skill://x/SKILL.md",
                "frontmatter": {"description": "nameless"},
            }]
        }));
        let skills = fetch_server_skills(
            "srv",
            &server as &dyn McpClientTrait,
            "s",
            CancellationToken::new(),
        )
        .await;
        assert!(skills.is_empty());
    }

    #[tokio::test]
    async fn test_discover_tolerates_failed_list() {
        let server = FakeSkillsServer {
            info: FakeSkillsServer::with_capability(),
            list_pages: HashMap::new(),
            get_entries: HashMap::new(),
            delay: None,
        };
        let skills = fetch_server_skills(
            "gh",
            &server as &dyn McpClientTrait,
            "s",
            CancellationToken::new(),
        )
        .await;
        assert!(skills.is_empty());
    }

    #[tokio::test]
    async fn test_discover_accepts_non_skill_scheme() {
        let server = FakeSkillsServer::single_page(serde_json::json!({
            "skills": [entry_json(
                "pull-requests",
                "github://github/repo/skills/pull-requests/SKILL.md"
            )]
        }));
        let skills = fetch_server_skills(
            "github",
            &server as &dyn McpClientTrait,
            "s",
            CancellationToken::new(),
        )
        .await;
        let entries = &skills.skills;
        assert_eq!(entries.len(), 1);
        assert_eq!(
            entries[0].uri,
            "github://github/repo/skills/pull-requests/SKILL.md"
        );
        assert_eq!(
            entries[0].skill_root_uri(),
            "github://github/repo/skills/pull-requests"
        );
    }

    #[tokio::test]
    async fn test_fetch_skill_entry_via_skills_get() {
        let uri = "skill://unlisted/SKILL.md";
        let server = FakeSkillsServer {
            info: FakeSkillsServer::with_capability(),
            list_pages: HashMap::new(),
            get_entries: HashMap::from([(uri.to_string(), entry_json("unlisted", uri))]),
            delay: None,
        };

        let entry = fetch_skill_entry(
            "srv",
            &server as &dyn McpClientTrait,
            "s",
            uri,
            CancellationToken::new(),
        )
        .await
        .unwrap();
        assert_eq!(entry.name, "unlisted");
        assert_eq!(entry.uri, uri);

        let err = fetch_skill_entry(
            "srv",
            &server as &dyn McpClientTrait,
            "s",
            "skill://nope/SKILL.md",
            CancellationToken::new(),
        )
        .await
        .unwrap_err();
        assert!(err.contains("skills/get failed"));
    }

    #[tokio::test]
    async fn test_discover_timeout_does_not_block() {
        let server = FakeSkillsServer {
            info: FakeSkillsServer::with_capability(),
            list_pages: HashMap::from([(None, serde_json::json!({"skills": []}))]),
            get_entries: HashMap::new(),
            delay: Some(LIST_FETCH_TIMEOUT + Duration::from_millis(500)),
        };

        let start = std::time::Instant::now();
        let skills = fetch_server_skills(
            "slow",
            &server as &dyn McpClientTrait,
            "s",
            CancellationToken::new(),
        )
        .await;
        let elapsed = start.elapsed();

        assert!(skills.is_empty());
        assert!(
            elapsed < LIST_FETCH_TIMEOUT + Duration::from_millis(500),
            "fetch took {:?}, should have timed out",
            elapsed
        );
    }
}
