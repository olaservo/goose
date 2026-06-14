//! MCP-served Agent Skills discovery, per SEP `io.modelcontextprotocol/skills`.
//!
//! Bridges skills served over MCP (via `skill://` or any index-listed URI)
//! into Goose's existing skills pipeline. This module is the discovery layer:
//! it reads a server's `skill://index.json`, parses skill entries, and returns
//! [`McpSkillEntry`] values that the skills platform extension caches and
//! surfaces in the system prompt.
//!
//! Scheme-agnostic: the SEP permits servers to list skills under a
//! domain-native URI scheme (e.g. `github://owner/repo/.../SKILL.md`) so long
//! as the entry appears in `skill://index.json`.
//!
//! Security: per the SEP, skill content from MCP servers is UNTRUSTED model
//! input. This module extracts only the skill's frontmatter and URI/digest
//! locators from the index — never execution-capable fields. Loaded SKILL.md
//! and archive content is verified against the index `digest` before use.

use rmcp::model::{InitializeResult, ResourceContents};
use serde::Deserialize;
use std::time::Duration;
use tokio_util::sync::CancellationToken;
use tracing::{debug, warn};

use crate::agents::mcp_client::McpClientTrait;
use crate::skills::SkillFrontmatter;

/// Extension identifier per the SEP.
pub(crate) const SKILLS_EXTENSION_ID: &str = "io.modelcontextprotocol/skills";

/// Well-known index resource URI. Fixed by the SEP — always read from
/// `skill://index.json` regardless of which scheme the listed skills use.
pub(crate) const INDEX_URI: &str = "skill://index.json";

/// How long to wait for a server's index fetch before giving up.
/// Applied at extension-registration time so a misbehaving server cannot
/// stall session startup indefinitely. An empty cache on timeout is
/// acceptable — a future `notifications/resources/list_changed` or
/// explicit UI refresh repopulates.
pub(crate) const INDEX_FETCH_TIMEOUT: Duration = Duration::from_secs(5);

/// One archive distribution form of a skill, per `skills[].archives[]`.
/// The archive unpacks into the skill's virtual file tree (`SKILL.md` at
/// the root). `digest` is verified before the bytes are unpacked.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ArchiveRef {
    pub url: String,
    pub media_type: String,
    pub digest: String,
}

/// A single indexed skill served over MCP, as surfaced to the skills
/// platform extension.
///
/// `name`/`description` are taken from the entry's verbatim `frontmatter`
/// block (per SEP, the index carries the full `SKILL.md` frontmatter).
///
/// An entry exposes its `SKILL.md` as an individually-addressable resource
/// (`url` + `digest`), as one or more `archives`, or both. `url` is `None`
/// for archive-only entries; the skill root for relative refs is then the
/// unpacked archive tree rather than a URI prefix.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct McpSkillEntry {
    pub server: String,
    pub name: String,
    pub description: String,
    /// Resource URI of the skill's `SKILL.md` (any scheme). `None` when the
    /// skill is distributed only as an archive.
    pub url: Option<String>,
    /// `sha256:<hex>` digest of the `SKILL.md` resource at `url`. Present
    /// whenever `url` is; the host MUST verify the loaded body against it.
    pub digest: Option<String>,
    /// Archive forms of the skill. Non-empty for archive-distributed skills.
    pub archives: Vec<ArchiveRef>,
}

impl McpSkillEntry {
    /// Skill root URI for individually-addressed skills: `url` truncated at
    /// (and including) the final `/`. Returns `None` for archive-only
    /// entries (which have no URI namespace). Matches the SEP's
    /// relative-resolution rule — the entry URI's directory is the base for
    /// relative refs, regardless of whether the trailing segment is the
    /// literal `SKILL.md` or a directory.
    pub fn skill_root_uri(&self) -> Option<&str> {
        let url = self.url.as_deref()?;
        // `idx` is the byte offset of an ASCII `/`, always a valid UTF-8 boundary.
        #[allow(clippy::string_slice)]
        match url.rfind('/') {
            Some(idx) => Some(&url[..=idx]),
            None => Some(url),
        }
    }

    /// Returns the first archive whose `mediaType` this host can unpack, if
    /// any. Used to load archive-distributed skills.
    pub fn supported_archive(&self) -> Option<&ArchiveRef> {
        self.archives
            .iter()
            .find(|a| crate::skills::archive::supports_media_type(&a.media_type))
    }
}

/// All MCP-served skills discovered from a single server's index.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct ServerSkills {
    pub concrete: Vec<McpSkillEntry>,
}

impl ServerSkills {
    pub fn is_empty(&self) -> bool {
        self.concrete.is_empty()
    }
}

/// Returns true if the server's initialize response declares the skills
/// extension capability. Informational only — per the SEP, hosts MUST
/// attempt `skill://index.json` regardless.
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

/// Index shape per the SEP. The index carries no `$schema`/version marker —
/// it is versioned by the extension capability. Unknown fields are ignored.
#[derive(Debug, Deserialize)]
struct IndexDoc {
    #[serde(default)]
    skills: Vec<IndexEntry>,
}

/// One `skills[]` entry. `frontmatter` is the verbatim `SKILL.md` YAML as
/// JSON (required, carries `name`/`description`). Every entry MUST carry a
/// `url` (+ `digest`), a non-empty `archives`, or both.
#[derive(Debug, Deserialize)]
struct IndexEntry {
    #[serde(default)]
    frontmatter: Option<serde_json::Value>,
    #[serde(default)]
    url: Option<String>,
    #[serde(default)]
    digest: Option<String>,
    #[serde(default)]
    archives: Vec<IndexArchive>,
}

#[derive(Debug, Deserialize)]
struct IndexArchive {
    url: String,
    #[serde(rename = "mimeType")]
    media_type: String,
    digest: String,
}

/// Verify `bytes` against a `sha256:<hex>` digest from the index. Per the
/// SEP, hosts MUST verify retrieved content against the index digest and
/// MUST NOT use content that fails to match.
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
            "digest mismatch: index advertised {} but content hashes to sha256:{}",
            expected, actual
        ))
    }
}

/// Fetches and parses `skill://index.json` from a single MCP server via
/// its client handle. Returns an empty [`ServerSkills`] (with a log) on
/// any failure — this function MUST NOT propagate errors because it runs
/// during extension registration and must not block the agent from
/// starting.
///
/// Caller supplies the server name (extension key) because it's stamped
/// into each returned entry's `server` field for later routing.
pub async fn fetch_server_skills(
    server: &str,
    client: &dyn McpClientTrait,
    session_id: &str,
    cancel: CancellationToken,
) -> ServerSkills {
    let fetch = async {
        let read = client
            .read_resource(session_id, INDEX_URI, cancel.clone())
            .await?;

        let text = read
            .contents
            .into_iter()
            .find_map(|c| match c {
                ResourceContents::TextResourceContents { text, .. } => Some(text),
                _ => None,
            })
            .ok_or("index resource contained no text contents")?;

        let doc: IndexDoc = serde_json::from_str(&text)
            .map_err(|e| format!("failed to parse {}: {}", INDEX_URI, e))?;

        Ok::<_, Box<dyn std::error::Error + Send + Sync>>(doc)
    };

    let doc = match tokio::time::timeout(INDEX_FETCH_TIMEOUT, fetch).await {
        Ok(Ok(doc)) => doc,
        Ok(Err(e)) => {
            debug!(server, error = %e, "skill index fetch: no usable index");
            return ServerSkills::default();
        }
        Err(_) => {
            warn!(
                server,
                timeout_secs = INDEX_FETCH_TIMEOUT.as_secs(),
                "skill index fetch timed out"
            );
            return ServerSkills::default();
        }
    };

    let mut out = ServerSkills::default();
    for raw in doc.skills {
        if let Some(entry) = parse_entry(server, raw) {
            out.concrete.push(entry);
        }
    }
    out
}

fn parse_entry(server: &str, raw: IndexEntry) -> Option<McpSkillEntry> {
    // `name`/`description` come from the verbatim frontmatter block.
    let Some(frontmatter) = raw.frontmatter else {
        warn!(server, "skipping index entry with no `frontmatter`");
        return None;
    };
    let fm: SkillFrontmatter = match serde_json::from_value(frontmatter) {
        Ok(fm) => fm,
        Err(e) => {
            warn!(server, error = %e, "skipping index entry with unparseable `frontmatter`");
            return None;
        }
    };
    let Some(name) = fm.name.filter(|s| !s.is_empty()) else {
        warn!(
            server,
            "skipping index entry whose frontmatter has no `name`"
        );
        return None;
    };

    let url = raw.url.filter(|s| !s.is_empty());
    let archives: Vec<ArchiveRef> = raw
        .archives
        .into_iter()
        .filter(|a| !a.url.is_empty())
        .map(|a| ArchiveRef {
            url: a.url,
            media_type: a.media_type,
            digest: a.digest,
        })
        .collect();

    // SEP MUST: every entry carries a `url`, a non-empty `archives`, or both.
    if url.is_none() && archives.is_empty() {
        warn!(
            server,
            name, "skipping index entry with neither `url` nor `archives`"
        );
        return None;
    }

    if let Some(u) = &url {
        if raw.digest.is_none() {
            debug!(server, name, url = %u, "skill entry has `url` but no `digest`; content will be loaded unverified");
        }
        if !u.ends_with("SKILL.md") && !u.ends_with('/') {
            debug!(server, name, url = %u, "skill entry `url` does not end in `SKILL.md` or `/`");
        }
    }

    Some(McpSkillEntry {
        server: server.to_string(),
        name,
        description: fm.description,
        url,
        digest: raw.digest,
        archives,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use async_trait::async_trait;
    use rmcp::model::{
        CallToolResult, ExtensionCapabilities, InitializeResult, JsonObject, ListResourcesResult,
        ListToolsResult, ReadResourceResult, ServerCapabilities, ServerNotification,
    };
    use std::collections::HashMap;
    use tokio::sync::mpsc;

    use crate::agents::mcp_client::Error;
    use crate::agents::ToolCallContext;

    /// Test double — exposes a fixed set of resources and no tools.
    pub struct FakeSkillsServer {
        pub info: InitializeResult,
        pub resources: HashMap<String, String>,
        pub delay: Option<Duration>,
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
    }

    #[async_trait]
    impl McpClientTrait for FakeSkillsServer {
        async fn list_tools(
            &self,
            _session_id: &str,
            _next_cursor: Option<String>,
            _cancel_token: CancellationToken,
        ) -> Result<ListToolsResult, Error> {
            Ok(ListToolsResult {
                tools: vec![],
                next_cursor: None,
                meta: None,
            })
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

        async fn list_resources(
            &self,
            _session_id: &str,
            _next_cursor: Option<String>,
            _cancel_token: CancellationToken,
        ) -> Result<ListResourcesResult, Error> {
            Ok(ListResourcesResult {
                resources: vec![],
                next_cursor: None,
                meta: None,
            })
        }

        async fn read_resource(
            &self,
            _session_id: &str,
            uri: &str,
            _cancel_token: CancellationToken,
        ) -> Result<ReadResourceResult, Error> {
            if let Some(delay) = self.delay {
                tokio::time::sleep(delay).await;
            }
            match self.resources.get(uri) {
                Some(text) => Ok(ReadResourceResult::new(vec![
                    ResourceContents::TextResourceContents {
                        uri: uri.to_string(),
                        mime_type: None,
                        text: text.clone(),
                        meta: None,
                    },
                ])),
                None => Err(Error::TransportClosed),
            }
        }

        async fn subscribe(&self) -> mpsc::Receiver<ServerNotification> {
            mpsc::channel(1).1
        }
    }

    /// Build an index entry with a `frontmatter` block from `name`/`description`
    /// plus the supplied extra JSON fields (e.g. `,"url":"…","digest":"…"`).
    fn entry(name: &str, description: &str, extra: &str) -> String {
        format!(
            r#"{{"frontmatter":{{"name":"{}","description":"{}"}}{}}}"#,
            name, description, extra
        )
    }

    fn index_with(entries: &str) -> String {
        format!(r#"{{"skills":[{}]}}"#, entries)
    }

    #[test]
    fn test_server_declares_capability() {
        let declared = FakeSkillsServer::with_capability();
        assert!(server_declares_skills_capability(&declared));

        let undeclared = FakeSkillsServer::without_capability();
        assert!(!server_declares_skills_capability(&undeclared));
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

    #[tokio::test]
    async fn test_discover_via_index_json() {
        let mut resources = HashMap::new();
        resources.insert(
            INDEX_URI.to_string(),
            index_with(&format!(
                "{},{}",
                entry(
                    "git-workflow",
                    "Git conventions",
                    r#","url":"skill://git-workflow/SKILL.md","digest":"sha256:abc""#
                ),
                entry(
                    "refunds",
                    "Process refunds",
                    r#","url":"skill://acme/billing/refunds/SKILL.md","digest":"sha256:def""#
                ),
            )),
        );
        let server = FakeSkillsServer {
            info: FakeSkillsServer::with_capability(),
            resources,
            delay: None,
        };

        let skills = fetch_server_skills(
            "gh",
            &server as &dyn McpClientTrait,
            "s",
            CancellationToken::new(),
        )
        .await;

        let entries = &skills.concrete;
        assert_eq!(entries.len(), 2);
        assert_eq!(entries[0].name, "git-workflow");
        assert_eq!(
            entries[0].url.as_deref(),
            Some("skill://git-workflow/SKILL.md")
        );
        assert_eq!(entries[0].digest.as_deref(), Some("sha256:abc"));
        assert_eq!(entries[0].skill_root_uri(), Some("skill://git-workflow/"));
        assert_eq!(entries[0].server, "gh");
        assert_eq!(entries[1].name, "refunds");
        assert_eq!(
            entries[1].skill_root_uri(),
            Some("skill://acme/billing/refunds/")
        );
    }

    #[tokio::test]
    async fn test_discover_archive_only_entry() {
        // An entry with no `url`, only an archive. It must surface (name +
        // description from frontmatter) with the archive recorded.
        let mut resources = HashMap::new();
        resources.insert(
            INDEX_URI.to_string(),
            index_with(&entry(
                "pdf-processing",
                "Process PDFs",
                r#","archives":[{"url":"skill://pdf-processing.tar.gz","mimeType":"application/gzip","digest":"sha256:aaa"}]"#,
            )),
        );
        let server = FakeSkillsServer {
            info: FakeSkillsServer::with_capability(),
            resources,
            delay: None,
        };

        let skills = fetch_server_skills(
            "srv",
            &server as &dyn McpClientTrait,
            "s",
            CancellationToken::new(),
        )
        .await;
        assert_eq!(skills.concrete.len(), 1);
        let e = &skills.concrete[0];
        assert_eq!(e.name, "pdf-processing");
        assert!(e.url.is_none());
        assert_eq!(e.skill_root_uri(), None);
        assert_eq!(e.archives.len(), 1);
        assert_eq!(e.archives[0].media_type, "application/gzip");
        assert!(e.supported_archive().is_some());
    }

    #[tokio::test]
    async fn test_discover_skips_entry_without_url_or_archives() {
        let mut resources = HashMap::new();
        resources.insert(
            INDEX_URI.to_string(),
            index_with(&format!(
                "{},{}",
                entry(
                    "ok",
                    "fine",
                    r#","url":"skill://ok/SKILL.md","digest":"sha256:1""#
                ),
                entry("bad", "no locator", ""),
            )),
        );
        let server = FakeSkillsServer {
            info: FakeSkillsServer::with_capability(),
            resources,
            delay: None,
        };
        let skills = fetch_server_skills(
            "srv",
            &server as &dyn McpClientTrait,
            "s",
            CancellationToken::new(),
        )
        .await;
        assert_eq!(skills.concrete.len(), 1);
        assert_eq!(skills.concrete[0].name, "ok");
    }

    #[tokio::test]
    async fn test_discover_skips_entry_without_frontmatter_name() {
        let mut resources = HashMap::new();
        resources.insert(
            INDEX_URI.to_string(),
            index_with(
                r#"{"frontmatter":{"description":"nameless"},"url":"skill://x/SKILL.md","digest":"sha256:1"}"#,
            ),
        );
        let server = FakeSkillsServer {
            info: FakeSkillsServer::with_capability(),
            resources,
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
    async fn test_discover_tolerates_missing_index() {
        let server = FakeSkillsServer {
            info: FakeSkillsServer::with_capability(),
            resources: HashMap::new(),
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
        let mut resources = HashMap::new();
        resources.insert(
            INDEX_URI.to_string(),
            index_with(&entry(
                "pull-requests",
                "PRs",
                r#","url":"github://github/repo/skills/pull-requests/SKILL.md","digest":"sha256:1""#,
            )),
        );
        let server = FakeSkillsServer {
            info: FakeSkillsServer::with_capability(),
            resources,
            delay: None,
        };

        let skills = fetch_server_skills(
            "github",
            &server as &dyn McpClientTrait,
            "s",
            CancellationToken::new(),
        )
        .await;
        let entries = &skills.concrete;
        assert_eq!(entries.len(), 1);
        assert_eq!(
            entries[0].url.as_deref(),
            Some("github://github/repo/skills/pull-requests/SKILL.md")
        );
        assert_eq!(
            entries[0].skill_root_uri(),
            Some("github://github/repo/skills/pull-requests/")
        );
    }

    #[tokio::test]
    async fn test_discover_directory_form_url() {
        let mut resources = HashMap::new();
        resources.insert(
            INDEX_URI.to_string(),
            index_with(&entry(
                "refunds",
                "",
                r#","url":"skill://acme/billing/refunds/","digest":"sha256:1""#,
            )),
        );
        let server = FakeSkillsServer {
            info: FakeSkillsServer::with_capability(),
            resources,
            delay: None,
        };

        let skills = fetch_server_skills(
            "acme",
            &server as &dyn McpClientTrait,
            "s",
            CancellationToken::new(),
        )
        .await;
        let entries = &skills.concrete;
        assert_eq!(entries.len(), 1);
        assert_eq!(
            entries[0].url.as_deref(),
            Some("skill://acme/billing/refunds/")
        );
        assert_eq!(
            entries[0].skill_root_uri(),
            Some("skill://acme/billing/refunds/")
        );
    }

    #[tokio::test]
    async fn test_discover_timeout_does_not_block() {
        let server = FakeSkillsServer {
            info: FakeSkillsServer::with_capability(),
            resources: HashMap::new(),
            delay: Some(INDEX_FETCH_TIMEOUT + Duration::from_millis(500)),
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
            elapsed < INDEX_FETCH_TIMEOUT + Duration::from_millis(500),
            "fetch took {:?}, should have timed out",
            elapsed
        );
    }
}
