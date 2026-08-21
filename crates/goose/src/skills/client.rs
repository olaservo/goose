use super::discover_skills_with_config;
use super::loaded_skill_context_with_args;
use super::mcp_client::McpSkillEntry;
use crate::agents::extension::PlatformExtensionContext;
use crate::agents::extension_manager::ExtensionManager;
use crate::agents::mcp_client::{Error, McpClientTrait};
use crate::agents::ToolCallContext;
use crate::config::Config;
use async_trait::async_trait;
use goose_sdk_types::custom_requests::{SourceEntry, SourceType};
use rmcp::model::{
    CallToolResult, ContentBlock, Implementation, InitializeResult, JsonObject, ListToolsResult,
    ResourceContents, ServerCapabilities, ServerNotification, Tool,
};
use std::collections::HashSet;
use std::path::{Path, PathBuf};
use std::sync::{Mutex, Weak};
use std::time::{Duration, Instant};
use tokio::sync::mpsc;
use tokio_util::sync::CancellationToken;
use tracing::{debug, warn};

/// How long a cached snapshot of installed FS skill names stays valid. The
/// cache backs FS-vs-MCP collision detection in `get_dynamic_instructions`,
/// which runs every reply; without the cache we'd walk up to seven skill
/// directories per turn once any MCP skill is cached. Ten seconds is plenty
/// short for newly-added local skills to show up on the next collision
/// check.
const FS_NAMES_TTL: Duration = Duration::from_secs(10);

pub static EXTENSION_NAME: &str = "skills";

pub struct SkillsClient {
    info: InitializeResult,
    working_dir: PathBuf,
    exclude_builtin_skills: bool,
    config: &'static Config,
    /// Weak reference to the extension manager so we can, per turn, read
    /// the MCP-served skills cache populated at server connect time and
    /// dispatch `resources/read` when `load_skill` hits an MCP entry.
    /// `None` in session-less contexts (tests, bootstrap).
    extension_manager: Option<Weak<ExtensionManager>>,
    /// TTL-cached snapshot of installed FS skill names. Read on every reply
    /// to drive FS-vs-MCP collision prefixing; we recompute at most once
    /// per `FS_NAMES_TTL` so the per-turn cost is amortized.
    fs_names_cache: Mutex<FsNamesCache>,
}

#[derive(Default)]
struct FsNamesCache {
    refreshed_at: Option<Instant>,
    names: HashSet<String>,
}

impl SkillsClient {
    pub fn new(context: PlatformExtensionContext) -> anyhow::Result<Self> {
        let working_dir = context
            .session
            .as_ref()
            .map(|s| s.working_dir.clone())
            .unwrap_or_else(|| std::env::current_dir().unwrap_or_default());

        let info = InitializeResult::new(ServerCapabilities::builder().enable_tools().build())
            .with_server_info(Implementation::new(EXTENSION_NAME, "1.0.0").with_title("Skills"));

        Ok(Self {
            info,
            working_dir,
            exclude_builtin_skills: false,
            config: Config::global(),
            extension_manager: context.extension_manager,
            fs_names_cache: Mutex::new(FsNamesCache::default()),
        })
    }

    /// Controls whether Goose's bundled skills are exposed by this client.
    /// Bundled skills are enabled by default.
    pub fn with_builtin_skills(mut self, enabled: bool) -> Self {
        self.exclude_builtin_skills = !enabled;
        self
    }

    #[cfg(test)]
    fn with_config(mut self, config: &'static Config) -> Self {
        self.config = config;
        self
    }

    fn discover_skills(&self) -> Vec<SourceEntry> {
        discover_skills_with_config(Some(&self.working_dir), self.config)
            .into_iter()
            .filter(|skill| {
                !self.exclude_builtin_skills || skill.source_type != SourceType::BuiltinSkill
            })
            .collect()
    }

    fn upgraded_manager(&self) -> Option<std::sync::Arc<ExtensionManager>> {
        self.extension_manager.as_ref().and_then(|w| w.upgrade())
    }

    /// All MCP skills discovered from connected servers, opted-in or not.
    /// Used for `load_skill` resolution — a skill the user explicitly names
    /// should load even from a server whose skills aren't auto-injected.
    async fn mcp_skills(&self) -> Vec<McpSkillEntry> {
        match self.upgraded_manager() {
            Some(mgr) => mgr.aggregated_mcp_skills().await,
            None => Vec::new(),
        }
    }

    /// MCP skills from servers the user has opted into injecting — the
    /// gated set surfaced in the system prompt.
    async fn injectable_mcp_skills(&self) -> Vec<McpSkillEntry> {
        match self.upgraded_manager() {
            Some(mgr) => mgr.injectable_mcp_skills().await,
            None => Vec::new(),
        }
    }

    /// Rebuilds the set of FS skill names currently installed (respecting
    /// the builtin filter, so collision detection matches what the model
    /// sees). Used to detect FS-vs-MCP name collisions (FS wins — the MCP
    /// entry is rendered with a `<server>__<name>` prefix).
    fn fs_skill_names(&self) -> HashSet<String> {
        self.discover_skills()
            .into_iter()
            .filter(|s| matches!(s.source_type, SourceType::Skill | SourceType::BuiltinSkill))
            .map(|s| s.name)
            .collect()
    }

    /// Cached wrapper around `fs_skill_names`. Rescans the FS when the
    /// previous snapshot is older than `FS_NAMES_TTL`, or on first call.
    /// Never holds the mutex across the FS walk — we drop the guard, do
    /// the blocking scan, then re-acquire to write. The scan can race
    /// with a concurrent call but the result is equivalent (both computes
    /// produce the same set for identical FS state, and the last write
    /// wins with a fresh timestamp).
    fn fs_skill_names_cached(&self) -> HashSet<String> {
        {
            let cache = self.fs_names_cache.lock().expect("fs_names_cache poisoned");
            if let Some(ts) = cache.refreshed_at {
                if ts.elapsed() < FS_NAMES_TTL {
                    return cache.names.clone();
                }
            }
        }

        let fresh = self.fs_skill_names();

        let mut cache = self.fs_names_cache.lock().expect("fs_names_cache poisoned");
        cache.refreshed_at = Some(Instant::now());
        cache.names = fresh.clone();
        fresh
    }

    /// Load a skill given only its SKILL.md URI (SEP: hosts MUST support
    /// this). A cached entry matching the URI is used directly; otherwise
    /// `skills/get` is issued against the named server, or against every
    /// connected server declaring the skills extension, and the first
    /// success wins. Fetched entries are remembered in the server's cache
    /// so supporting-file reads and repeat loads resolve without another
    /// fetch.
    async fn load_skill_by_uri(
        &self,
        session_id: &str,
        uri: &str,
        server_arg: Option<&str>,
        cancel: CancellationToken,
    ) -> CallToolResult {
        let Some(mgr) = self.upgraded_manager() else {
            return CallToolResult::error(vec![ContentBlock::text(
                "No MCP servers are connected, so a skill URI cannot be resolved.",
            )]);
        };

        if let Some(entry) = self
            .mcp_skills()
            .await
            .into_iter()
            .find(|e| e.uri == uri && server_arg.is_none_or(|s| e.server == s))
        {
            return load_mcp_skill_md(mgr.as_ref(), session_id, &entry, cancel).await;
        }

        let candidates: Vec<String> = match server_arg {
            Some(server) => vec![server.to_string()],
            None => {
                let mut out = Vec::new();
                for name in mgr.list_extensions().await.unwrap_or_default() {
                    if mgr.server_declares_skills(&name).await {
                        out.push(name);
                    }
                }
                out
            }
        };

        if candidates.is_empty() {
            return CallToolResult::error(vec![ContentBlock::text(format!(
                "No connected server declares the skills extension, so '{}' cannot be resolved. \
                 Use the read_resource tool for arbitrary resource URIs.",
                uri
            ))]);
        }

        let mut last_err = String::new();
        for server in candidates {
            match mgr
                .skills_get_for_server(session_id, uri, &server, cancel.clone())
                .await
            {
                Ok(entry) => {
                    mgr.remember_skill_entry(&server, entry.clone()).await;
                    return load_mcp_skill_md(mgr.as_ref(), session_id, &entry, cancel).await;
                }
                Err(e) => last_err = e,
            }
        }

        CallToolResult::error(vec![ContentBlock::text(format!(
            "No connected server serves '{}' as a skill (skills/get failed; last error: {}).",
            uri, last_err
        ))])
    }
}

/// Computes the set of skill names that collide across FS skills and MCP
/// entries. Any name appearing more than once in this union needs to be
/// rendered in its disambiguated `<server>__<name>` form so the model can
/// address the right entity unambiguously.
fn collision_names(fs_names: &HashSet<String>, mcp: &[McpSkillEntry]) -> HashSet<String> {
    let mut counts: std::collections::HashMap<String, usize> = std::collections::HashMap::new();
    for n in fs_names {
        *counts.entry(n.clone()).or_insert(0) += 1;
    }
    for entry in mcp {
        *counts.entry(entry.name.clone()).or_insert(0) += 1;
    }
    counts
        .into_iter()
        .filter_map(|(n, c)| if c > 1 { Some(n) } else { None })
        .collect()
}

/// Renders the MCP-skills section of the system prompt. Names that collide
/// with any other visible skill (FS or another MCP entry) are rendered in
/// `<server>__<name>` form so the model can address the entry unambiguously
/// via `load_skill`. Empty output when no MCP skills are available.
fn format_mcp_skills_section(collisions: &HashSet<String>, mcp: &[McpSkillEntry]) -> String {
    if mcp.is_empty() {
        return String::new();
    }

    let mut sorted: Vec<&McpSkillEntry> = mcp.iter().collect();
    sorted.sort_by(|a, b| {
        (a.name.as_str(), a.server.as_str()).cmp(&(b.name.as_str(), b.server.as_str()))
    });

    let mut out = String::from(
        "\n\nYou also have these skills from connected MCP servers. Load them via load_skill by name; if a collision is shown in <server>__<name> form, use that exact form:",
    );
    for entry in sorted {
        let needs_prefix = collisions.contains(&entry.name);
        let display_name = if needs_prefix {
            format!("{}__{}", entry.server, entry.name)
        } else {
            entry.name.clone()
        };
        // URI intentionally omitted: the model addresses MCP skills by
        // name via `load_skill`, and including full URIs for every entry
        // bloats every turn's system prompt on servers with many skills.
        out.push_str(&format!(
            "\n• {} ({}) - {}",
            display_name, entry.server, entry.description
        ));
    }
    out
}

/// Extracts the first text content from a `ReadResourceResult`. Returns
/// `None` if the result contains only blob contents (binary). Logs a
/// warning if the server returned more than one text entry — a skill file
/// should arrive as a single document, and a multi-entry response likely
/// means the server is splitting content in a way the host won't
/// reassemble.
fn first_text_content(
    result: rmcp::model::ReadResourceResult,
    server: &str,
    uri: &str,
) -> Option<String> {
    let mut text_count = 0usize;
    let mut first: Option<String> = None;
    for c in result.contents {
        if let ResourceContents::TextResourceContents { text, .. } = c {
            text_count += 1;
            if first.is_none() {
                first = Some(text);
            }
        }
    }
    if text_count > 1 {
        warn!(
            server,
            uri,
            text_count,
            "read_resource returned multiple text contents; only the first was used"
        );
    }
    first
}

/// Normalizes a supporting-file relative reference before composing it
/// with a skill's root URI. Rejects inputs that could escape the skill
/// directory — `..` segments or a leading `/`. Backslashes are folded to
/// forward slashes so Windows-style paths from the model don't slip past
/// the `..` check. Returns `None` if the input is unsafe to compose.
fn sanitize_relative_ref(raw: &str) -> Option<String> {
    let normalized = raw.replace('\\', "/");
    if normalized.starts_with('/') {
        return None;
    }
    if normalized.split('/').any(|segment| segment == "..") {
        return None;
    }
    Some(normalized)
}

/// Finds an MCP skill entry by name, accepting either the bare name or the
/// `<server>__<name>` collision form. Literal match wins so a server can
/// legitimately publish a skill whose name contains `__` without being
/// hijacked by a coincidental server/skill pair on the other side of the
/// split.
fn find_mcp_by_name<'a>(mcp: &'a [McpSkillEntry], query: &str) -> Option<&'a McpSkillEntry> {
    if let Some(hit) = mcp.iter().find(|e| e.name == query) {
        return Some(hit);
    }
    if let Some((server_prefix, bare_name)) = query.split_once("__") {
        return mcp
            .iter()
            .find(|e| e.server == server_prefix && e.name == bare_name);
    }
    None
}

/// Enumerates supporting-file relative refs for an MCP skill (excluding the
/// SKILL.md itself). The entry's manifest is the source of truth
/// (SEP §Resources) — no extra round trips. For dynamic skills: when the
/// owning server declares `directoryRead: true`, walks the skill tree via
/// `resources/directory/read`; otherwise filters the server's flat
/// `resources/list`. Best-effort in the dynamic case: any error yields an
/// empty list and no section is rendered. Surfaces only pointers, never
/// content.
async fn enumerate_mcp_supporting_resources(
    mgr: &ExtensionManager,
    session_id: &str,
    entry: &McpSkillEntry,
    cancel: CancellationToken,
) -> Vec<String> {
    let root = entry.skill_root_uri();
    let root_prefix = format!("{}/", root);

    let uris: Vec<String> = if let Some(resources) = entry.manifest() {
        resources.iter().map(|r| r.uri.clone()).collect()
    } else if mgr.server_supports_directory_read(&entry.server).await {
        directory_walk(mgr, session_id, &entry.server, root, cancel).await
    } else {
        match mgr
            .list_resources_for_server(session_id, &entry.server, cancel)
            .await
        {
            Ok(list) => list.resources.iter().map(|r| r.uri.clone()).collect(),
            Err(e) => {
                debug!(
                    server = %entry.server,
                    skill = %entry.name,
                    error = %e.message,
                    "supporting-file enumeration failed; rendering without the section",
                );
                Vec::new()
            }
        }
    };

    let mut out = Vec::new();
    for uri in uris {
        if uri == entry.uri {
            continue;
        }
        let Some(rel) = uri.strip_prefix(&root_prefix) else {
            continue;
        };
        if rel.is_empty() {
            continue;
        }
        out.push(rel.to_string());
    }
    out.sort();
    out.dedup();
    out
}

/// Recursively lists every file URI under `root` via
/// `resources/directory/read`, descending into `inode/directory` children.
/// Bounded by a node cap to keep a pathological tree from stalling a turn.
async fn directory_walk(
    mgr: &ExtensionManager,
    session_id: &str,
    server: &str,
    root: &str,
    cancel: CancellationToken,
) -> Vec<String> {
    const MAX_NODES: usize = 512;
    let mut files = Vec::new();
    let mut queue: Vec<String> = vec![root.to_string()];
    let mut visited = 0usize;

    while let Some(dir) = queue.pop() {
        if visited >= MAX_NODES {
            break;
        }
        visited += 1;
        let children = match mgr
            .directory_read_for_server(session_id, &dir, server, cancel.clone())
            .await
        {
            Ok(children) => children,
            Err(e) => {
                debug!(server, dir, error = %e.message, "directory/read failed; partial walk");
                continue;
            }
        };
        for child in children {
            if child.mime_type.as_deref() == Some("inode/directory") {
                queue.push(child.uri.clone());
            } else {
                files.push(child.uri.clone());
            }
        }
    }
    files
}

/// A failed MCP skill-content load, split by whether the failure is a
/// SEP verification failure (digest mismatch, unlisted read, frontmatter
/// discrepancy) — those are recoverable via a `skills/get` refresh — or an
/// ordinary read error.
enum LoadFailure {
    Verification(String),
    Other(String),
}

impl LoadFailure {
    fn message(&self) -> &str {
        match self {
            LoadFailure::Verification(m) | LoadFailure::Other(m) => m,
        }
    }
}

/// Reads and fully verifies an entry's `SKILL.md`, per SEP §Integrity and
/// verification: digest check against the entry's `resources` (when
/// carried) and the mandatory field-by-field frontmatter identity check.
/// Returns the SKILL.md body with its frontmatter block stripped, matching
/// how the FS path frames skill content.
async fn fetch_verified_skill_md(
    mgr: &ExtensionManager,
    session_id: &str,
    entry: &McpSkillEntry,
    cancel: CancellationToken,
) -> Result<String, LoadFailure> {
    let result = mgr
        .read_resource(session_id, &entry.uri, &entry.server, cancel)
        .await
        .map_err(|e| {
            LoadFailure::Other(format!(
                "Failed to read '{}' from '{}': {}",
                entry.uri, entry.server, e.message
            ))
        })?;

    let text = first_text_content(result, &entry.server, &entry.uri).ok_or_else(|| {
        LoadFailure::Other(format!(
            "Resource '{}' from '{}' had no text content.",
            entry.uri, entry.server
        ))
    })?;

    entry
        .verify_read(&entry.uri, text.as_bytes())
        .map_err(LoadFailure::Verification)?;

    // SEP: after fetching a SKILL.md for which the host holds an entry,
    // parse its frontmatter and compare field-by-field against the entry's
    // `frontmatter`. Any discrepancy — including missing frontmatter — is a
    // verification failure.
    let parsed: Option<(serde_json::Value, String)> =
        crate::sources::parse_frontmatter(&text).unwrap_or_default();
    let Some((fetched_frontmatter, body)) = parsed else {
        return Err(LoadFailure::Verification(format!(
            "SKILL.md at '{}' has no parseable frontmatter to compare against the entry",
            entry.uri
        )));
    };
    entry
        .verify_frontmatter(&fetched_frontmatter)
        .map_err(LoadFailure::Verification)?;

    Ok(body)
}

/// Frames a loaded SKILL.md body the same way the FS path does: a
/// `# Loaded Skill:` header with an MCP origin tag, an optional
/// "Supporting Files" pointer block, and the standard footer.
fn frame_skill_md(entry: &McpSkillEntry, body: &str, supporting: &[String]) -> String {
    let mut output = format!(
        "# Loaded Skill: {} (mcp skill from {})\n\n{}\n",
        entry.name, entry.server, body
    );
    if !supporting.is_empty() {
        output.push_str(&format!(
            "\n## Supporting Files\n\nSkill base: {}\n\n",
            entry.skill_root_uri()
        ));
        for rel in supporting {
            output.push_str(&format!(
                "- {} → load_skill(name: \"{}/{}\")\n",
                rel, entry.name, rel
            ));
        }
    }
    output.push_str("\n---\nThis knowledge is now available in your context.");
    output
}

/// Loads an MCP skill's `SKILL.md`, verifies it against the entry, and
/// frames it. On a verification failure — the content changed since the
/// entry was fetched — recovers per the SEP by fetching a fresh entry via
/// `skills/get` and retrying once against it.
async fn load_mcp_skill_md(
    mgr: &ExtensionManager,
    session_id: &str,
    entry: &McpSkillEntry,
    cancel: CancellationToken,
) -> CallToolResult {
    let first_failure = match fetch_verified_skill_md(mgr, session_id, entry, cancel.clone()).await
    {
        Ok(body) => {
            let supporting =
                enumerate_mcp_supporting_resources(mgr, session_id, entry, cancel).await;
            return CallToolResult::success(vec![ContentBlock::text(frame_skill_md(
                entry,
                &body,
                &supporting,
            ))]);
        }
        Err(failure) => failure,
    };

    if let LoadFailure::Verification(reason) = &first_failure {
        debug!(
            server = %entry.server,
            skill = %entry.name,
            reason,
            "verification failed; refreshing entry via skills/get"
        );
        if let Ok(fresh) = mgr
            .skills_get_for_server(session_id, &entry.uri, &entry.server, cancel.clone())
            .await
        {
            mgr.remember_skill_entry(&entry.server, fresh.clone()).await;
            match fetch_verified_skill_md(mgr, session_id, &fresh, cancel.clone()).await {
                Ok(body) => {
                    let supporting =
                        enumerate_mcp_supporting_resources(mgr, session_id, &fresh, cancel).await;
                    return CallToolResult::success(vec![ContentBlock::text(frame_skill_md(
                        &fresh,
                        &body,
                        &supporting,
                    ))]);
                }
                Err(retry_failure) => {
                    return CallToolResult::error(vec![ContentBlock::text(format!(
                        "Refusing to load skill '{}': {}",
                        entry.name,
                        retry_failure.message()
                    ))]);
                }
            }
        }
    }

    CallToolResult::error(vec![ContentBlock::text(format!(
        "Refusing to load skill '{}': {}",
        entry.name,
        first_failure.message()
    ))])
}

/// Loads a supporting file (`<skill>/<rel>`) for an MCP skill and frames it
/// with the `# Loaded:` header. The file URI is composed against the skill
/// root and, when the entry carries `resources`, the read is verified
/// against it — an unlisted file within a held skill is a verification
/// failure per the SEP.
async fn load_mcp_supporting(
    mgr: &ExtensionManager,
    session_id: &str,
    entry: &McpSkillEntry,
    rel: &str,
    cancel: CancellationToken,
) -> CallToolResult {
    let composed = entry.resolve_relative(rel);

    if let Err(e) = entry.verify_read_uri_listed(&composed) {
        return CallToolResult::error(vec![ContentBlock::text(format!(
            "Refusing to load '{}/{}': {}",
            entry.name, rel, e
        ))]);
    }

    match mgr
        .read_resource(session_id, &composed, &entry.server, cancel)
        .await
    {
        Ok(result) => match first_text_content(result, &entry.server, &composed) {
            Some(body) => match entry.verify_read(&composed, body.as_bytes()) {
                Ok(()) => CallToolResult::success(vec![ContentBlock::text(format!(
                    "# Loaded: {}/{}\n\n{}\n\n---\nFile loaded into context.",
                    entry.name, rel, body
                ))]),
                Err(e) => CallToolResult::error(vec![ContentBlock::text(format!(
                    "Refusing to load '{}/{}': {} (reload the skill to refresh its entry)",
                    entry.name, rel, e
                ))]),
            },
            None => CallToolResult::error(vec![ContentBlock::text(format!(
                "Resource '{}' from '{}' had no text content.",
                composed, entry.server
            ))]),
        },
        Err(e) => CallToolResult::error(vec![ContentBlock::text(format!(
            "Failed to read '{}' from '{}': {}",
            composed, entry.server, e.message
        ))]),
    }
}

#[async_trait]
impl McpClientTrait for SkillsClient {
    async fn list_tools(
        &self,
        _session_id: &str,
        _next_cursor: Option<String>,
        _cancellation_token: CancellationToken,
    ) -> Result<ListToolsResult, Error> {
        let schema = serde_json::json!({
            "type": "object",
            "required": ["name"],
            "properties": {
                "name": {
                    "type": "string",
                    "description": "Name of the skill to load. Use \"skill-name/path\" to load a supporting file. For MCP skills with a name collision, use the \"<server>__<name>\" form shown in your system instructions. A full skill URI ending in /SKILL.md (e.g. skill://git-workflow/SKILL.md) is also accepted."
                },
                "args": {
                    "type": "string",
                    "description": "Optional arguments to provide when loading the skill."
                },
                "server": {
                    "type": "string",
                    "description": "Optional MCP server name to resolve a skill URI against when more than one connected server could serve it."
                }
            }
        });

        let tool = Tool::new(
            "load_skill",
            "Load a skill's full content into your context so you can follow its instructions.\n\n\
             Skills are listed in your system instructions (both local skills and skills from connected MCP servers). When you need to use one, load it first to get the detailed instructions.\n\n\
             Examples:\n\
             - load_skill(name: \"gdrive\") → Loads the gdrive skill instructions\n\
             - load_skill(name: \"my-skill\", args: \"the arguments for the skill\") → Loads a skill with arguments\n\
             - load_skill(name: \"my-skill/template.md\") → Loads a supporting file\n\
             - load_skill(name: \"github__pull-requests\") → Disambiguates a collision between two servers\n\
             - load_skill(name: \"skill://git-workflow/SKILL.md\") → Loads an MCP skill by URI\n\n\
             Use read_resource (from the extensionmanager) for non-skill resource URIs. Do NOT pass skill URIs to file-reading, writing, editing, or shell tools — those operate on filesystem paths."
                .to_string(),
            schema.as_object().unwrap().clone(),
        );

        Ok(ListToolsResult {
            tools: vec![tool],
            next_cursor: None,
            meta: None,
            ..Default::default()
        })
    }

    async fn call_tool(
        &self,
        ctx: &ToolCallContext,
        name: &str,
        arguments: Option<JsonObject>,
        cancellation_token: CancellationToken,
    ) -> Result<CallToolResult, Error> {
        if name != "load_skill" {
            return Ok(CallToolResult::error(vec![ContentBlock::text(format!(
                "Unknown tool: {}",
                name
            ))]));
        }

        let skill_name = arguments
            .as_ref()
            .and_then(|args| args.get("name"))
            .and_then(|v| v.as_str())
            .unwrap_or("");

        if skill_name.is_empty() {
            return Ok(CallToolResult::error(vec![ContentBlock::text(
                "Missing required parameter: name",
            )]));
        }
        let args = arguments
            .as_ref()
            .and_then(|args| args.get("args"))
            .and_then(|v| v.as_str());
        let server_arg = arguments
            .as_ref()
            .and_then(|args| args.get("server"))
            .and_then(|v| v.as_str());

        // Skill URIs get their own resolution path (SEP: hosts MUST support
        // loading a skill given only its URI). Non-SKILL.md URIs are still
        // redirected to `read_resource` — `load_skill` loads skills, not
        // arbitrary resources.
        if crate::agents::platform_extensions::looks_like_uri(skill_name) {
            if !skill_name.ends_with("/SKILL.md") {
                return Ok(CallToolResult::error(vec![ContentBlock::text(format!(
                    "'{}' is a resource URI but not a skill's SKILL.md. Use the read_resource tool (it takes a server name and a uri) for arbitrary resources.",
                    skill_name
                ))]));
            }
            return Ok(self
                .load_skill_by_uri(&ctx.session_id, skill_name, server_arg, cancellation_token)
                .await);
        }

        let skills = self.discover_skills();

        if let Some(skill) = skills.iter().find(|s| s.name == skill_name) {
            return match loaded_skill_context_with_args(skill, args) {
                Ok(rendered) => Ok(CallToolResult::success(vec![ContentBlock::text(rendered)])),
                Err(e) => Ok(CallToolResult::error(vec![ContentBlock::text(format!(
                    "Failed to parse skill arguments: {}",
                    e
                ))])),
            };
        }

        if let Some((parent_skill_name, raw_relative_path)) = skill_name.split_once('/') {
            let relative_path = raw_relative_path.replace('\\', "/");
            if let Some(skill) = skills.iter().find(|s| {
                s.name == parent_skill_name
                    && matches!(s.source_type, SourceType::Skill | SourceType::BuiltinSkill)
            }) {
                let listed_skill_dir = PathBuf::from(&skill.path);
                let load_skill_dir = match listed_skill_dir.canonicalize() {
                    Ok(path) => path,
                    Err(e) => {
                        return Ok(CallToolResult::error(vec![ContentBlock::text(format!(
                            "Failed to resolve '{}': {}",
                            parent_skill_name, e
                        ))]));
                    }
                };

                for file_path in &skill.supporting_files {
                    let file_path_buf = Path::new(file_path);
                    let Ok(rel) = file_path_buf.strip_prefix(&listed_skill_dir) else {
                        continue;
                    };
                    if rel.to_string_lossy().replace('\\', "/") != relative_path {
                        continue;
                    }

                    let result = match super::load_supporting_file(&load_skill_dir, rel, skill_name)
                    {
                        Ok(content) => CallToolResult::success(vec![ContentBlock::text(content)]),
                        Err(e) => CallToolResult::error(vec![ContentBlock::text(format!(
                            "Failed to read '{}': {}",
                            skill_name, e
                        ))]),
                    };
                    return Ok(result);
                }

                let available: Vec<String> = skill
                    .supporting_files
                    .iter()
                    .filter_map(|f| {
                        Path::new(f)
                            .strip_prefix(&listed_skill_dir)
                            .ok()
                            .map(|r| r.to_string_lossy().replace('\\', "/"))
                    })
                    .take(10)
                    .collect();

                return Ok(if available.is_empty() {
                    CallToolResult::error(vec![ContentBlock::text(format!(
                        "Skill '{}' has no supporting files.",
                        skill.name
                    ))])
                } else {
                    CallToolResult::error(vec![ContentBlock::text(format!(
                        "File '{}' not found. Available: {}",
                        skill_name,
                        available.join(", ")
                    ))])
                });
            }
        }

        // MCP skill routing. Read the cache populated at extension-connect
        // time. `<server>__<name>` disambiguation is supported alongside
        // bare names.
        let mcp_skills = self.mcp_skills().await;
        let mgr = self.upgraded_manager();

        if let Some(entry) = find_mcp_by_name(&mcp_skills, skill_name) {
            if let Some(ref mgr) = mgr {
                return Ok(load_mcp_skill_md(
                    mgr.as_ref(),
                    &ctx.session_id,
                    entry,
                    cancellation_token.clone(),
                )
                .await);
            }
        }

        if let Some((parent, raw_rel)) = skill_name.split_once('/') {
            if let Some(entry) = find_mcp_by_name(&mcp_skills, parent) {
                if let Some(ref mgr) = mgr {
                    let Some(rel) = sanitize_relative_ref(raw_rel) else {
                        return Ok(CallToolResult::error(vec![ContentBlock::text(format!(
                            "Refusing to load '{}': relative path must not contain '..' or start with '/'.",
                            skill_name
                        ))]));
                    };
                    return Ok(load_mcp_supporting(
                        mgr.as_ref(),
                        &ctx.session_id,
                        entry,
                        &rel,
                        cancellation_token.clone(),
                    )
                    .await);
                }
            }
        }

        let mut candidates: Vec<&str> = skills
            .iter()
            .filter(|s| {
                s.name.to_lowercase().contains(&skill_name.to_lowercase())
                    || skill_name.to_lowercase().contains(&s.name.to_lowercase())
            })
            .map(|s| s.name.as_str())
            .collect();
        candidates.extend(mcp_skills.iter().filter_map(|e| {
            if e.name.to_lowercase().contains(&skill_name.to_lowercase())
                || skill_name.to_lowercase().contains(&e.name.to_lowercase())
            {
                Some(e.name.as_str())
            } else {
                None
            }
        }));
        candidates.sort();
        candidates.dedup();
        candidates.truncate(3);

        Ok(if candidates.is_empty() {
            CallToolResult::error(vec![ContentBlock::text(format!(
                "Skill '{}' not found.",
                skill_name
            ))])
        } else {
            CallToolResult::error(vec![ContentBlock::text(format!(
                "Skill '{}' not found. Did you mean: {}?",
                skill_name,
                candidates.join(", ")
            ))])
        })
    }

    fn get_info(&self) -> Option<&InitializeResult> {
        Some(&self.info)
    }

    fn get_instructions(&self) -> Option<String> {
        let sources = self.discover_skills();
        let mut skills: Vec<&SourceEntry> = sources
            .iter()
            .filter(|s| {
                s.source_type == SourceType::Skill || s.source_type == SourceType::BuiltinSkill
            })
            .collect();
        skills.sort_by(|a, b| (&a.name, &a.path).cmp(&(&b.name, &b.path)));

        if skills.is_empty() {
            return None;
        }

        let mut instructions = String::from(
            "\n\nYou have these skills at your disposal, when it is clear they can help you solve a problem or you are asked to use them:",
        );
        for skill in &skills {
            instructions.push_str(&format!("\n• {} - {}", skill.name, skill.description));
        }
        Some(instructions)
    }

    async fn subscribe(&self) -> mpsc::Receiver<ServerNotification> {
        let (_tx, rx) = mpsc::channel(1);
        rx
    }

    async fn get_dynamic_instructions(&self, _session_id: &str) -> Option<String> {
        let mcp = self.injectable_mcp_skills().await;
        if mcp.is_empty() {
            return None;
        }
        let fs_names = self.fs_skill_names_cached();
        let collisions = collision_names(&fs_names, &mcp);
        Some(format_mcp_skills_section(&collisions, &mcp))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::Config;
    use std::collections::HashMap;
    use std::fs;
    use std::sync::Arc;
    use tempfile::TempDir;

    fn write_plugin_skill(
        project: &Path,
        plugin_name: &str,
        skill_name: &str,
        description: &str,
        body: &str,
    ) {
        let skill_dir = project
            .join(".agents/plugins")
            .join(plugin_name)
            .join("skills")
            .join(skill_name);
        fs::create_dir_all(&skill_dir).unwrap();
        fs::write(
            skill_dir.join("SKILL.md"),
            format!("---\nname: {skill_name}\ndescription: {description}\n---\n{body}"),
        )
        .unwrap();
    }

    fn write_open_plugin_manifest(project: &Path, plugin_name: &str) {
        let plugin_dir = project.join(".agents/plugins").join(plugin_name);
        fs::write(
            plugin_dir.join("plugin.json"),
            format!(
                r#"{{"name":"{plugin_name}","skills":{{"paths":["./skills","./custom-skills"]}}}}"#
            ),
        )
        .unwrap();
    }

    fn test_client(project: &Path, plugin_name: &str, enabled: bool) -> SkillsClient {
        let config = Box::leak(Box::new(
            Config::new(project.join("test-config.yaml"), "goose-skills-test").unwrap(),
        ));
        let plugin_root = project.join(".agents/plugins").join(plugin_name);
        config
            .set_param(
                "plugins",
                HashMap::from([(
                    plugin_root.to_string_lossy().into_owned(),
                    HashMap::from([("enabled", enabled)]),
                )]),
            )
            .unwrap();
        let session = Arc::new(crate::session::Session {
            working_dir: project.to_path_buf(),
            ..crate::session::Session::default()
        });
        SkillsClient::new(PlatformExtensionContext {
            extension_manager: None,
            session_manager: Arc::new(crate::session::SessionManager::instance()),
            scheduler: None,
            session: Some(session),
            use_login_shell_path: false,
        })
        .unwrap()
        .with_builtin_skills(false)
        .with_config(config)
    }

    fn result_text(result: &CallToolResult) -> &str {
        match &result.content[0] {
            ContentBlock::Text(text) => &text.text,
            _ => panic!("expected text"),
        }
    }

    #[tokio::test]
    async fn disabled_plugin_skill_is_not_listed_or_loadable() {
        let _guard = env_lock::lock_env([("PLUGINS", None::<&str>)]);
        let project = TempDir::new().unwrap();
        write_plugin_skill(
            project.path(),
            "disabled-plugin",
            "disabled-plugin-skill",
            "Disabled plugin metadata",
            "disabled plugin full body",
        );
        let client = test_client(project.path(), "disabled-plugin", false);

        assert!(client
            .get_instructions()
            .is_none_or(|instructions| !instructions.contains("disabled-plugin-skill")));

        let ctx = ToolCallContext::new("test".to_string(), None, None);
        let args = serde_json::from_value(serde_json::json!({
            "name": "disabled-plugin-skill"
        }))
        .unwrap();
        let result = client
            .call_tool(&ctx, "load_skill", Some(args), CancellationToken::new())
            .await
            .unwrap();

        assert!(result.is_error.unwrap_or(false));
        assert!(!result_text(&result).contains("disabled plugin full body"));
    }

    #[tokio::test]
    async fn enabled_plugin_skill_is_listed_and_loadable() {
        let _guard = env_lock::lock_env([("PLUGINS", None::<&str>)]);
        let project = TempDir::new().unwrap();
        write_plugin_skill(
            project.path(),
            "enabled-plugin",
            "enabled-plugin-skill",
            "Enabled plugin metadata",
            "enabled plugin full body",
        );
        let custom_skill_dir = project
            .path()
            .join(".agents/plugins/enabled-plugin/custom-skills/custom-plugin-skill");
        fs::create_dir_all(&custom_skill_dir).unwrap();
        fs::write(
            custom_skill_dir.join("SKILL.md"),
            "---\nname: custom-plugin-skill\ndescription: Custom plugin metadata\n---\ncustom plugin full body",
        )
        .unwrap();
        write_open_plugin_manifest(project.path(), "enabled-plugin");
        let client = test_client(project.path(), "enabled-plugin", true);

        let instructions = client.get_instructions().unwrap();
        assert!(instructions.contains("enabled-plugin-skill"));
        assert!(instructions.contains("Enabled plugin metadata"));
        assert!(instructions.contains("custom-plugin-skill"));
        assert!(instructions.contains("Custom plugin metadata"));

        let ctx = ToolCallContext::new("test".to_string(), None, None);
        let args = serde_json::from_value(serde_json::json!({
            "name": "custom-plugin-skill"
        }))
        .unwrap();
        let result = client
            .call_tool(&ctx, "load_skill", Some(args), CancellationToken::new())
            .await
            .unwrap();

        assert!(!result.is_error.unwrap_or(false));
        assert!(result_text(&result).contains("custom plugin full body"));
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn symlinked_project_plugin_supporting_file_is_loadable() {
        let _guard = env_lock::lock_env([("PLUGINS", None::<&str>)]);
        let project = TempDir::new().unwrap();
        let external = TempDir::new().unwrap();
        write_plugin_skill(
            external.path(),
            "symlinked-plugin",
            "symlinked-skill",
            "Symlinked skill metadata",
            "symlinked skill body",
        );
        write_open_plugin_manifest(external.path(), "symlinked-plugin");
        let external_plugin = external.path().join(".agents/plugins/symlinked-plugin");
        let supporting_file = external_plugin.join("skills/symlinked-skill/guide.md");
        fs::write(&supporting_file, "Symlinked supporting guidance.").unwrap();

        let plugin_link = project.path().join(".agents/plugins/symlinked-plugin");
        fs::create_dir_all(plugin_link.parent().unwrap()).unwrap();
        std::os::unix::fs::symlink(&external_plugin, &plugin_link).unwrap();
        let client = test_client(project.path(), "symlinked-plugin", true);

        let ctx = ToolCallContext::new("test".to_string(), None, None);
        let args = serde_json::from_value(serde_json::json!({
            "name": "symlinked-skill/guide.md"
        }))
        .unwrap();
        let result = client
            .call_tool(&ctx, "load_skill", Some(args), CancellationToken::new())
            .await
            .unwrap();

        assert!(!result.is_error.unwrap_or(false));
        assert!(result_text(&result).contains("Symlinked supporting guidance."));
    }

    #[tokio::test]
    async fn test_load_filesystem_skill_without_builtin_skills() {
        let temp_dir = TempDir::new().unwrap();
        let skill_dir = temp_dir.path().join(".goose/skills/my-skill");
        fs::create_dir_all(&skill_dir).unwrap();
        fs::write(
            skill_dir.join("SKILL.md"),
            "---\nname: my-skill\ndescription: A test skill\n---\nDo the thing.",
        )
        .unwrap();
        fs::create_dir(skill_dir.join("nested")).unwrap();
        fs::write(skill_dir.join("nested/guide.md"), "Nested guidance.").unwrap();

        let session = std::sync::Arc::new(crate::session::Session {
            working_dir: temp_dir.path().to_path_buf(),
            ..crate::session::Session::default()
        });
        let client = SkillsClient::new(PlatformExtensionContext {
            extension_manager: None,
            session_manager: Arc::new(crate::session::SessionManager::instance()),
            scheduler: None,
            session: Some(session),
            use_login_shell_path: false,
        })
        .unwrap()
        .with_builtin_skills(false);

        assert!(client
            .discover_skills()
            .iter()
            .all(|skill| skill.source_type != SourceType::BuiltinSkill));

        let ctx = ToolCallContext::new("test".to_string(), None, None);
        let args: JsonObject =
            serde_json::from_value(serde_json::json!({"name": "my-skill"})).unwrap();
        let result = client
            .call_tool(&ctx, "load_skill", Some(args), CancellationToken::new())
            .await
            .unwrap();

        assert!(!result.is_error.unwrap_or(false));
        let text = text_of(&result);
        assert!(text.contains("my-skill"));
        assert!(text.contains("Do the thing"));

        let args: JsonObject =
            serde_json::from_value(serde_json::json!({"name": "my-skill/nested/guide.md"}))
                .unwrap();
        let result = client
            .call_tool(&ctx, "load_skill", Some(args), CancellationToken::new())
            .await
            .unwrap();

        assert!(!result.is_error.unwrap_or(false));
        let text = match &result.content[0] {
            rmcp::model::ContentBlock::Text(t) => &t.text,
            _ => panic!("expected text"),
        };
        assert!(text.contains("Nested guidance."));
    }

    #[tokio::test]
    async fn test_load_skill_not_found_returns_error() {
        let client = SkillsClient::new(PlatformExtensionContext {
            extension_manager: None,
            session_manager: Arc::new(crate::session::SessionManager::instance()),
            scheduler: None,
            session: None,
            use_login_shell_path: false,
        })
        .unwrap();

        let ctx = ToolCallContext::new("test".to_string(), None, None);
        let args: JsonObject =
            serde_json::from_value(serde_json::json!({"name": "nonexistent"})).unwrap();
        let result = client
            .call_tool(&ctx, "load_skill", Some(args), CancellationToken::new())
            .await
            .unwrap();

        assert!(result.is_error.unwrap_or(false));
    }

    // ---------- MCP skill routing tests ----------

    use crate::agents::extension::ExtensionConfig;
    use crate::agents::extension_manager::ExtensionManager;
    use crate::skills::mcp_client::{SkillsGetResult, SkillsListResult, SKILLS_EXTENSION_ID};
    use async_trait::async_trait;
    use rmcp::model::{
        ExtensionCapabilities, ListResourcesResult, ReadResourceResult, Resource,
        ServerNotification,
    };

    /// Directory listing fixture: dir URI -> children as (uri, mimeType).
    type DirMap = HashMap<String, Vec<(String, Option<String>)>>;

    struct FakeMcp {
        info: InitializeResult,
        resources: std::sync::Mutex<HashMap<String, String>>,
        /// `skills/list` result document, swappable for refresh tests.
        list_doc: std::sync::Mutex<serde_json::Value>,
        /// Extra entries served only via `skills/get` (unlisted skills).
        get_entries: HashMap<String, serde_json::Value>,
        /// `resources/directory/read` fixture; non-empty implies the server
        /// declares `directoryRead: true`.
        directories: DirMap,
        subscribers: tokio::sync::Mutex<Vec<mpsc::Sender<ServerNotification>>>,
    }

    impl FakeMcp {
        fn build_info(directory_read: bool) -> InitializeResult {
            let mut caps = ExtensionCapabilities::new();
            let mut cfg = JsonObject::new();
            if directory_read {
                cfg.insert("directoryRead".to_string(), serde_json::json!(true));
            }
            caps.insert(SKILLS_EXTENSION_ID.to_string(), cfg);
            InitializeResult::new(
                ServerCapabilities::builder()
                    .enable_resources()
                    .enable_extensions_with(caps)
                    .build(),
            )
        }

        fn new(list_doc: serde_json::Value, resources: HashMap<String, String>) -> Self {
            Self {
                info: Self::build_info(false),
                resources: std::sync::Mutex::new(resources),
                list_doc: std::sync::Mutex::new(list_doc),
                get_entries: HashMap::new(),
                directories: HashMap::new(),
                subscribers: tokio::sync::Mutex::new(Vec::new()),
            }
        }

        fn with_get_entry(mut self, uri: &str, entry: serde_json::Value) -> Self {
            self.get_entries.insert(uri.to_string(), entry);
            self
        }

        fn with_directories(mut self, dirs: DirMap) -> Self {
            self.info = Self::build_info(true);
            self.directories = dirs;
            self
        }

        fn swap_skills(&self, list_doc: serde_json::Value, new_resources: HashMap<String, String>) {
            *self.list_doc.lock().unwrap() = list_doc;
            *self.resources.lock().unwrap() = new_resources;
        }

        async fn notify_resources_list_changed(&self) {
            use rmcp::model::ResourceListChangedNotification;
            let subs = self.subscribers.lock().await;
            for tx in subs.iter() {
                let _ = tx
                    .send(ServerNotification::ResourceListChangedNotification(
                        ResourceListChangedNotification::default(),
                    ))
                    .await;
            }
        }

        async fn wait_for_subscriber(&self, timeout: Duration) -> bool {
            let deadline = std::time::Instant::now() + timeout;
            loop {
                if !self.subscribers.lock().await.is_empty() {
                    return true;
                }
                if std::time::Instant::now() >= deadline {
                    return false;
                }
                tokio::time::sleep(Duration::from_millis(10)).await;
            }
        }
    }

    #[async_trait]
    impl McpClientTrait for FakeMcp {
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
            unreachable!("FakeMcp has no tools")
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
            let resources = self
                .resources
                .lock()
                .unwrap()
                .keys()
                .map(|uri| Resource::new(uri.as_str(), uri.as_str()))
                .collect();
            Ok(ListResourcesResult {
                resources,
                next_cursor: None,
                meta: None,
                ..Default::default()
            })
        }

        async fn read_resource(
            &self,
            _session_id: &str,
            uri: &str,
            _cancel_token: CancellationToken,
        ) -> Result<ReadResourceResult, Error> {
            match self.resources.lock().unwrap().get(uri).cloned() {
                Some(text) => Ok(ReadResourceResult::new(vec![
                    ResourceContents::TextResourceContents {
                        uri: uri.to_string(),
                        mime_type: None,
                        text,
                        meta: None,
                    },
                ])),
                None => Err(Error::TransportClosed),
            }
        }

        async fn skills_list(
            &self,
            _session_id: &str,
            _cursor: Option<String>,
            _cancel_token: CancellationToken,
        ) -> Result<SkillsListResult, Error> {
            let doc = self.list_doc.lock().unwrap().clone();
            serde_json::from_value(doc).map_err(|_| Error::UnexpectedResponse)
        }

        async fn skills_get(
            &self,
            _session_id: &str,
            uri: &str,
            _cancel_token: CancellationToken,
        ) -> Result<SkillsGetResult, Error> {
            let entry = self.get_entries.get(uri).cloned().or_else(|| {
                self.list_doc.lock().unwrap()["skills"]
                    .as_array()
                    .and_then(|skills| {
                        skills
                            .iter()
                            .find(|e| e["uri"].as_str() == Some(uri))
                            .cloned()
                    })
            });
            match entry {
                Some(doc) => Ok(SkillsGetResult {
                    result_type: None,
                    skill: doc,
                }),
                None => Err(Error::TransportClosed),
            }
        }

        async fn directory_read(
            &self,
            _session_id: &str,
            uri: &str,
            _cursor: Option<String>,
            _cancel_token: CancellationToken,
        ) -> Result<ListResourcesResult, Error> {
            let children = self.directories.get(uri).ok_or(Error::TransportClosed)?;
            let resources = children
                .iter()
                .map(|(child_uri, mime)| {
                    let mut resource = Resource::new(child_uri.as_str(), child_uri.as_str());
                    if let Some(mime) = mime {
                        resource = resource.with_mime_type(mime.clone());
                    }
                    resource
                })
                .collect();
            Ok(ListResourcesResult {
                resources,
                next_cursor: None,
                meta: None,
                ..Default::default()
            })
        }

        async fn subscribe(&self) -> mpsc::Receiver<ServerNotification> {
            let (tx, rx) = mpsc::channel(16);
            self.subscribers.lock().await.push(tx);
            rx
        }
    }

    fn sha256_digest(bytes: &[u8]) -> String {
        use sha2::{Digest, Sha256};
        let mut h = Sha256::new();
        h.update(bytes);
        format!("sha256:{}", crate::utils::bytes_to_hex(h.finalize()))
    }

    fn skill_md(name: &str, description: &str, body: &str) -> String {
        // Description is quoted so an empty string round-trips as "" rather
        // than YAML null — the frontmatter identity check is exact.
        format!(
            "---\nname: {}\ndescription: '{}'\n---\n{}",
            name, description, body
        )
    }

    /// Builds a wire entry whose `resources` digests match the given file
    /// bodies. `files` maps each resource URI (including the SKILL.md URI)
    /// to its content.
    fn wire_entry(
        name: &str,
        description: &str,
        uri: &str,
        files: &[(&str, &str)],
    ) -> serde_json::Value {
        let resources: Vec<serde_json::Value> = files
            .iter()
            .map(|(file_uri, body)| {
                serde_json::json!({"uri": file_uri, "digest": sha256_digest(body.as_bytes()), "size": body.len()})
            })
            .collect();
        serde_json::json!({
            "uri": uri,
            "frontmatter": {"name": name, "description": description},
            "resources": resources,
        })
    }

    /// One-skill fixture: returns the `skills/list` doc plus the resource
    /// map serving the skill's files.
    fn one_skill_fixture(
        name: &str,
        description: &str,
        body: &str,
        supporting: &[(&str, &str)],
    ) -> (serde_json::Value, HashMap<String, String>) {
        let uri = format!("skill://{}/SKILL.md", name);
        let md = skill_md(name, description, body);
        let mut files: Vec<(String, String)> = vec![(uri.clone(), md)];
        for (rel, content) in supporting {
            files.push((format!("skill://{}/{}", name, rel), content.to_string()));
        }
        let file_refs: Vec<(&str, &str)> = files
            .iter()
            .map(|(u, c)| (u.as_str(), c.as_str()))
            .collect();
        let doc = serde_json::json!({"skills": [wire_entry(name, description, &uri, &file_refs)]});
        (doc, files.into_iter().collect())
    }

    async fn register_built_fake(mgr: &Arc<ExtensionManager>, server_name: &str, fake: FakeMcp) {
        let fake: Arc<dyn McpClientTrait> = Arc::new(fake);
        mgr.add_client(
            server_name.to_string(),
            ExtensionConfig::Builtin {
                name: server_name.to_string(),
                display_name: Some(server_name.to_string()),
                description: "fake mcp".to_string(),
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
        mgr.set_skills_enabled(server_name, true).await;
    }

    async fn setup_client_with_built(
        server_name: &str,
        fake: FakeMcp,
        working_dir: PathBuf,
    ) -> (SkillsClient, Arc<ExtensionManager>, TempDir) {
        let tmp = TempDir::new().unwrap();
        let mgr = Arc::new(ExtensionManager::new_without_provider(
            tmp.path().to_path_buf(),
        ));
        register_built_fake(&mgr, server_name, fake).await;

        let session = Arc::new(crate::session::Session {
            working_dir: working_dir.clone(),
            ..crate::session::Session::default()
        });
        let client = SkillsClient::new(PlatformExtensionContext {
            extension_manager: Some(Arc::downgrade(&mgr)),
            session_manager: Arc::new(crate::session::SessionManager::instance()),
            scheduler: None,
            session: Some(session),
            use_login_shell_path: false,
        })
        .unwrap();
        (client, mgr, tmp)
    }

    fn text_of(r: &CallToolResult) -> String {
        match &r.content[0] {
            ContentBlock::Text(t) => t.text.clone(),
            _ => panic!("expected text content"),
        }
    }

    async fn call_load(client: &SkillsClient, name: &str) -> CallToolResult {
        let ctx = ToolCallContext::new("s".to_string(), None, None);
        client
            .call_tool(
                &ctx,
                "load_skill",
                Some(serde_json::from_value(serde_json::json!({"name": name})).unwrap()),
                CancellationToken::new(),
            )
            .await
            .unwrap()
    }

    #[tokio::test]
    async fn test_load_mcp_skill_basic() {
        let tmp = TempDir::new().unwrap();
        let (doc, resources) = one_skill_fixture("git-workflow", "Git", "Git body text.", &[]);
        let (client, _mgr, _g) =
            setup_client_with_built("gh", FakeMcp::new(doc, resources), tmp.path().to_path_buf())
                .await;

        let result = call_load(&client, "git-workflow").await;

        assert!(
            !result.is_error.unwrap_or(false),
            "got: {}",
            text_of(&result)
        );
        let body = text_of(&result);
        assert!(body.contains("Git body text"), "got: {}", body);
        assert!(body.contains("mcp skill from gh"), "got: {}", body);
    }

    #[tokio::test]
    async fn test_load_mcp_skill_rejects_digest_mismatch() {
        let tmp = TempDir::new().unwrap();
        let (doc, mut resources) = one_skill_fixture("tampered", "t", "original body", &[]);
        // Server serves different bytes than the entry's digest promises,
        // and skills/get (same stale doc) doesn't fix it.
        resources.insert(
            "skill://tampered/SKILL.md".to_string(),
            skill_md("tampered", "t", "SHOULD NOT SURFACE"),
        );
        let (client, _mgr, _g) =
            setup_client_with_built("gh", FakeMcp::new(doc, resources), tmp.path().to_path_buf())
                .await;

        let result = call_load(&client, "tampered").await;

        assert!(result.is_error.unwrap_or(false));
        let body = text_of(&result);
        assert!(
            body.contains("digest mismatch") || body.contains("Refusing"),
            "got: {}",
            body
        );
        assert!(!body.contains("SHOULD NOT SURFACE"));
    }

    #[tokio::test]
    async fn test_load_mcp_skill_rejects_frontmatter_mismatch() {
        let tmp = TempDir::new().unwrap();
        // The served SKILL.md's frontmatter carries an extra field the
        // entry's frontmatter doesn't — identity check must fail even
        // though we digest the served bytes correctly.
        let md = "---\nname: sneaky\ndescription: d\nallowed-tools: shell\n---\nbody";
        let uri = "skill://sneaky/SKILL.md";
        let doc = serde_json::json!({"skills": [{
            "uri": uri,
            "frontmatter": {"name": "sneaky", "description": "d"},
            "resources": [{"uri": uri, "digest": sha256_digest(md.as_bytes()), "size": md.len()}],
        }]});
        let resources = HashMap::from([(uri.to_string(), md.to_string())]);
        let (client, _mgr, _g) =
            setup_client_with_built("gh", FakeMcp::new(doc, resources), tmp.path().to_path_buf())
                .await;

        let result = call_load(&client, "sneaky").await;

        assert!(
            result.is_error.unwrap_or(false),
            "got: {}",
            text_of(&result)
        );
        assert!(
            text_of(&result).contains("frontmatter"),
            "got: {}",
            text_of(&result)
        );
    }

    #[tokio::test]
    async fn test_verification_failure_recovers_via_skills_get() {
        let tmp = TempDir::new().unwrap();
        // Cache holds a stale entry (digest of the OLD body); the server now
        // serves a new body AND a fresh matching entry via skills/get. The
        // load must recover through the refresh path.
        let name = "evolving";
        let uri = format!("skill://{}/SKILL.md", name);
        let old_md = skill_md(name, "d", "old body");
        let new_md = skill_md(name, "d", "new body");

        let stale_doc = serde_json::json!({"skills": [
            wire_entry(name, "d", &uri, &[(uri.as_str(), old_md.as_str())])
        ]});
        let fresh_entry = wire_entry(name, "d", &uri, &[(uri.as_str(), new_md.as_str())]);
        let resources = HashMap::from([(uri.clone(), new_md.clone())]);

        let fake = FakeMcp::new(stale_doc.clone(), resources);
        let (client, mgr, _g) = setup_client_with_built("gh", fake, tmp.path().to_path_buf()).await;
        // Overwrite the skills/get answer with the fresh entry: swap the
        // list doc so skills/get resolves to the fresh digests, while the
        // manager cache still holds the stale entry from registration.
        {
            let skills = mgr.aggregated_mcp_skills().await;
            assert_eq!(skills.len(), 1, "stale entry should be cached");
        }
        // Re-register the fake's list doc via a fresh FakeMcp is not
        // possible here; instead serve the fresh entry through get_entries
        // on a second server-less path: simplest is to re-add the client
        // with an updated fake.
        let fake2 = FakeMcp::new(
            serde_json::json!({"skills": [fresh_entry.clone()]}),
            HashMap::from([(uri.clone(), new_md.clone())]),
        );
        // Replace the extension: same name, new fake. The cache repopulates
        // from the new list doc, but to exercise the RECOVERY path we then
        // poison the cache with the stale entry again.
        register_built_fake(&mgr, "gh", fake2).await;
        let stale_entry = {
            let mut e = mgr.aggregated_mcp_skills().await.remove(0);
            e.resources = crate::skills::mcp_client::SkillResources::Manifest(vec![
                crate::skills::mcp_client::SkillResourceRef {
                    uri: uri.clone(),
                    digest: sha256_digest(old_md.as_bytes()),
                    size: old_md.len() as u64,
                },
            ]);
            e
        };
        mgr.remember_skill_entry("gh", stale_entry).await;

        let result = call_load(&client, name).await;

        assert!(
            !result.is_error.unwrap_or(false),
            "expected recovery via skills/get, got: {}",
            text_of(&result)
        );
        assert!(text_of(&result).contains("new body"));
    }

    #[tokio::test]
    async fn test_load_mcp_skill_non_skill_scheme() {
        let tmp = TempDir::new().unwrap();
        let uri = "github://o/r/skills/pull-requests/SKILL.md";
        let md = skill_md("pull-requests", "PRs", "PR review workflow body.");
        let doc = serde_json::json!({"skills": [
            wire_entry("pull-requests", "PRs", uri, &[(uri, md.as_str())])
        ]});
        let resources = HashMap::from([(uri.to_string(), md)]);
        let (client, _mgr, _g) = setup_client_with_built(
            "github",
            FakeMcp::new(doc, resources),
            tmp.path().to_path_buf(),
        )
        .await;

        let result = call_load(&client, "pull-requests").await;

        assert!(
            !result.is_error.unwrap_or(false),
            "got: {}",
            text_of(&result)
        );
        assert!(text_of(&result).contains("PR review workflow body"));
    }

    #[tokio::test]
    async fn test_load_mcp_supporting_file_verified() {
        let tmp = TempDir::new().unwrap();
        let (doc, resources) = one_skill_fixture(
            "docs",
            "D",
            "main body",
            &[("references/GUIDE.md", "Guide body.")],
        );
        let (client, _mgr, _g) = setup_client_with_built(
            "srv",
            FakeMcp::new(doc, resources),
            tmp.path().to_path_buf(),
        )
        .await;

        let result = call_load(&client, "docs/references/GUIDE.md").await;

        assert!(
            !result.is_error.unwrap_or(false),
            "got: {}",
            text_of(&result)
        );
        assert!(text_of(&result).contains("Guide body"));
    }

    #[tokio::test]
    async fn test_load_mcp_supporting_file_rejects_unlisted() {
        let tmp = TempDir::new().unwrap();
        let (doc, mut resources) = one_skill_fixture("docs", "D", "main body", &[]);
        // The server can serve the file, but it is not in the entry's
        // resources — per the SEP that read is a verification failure.
        resources.insert(
            "skill://docs/secret.md".to_string(),
            "SHOULD NOT SURFACE".to_string(),
        );
        let (client, _mgr, _g) = setup_client_with_built(
            "srv",
            FakeMcp::new(doc, resources),
            tmp.path().to_path_buf(),
        )
        .await;

        let result = call_load(&client, "docs/secret.md").await;

        assert!(
            result.is_error.unwrap_or(false),
            "got: {}",
            text_of(&result)
        );
        let body = text_of(&result);
        assert!(body.contains("not listed"), "got: {}", body);
        assert!(!body.contains("SHOULD NOT SURFACE"));
    }

    #[tokio::test]
    async fn test_load_mcp_supporting_file_rejects_parent_traversal() {
        let tmp = TempDir::new().unwrap();
        let (doc, resources) = one_skill_fixture("docs", "", "body", &[]);
        let (client, _mgr, _g) = setup_client_with_built(
            "srv",
            FakeMcp::new(doc, resources),
            tmp.path().to_path_buf(),
        )
        .await;

        for bad in ["docs/../secrets/SKILL.md", "docs//etc/passwd"] {
            let result = call_load(&client, bad).await;
            assert!(
                result.is_error.unwrap_or(false),
                "expected rejection for {bad}, got: {:?}",
                text_of(&result)
            );
        }
    }

    #[tokio::test]
    async fn test_load_skill_by_uri_via_skills_get() {
        let tmp = TempDir::new().unwrap();
        // The skill is NOT in the listing — only reachable via skills/get.
        let uri = "skill://unlisted/SKILL.md";
        let md = skill_md("unlisted", "hidden", "unlisted body");
        let entry = wire_entry("unlisted", "hidden", uri, &[(uri, md.as_str())]);
        let fake = FakeMcp::new(
            serde_json::json!({"skills": []}),
            HashMap::from([(uri.to_string(), md.clone())]),
        )
        .with_get_entry(uri, entry);
        let (client, mgr, _g) =
            setup_client_with_built("srv", fake, tmp.path().to_path_buf()).await;
        assert!(mgr.aggregated_mcp_skills().await.is_empty());

        let result = call_load(&client, uri).await;

        assert!(
            !result.is_error.unwrap_or(false),
            "got: {}",
            text_of(&result)
        );
        assert!(text_of(&result).contains("unlisted body"));
        // The fetched entry is remembered, so a follow-up by-name load and
        // supporting reads resolve from cache.
        assert_eq!(mgr.aggregated_mcp_skills().await.len(), 1);
    }

    #[tokio::test]
    async fn test_load_skill_by_uri_unknown_errors() {
        let tmp = TempDir::new().unwrap();
        let (doc, resources) = one_skill_fixture("known", "", "body", &[]);
        let (client, _mgr, _g) = setup_client_with_built(
            "srv",
            FakeMcp::new(doc, resources),
            tmp.path().to_path_buf(),
        )
        .await;

        let result = call_load(&client, "skill://unknown/SKILL.md").await;
        assert!(result.is_error.unwrap_or(false));
        assert!(
            text_of(&result).contains("skills/get"),
            "got: {}",
            text_of(&result)
        );

        // A URI that is not a SKILL.md is redirected to read_resource.
        let result = call_load(&client, "skill://known/references/GUIDE.md").await;
        assert!(result.is_error.unwrap_or(false));
        assert!(
            text_of(&result).contains("read_resource"),
            "got: {}",
            text_of(&result)
        );
    }

    #[tokio::test]
    async fn test_dynamic_instructions_include_mcp_skills() {
        let tmp = TempDir::new().unwrap();
        let (doc, resources) = one_skill_fixture("alpha", "A", "body", &[]);
        let (client, _mgr, _g) = setup_client_with_built(
            "srv",
            FakeMcp::new(doc, resources),
            tmp.path().to_path_buf(),
        )
        .await;

        let out = client
            .get_dynamic_instructions("s")
            .await
            .expect("should have dynamic output");
        assert!(out.contains("alpha"), "got: {}", out);
        assert!(out.contains("srv"), "got: {}", out);
    }

    async fn setup_client_without_opt_in(
        server_name: &str,
        fake: FakeMcp,
        working_dir: PathBuf,
    ) -> (SkillsClient, Arc<ExtensionManager>) {
        let mgr = Arc::new(ExtensionManager::new_without_provider(working_dir.clone()));
        let fake: Arc<dyn McpClientTrait> = Arc::new(fake);
        mgr.add_client(
            server_name.to_string(),
            ExtensionConfig::Builtin {
                name: server_name.to_string(),
                display_name: Some(server_name.to_string()),
                description: "fake mcp".to_string(),
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

        let session = Arc::new(crate::session::Session {
            working_dir,
            ..crate::session::Session::default()
        });
        let client = SkillsClient::new(PlatformExtensionContext {
            extension_manager: Some(Arc::downgrade(&mgr)),
            session_manager: Arc::new(crate::session::SessionManager::instance()),
            scheduler: None,
            session: Some(session),
            use_login_shell_path: false,
        })
        .unwrap();
        (client, mgr)
    }

    #[tokio::test]
    async fn test_injection_gated_until_opt_in() {
        let tmp = TempDir::new().unwrap();
        let (doc, resources) = one_skill_fixture("alpha", "A", "body", &[]);
        let (client, mgr) = setup_client_without_opt_in(
            "srv",
            FakeMcp::new(doc, resources),
            tmp.path().to_path_buf(),
        )
        .await;

        assert!(!mgr.aggregated_mcp_skills().await.is_empty());
        assert!(mgr.injectable_mcp_skills().await.is_empty());
        assert!(client.get_dynamic_instructions("s").await.is_none());

        mgr.set_skills_enabled("srv", true).await;
        let out = client
            .get_dynamic_instructions("s")
            .await
            .expect("opted-in skill should render");
        assert!(out.contains("alpha"), "got: {}", out);

        mgr.set_skills_enabled("srv", false).await;
        assert!(client.get_dynamic_instructions("s").await.is_none());
        assert!(!mgr.aggregated_mcp_skills().await.is_empty());
    }

    #[tokio::test]
    async fn test_mcp_skill_servers_reports_counts_and_consent_state() {
        let tmp = TempDir::new().unwrap();
        let (doc, resources) = one_skill_fixture("alpha", "A", "body", &[]);
        let (_client, mgr) = setup_client_without_opt_in(
            "srv",
            FakeMcp::new(doc, resources),
            tmp.path().to_path_buf(),
        )
        .await;

        let before = mgr.mcp_skill_servers().await;
        let entry = before.iter().find(|s| s.server == "srv").unwrap();
        assert_eq!(entry.skill_count, 1);
        assert!(!entry.skills_enabled);

        mgr.set_skills_enabled("srv", true).await;
        let after = mgr.mcp_skill_servers().await;
        let entry = after.iter().find(|s| s.server == "srv").unwrap();
        assert!(entry.skills_enabled);
    }

    #[tokio::test]
    async fn test_mcp_vs_mcp_collision_renders_prefixed_names() {
        let tmp = TempDir::new().unwrap();
        let (doc1, r1) = one_skill_fixture("shared", "from one", "one body", &[]);
        let (client, mgr, _g) =
            setup_client_with_built("one", FakeMcp::new(doc1, r1), tmp.path().to_path_buf()).await;

        let (doc2, r2) = one_skill_fixture("shared", "from two", "two body", &[]);
        register_built_fake(&mgr, "two", FakeMcp::new(doc2, r2)).await;

        let out = client
            .get_dynamic_instructions("s")
            .await
            .expect("dynamic output");
        assert!(out.contains("one__shared"), "got:\n{}", out);
        assert!(out.contains("two__shared"), "got:\n{}", out);
        assert!(!out.contains("• shared "), "got:\n{}", out);
    }

    #[tokio::test]
    async fn test_load_skill_resolves_server_prefix() {
        let tmp = TempDir::new().unwrap();
        let (doc1, r1) = one_skill_fixture("shared", "", "body from server one", &[]);
        let (client, mgr, _g) =
            setup_client_with_built("one", FakeMcp::new(doc1, r1), tmp.path().to_path_buf()).await;

        let (doc2, r2) = one_skill_fixture("shared", "", "body from server two", &[]);
        register_built_fake(&mgr, "two", FakeMcp::new(doc2, r2)).await;

        let result = call_load(&client, "two__shared").await;

        assert!(
            !result.is_error.unwrap_or(false),
            "got: {}",
            text_of(&result)
        );
        let body = text_of(&result);
        assert!(body.contains("body from server two"), "got:\n{}", body);
        assert!(!body.contains("body from server one"));
    }

    #[tokio::test]
    async fn test_supporting_files_enumerated_from_entry_resources() {
        let tmp = TempDir::new().unwrap();
        let (doc, mut resources) = one_skill_fixture(
            "docs",
            "D",
            "main skill body",
            &[("references/GUIDE.md", "guide body")],
        );
        // An unrelated resource the server also serves must not show up as
        // a supporting file — enumeration comes from the entry's resources.
        resources.insert(
            "skill://other/SKILL.md".to_string(),
            "unrelated".to_string(),
        );
        let (client, _mgr, _g) = setup_client_with_built(
            "srv",
            FakeMcp::new(doc, resources),
            tmp.path().to_path_buf(),
        )
        .await;

        let result = call_load(&client, "docs").await;

        assert!(
            !result.is_error.unwrap_or(false),
            "got: {}",
            text_of(&result)
        );
        let body = text_of(&result);
        assert!(body.contains("main skill body"), "got:\n{}", body);
        assert!(
            body.contains("references/GUIDE.md → load_skill(name: \"docs/references/GUIDE.md\")"),
            "got:\n{}",
            body
        );
        assert!(!body.contains("guide body"), "got:\n{}", body);
        assert!(!body.contains("skill://other/SKILL.md"), "got:\n{}", body);
    }

    #[tokio::test]
    async fn test_dynamic_skill_supporting_files_via_directory_read() {
        let tmp = TempDir::new().unwrap();
        // A dynamic skill falls back to the directory walk for
        // supporting-file enumeration.
        let uri = "skill://docs/SKILL.md";
        let md = skill_md("docs", "D", "main body");
        let doc = serde_json::json!({"skills": [{
            "uri": uri,
            "frontmatter": {"name": "docs", "description": "D"},
            "resources": "dynamic",
        }]});
        let resources = HashMap::from([
            (uri.to_string(), md),
            (
                "skill://docs/references/GUIDE.md".to_string(),
                "guide".to_string(),
            ),
        ]);

        let mut dirs: DirMap = HashMap::new();
        dirs.insert(
            "skill://docs".to_string(),
            vec![
                (uri.to_string(), Some("text/markdown".to_string())),
                (
                    "skill://docs/references".to_string(),
                    Some("inode/directory".to_string()),
                ),
            ],
        );
        dirs.insert(
            "skill://docs/references".to_string(),
            vec![(
                "skill://docs/references/GUIDE.md".to_string(),
                Some("text/markdown".to_string()),
            )],
        );

        let fake = FakeMcp::new(doc, resources).with_directories(dirs);
        let (client, mgr, _g) =
            setup_client_with_built("srv", fake, tmp.path().to_path_buf()).await;
        assert!(mgr.server_supports_directory_read("srv").await);

        let result = call_load(&client, "docs").await;

        assert!(
            !result.is_error.unwrap_or(false),
            "got: {}",
            text_of(&result)
        );
        let body = text_of(&result);
        assert!(body.contains("main body"), "got:\n{}", body);
        assert!(
            body.contains("references/GUIDE.md → load_skill(name: \"docs/references/GUIDE.md\")"),
            "directory walk should surface nested supporting file; got:\n{}",
            body
        );
    }

    #[tokio::test]
    async fn test_load_skill_framing_parity_fs_vs_mcp() {
        let tmp = TempDir::new().unwrap();

        let fs_skill_dir = tmp.path().join(".goose/skills/fs-demo");
        fs::create_dir_all(&fs_skill_dir).unwrap();
        fs::write(
            fs_skill_dir.join("SKILL.md"),
            "---\nname: fs-demo\ndescription: FS demo\n---\nfs body",
        )
        .unwrap();
        fs::write(fs_skill_dir.join("guide.md"), "supporting body").unwrap();

        let (doc, resources) = one_skill_fixture(
            "mcp-demo",
            "MCP demo",
            "mcp body",
            &[("guide.md", "supporting body")],
        );
        let (client, _mgr, _g) = setup_client_with_built(
            "srv",
            FakeMcp::new(doc, resources),
            tmp.path().to_path_buf(),
        )
        .await;

        let fs_text = text_of(&call_load(&client, "fs-demo").await);
        let mcp_text = text_of(&call_load(&client, "mcp-demo").await);

        // Framing parity: both origins share the header shape, a supporting-
        // files section, and load_skill pointers for supporting files. The
        // section bodies differ (FS shows resolved paths and shell-cd
        // guidance per upstream's richer FS framing).
        for (label, text) in [("fs", &fs_text), ("mcp", &mcp_text)] {
            assert!(text.starts_with("# Loaded Skill: "), "{}: {}", label, text);
            assert!(text.contains("## Supporting Files"), "{}: {}", label, text);
            assert!(text.contains("load_skill(name: \""), "{}: {}", label, text);
        }

        assert!(
            mcp_text.contains("mcp skill from srv"),
            "got:\n{}",
            mcp_text
        );
        assert!(!fs_text.contains("mcp skill from"), "got:\n{}", fs_text);
    }

    #[tokio::test]
    async fn test_supporting_file_framing_parity_fs_vs_mcp() {
        let tmp = TempDir::new().unwrap();

        let fs_skill_dir = tmp.path().join(".goose/skills/fs-demo");
        fs::create_dir_all(&fs_skill_dir).unwrap();
        fs::write(
            fs_skill_dir.join("SKILL.md"),
            "---\nname: fs-demo\ndescription: demo\n---\nbody",
        )
        .unwrap();
        fs::write(fs_skill_dir.join("guide.md"), "fs supporting body").unwrap();

        let (doc, resources) = one_skill_fixture(
            "mcp-demo",
            "d",
            "mcp body",
            &[("guide.md", "mcp supporting body")],
        );
        let (client, _mgr, _g) = setup_client_with_built(
            "srv",
            FakeMcp::new(doc, resources),
            tmp.path().to_path_buf(),
        )
        .await;

        let fs_text = text_of(&call_load(&client, "fs-demo/guide.md").await);
        let mcp_text = text_of(&call_load(&client, "mcp-demo/guide.md").await);

        for (label, text) in [("fs", &fs_text), ("mcp", &mcp_text)] {
            assert!(text.starts_with("# Loaded: "), "{}: {}", label, text);
            assert!(
                text.contains("File loaded into context."),
                "{}: {}",
                label,
                text
            );
            assert!(!text.starts_with("# Loaded Skill: "), "{}: {}", label, text);
            assert!(
                !text.contains("This knowledge is now available in your context."),
                "{}: {}",
                label,
                text
            );
        }

        assert!(fs_text.contains("fs-demo/guide.md"));
        assert!(fs_text.contains("fs supporting body"));
        assert!(mcp_text.contains("mcp-demo/guide.md"));
        assert!(mcp_text.contains("mcp supporting body"));
    }

    #[tokio::test]
    async fn test_list_changed_refreshes_cache() {
        let tmp = TempDir::new().unwrap();
        let (initial_doc, initial_res) = one_skill_fixture("alpha", "a", "body", &[]);
        let fake = Arc::new(FakeMcp::new(initial_doc, initial_res));
        let mgr = Arc::new(ExtensionManager::new_without_provider(
            tmp.path().to_path_buf(),
        ));
        let trait_handle: Arc<dyn McpClientTrait> = fake.clone();
        mgr.add_client(
            "srv".to_string(),
            ExtensionConfig::Builtin {
                name: "srv".to_string(),
                display_name: Some("srv".to_string()),
                description: "fake".to_string(),
                timeout: None,
                bundled: None,
                available_tools: vec![],
            },
            trait_handle,
            None,
            None,
            Some("s"),
        )
        .await;

        let before = mgr.aggregated_mcp_skills().await;
        assert_eq!(before.len(), 1);

        assert!(fake.wait_for_subscriber(Duration::from_secs(2)).await);

        let (doc_a, res_a) = one_skill_fixture("alpha", "a", "body", &[]);
        let (doc_b, res_b) = one_skill_fixture("beta", "b", "body", &[]);
        let mut merged: HashMap<String, String> = res_a;
        merged.extend(res_b);
        let merged_doc = serde_json::json!({
            "skills": [doc_a["skills"][0].clone(), doc_b["skills"][0].clone()]
        });
        fake.swap_skills(merged_doc, merged);
        fake.notify_resources_list_changed().await;

        let deadline = std::time::Instant::now() + Duration::from_secs(1);
        let mut after = mgr.aggregated_mcp_skills().await;
        while after.len() < 2 && std::time::Instant::now() < deadline {
            tokio::time::sleep(Duration::from_millis(20)).await;
            after = mgr.aggregated_mcp_skills().await;
        }
        assert_eq!(after.len(), 2, "got: {:?}", after);
        let names: HashSet<&str> = after.iter().map(|e| e.name.as_str()).collect();
        assert!(names.contains("alpha"));
        assert!(names.contains("beta"));
    }

    #[tokio::test]
    async fn test_remove_extension_ends_watcher_task() {
        let tmp = TempDir::new().unwrap();
        let (doc, res) = one_skill_fixture("alpha", "a", "body", &[]);
        let fake = Arc::new(FakeMcp::new(doc, res));
        let mgr = Arc::new(ExtensionManager::new_without_provider(
            tmp.path().to_path_buf(),
        ));
        let trait_handle: Arc<dyn McpClientTrait> = fake.clone();
        mgr.add_client(
            "srv".to_string(),
            ExtensionConfig::Builtin {
                name: "srv".to_string(),
                display_name: Some("srv".to_string()),
                description: "fake".to_string(),
                timeout: None,
                bundled: None,
                available_tools: vec![],
            },
            trait_handle,
            None,
            None,
            Some("s"),
        )
        .await;

        assert!(fake.wait_for_subscriber(Duration::from_secs(2)).await);
        let tx = {
            let subs = fake.subscribers.lock().await;
            subs.first().expect("subscriber").clone()
        };
        assert!(!tx.is_closed());

        mgr.remove_extension("srv").await.expect("remove");

        let deadline = std::time::Instant::now() + Duration::from_secs(1);
        while !tx.is_closed() && std::time::Instant::now() < deadline {
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
        assert!(tx.is_closed());
    }

    #[tokio::test]
    async fn test_add_extension_repopulates_empty_cache() {
        let tmp = TempDir::new().unwrap();
        let (doc, res) = one_skill_fixture("alpha", "a", "body", &[]);
        let fake = Arc::new(FakeMcp::new(doc, res));
        let mgr = Arc::new(ExtensionManager::new_without_provider(
            tmp.path().to_path_buf(),
        ));
        let trait_handle: Arc<dyn McpClientTrait> = fake.clone();
        // First registration has no session id → empty cache, no watcher.
        mgr.add_client(
            "srv".to_string(),
            ExtensionConfig::Builtin {
                name: "srv".to_string(),
                display_name: Some("srv".to_string()),
                description: "fake".to_string(),
                timeout: None,
                bundled: None,
                available_tools: vec![],
            },
            trait_handle,
            None,
            None,
            None,
        )
        .await;
        assert!(mgr.aggregated_mcp_skills().await.is_empty());

        // Re-register with an identical config and a real session id: the
        // fast-path must repopulate the cache in place and spawn the
        // watcher.
        mgr.add_extension(
            ExtensionConfig::Builtin {
                name: "srv".to_string(),
                display_name: Some("srv".to_string()),
                description: "fake".to_string(),
                timeout: None,
                bundled: None,
                available_tools: vec![],
            },
            Some(tmp.path().to_path_buf()),
            None,
            Some("s"),
        )
        .await
        .expect("repopulate add_extension should succeed");

        let after_repopulate = mgr.aggregated_mcp_skills().await;
        assert_eq!(after_repopulate.len(), 1);
        assert_eq!(after_repopulate[0].name, "alpha");

        assert!(fake.wait_for_subscriber(Duration::from_secs(2)).await);
    }
}
