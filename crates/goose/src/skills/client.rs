use super::discover_skills;
use super::mcp_client::McpSkillEntry;
use crate::agents::extension::PlatformExtensionContext;
use crate::agents::extension_manager::ExtensionManager;
use crate::agents::mcp_client::{Error, McpClientTrait};
use crate::agents::ToolCallContext;
use async_trait::async_trait;
use goose_sdk::custom_requests::{SourceEntry, SourceType};
use rmcp::model::{
    CallToolResult, Content, Implementation, InitializeResult, JsonObject, ListToolsResult,
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
            extension_manager: context.extension_manager,
            fs_names_cache: Mutex::new(FsNamesCache::default()),
        })
    }

    /// All concrete MCP skills discovered from connected servers, opted-in or
    /// not. Used for `load_skill` resolution — a skill the user explicitly
    /// names should load even from a server whose skills aren't auto-injected.
    async fn mcp_skills(&self) -> Vec<McpSkillEntry> {
        match self.extension_manager.as_ref().and_then(|w| w.upgrade()) {
            Some(mgr) => mgr.aggregated_mcp_skills().await,
            None => Vec::new(),
        }
    }

    /// Concrete MCP skills from servers the user has opted into injecting —
    /// the gated set surfaced in the system prompt.
    async fn injectable_mcp_skills(&self) -> Vec<McpSkillEntry> {
        match self.extension_manager.as_ref().and_then(|w| w.upgrade()) {
            Some(mgr) => mgr.injectable_mcp_skills().await,
            None => Vec::new(),
        }
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

        let fresh = fs_skill_names(&self.working_dir);

        let mut cache = self.fs_names_cache.lock().expect("fs_names_cache poisoned");
        cache.refreshed_at = Some(Instant::now());
        cache.names = fresh.clone();
        fresh
    }
}

/// Rebuilds the list of FS skill names currently installed. Used to detect
/// FS-vs-MCP name collisions (FS wins — the MCP entry is rendered with a
/// `<server>__<name>` prefix).
fn fs_skill_names(working_dir: &Path) -> HashSet<String> {
    discover_skills(Some(working_dir))
        .into_iter()
        .filter(|s| matches!(s.source_type, SourceType::Skill | SourceType::BuiltinSkill))
        .map(|s| s.name)
        .collect()
}

/// Computes the set of skill names that collide across FS skills and MCP
/// concrete entries. Any name appearing more than once in this union needs
/// to be rendered in its disambiguated `<server>__<name>` form so the model
/// can address the right entity unambiguously.
fn collision_names(fs_names: &HashSet<String>, concrete: &[McpSkillEntry]) -> HashSet<String> {
    let mut counts: std::collections::HashMap<String, usize> = std::collections::HashMap::new();
    for n in fs_names {
        *counts.entry(n.clone()).or_insert(0) += 1;
    }
    for entry in concrete {
        *counts.entry(entry.name.clone()).or_insert(0) += 1;
    }
    counts
        .into_iter()
        .filter_map(|(n, c)| if c > 1 { Some(n) } else { None })
        .collect()
}

/// Renders the concrete-MCP-skills section of the system prompt. Names
/// that collide with any other visible skill (FS, another concrete MCP
/// entry, or a template) are rendered in `<server>__<name>` form so the
/// model can address the entry unambiguously via `load_skill`. Empty
/// output when no concrete MCP skills are available.
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
        // URL intentionally omitted: the model addresses MCP skills by
        // name via `load_skill`, and including full URLs for every entry
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
/// warning if the server returned more than one text entry — the SEP
/// expects SKILL.md to arrive as a single document, and a multi-entry
/// response likely means the server is splitting content in a way the
/// host won't reassemble.
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

/// Extracts the first resource content as raw bytes — text contents as
/// their UTF-8 bytes, blob contents base64-decoded. Used for archive
/// payloads and (text) digest verification.
fn first_content_bytes(result: rmcp::model::ReadResourceResult) -> Option<Vec<u8>> {
    use base64::Engine;
    for c in result.contents {
        match c {
            ResourceContents::TextResourceContents { text, .. } => return Some(text.into_bytes()),
            ResourceContents::BlobResourceContents { blob, .. } => {
                if let Ok(bytes) = base64::engine::general_purpose::STANDARD.decode(&blob) {
                    return Some(bytes);
                }
            }
        }
    }
    None
}

/// Enumerates supporting-file relative refs under a URI-addressed skill's
/// root (excluding the SKILL.md itself). When the owning server declares
/// `directoryRead: true`, walks the skill tree via `resources/directory/read`
/// (scoped to the root, descending into `inode/directory` children);
/// otherwise filters the server's flat `resources/list`. Best-effort: any
/// error yields an empty list and no section is rendered. Surfaces only
/// pointers, never content.
async fn enumerate_mcp_supporting_resources(
    mgr: &ExtensionManager,
    session_id: &str,
    entry: &McpSkillEntry,
    cancel: CancellationToken,
) -> Vec<String> {
    let Some(root) = entry.skill_root_uri() else {
        return Vec::new();
    };

    let uris = if mgr.server_supports_directory_read(&entry.server).await {
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

    // Directory-form entries (url ends in '/') and `…/SKILL.md` forms both
    // denote the same skill body — skip both so neither shows up as its own
    // supporting-file bullet.
    let skill_md_alias = entry
        .url
        .as_deref()
        .filter(|u| u.ends_with('/'))
        .map(|_| format!("{}SKILL.md", root));

    let mut out = Vec::new();
    for uri in uris {
        if entry.url.as_deref() == Some(uri.as_str())
            || skill_md_alias.as_deref() == Some(uri.as_str())
        {
            continue;
        }
        let Some(rel) = uri.strip_prefix(root) else {
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

/// Fetches and unpacks the first supported archive form of an entry,
/// verifying the archive `digest` before unpacking. Returns the in-memory
/// skill file tree.
async fn materialize_skill_archive(
    mgr: &ExtensionManager,
    session_id: &str,
    entry: &McpSkillEntry,
    cancel: CancellationToken,
) -> Result<crate::skills::archive::SkillTree, String> {
    let Some(arch) = entry.supported_archive() else {
        let offered: Vec<&str> = entry
            .archives
            .iter()
            .map(|a| a.media_type.as_str())
            .collect();
        return Err(format!(
            "no supported archive form for skill '{}' (server offers: [{}]; host supports tar.gz and zip)",
            entry.name,
            offered.join(", ")
        ));
    };
    let result = mgr
        .read_resource(session_id, &arch.url, &entry.server, cancel)
        .await
        .map_err(|e| format!("failed to read archive '{}': {}", arch.url, e.message))?;
    let bytes = first_content_bytes(result)
        .ok_or_else(|| format!("archive '{}' had no readable content", arch.url))?;
    crate::skills::mcp_client::verify_digest(&arch.digest, &bytes)
        .map_err(|e| format!("refusing to use archive '{}': {}", arch.url, e))?;
    crate::skills::archive::unpack_skill_archive(&bytes, &arch.media_type)
}

/// Frames a loaded SKILL.md body the same way the FS path does: a
/// `# Loaded Skill:` header with an MCP origin tag, an optional
/// "Supporting Files" pointer block, and the standard footer.
fn frame_skill_md(entry: &McpSkillEntry, base: &str, body: &str, supporting: &[String]) -> String {
    let mut output = format!(
        "# Loaded Skill: {} (mcp skill from {})\n\n{}\n",
        entry.name, entry.server, body
    );
    if !supporting.is_empty() {
        output.push_str(&format!(
            "\n## Supporting Files\n\nSkill base: {}\n\n",
            base
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

/// Loads an MCP skill's `SKILL.md` and frames it. Handles both
/// individually-addressed skills (read via `resources/read`, verified
/// against the index `digest`) and archive-distributed skills (fetch,
/// verify, unpack, read `SKILL.md` from the tree).
async fn load_mcp_skill_md(
    mgr: &ExtensionManager,
    session_id: &str,
    entry: &McpSkillEntry,
    cancel: CancellationToken,
) -> CallToolResult {
    if let Some(url) = entry.url.clone() {
        match mgr
            .read_resource(session_id, &url, &entry.server, cancel.clone())
            .await
        {
            Ok(result) => match first_text_content(result, &entry.server, &url) {
                Some(body) => {
                    if let Some(digest) = &entry.digest {
                        if let Err(e) =
                            crate::skills::mcp_client::verify_digest(digest, body.as_bytes())
                        {
                            return CallToolResult::error(vec![Content::text(format!(
                                "Refusing to load skill '{}': {}",
                                entry.name, e
                            ))]);
                        }
                    }
                    let supporting =
                        enumerate_mcp_supporting_resources(mgr, session_id, entry, cancel).await;
                    let base = entry.skill_root_uri().unwrap_or(&url);
                    CallToolResult::success(vec![Content::text(frame_skill_md(
                        entry,
                        base,
                        &body,
                        &supporting,
                    ))])
                }
                None => CallToolResult::error(vec![Content::text(format!(
                    "Resource '{}' from '{}' had no text content.",
                    url, entry.server
                ))]),
            },
            Err(e) => CallToolResult::error(vec![Content::text(format!(
                "Failed to read '{}' from '{}': {}",
                url, entry.server, e.message
            ))]),
        }
    } else {
        match materialize_skill_archive(mgr, session_id, entry, cancel).await {
            Ok(tree) => {
                let Some(body) = tree.get("SKILL.md") else {
                    return CallToolResult::error(vec![Content::text(format!(
                        "Archive for skill '{}' has no SKILL.md at its root.",
                        entry.name
                    ))]);
                };
                let body = String::from_utf8_lossy(body).into_owned();
                let mut supporting: Vec<String> = tree
                    .keys()
                    .filter(|k| k.as_str() != "SKILL.md")
                    .cloned()
                    .collect();
                supporting.sort();
                CallToolResult::success(vec![Content::text(frame_skill_md(
                    entry,
                    "(archive)",
                    &body,
                    &supporting,
                ))])
            }
            Err(e) => CallToolResult::error(vec![Content::text(e)]),
        }
    }
}

/// Loads a supporting file (`<skill>/<rel>`) for an MCP skill and frames it
/// with the `# Loaded:` header. URI-addressed skills compose the file URI
/// against the skill root; archive-distributed skills read it from the
/// unpacked tree.
async fn load_mcp_supporting(
    mgr: &ExtensionManager,
    session_id: &str,
    entry: &McpSkillEntry,
    rel: &str,
    cancel: CancellationToken,
) -> CallToolResult {
    let frame = |body: &str| {
        CallToolResult::success(vec![Content::text(format!(
            "# Loaded: {}/{}\n\n{}\n\n---\nFile loaded into context.",
            entry.name, rel, body
        ))])
    };

    if let Some(root) = entry.skill_root_uri() {
        let composed = format!("{}{}", root, rel);
        match mgr
            .read_resource(session_id, &composed, &entry.server, cancel)
            .await
        {
            Ok(result) => match first_text_content(result, &entry.server, &composed) {
                Some(body) => frame(&body),
                None => CallToolResult::error(vec![Content::text(format!(
                    "Resource '{}' from '{}' had no text content.",
                    composed, entry.server
                ))]),
            },
            Err(e) => CallToolResult::error(vec![Content::text(format!(
                "Failed to read '{}' from '{}': {}",
                composed, entry.server, e.message
            ))]),
        }
    } else {
        match materialize_skill_archive(mgr, session_id, entry, cancel).await {
            Ok(tree) => match tree.get(rel) {
                Some(body) => frame(&String::from_utf8_lossy(body)),
                None => {
                    let mut available: Vec<&str> = tree
                        .keys()
                        .filter(|k| k.as_str() != "SKILL.md")
                        .map(|k| k.as_str())
                        .collect();
                    available.sort();
                    available.truncate(10);
                    CallToolResult::error(vec![Content::text(format!(
                        "File '{}/{}' not found in archive. Available: {}",
                        entry.name,
                        rel,
                        available.join(", ")
                    ))])
                }
            },
            Err(e) => CallToolResult::error(vec![Content::text(e)]),
        }
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
        let load_skill_schema = serde_json::json!({
            "type": "object",
            "required": ["name"],
            "properties": {
                "name": {
                    "type": "string",
                    "description": "Name of the skill to load. Use \"skill-name/path\" to load a supporting file. For MCP skills with a name collision, use the \"<server>__<name>\" form shown in your system instructions. Do NOT pass a URI here — use the read_resource tool (on the extensionmanager) if you only have a URI."
                },
                "args": {
                    "type": "string",
                    "description": "Optional arguments to provide when loading the skill."
                }
            }
        });

        let load_skill = Tool::new(
            "load_skill",
            "Load a skill's full content into your context so you can follow its instructions.\n\n\
             Skills are listed in your system instructions (both local skills and skills from connected MCP servers). When you need to use one, load it first to get the detailed instructions.\n\n\
             Examples:\n\
             - load_skill(name: \"gdrive\") → Loads the gdrive skill instructions\n\
             - load_skill(name: \"my-skill\", args: \"the arguments for the skill\") → Loads a skill with arguments\n\
             - load_skill(name: \"my-skill/template.md\") → Loads a supporting file\n\
             - load_skill(name: \"github__pull-requests\") → Disambiguates a collision between two servers\n\n\
             Use read_resource (from the extensionmanager) if you only have a raw URI. Do NOT pass skill URIs to file-reading, writing, editing, or shell tools — those operate on filesystem paths."
                .to_string(),
            load_skill_schema.as_object().unwrap().clone(),
        );

        let tools = vec![load_skill];

        Ok(ListToolsResult {
            tools,
            next_cursor: None,
            meta: None,
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
            return Ok(CallToolResult::error(vec![Content::text(format!(
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
            return Ok(CallToolResult::error(vec![Content::text(
                "Missing required parameter: name",
            )]));
        }
        let args = arguments
            .as_ref()
            .and_then(|args| args.get("args"))
            .and_then(|v| v.as_str());

        // Reject raw URIs — they go through `read_resource` (a separate
        // tool) rather than `load_skill`. Shares `looks_like_uri` with
        // `developer::edit::reject_uri_path` so the two guardrails can't
        // drift apart on scheme shape.
        if crate::agents::platform_extensions::looks_like_uri(skill_name) {
            return Ok(CallToolResult::error(vec![Content::text(format!(
                "'{}' looks like a URI. Use the read_resource tool instead (it takes a server name and a uri). load_skill takes a skill name or <skill>/<relative/path>.",
                skill_name
            ))]));
        }

        let skills = discover_skills(Some(&self.working_dir));

        if let Some(skill) = skills.iter().find(|s| s.name == skill_name) {
            let content = if let Some(a) = args {
                match super::apply_skill_arguments(
                    &skill.content,
                    a,
                    &super::skill_argument_names(skill),
                ) {
                    Ok(c) => c,
                    Err(e) => {
                        return Ok(CallToolResult::error(vec![Content::text(format!(
                            "Failed to parse skill arguments: {}",
                            e
                        ))]));
                    }
                }
            } else {
                skill.content.clone()
            };

            let mut output = format!(
                "# Loaded Skill: {} ({})\n\n## {} ({})\n\n{}\n\n### Content\n\n{}\n",
                skill.name,
                skill.source_type,
                skill.name,
                skill.source_type,
                skill.description,
                content,
            );

            if !skill.supporting_files.is_empty() {
                let skill_dir = Path::new(&skill.path);
                output.push_str(&format!(
                    "\n## Supporting Files\n\nSkill base: {}\n\n",
                    skill.path
                ));
                for file in &skill.supporting_files {
                    if let Ok(relative) = Path::new(file).strip_prefix(skill_dir) {
                        let rel_str = relative.to_string_lossy().replace('\\', "/");
                        output.push_str(&format!(
                            "- {} → load_skill(name: \"{}/{}\")\n",
                            rel_str, skill.name, rel_str
                        ));
                    }
                }
            }

            output.push_str("\n---\nThis knowledge is now available in your context.");
            return Ok(CallToolResult::success(vec![Content::text(output)]));
        }

        if let Some((parent_skill_name, raw_relative_path)) = skill_name.split_once('/') {
            let relative_path = raw_relative_path.replace('\\', "/");
            if let Some(skill) = skills.iter().find(|s| {
                s.name == parent_skill_name
                    && matches!(s.source_type, SourceType::Skill | SourceType::BuiltinSkill)
            }) {
                let skill_dir = PathBuf::from(&skill.path);
                let canonical_skill_dir = skill_dir
                    .canonicalize()
                    .unwrap_or_else(|_| skill_dir.clone());

                for file_path in &skill.supporting_files {
                    let file_path_buf = Path::new(file_path);
                    let Ok(rel) = file_path_buf.strip_prefix(&skill_dir) else {
                        continue;
                    };
                    if rel.to_string_lossy().replace('\\', "/") != relative_path {
                        continue;
                    }

                    return Ok(match file_path_buf.canonicalize() {
                        Ok(canonical) if canonical.starts_with(&canonical_skill_dir) => {
                            match std::fs::read_to_string(&canonical) {
                                Ok(content) => {
                                    CallToolResult::success(vec![Content::text(format!(
                                        "# Loaded: {}\n\n{}\n\n---\nFile loaded into context.",
                                        skill_name, content
                                    ))])
                                }
                                Err(e) => CallToolResult::error(vec![Content::text(format!(
                                    "Failed to read '{}': {}",
                                    skill_name, e
                                ))]),
                            }
                        }
                        Ok(_) => CallToolResult::error(vec![Content::text(format!(
                            "Refusing to load '{}': resolves outside the skill directory",
                            skill_name
                        ))]),
                        Err(e) => CallToolResult::error(vec![Content::text(format!(
                            "Failed to resolve '{}': {}",
                            skill_name, e
                        ))]),
                    });
                }

                let available: Vec<String> = skill
                    .supporting_files
                    .iter()
                    .filter_map(|f| {
                        Path::new(f)
                            .strip_prefix(&skill_dir)
                            .ok()
                            .map(|r| r.to_string_lossy().replace('\\', "/"))
                    })
                    .take(10)
                    .collect();

                return Ok(if available.is_empty() {
                    CallToolResult::error(vec![Content::text(format!(
                        "Skill '{}' has no supporting files.",
                        skill.name
                    ))])
                } else {
                    CallToolResult::error(vec![Content::text(format!(
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
        let mgr = self.extension_manager.as_ref().and_then(|w| w.upgrade());

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
                        return Ok(CallToolResult::error(vec![Content::text(format!(
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
            CallToolResult::error(vec![Content::text(format!(
                "Skill '{}' not found.",
                skill_name
            ))])
        } else {
            CallToolResult::error(vec![Content::text(format!(
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
        let sources = discover_skills(Some(&self.working_dir));
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
    use std::fs;
    use std::sync::Arc;
    use tempfile::TempDir;

    #[tokio::test]
    async fn test_load_skill_from_filesystem() {
        let temp_dir = TempDir::new().unwrap();
        let skill_dir = temp_dir.path().join(".goose/skills/my-skill");
        fs::create_dir_all(&skill_dir).unwrap();
        fs::write(
            skill_dir.join("SKILL.md"),
            "---\nname: my-skill\ndescription: A test skill\n---\nDo the thing.",
        )
        .unwrap();

        let session = std::sync::Arc::new(crate::session::Session {
            working_dir: temp_dir.path().to_path_buf(),
            ..crate::session::Session::default()
        });
        let client = SkillsClient::new(PlatformExtensionContext {
            extension_manager: None,
            session_manager: Arc::new(crate::session::SessionManager::instance()),
            session: Some(session),
            use_login_shell_path: false,
        })
        .unwrap();

        let ctx = ToolCallContext::new("test".to_string(), None, None);
        let args: JsonObject =
            serde_json::from_value(serde_json::json!({"name": "my-skill"})).unwrap();
        let result = client
            .call_tool(&ctx, "load_skill", Some(args), CancellationToken::new())
            .await
            .unwrap();

        assert!(!result.is_error.unwrap_or(false));
        let text = match &result.content[0].raw {
            rmcp::model::RawContent::Text(t) => &t.text,
            _ => panic!("expected text"),
        };
        assert!(text.contains("my-skill"));
        assert!(text.contains("Do the thing"));
    }

    #[tokio::test]
    async fn test_load_skill_not_found_returns_error() {
        let client = SkillsClient::new(PlatformExtensionContext {
            extension_manager: None,
            session_manager: Arc::new(crate::session::SessionManager::instance()),
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
    use async_trait::async_trait;
    use rmcp::model::{
        Annotated, ExtensionCapabilities, ListResourcesResult, RawResource, ReadResourceResult,
        ServerNotification,
    };
    use std::collections::HashMap;

    /// Directory listing fixture: dir URI -> children as (uri, mimeType).
    type DirMap = HashMap<String, Vec<(String, Option<String>)>>;

    struct FakeMcp {
        info: InitializeResult,
        resources: std::sync::Mutex<HashMap<String, String>>,
        /// Binary resources (e.g. skill archives), returned as blob contents.
        blobs: HashMap<String, Vec<u8>>,
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
            caps.insert(
                super::super::mcp_client::SKILLS_EXTENSION_ID.to_string(),
                cfg,
            );
            InitializeResult::new(
                ServerCapabilities::builder()
                    .enable_resources()
                    .enable_extensions_with(caps)
                    .build(),
            )
        }

        fn new(resources: HashMap<String, String>) -> Self {
            Self {
                info: Self::build_info(false),
                resources: std::sync::Mutex::new(resources),
                blobs: HashMap::new(),
                directories: HashMap::new(),
                subscribers: tokio::sync::Mutex::new(Vec::new()),
            }
        }

        fn with_blob(mut self, uri: &str, bytes: Vec<u8>) -> Self {
            self.blobs.insert(uri.to_string(), bytes);
            self
        }

        fn with_directories(mut self, dirs: DirMap) -> Self {
            self.info = Self::build_info(true);
            self.directories = dirs;
            self
        }

        fn swap_resources(&self, new_resources: HashMap<String, String>) {
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
                .filter(|uri| uri.as_str() != super::super::mcp_client::INDEX_URI)
                .map(|uri| Annotated::new(RawResource::new(uri.as_str(), uri.as_str()), None))
                .collect();
            Ok(ListResourcesResult {
                resources,
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
            if let Some(bytes) = self.blobs.get(uri) {
                use base64::Engine;
                let blob = base64::engine::general_purpose::STANDARD.encode(bytes);
                return Ok(ReadResourceResult::new(vec![
                    ResourceContents::BlobResourceContents {
                        uri: uri.to_string(),
                        mime_type: Some("application/octet-stream".to_string()),
                        blob,
                        meta: None,
                    },
                ]));
            }
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
                    let mut raw = RawResource::new(child_uri.as_str(), child_uri.as_str());
                    raw.mime_type = mime.clone();
                    Annotated::new(raw, None)
                })
                .collect();
            Ok(ListResourcesResult {
                resources,
                next_cursor: None,
                meta: None,
            })
        }

        async fn subscribe(&self) -> mpsc::Receiver<ServerNotification> {
            let (tx, rx) = mpsc::channel(16);
            self.subscribers.lock().await.push(tx);
            rx
        }
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
            session: Some(session),
            use_login_shell_path: false,
        })
        .unwrap();
        (client, mgr, tmp)
    }

    async fn setup_client_with_fake(
        server_name: &str,
        resources: HashMap<String, String>,
        working_dir: PathBuf,
    ) -> (SkillsClient, Arc<ExtensionManager>, TempDir) {
        setup_client_with_built(server_name, FakeMcp::new(resources), working_dir).await
    }

    /// Build an index entry carrying a frontmatter block plus extra JSON
    /// (e.g. `,"url":"...","digest":"..."`).
    fn fm_entry(name: &str, description: &str, extra: &str) -> String {
        format!(
            r#"{{"frontmatter":{{"name":"{}","description":"{}"}}{}}}"#,
            name, description, extra
        )
    }

    fn index_json(entries: &str) -> String {
        format!(r#"{{"skills":[{}]}}"#, entries)
    }

    fn sha256_hex(bytes: &[u8]) -> String {
        use sha2::{Digest, Sha256};
        let mut h = Sha256::new();
        h.update(bytes);
        let out = h.finalize();
        let mut s = String::new();
        for b in out {
            s.push_str(&format!("{:02x}", b));
        }
        s
    }

    fn make_tar_gz(files: &[(&str, &[u8])]) -> Vec<u8> {
        use flate2::write::GzEncoder;
        use flate2::Compression;
        let mut builder = tar::Builder::new(GzEncoder::new(Vec::new(), Compression::default()));
        for (name, content) in files {
            let mut header = tar::Header::new_gnu();
            header.set_size(content.len() as u64);
            header.set_mode(0o644);
            header.set_cksum();
            builder.append_data(&mut header, name, *content).unwrap();
        }
        builder.into_inner().unwrap().finish().unwrap()
    }

    fn text_of(r: &CallToolResult) -> String {
        match &r.content[0].raw {
            rmcp::model::RawContent::Text(t) => t.text.clone(),
            _ => panic!("expected text content"),
        }
    }

    #[tokio::test]
    async fn test_load_mcp_skill_basic() {
        let tmp = TempDir::new().unwrap();
        let mut resources = HashMap::new();
        resources.insert(
            "skill://index.json".to_string(),
            index_json(&fm_entry(
                "git-workflow",
                "Git",
                r#","url":"skill://git-workflow/SKILL.md""#,
            )),
        );
        resources.insert(
            "skill://git-workflow/SKILL.md".to_string(),
            "Git body text.".to_string(),
        );

        let (client, _mgr, _tmp_guard) =
            setup_client_with_fake("gh", resources, tmp.path().to_path_buf()).await;

        let ctx = ToolCallContext::new("s".to_string(), None, None);
        let args: JsonObject =
            serde_json::from_value(serde_json::json!({"name": "git-workflow"})).unwrap();
        let result = client
            .call_tool(&ctx, "load_skill", Some(args), CancellationToken::new())
            .await
            .unwrap();

        assert!(!result.is_error.unwrap_or(false));
        let body = text_of(&result);
        assert!(body.contains("Git body text"), "got: {}", body);
        assert!(body.contains("mcp skill from gh"), "got: {}", body);
    }

    #[tokio::test]
    async fn test_load_mcp_skill_verifies_digest_and_rejects_mismatch() {
        let tmp = TempDir::new().unwrap();
        let body = "verified skill body";
        let good_digest = format!("sha256:{}", sha256_hex(body.as_bytes()));

        let mut resources = HashMap::new();
        resources.insert(
            "skill://index.json".to_string(),
            index_json(&fm_entry(
                "verified",
                "v",
                &format!(
                    r#","url":"skill://verified/SKILL.md","digest":"{}""#,
                    good_digest
                ),
            )),
        );
        resources.insert("skill://verified/SKILL.md".to_string(), body.to_string());
        let (client, _mgr, _g) =
            setup_client_with_fake("gh", resources, tmp.path().to_path_buf()).await;
        let ctx = ToolCallContext::new("s".to_string(), None, None);
        let ok = client
            .call_tool(
                &ctx,
                "load_skill",
                Some(serde_json::from_value(serde_json::json!({"name": "verified"})).unwrap()),
                CancellationToken::new(),
            )
            .await
            .unwrap();
        assert!(!ok.is_error.unwrap_or(false), "got: {}", text_of(&ok));
        assert!(text_of(&ok).contains("verified skill body"));

        let mut resources = HashMap::new();
        resources.insert(
            "skill://index.json".to_string(),
            index_json(&fm_entry(
                "tampered",
                "t",
                r#","url":"skill://tampered/SKILL.md","digest":"sha256:deadbeef""#,
            )),
        );
        resources.insert(
            "skill://tampered/SKILL.md".to_string(),
            "SHOULD NOT SURFACE".to_string(),
        );
        let (client, _mgr, _g) =
            setup_client_with_fake("gh", resources, tmp.path().to_path_buf()).await;
        let bad = client
            .call_tool(
                &ctx,
                "load_skill",
                Some(serde_json::from_value(serde_json::json!({"name": "tampered"})).unwrap()),
                CancellationToken::new(),
            )
            .await
            .unwrap();
        assert!(bad.is_error.unwrap_or(false));
        let body = text_of(&bad);
        assert!(
            body.contains("digest mismatch") || body.contains("Refusing"),
            "got: {}",
            body
        );
        assert!(!body.contains("SHOULD NOT SURFACE"));
    }

    #[tokio::test]
    async fn test_load_mcp_skill_from_archive() {
        let tmp = TempDir::new().unwrap();
        let archive = make_tar_gz(&[
            ("SKILL.md", b"archived skill body"),
            ("references/GUIDE.md", b"archived guide"),
        ]);
        let digest = format!("sha256:{}", sha256_hex(&archive));
        let mut resources = HashMap::new();
        resources.insert(
            "skill://index.json".to_string(),
            index_json(&fm_entry(
                "pdf-processing",
                "PDFs",
                &format!(
                    r#","archives":[{{"url":"skill://pdf-processing.tar.gz","mimeType":"application/gzip","digest":"{}"}}]"#,
                    digest
                ),
            )),
        );
        let fake = FakeMcp::new(resources).with_blob("skill://pdf-processing.tar.gz", archive);
        let (client, _mgr, _g) =
            setup_client_with_built("srv", fake, tmp.path().to_path_buf()).await;

        let ctx = ToolCallContext::new("s".to_string(), None, None);
        let result = client
            .call_tool(
                &ctx,
                "load_skill",
                Some(
                    serde_json::from_value(serde_json::json!({"name": "pdf-processing"})).unwrap(),
                ),
                CancellationToken::new(),
            )
            .await
            .unwrap();
        assert!(
            !result.is_error.unwrap_or(false),
            "got: {}",
            text_of(&result)
        );
        let body = text_of(&result);
        assert!(body.contains("archived skill body"), "got: {}", body);
        assert!(
            body.contains("references/GUIDE.md → load_skill"),
            "archive supporting files should be listed; got: {}",
            body
        );
        assert!(!body.contains("archived guide"), "got: {}", body);

        let sf = client
            .call_tool(
                &ctx,
                "load_skill",
                Some(
                    serde_json::from_value(
                        serde_json::json!({"name": "pdf-processing/references/GUIDE.md"}),
                    )
                    .unwrap(),
                ),
                CancellationToken::new(),
            )
            .await
            .unwrap();
        assert!(!sf.is_error.unwrap_or(false), "got: {}", text_of(&sf));
        assert!(text_of(&sf).contains("archived guide"));
    }

    #[tokio::test]
    async fn test_load_mcp_skill_non_skill_scheme() {
        let tmp = TempDir::new().unwrap();
        let mut resources = HashMap::new();
        resources.insert(
            "skill://index.json".to_string(),
            index_json(&fm_entry(
                "pull-requests",
                "PRs",
                r#","url":"github://o/r/skills/pull-requests/SKILL.md""#,
            )),
        );
        resources.insert(
            "github://o/r/skills/pull-requests/SKILL.md".to_string(),
            "PR review workflow body.".to_string(),
        );

        let (client, _mgr, _tmp_guard) =
            setup_client_with_fake("github", resources, tmp.path().to_path_buf()).await;

        let ctx = ToolCallContext::new("s".to_string(), None, None);
        let args: JsonObject =
            serde_json::from_value(serde_json::json!({"name": "pull-requests"})).unwrap();
        let result = client
            .call_tool(&ctx, "load_skill", Some(args), CancellationToken::new())
            .await
            .unwrap();

        assert!(!result.is_error.unwrap_or(false));
        assert!(text_of(&result).contains("PR review workflow body"));
    }

    #[tokio::test]
    async fn test_load_mcp_supporting_file() {
        let tmp = TempDir::new().unwrap();
        let mut resources = HashMap::new();
        resources.insert(
            "skill://index.json".to_string(),
            index_json(&fm_entry("docs", "", r#","url":"skill://docs/SKILL.md""#)),
        );
        resources.insert(
            "skill://docs/references/GUIDE.md".to_string(),
            "Guide body.".to_string(),
        );

        let (client, _mgr, _tmp_guard) =
            setup_client_with_fake("srv", resources, tmp.path().to_path_buf()).await;

        let ctx = ToolCallContext::new("s".to_string(), None, None);
        let args: JsonObject =
            serde_json::from_value(serde_json::json!({"name": "docs/references/GUIDE.md"}))
                .unwrap();
        let result = client
            .call_tool(&ctx, "load_skill", Some(args), CancellationToken::new())
            .await
            .unwrap();

        assert!(!result.is_error.unwrap_or(false));
        assert!(text_of(&result).contains("Guide body"));
    }

    #[tokio::test]
    async fn test_load_mcp_supporting_file_rejects_parent_traversal() {
        let tmp = TempDir::new().unwrap();
        let mut resources = HashMap::new();
        resources.insert(
            "skill://index.json".to_string(),
            index_json(&fm_entry("docs", "", r#","url":"skill://docs/SKILL.md""#)),
        );
        let (client, _mgr, _tmp_guard) =
            setup_client_with_fake("srv", resources, tmp.path().to_path_buf()).await;

        let ctx = ToolCallContext::new("s".to_string(), None, None);
        for bad in ["docs/../secrets/SKILL.md", "docs//etc/passwd"] {
            let args: JsonObject =
                serde_json::from_value(serde_json::json!({"name": bad})).unwrap();
            let result = client
                .call_tool(&ctx, "load_skill", Some(args), CancellationToken::new())
                .await
                .unwrap();
            assert!(
                result.is_error.unwrap_or(false),
                "expected rejection for {bad}, got: {:?}",
                text_of(&result)
            );
            let body = text_of(&result);
            assert!(
                body.contains("Refusing to load") || body.contains(".."),
                "unexpected rejection message for {bad}: {body}"
            );
        }
    }

    #[tokio::test]
    async fn test_load_skill_uri_input_redirects_to_read_resource() {
        let tmp = TempDir::new().unwrap();
        let (client, _mgr, _tmp_guard) =
            setup_client_with_fake("srv", HashMap::new(), tmp.path().to_path_buf()).await;

        let ctx = ToolCallContext::new("s".to_string(), None, None);
        let args: JsonObject =
            serde_json::from_value(serde_json::json!({"name": "skill://unknown/SKILL.md"}))
                .unwrap();
        let result = client
            .call_tool(&ctx, "load_skill", Some(args), CancellationToken::new())
            .await
            .unwrap();

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
        let mut resources = HashMap::new();
        resources.insert(
            "skill://index.json".to_string(),
            index_json(&fm_entry(
                "alpha",
                "A",
                r#","url":"skill://alpha/SKILL.md""#,
            )),
        );
        let (client, _mgr, _tmp_guard) =
            setup_client_with_fake("srv", resources, tmp.path().to_path_buf()).await;

        let out = client
            .get_dynamic_instructions("s")
            .await
            .expect("should have dynamic output");
        assert!(out.contains("alpha"), "got: {}", out);
        assert!(out.contains("srv"), "got: {}", out);
    }

    async fn setup_client_without_opt_in(
        server_name: &str,
        resources: HashMap<String, String>,
        working_dir: PathBuf,
    ) -> (SkillsClient, Arc<ExtensionManager>) {
        let mgr = Arc::new(ExtensionManager::new_without_provider(working_dir.clone()));
        let fake: Arc<dyn McpClientTrait> = Arc::new(FakeMcp::new(resources));
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
            session: Some(session),
            use_login_shell_path: false,
        })
        .unwrap();
        (client, mgr)
    }

    #[tokio::test]
    async fn test_injection_gated_until_opt_in() {
        let tmp = TempDir::new().unwrap();
        let mut resources = HashMap::new();
        resources.insert(
            "skill://index.json".to_string(),
            index_json(&fm_entry(
                "alpha",
                "A",
                r#","url":"skill://alpha/SKILL.md""#,
            )),
        );
        let (client, mgr) =
            setup_client_without_opt_in("srv", resources, tmp.path().to_path_buf()).await;

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
        let mut resources = HashMap::new();
        resources.insert(
            "skill://index.json".to_string(),
            index_json(&fm_entry(
                "alpha",
                "A",
                r#","url":"skill://alpha/SKILL.md""#,
            )),
        );
        let (_client, mgr) =
            setup_client_without_opt_in("srv", resources, tmp.path().to_path_buf()).await;

        let before = mgr.mcp_skill_servers().await;
        let entry = before.iter().find(|s| s.server == "srv").unwrap();
        assert_eq!(entry.concrete_count, 1);
        assert!(!entry.skills_enabled);

        mgr.set_skills_enabled("srv", true).await;
        let after = mgr.mcp_skill_servers().await;
        let entry = after.iter().find(|s| s.server == "srv").unwrap();
        assert!(entry.skills_enabled);
    }

    async fn register_fake(
        mgr: &Arc<ExtensionManager>,
        server_name: &str,
        resources: HashMap<String, String>,
    ) {
        register_built_fake(mgr, server_name, FakeMcp::new(resources)).await;
    }

    #[tokio::test]
    async fn test_mcp_vs_mcp_collision_renders_prefixed_names() {
        let tmp = TempDir::new().unwrap();
        let mut r1 = HashMap::new();
        r1.insert(
            "skill://index.json".to_string(),
            index_json(&fm_entry(
                "shared",
                "from one",
                r#","url":"skill://shared/SKILL.md""#,
            )),
        );
        let (client, mgr, _tmp) = setup_client_with_fake("one", r1, tmp.path().to_path_buf()).await;

        let mut r2 = HashMap::new();
        r2.insert(
            "skill://index.json".to_string(),
            index_json(&fm_entry(
                "shared",
                "from two",
                r#","url":"skill://shared/SKILL.md""#,
            )),
        );
        register_fake(&mgr, "two", r2).await;

        let out = client
            .get_dynamic_instructions("s")
            .await
            .expect("dynamic output");
        assert!(out.contains("one__shared"), "got:\n{}", out);
        assert!(out.contains("two__shared"), "got:\n{}", out);
        assert!(!out.contains("• shared "), "got:\n{}", out);
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

        let mut resources = HashMap::new();
        resources.insert(
            "skill://index.json".to_string(),
            index_json(&fm_entry(
                "mcp-demo",
                "MCP demo",
                r#","url":"skill://mcp-demo/SKILL.md""#,
            )),
        );
        resources.insert(
            "skill://mcp-demo/SKILL.md".to_string(),
            "mcp body".to_string(),
        );
        resources.insert(
            "skill://mcp-demo/guide.md".to_string(),
            "supporting body".to_string(),
        );
        let (client, _mgr, _tmp) =
            setup_client_with_fake("srv", resources, tmp.path().to_path_buf()).await;

        let ctx = ToolCallContext::new("s".to_string(), None, None);
        let fs_result = client
            .call_tool(
                &ctx,
                "load_skill",
                Some(
                    serde_json::from_value::<JsonObject>(serde_json::json!({"name": "fs-demo"}))
                        .unwrap(),
                ),
                CancellationToken::new(),
            )
            .await
            .unwrap();
        let fs_text = text_of(&fs_result);

        let mcp_result = client
            .call_tool(
                &ctx,
                "load_skill",
                Some(
                    serde_json::from_value::<JsonObject>(serde_json::json!({"name": "mcp-demo"}))
                        .unwrap(),
                ),
                CancellationToken::new(),
            )
            .await
            .unwrap();
        let mcp_text = text_of(&mcp_result);

        for (label, text) in [("fs", &fs_text), ("mcp", &mcp_text)] {
            assert!(text.starts_with("# Loaded Skill: "), "{}: {}", label, text);
            assert!(text.contains("## Supporting Files"), "{}: {}", label, text);
            assert!(text.contains("Skill base: "), "{}: {}", label, text);
            assert!(!text.contains("Skill directory:"), "{}: {}", label, text);
            assert!(
                text.contains("This knowledge is now available in your context."),
                "{}: {}",
                label,
                text
            );
            assert!(
                text.contains("→ load_skill(name: \""),
                "{}: {}",
                label,
                text
            );
        }

        assert!(
            mcp_text.contains("mcp skill from srv"),
            "got:\n{}",
            mcp_text
        );
        assert!(!fs_text.contains("mcp skill from"), "got:\n{}", fs_text);
    }

    #[tokio::test]
    async fn test_watcher_spawned_on_session_carrying_repopulate() {
        let tmp = TempDir::new().unwrap();
        let mut initial = HashMap::new();
        initial.insert(
            "skill://index.json".to_string(),
            index_json(&fm_entry(
                "alpha",
                "a",
                r#","url":"skill://alpha/SKILL.md""#,
            )),
        );
        let fake = Arc::new(FakeMcp::new(initial));
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
            None,
        )
        .await;
        assert!(mgr.aggregated_mcp_skills().await.is_empty());

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

        let mut updated = HashMap::new();
        updated.insert(
            "skill://index.json".to_string(),
            index_json(&format!(
                "{},{}",
                fm_entry("alpha", "a", r#","url":"skill://alpha/SKILL.md""#),
                fm_entry("beta", "b", r#","url":"skill://beta/SKILL.md""#),
            )),
        );
        fake.swap_resources(updated);
        fake.notify_resources_list_changed().await;

        let deadline = std::time::Instant::now() + Duration::from_secs(1);
        let mut refreshed = mgr.aggregated_mcp_skills().await;
        while refreshed.len() < 2 && std::time::Instant::now() < deadline {
            tokio::time::sleep(Duration::from_millis(20)).await;
            refreshed = mgr.aggregated_mcp_skills().await;
        }
        assert_eq!(refreshed.len(), 2, "got: {:?}", refreshed);
    }

    #[tokio::test]
    async fn test_list_changed_refreshes_cache() {
        let tmp = TempDir::new().unwrap();
        let mut initial = HashMap::new();
        initial.insert(
            "skill://index.json".to_string(),
            index_json(&fm_entry(
                "alpha",
                "a",
                r#","url":"skill://alpha/SKILL.md""#,
            )),
        );
        let fake = Arc::new(FakeMcp::new(initial));
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

        let mut updated = HashMap::new();
        updated.insert(
            "skill://index.json".to_string(),
            index_json(&format!(
                "{},{}",
                fm_entry("alpha", "a", r#","url":"skill://alpha/SKILL.md""#),
                fm_entry("beta", "b", r#","url":"skill://beta/SKILL.md""#),
            )),
        );
        fake.swap_resources(updated);
        fake.notify_resources_list_changed().await;

        let deadline = std::time::Instant::now() + Duration::from_secs(1);
        let mut after = mgr.aggregated_mcp_skills().await;
        while after.len() < 2 && std::time::Instant::now() < deadline {
            tokio::time::sleep(Duration::from_millis(20)).await;
            after = mgr.aggregated_mcp_skills().await;
        }
        assert_eq!(after.len(), 2, "got: {:?}", after);
        let names: std::collections::HashSet<&str> =
            after.iter().map(|e| e.name.as_str()).collect();
        assert!(names.contains("alpha"));
        assert!(names.contains("beta"));
    }

    #[tokio::test]
    async fn test_remove_extension_ends_watcher_task() {
        let tmp = TempDir::new().unwrap();
        let mut initial = HashMap::new();
        initial.insert(
            "skill://index.json".to_string(),
            index_json(&fm_entry(
                "alpha",
                "a",
                r#","url":"skill://alpha/SKILL.md""#,
            )),
        );
        let fake = Arc::new(FakeMcp::new(initial));
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
    async fn test_load_skill_resolves_server_prefix() {
        let tmp = TempDir::new().unwrap();
        let mut r1 = HashMap::new();
        r1.insert(
            "skill://index.json".to_string(),
            index_json(&fm_entry(
                "shared",
                "",
                r#","url":"skill://shared/SKILL.md""#,
            )),
        );
        r1.insert(
            "skill://shared/SKILL.md".to_string(),
            "body from server one".to_string(),
        );
        let (client, mgr, _tmp) = setup_client_with_fake("one", r1, tmp.path().to_path_buf()).await;

        let mut r2 = HashMap::new();
        r2.insert(
            "skill://index.json".to_string(),
            index_json(&fm_entry(
                "shared",
                "",
                r#","url":"skill://shared/SKILL.md""#,
            )),
        );
        r2.insert(
            "skill://shared/SKILL.md".to_string(),
            "body from server two".to_string(),
        );
        register_fake(&mgr, "two", r2).await;

        let ctx = ToolCallContext::new("s".to_string(), None, None);
        let args: JsonObject =
            serde_json::from_value(serde_json::json!({"name": "two__shared"})).unwrap();
        let result = client
            .call_tool(&ctx, "load_skill", Some(args), CancellationToken::new())
            .await
            .unwrap();

        assert!(!result.is_error.unwrap_or(false));
        let body = text_of(&result);
        assert!(body.contains("body from server two"), "got:\n{}", body);
        assert!(!body.contains("body from server one"));
    }

    #[tokio::test]
    async fn test_load_mcp_skill_lists_supporting_files_via_resources_list() {
        let tmp = TempDir::new().unwrap();
        let mut resources = HashMap::new();
        resources.insert(
            "skill://index.json".to_string(),
            index_json(&fm_entry("docs", "D", r#","url":"skill://docs/SKILL.md""#)),
        );
        resources.insert(
            "skill://docs/SKILL.md".to_string(),
            "main skill body".to_string(),
        );
        resources.insert(
            "skill://docs/references/GUIDE.md".to_string(),
            "SHOULD NOT APPEAR IN OUTPUT".to_string(),
        );
        resources.insert(
            "skill://other/SKILL.md".to_string(),
            "unrelated".to_string(),
        );

        let (client, _mgr, _tmp_guard) =
            setup_client_with_fake("srv", resources, tmp.path().to_path_buf()).await;

        let ctx = ToolCallContext::new("s".to_string(), None, None);
        let args: JsonObject = serde_json::from_value(serde_json::json!({"name": "docs"})).unwrap();
        let result = client
            .call_tool(&ctx, "load_skill", Some(args), CancellationToken::new())
            .await
            .unwrap();

        assert!(!result.is_error.unwrap_or(false));
        let body = text_of(&result);
        assert!(body.contains("main skill body"), "got:\n{}", body);
        assert!(
            body.contains("references/GUIDE.md → load_skill(name: \"docs/references/GUIDE.md\")"),
            "got:\n{}",
            body
        );
        assert!(
            !body.contains("SHOULD NOT APPEAR IN OUTPUT"),
            "got:\n{}",
            body
        );
        assert!(!body.contains("skill://other/SKILL.md"), "got:\n{}", body);
    }

    #[tokio::test]
    async fn test_supporting_files_via_directory_read() {
        let tmp = TempDir::new().unwrap();
        let mut resources = HashMap::new();
        resources.insert(
            "skill://index.json".to_string(),
            index_json(&fm_entry("docs", "D", r#","url":"skill://docs/SKILL.md""#)),
        );
        resources.insert("skill://docs/SKILL.md".to_string(), "main body".to_string());
        resources.insert(
            "skill://docs/references/GUIDE.md".to_string(),
            "guide".to_string(),
        );

        let mut dirs: DirMap = HashMap::new();
        dirs.insert(
            "skill://docs/".to_string(),
            vec![
                (
                    "skill://docs/SKILL.md".to_string(),
                    Some("text/markdown".to_string()),
                ),
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

        let fake = FakeMcp::new(resources).with_directories(dirs);
        let (client, mgr, _g) =
            setup_client_with_built("srv", fake, tmp.path().to_path_buf()).await;
        assert!(mgr.server_supports_directory_read("srv").await);

        let ctx = ToolCallContext::new("s".to_string(), None, None);
        let result = client
            .call_tool(
                &ctx,
                "load_skill",
                Some(serde_json::from_value(serde_json::json!({"name": "docs"})).unwrap()),
                CancellationToken::new(),
            )
            .await
            .unwrap();
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

        let mut resources = HashMap::new();
        resources.insert(
            "skill://index.json".to_string(),
            index_json(&fm_entry(
                "mcp-demo",
                "d",
                r#","url":"skill://mcp-demo/SKILL.md""#,
            )),
        );
        resources.insert(
            "skill://mcp-demo/SKILL.md".to_string(),
            "mcp body".to_string(),
        );
        resources.insert(
            "skill://mcp-demo/guide.md".to_string(),
            "mcp supporting body".to_string(),
        );
        let (client, _mgr, _tmp_guard) =
            setup_client_with_fake("srv", resources, tmp.path().to_path_buf()).await;

        let ctx = ToolCallContext::new("s".to_string(), None, None);
        let fs_result = client
            .call_tool(
                &ctx,
                "load_skill",
                Some(
                    serde_json::from_value::<JsonObject>(
                        serde_json::json!({"name": "fs-demo/guide.md"}),
                    )
                    .unwrap(),
                ),
                CancellationToken::new(),
            )
            .await
            .unwrap();
        let fs_text = text_of(&fs_result);

        let mcp_result = client
            .call_tool(
                &ctx,
                "load_skill",
                Some(
                    serde_json::from_value::<JsonObject>(
                        serde_json::json!({"name": "mcp-demo/guide.md"}),
                    )
                    .unwrap(),
                ),
                CancellationToken::new(),
            )
            .await
            .unwrap();
        let mcp_text = text_of(&mcp_result);

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
}
