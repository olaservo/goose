use super::*;

impl GooseAcpAgent {
    pub(super) async fn on_create_source(
        &self,
        req: CreateSourceRequest,
    ) -> Result<CreateSourceResponse, agent_client_protocol::Error> {
        let (global, project_dir) = resolve_source_scope(&req.target)?;
        let source = crate::sources::create_source(
            req.source_type,
            &req.name,
            &req.description,
            &req.content,
            global,
            project_dir.as_deref(),
            req.properties,
        )?;
        Ok(CreateSourceResponse { source })
    }

    pub(super) async fn on_list_sources(
        &self,
        req: ListSourcesRequest,
    ) -> Result<ListSourcesResponse, agent_client_protocol::Error> {
        let mut sources = crate::sources::list_sources_with_roots(
            req.source_type,
            req.project_dir.as_deref(),
            req.include_project_sources,
            &self.additional_source_roots,
        )?;
        if matches!(req.source_type, None | Some(SourceType::Skill)) {
            sources.extend(self.live_mcp_skill_sources().await);
        }
        Ok(ListSourcesResponse { sources })
    }

    /// MCP-served skills exist only in a live session's extension manager, so
    /// they are listed from every open session and deduplicated by URI.
    async fn live_mcp_skill_sources(&self) -> Vec<SourceEntry> {
        let agents: Vec<(String, Arc<Agent>)> = self
            .sessions
            .lock()
            .await
            .iter()
            .map(|(id, session)| (id.clone(), session.agent.clone()))
            .collect();

        let mut seen = HashSet::new();
        let mut sources = Vec::new();
        for (session_id, agent) in agents {
            for entry in agent.extension_manager.aggregated_mcp_skills().await {
                if !seen.insert((entry.server.clone(), entry.uri.clone())) {
                    continue;
                }
                let attribution = crate::skills::attribution::grade_entry(
                    &agent.extension_manager,
                    &entry,
                    &session_id,
                )
                .await;
                let properties = HashMap::from([
                    ("mcpServer".to_string(), serde_json::json!(entry.server)),
                    (
                        "complianceStatus".to_string(),
                        serde_json::json!(attribution.compliance),
                    ),
                    (
                        "attributionSummary".to_string(),
                        serde_json::json!(attribution.summary),
                    ),
                    (
                        "sealStatus".to_string(),
                        serde_json::json!(attribution.seal),
                    ),
                    (
                        "sealSummary".to_string(),
                        serde_json::json!(attribution.seal_summary),
                    ),
                ]);
                sources.push(SourceEntry {
                    source_type: SourceType::Skill,
                    name: entry.name,
                    description: entry.description,
                    path: entry.uri,
                    properties,
                    ..Default::default()
                });
            }
        }
        sources
    }

    pub(super) async fn on_update_source(
        &self,
        req: UpdateSourceRequest,
    ) -> Result<UpdateSourceResponse, agent_client_protocol::Error> {
        let source = crate::sources::update_source_with_roots(
            req.source_type,
            &req.path,
            &req.name,
            &req.description,
            &req.content,
            crate::sources::UpdateSourceOptions {
                properties: req.properties,
                additional_roots: &self.additional_source_roots,
            },
        )?;
        Ok(UpdateSourceResponse { source })
    }

    pub(super) async fn on_delete_source(
        &self,
        req: DeleteSourceRequest,
    ) -> Result<EmptyResponse, agent_client_protocol::Error> {
        crate::sources::delete_source_with_roots(
            req.source_type,
            &req.path,
            &self.additional_source_roots,
        )?;
        Ok(EmptyResponse {})
    }

    pub(super) async fn on_export_source(
        &self,
        req: ExportSourceRequest,
    ) -> Result<ExportSourceResponse, agent_client_protocol::Error> {
        let (json, filename) = crate::sources::export_source_with_roots(
            req.source_type,
            &req.path,
            &self.additional_source_roots,
        )?;
        Ok(ExportSourceResponse { json, filename })
    }

    pub(super) async fn on_import_sources(
        &self,
        req: ImportSourcesRequest,
    ) -> Result<ImportSourcesResponse, agent_client_protocol::Error> {
        let (global, project_dir) = resolve_source_scope(&req.target)?;
        let sources = crate::sources::import_sources(&req.data, global, project_dir.as_deref())?;
        Ok(ImportSourcesResponse { sources })
    }
}

fn resolve_source_scope(
    target: &SourceScope,
) -> Result<(bool, Option<String>), agent_client_protocol::Error> {
    match target {
        SourceScope::Global => Ok((true, None)),
        SourceScope::ProjectDir { project_dir } => Ok((false, Some(project_dir.clone()))),
        SourceScope::ProjectId { project_id } => {
            let dirs = crate::sources::project_working_dirs(project_id);
            let project_dir = dirs.into_iter().next().ok_or_else(|| {
                agent_client_protocol::Error::invalid_params().data(format!(
                    "Project \"{project_id}\" has no working directories configured"
                ))
            })?;
            Ok((false, Some(project_dir)))
        }
    }
}
