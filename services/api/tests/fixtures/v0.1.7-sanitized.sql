-- Sanitized SQLite dump produced from LM Atelier v0.1.7 at Alembic
-- revision f7a2c9e51b40. It contains synthetic records and no user data.
BEGIN TRANSACTION;
CREATE TABLE alembic_version (
    version_num VARCHAR(32) NOT NULL,
    CONSTRAINT alembic_version_pkc PRIMARY KEY (version_num)
);
INSERT INTO alembic_version VALUES ('f7a2c9e51b40');
CREATE TABLE app_settings (
    "key" VARCHAR(200) NOT NULL,
    value_json JSON NOT NULL,
    created_at DATETIME NOT NULL,
    updated_at DATETIME NOT NULL,
    PRIMARY KEY ("key")
);
CREATE TABLE artifacts (
    id VARCHAR(80) NOT NULL,
    sha256 VARCHAR(64) NOT NULL,
    kind VARCHAR(24) NOT NULL,
    media_type VARCHAR(120) NOT NULL,
    size_bytes INTEGER NOT NULL,
    relative_path TEXT NOT NULL,
    original_name VARCHAR(500),
    metadata_json JSON NOT NULL,
    created_at DATETIME NOT NULL,
    updated_at DATETIME NOT NULL,
    PRIMARY KEY (id),
    UNIQUE (relative_path)
);
CREATE TABLE chats (
    id VARCHAR(40) NOT NULL,
    project_id VARCHAR(40),
    title VARCHAR(240) NOT NULL,
    archived BOOLEAN NOT NULL,
    routing_mode VARCHAR(16) NOT NULL,
    confirm_uncertain_media BOOLEAN NOT NULL,
    active_chat_profile_id VARCHAR(40),
    active_image_profile_id VARCHAR(40),
    active_video_profile_id VARCHAR(40),
    created_at DATETIME NOT NULL,
    updated_at DATETIME NOT NULL,
    active_head_message_id VARCHAR(40),
    PRIMARY KEY (id),
    FOREIGN KEY(project_id) REFERENCES projects (id) ON DELETE SET NULL
);
CREATE TABLE custom_node_installs (
    id VARCHAR(40) NOT NULL,
    name VARCHAR(240) NOT NULL,
    source_url VARCHAR(1000) NOT NULL,
    revision VARCHAR(40) NOT NULL,
    previous_revision VARCHAR(40),
    installed_path TEXT NOT NULL,
    tree_hash VARCHAR(64) NOT NULL,
    trusted BOOLEAN DEFAULT 0 NOT NULL,
    active BOOLEAN DEFAULT 1 NOT NULL,
    security_json JSON NOT NULL,
    created_at DATETIME NOT NULL,
    updated_at DATETIME NOT NULL,
    PRIMARY KEY (id),
    UNIQUE (source_url)
);
CREATE TABLE generation_presets (
    id VARCHAR(40) NOT NULL,
    name VARCHAR(200) NOT NULL,
    role VARCHAR(16) NOT NULL,
    settings_json JSON NOT NULL,
    is_default BOOLEAN NOT NULL,
    created_at DATETIME NOT NULL,
    updated_at DATETIME NOT NULL,
    PRIMARY KEY (id),
    CONSTRAINT uq_preset_role_name UNIQUE (role, name)
);
CREATE TABLE jobs (
    id VARCHAR(40) NOT NULL,
    kind VARCHAR(16) NOT NULL,
    status VARCHAR(16) NOT NULL,
    run_id VARCHAR(40),
    progress FLOAT NOT NULL,
    phase VARCHAR(120) NOT NULL,
    payload_json JSON NOT NULL,
    result_json JSON NOT NULL,
    error TEXT,
    attempt INTEGER NOT NULL,
    cancellable BOOLEAN NOT NULL,
    started_at DATETIME,
    completed_at DATETIME,
    created_at DATETIME NOT NULL,
    updated_at DATETIME NOT NULL,
    PRIMARY KEY (id),
    FOREIGN KEY(run_id) REFERENCES runs (id) ON DELETE SET NULL
);
CREATE TABLE message_parts (
    id VARCHAR(40) NOT NULL,
    message_id VARCHAR(40) NOT NULL,
    position INTEGER NOT NULL,
    type VARCHAR(32) NOT NULL,
    text TEXT,
    artifact_id VARCHAR(80),
    metadata_json JSON NOT NULL,
    created_at DATETIME NOT NULL,
    updated_at DATETIME NOT NULL,
    PRIMARY KEY (id),
    FOREIGN KEY(artifact_id) REFERENCES artifacts (id) ON DELETE SET NULL,
    FOREIGN KEY(message_id) REFERENCES messages (id) ON DELETE CASCADE,
    CONSTRAINT uq_message_part_position UNIQUE (message_id, position)
);
CREATE TABLE messages (
    id VARCHAR(40) NOT NULL,
    chat_id VARCHAR(40) NOT NULL,
    parent_id VARCHAR(40),
    role VARCHAR(16) NOT NULL,
    status VARCHAR(16) NOT NULL,
    created_at DATETIME NOT NULL,
    updated_at DATETIME NOT NULL,
    PRIMARY KEY (id),
    FOREIGN KEY(chat_id) REFERENCES chats (id) ON DELETE CASCADE,
    FOREIGN KEY(parent_id) REFERENCES messages (id) ON DELETE SET NULL
);
CREATE TABLE model_installs (
    id VARCHAR(40) NOT NULL,
    source_id VARCHAR(40),
    name VARCHAR(300) NOT NULL,
    role VARCHAR(16) NOT NULL,
    engine VARCHAR(32) NOT NULL,
    local_path TEXT NOT NULL,
    size_bytes INTEGER NOT NULL,
    compatibility VARCHAR(24) NOT NULL,
    manifest_json JSON NOT NULL,
    active BOOLEAN NOT NULL,
    created_at DATETIME NOT NULL,
    updated_at DATETIME NOT NULL,
    PRIMARY KEY (id),
    FOREIGN KEY(source_id) REFERENCES model_sources (id) ON DELETE SET NULL
);
CREATE TABLE model_profiles (
    id VARCHAR(40) NOT NULL,
    model_install_id VARCHAR(40),
    name VARCHAR(200) NOT NULL,
    role VARCHAR(16) NOT NULL,
    engine VARCHAR(32) NOT NULL,
    load_settings_json JSON NOT NULL,
    request_settings_json JSON NOT NULL,
    is_default BOOLEAN NOT NULL,
    created_at DATETIME NOT NULL,
    updated_at DATETIME NOT NULL,
    use_case TEXT DEFAULT '' NOT NULL,
    PRIMARY KEY (id),
    FOREIGN KEY(model_install_id) REFERENCES model_installs (id) ON DELETE CASCADE
);
CREATE TABLE model_sources (
    id VARCHAR(40) NOT NULL,
    provider VARCHAR(32) NOT NULL,
    remote_id VARCHAR(500) NOT NULL,
    revision VARCHAR(200) NOT NULL,
    metadata_json JSON NOT NULL,
    created_at DATETIME NOT NULL,
    updated_at DATETIME NOT NULL,
    PRIMARY KEY (id),
    CONSTRAINT uq_model_source UNIQUE (provider, remote_id, revision)
);
CREATE TABLE projects (
    id VARCHAR(40) NOT NULL,
    name VARCHAR(200) NOT NULL,
    description TEXT NOT NULL,
    instructions TEXT NOT NULL,
    archived BOOLEAN NOT NULL,
    created_at DATETIME NOT NULL,
    updated_at DATETIME NOT NULL,
    image_workflow_revision_id VARCHAR(40),
    video_workflow_revision_id VARCHAR(40),
    PRIMARY KEY (id),
    CONSTRAINT fk_projects_image_workflow_revision
        FOREIGN KEY(image_workflow_revision_id)
        REFERENCES workflow_revisions (id) ON DELETE SET NULL,
    CONSTRAINT fk_projects_video_workflow_revision
        FOREIGN KEY(video_workflow_revision_id)
        REFERENCES workflow_revisions (id) ON DELETE SET NULL
);
CREATE TABLE runs (
    id VARCHAR(40) NOT NULL,
    idempotency_key VARCHAR(200),
    chat_id VARCHAR(40) NOT NULL,
    user_message_id VARCHAR(40) NOT NULL,
    assistant_message_id VARCHAR(40) NOT NULL,
    operation VARCHAR(32) NOT NULL,
    status VARCHAR(16) NOT NULL,
    standalone_prompt TEXT NOT NULL,
    profile_id VARCHAR(40),
    workflow_revision_id VARCHAR(40),
    settings_json JSON NOT NULL,
    provenance_json JSON NOT NULL,
    error TEXT,
    started_at DATETIME,
    completed_at DATETIME,
    duration_ms INTEGER,
    created_at DATETIME NOT NULL,
    updated_at DATETIME NOT NULL,
    PRIMARY KEY (id),
    FOREIGN KEY(assistant_message_id) REFERENCES messages (id) ON DELETE CASCADE,
    FOREIGN KEY(chat_id) REFERENCES chats (id) ON DELETE CASCADE,
    FOREIGN KEY(user_message_id) REFERENCES messages (id) ON DELETE CASCADE,
    UNIQUE (assistant_message_id),
    UNIQUE (idempotency_key)
);
CREATE TABLE workflow_definitions (
    id VARCHAR(40) NOT NULL,
    name VARCHAR(240) NOT NULL,
    operation VARCHAR(32) NOT NULL,
    description TEXT NOT NULL,
    current_revision_id VARCHAR(40),
    created_at DATETIME NOT NULL,
    updated_at DATETIME NOT NULL,
    PRIMARY KEY (id)
);
CREATE TABLE workflow_revisions (
    id VARCHAR(40) NOT NULL,
    workflow_id VARCHAR(40) NOT NULL,
    version INTEGER NOT NULL,
    engine VARCHAR(32) NOT NULL,
    engine_version VARCHAR(100),
    ui_graph_json JSON NOT NULL,
    api_graph_json JSON NOT NULL,
    input_schema_json JSON NOT NULL,
    dependencies_json JSON NOT NULL,
    trusted BOOLEAN NOT NULL,
    created_at DATETIME NOT NULL,
    updated_at DATETIME NOT NULL,
    PRIMARY KEY (id),
    FOREIGN KEY(workflow_id) REFERENCES workflow_definitions (id) ON DELETE CASCADE,
    CONSTRAINT uq_workflow_version UNIQUE (workflow_id, version)
);
CREATE UNIQUE INDEX ix_artifacts_sha256 ON artifacts (sha256);
CREATE INDEX ix_workflow_definitions_name ON workflow_definitions (name);
CREATE INDEX ix_chats_archived ON chats (archived);
CREATE INDEX ix_chats_project_id ON chats (project_id);
CREATE INDEX ix_model_installs_name ON model_installs (name);
CREATE INDEX ix_workflow_revisions_workflow_id ON workflow_revisions (workflow_id);
CREATE INDEX ix_messages_chat_id ON messages (chat_id);
CREATE INDEX ix_messages_parent_id ON messages (parent_id);
CREATE INDEX ix_model_profiles_name ON model_profiles (name);
CREATE INDEX ix_message_parts_message_id ON message_parts (message_id);
CREATE INDEX ix_runs_chat_id ON runs (chat_id);
CREATE INDEX ix_runs_status ON runs (status);
CREATE INDEX ix_jobs_run_id ON jobs (run_id);
CREATE INDEX ix_jobs_status ON jobs (status);
CREATE INDEX ix_generation_presets_name ON generation_presets (name);
CREATE INDEX ix_generation_presets_role ON generation_presets (role);
CREATE INDEX ix_projects_name ON projects (name);
CREATE INDEX ix_projects_archived ON projects (archived);
CREATE INDEX ix_custom_node_installs_name ON custom_node_installs (name);
CREATE INDEX ix_custom_node_installs_trusted ON custom_node_installs (trusted);
CREATE INDEX ix_custom_node_installs_active ON custom_node_installs (active);
INSERT INTO app_settings VALUES (
    'last_chat_profile_id',
    '"profile_v017"',
    '2026-07-25 00:00:00',
    '2026-07-25 00:00:00'
);
INSERT INTO artifacts VALUES (
    'artifact_v017',
    'aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa',
    'image',
    'image/png',
    12,
    'sha256/aa/aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa',
    'synthetic.png',
    '{"synthetic":true}',
    '2026-07-25 00:00:00',
    '2026-07-25 00:00:00'
);
INSERT INTO model_sources VALUES (
    'source_v017',
    'huggingface',
    'example/synthetic-model',
    'bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb',
    '{"license":"Apache-2.0"}',
    '2026-07-25 00:00:00',
    '2026-07-25 00:00:00'
);
INSERT INTO model_installs VALUES (
    'install_v017',
    'source_v017',
    'Synthetic model',
    'chat',
    'llama.cpp',
    'C:\sanitized\models\synthetic.gguf',
    4096,
    'likely',
    '{"files":["synthetic.gguf"]}',
    1,
    '2026-07-25 00:00:00',
    '2026-07-25 00:00:00'
);
INSERT INTO model_profiles VALUES (
    'profile_v017',
    'install_v017',
    'Synthetic profile',
    'chat',
    'llama.cpp',
    '{"context_length":4096}',
    '{"max_tokens":256}',
    1,
    '2026-07-25 00:00:00',
    '2026-07-25 00:00:00',
    'writing'
);
INSERT INTO generation_presets VALUES (
    'preset_v017',
    'Synthetic preset',
    'chat',
    '{"temperature":0.5}',
    1,
    '2026-07-25 00:00:00',
    '2026-07-25 00:00:00'
);
INSERT INTO workflow_definitions VALUES (
    'workflow_v017',
    'Synthetic image workflow',
    'text_to_image',
    'Sanitized upgrade fixture',
    'revision_v017',
    '2026-07-25 00:00:00',
    '2026-07-25 00:00:00'
);
INSERT INTO workflow_revisions VALUES (
    'revision_v017',
    'workflow_v017',
    1,
    'comfyui',
    '0.28.0',
    '{}',
    '{"1":{"class_type":"SaveImage","inputs":{}}}',
    '{}',
    '{"models":[]}',
    1,
    '2026-07-25 00:00:00',
    '2026-07-25 00:00:00'
);
INSERT INTO projects VALUES (
    'project_v017',
    'Synthetic project',
    'No private data',
    'Keep answers concise',
    0,
    '2026-07-25 00:00:00',
    '2026-07-25 00:00:00',
    'revision_v017',
    NULL
);
INSERT INTO chats VALUES (
    'chat_v017',
    'project_v017',
    'Synthetic chat',
    0,
    'auto',
    1,
    'profile_v017',
    NULL,
    NULL,
    '2026-07-25 00:00:00',
    '2026-07-25 00:00:00',
    'assistant_v017'
);
INSERT INTO messages VALUES (
    'user_v017',
    'chat_v017',
    NULL,
    'user',
    'complete',
    '2026-07-25 00:00:00',
    '2026-07-25 00:00:00'
);
INSERT INTO messages VALUES (
    'assistant_v017',
    'chat_v017',
    'user_v017',
    'assistant',
    'complete',
    '2026-07-25 00:00:01',
    '2026-07-25 00:00:01'
);
INSERT INTO message_parts VALUES (
    'part_user_v017',
    'user_v017',
    0,
    'text',
    'Synthetic prompt',
    NULL,
    '{}',
    '2026-07-25 00:00:00',
    '2026-07-25 00:00:00'
);
INSERT INTO message_parts VALUES (
    'part_assistant_v017',
    'assistant_v017',
    0,
    'text',
    'Synthetic response',
    NULL,
    '{}',
    '2026-07-25 00:00:01',
    '2026-07-25 00:00:01'
);
INSERT INTO runs VALUES (
    'run_v017',
    'legacy-idempotency-key',
    'chat_v017',
    'user_v017',
    'assistant_v017',
    'text',
    'complete',
    'Synthetic prompt',
    'profile_v017',
    NULL,
    '{"max_tokens":256}',
    '{"input_artifact_ids":["artifact_v017"],"synthetic":true}',
    NULL,
    '2026-07-25 00:00:00',
    '2026-07-25 00:00:01',
    1000,
    '2026-07-25 00:00:00',
    '2026-07-25 00:00:01'
);
INSERT INTO jobs VALUES (
    'job_v017',
    'chat',
    'complete',
    'run_v017',
    1.0,
    'complete',
    '{"operation":"text"}',
    '{"finish_reason":"stop"}',
    NULL,
    1,
    0,
    '2026-07-25 00:00:00',
    '2026-07-25 00:00:01',
    '2026-07-25 00:00:00',
    '2026-07-25 00:00:01'
);
INSERT INTO custom_node_installs VALUES (
    'node_v017',
    'Synthetic node',
    'https://github.com/example/synthetic-node.git',
    'cccccccccccccccccccccccccccccccccccccccc',
    NULL,
    'lm-atelier-node_synthetic',
    'dddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddd',
    0,
    0,
    '{"review_required":true}',
    '2026-07-25 00:00:00',
    '2026-07-25 00:00:00'
);
COMMIT;
