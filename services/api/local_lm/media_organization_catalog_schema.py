"""Keep catalog pages on one revision across metadata and membership changes."""

CATALOG_SEED_SQL = (
    "INSERT OR IGNORE INTO media_organization_catalog_revisions (kind, revision) "
    "VALUES ('albums', 1), ('tags', 1)"
)

CATALOG_GUARD_SQL = (
    """
CREATE TRIGGER media_catalog_revision_insert_guard
BEFORE INSERT ON media_organization_catalog_revisions
BEGIN
  SELECT CASE WHEN NEW.revision != 1
    THEN RAISE(ABORT, 'media catalog initial revision is invalid') END;
END
""",
    """
CREATE TRIGGER media_catalog_revision_update_guard
BEFORE UPDATE ON media_organization_catalog_revisions
BEGIN
  SELECT CASE WHEN NEW.kind != OLD.kind OR NEW.revision != OLD.revision + 1
    THEN RAISE(ABORT, 'media catalog revision is stale') END;
END
""",
    """
CREATE TRIGGER media_catalog_revision_delete_guard
BEFORE DELETE ON media_organization_catalog_revisions
BEGIN
  SELECT RAISE(ABORT, 'media catalog revision is required');
END
""",
)

CATALOG_CHANGE_SQL = tuple(
    f"""
CREATE TRIGGER media_catalog_{kind}_{operation.lower()}
AFTER {operation} ON {table}
BEGIN
  UPDATE media_organization_catalog_revisions
  SET revision = revision + 1 WHERE kind = '{kind}';
  SELECT CASE WHEN changes() != 1
    THEN RAISE(ABORT, 'media catalog revision is required') END;
END
"""
    for kind, table in (("albums", "media_collections"), ("tags", "media_tags"))
    for operation in ("INSERT", "UPDATE", "DELETE")
)

CREATE_CATALOG_TRIGGER_SQL = CATALOG_GUARD_SQL + CATALOG_CHANGE_SQL
DROP_CATALOG_TRIGGER_SQL = tuple(
    f"DROP TRIGGER IF EXISTS {statement.split()[2]}"
    for statement in reversed(CREATE_CATALOG_TRIGGER_SQL)
)
