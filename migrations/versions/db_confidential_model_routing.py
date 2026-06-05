"""add db backed confidential model routing state

Revision ID: db_conf_route_001
Revises: cli_tokens_001
Create Date: 2026-06-04 00:00:00.000000
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision = "db_conf_route_001"
down_revision = "cli_tokens_001"
branch_labels = None
depends_on = None


def upgrade() -> None:
    conn = op.get_bind()
    inspector = sa.inspect(conn)
    columns = [c["name"] for c in inspector.get_columns("models")]

    if "last_seen_at" not in columns:
        op.add_column("models", sa.Column("last_seen_at", sa.Integer(), nullable=True))
    if "availability_status" not in columns:
        op.add_column(
            "models", sa.Column("availability_status", sa.String(), nullable=True)
        )
    if "capabilities_json" not in columns:
        op.add_column(
            "models", sa.Column("capabilities_json", sa.Text(), nullable=True)
        )

    if "provider_model_attestations" not in inspector.get_table_names():
        op.create_table(
            "provider_model_attestations",
            sa.Column("provider_id", sa.Integer(), nullable=False),
            sa.Column("model_id", sa.String(), nullable=False),
            sa.Column("mode", sa.String(), nullable=False),
            sa.Column("verified", sa.Boolean(), nullable=False, server_default="0"),
            sa.Column("verified_at", sa.Integer(), nullable=True),
            sa.Column("expires_at", sa.Integer(), nullable=True),
            sa.Column("policy_digest", sa.String(), nullable=True),
            sa.Column("evidence_digest", sa.String(), nullable=True),
            sa.Column("verifier", sa.String(), nullable=True),
            sa.Column("claims_json", sa.Text(), nullable=True),
            sa.Column("failure_reason", sa.Text(), nullable=True),
            sa.Column("updated_at", sa.Integer(), nullable=False),
            sa.PrimaryKeyConstraint("provider_id", "model_id", "mode"),
            sa.ForeignKeyConstraint(
                ["provider_id"], ["upstream_providers.id"], ondelete="CASCADE"
            ),
        )


def downgrade() -> None:
    conn = op.get_bind()
    inspector = sa.inspect(conn)

    if "provider_model_attestations" in inspector.get_table_names():
        op.drop_table("provider_model_attestations")

    columns = [c["name"] for c in inspector.get_columns("models")]
    if "capabilities_json" in columns:
        op.drop_column("models", "capabilities_json")
    if "availability_status" in columns:
        op.drop_column("models", "availability_status")
    if "last_seen_at" in columns:
        op.drop_column("models", "last_seen_at")
