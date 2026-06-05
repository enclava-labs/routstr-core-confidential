"""merge confidential routing and api key created_at heads

Revision ID: merge_conf_route_created_at
Revises: db_conf_route_001, f1a2b3c4d5e6
Create Date: 2026-06-05 00:00:00.000000
"""

from __future__ import annotations

revision = "merge_conf_route_created_at"
down_revision = ("db_conf_route_001", "f1a2b3c4d5e6")
branch_labels = None
depends_on = None


def upgrade() -> None:
    pass


def downgrade() -> None:
    pass
