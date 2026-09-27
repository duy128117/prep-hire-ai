"""reconcile deployed Supabase migration history

Revision ID: f6a2b8c9d1e4
Revises: c2f8a7b1d9e3
Create Date: 2026-05-27 00:00:00.000000

The deployed database already records this revision, but its original migration
file is no longer present in the repository. The current Supabase schema was
verified to contain all columns from c2f8a7b1d9e3, so this compatibility marker
restores the Alembic graph without changing existing data or schema.
"""

from typing import Sequence, Union


revision: str = "f6a2b8c9d1e4"
down_revision: Union[str, None] = "c2f8a7b1d9e3"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    pass


def downgrade() -> None:
    pass
