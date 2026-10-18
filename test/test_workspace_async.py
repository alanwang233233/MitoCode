import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from mitocode.core.workspace.workspace import Workspace
from mitocode.core.workspace.migration import BasicMigration


@pytest.mark.asyncio
async def test_workspace_supports_async_lifecycle(tmp_path):
    root = tmp_path / "async_workspace"
    root.mkdir()

    workspace = await Workspace.new_workspace(root, "async-agent", "async workspace")
    assert await workspace.validate() is True
    assert await workspace.is_outdated() is False

    workspace.config["description"] = "updated"
    assert await workspace.dump() is True
    assert await workspace.is_outdated() is False

    workspace.config["description"] = "memory-only-change"
    assert await workspace.is_outdated() is True

    assert await workspace.reload() is True
    assert workspace.config["description"] == "updated"

    loaded_workspace = await Workspace.init_from_path(root)
    assert loaded_workspace.config["name"] == "async-agent"


@pytest.mark.asyncio
async def test_workspace_migration_supports_async_call(tmp_path):
    root = tmp_path / "migration_workspace"
    root.mkdir()
    await Workspace.new_workspace(root, "migration-agent", "migration workspace")
    assert await BasicMigration.migrate(root) is True
