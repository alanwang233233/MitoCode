import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from mitocode import cli


def test_workspace_init_and_info_commands(tmp_path, capsys):
    exit_code = cli.main(
        [
            "workspace",
            "init",
            "--path",
            str(tmp_path),
            "--name",
            "demo-workspace",
            "--description",
            "demo",
        ]
    )
    assert exit_code == 0
    assert (tmp_path / ".mitocode" / "config.json").exists()
    capsys.readouterr()

    exit_code = cli.main(["workspace", "info", "--path", str(tmp_path)])
    assert exit_code == 0
    info = json.loads(capsys.readouterr().out)
    assert info["name"] == "demo-workspace"
    assert info["description"] == "demo"


def test_workspace_validate_and_outdated_commands(tmp_path, capsys):
    assert (
        cli.main(
            [
                "workspace",
                "init",
                "--path",
                str(tmp_path),
                "--name",
                "validate-workspace",
            ]
        )
        == 0
    )

    assert cli.main(["workspace", "validate", "--path", str(tmp_path)]) == 0
    out = capsys.readouterr().out
    assert "valid" in out

    assert cli.main(["workspace", "is-outdated", "--path", str(tmp_path)]) == 0
    assert "false" in capsys.readouterr().out


def test_workspace_migrate_command(tmp_path, capsys):
    assert (
        cli.main(["workspace", "init", "--path", str(tmp_path), "--name", "migrate"])
        == 0
    )
    assert cli.main(["workspace", "migrate", "--path", str(tmp_path)]) == 0
    assert "migrated" in capsys.readouterr().out
