"""Tests for mitocode.core.agent.template.template_manager (template discovery).

The manager scans ordered roots and serves a catalogue, exactly like
``SkillsManager``: a root that does not exist contributes nothing, an entry that
cannot be read costs only itself, and a name offered by two roots goes to the
first one.  Precedence is what lets a workspace override a template that ships
with the package, so it is tested here rather than assumed.
"""

import asyncio
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import pytest

from mitocode.core.agent.template.template import TemplateRoot
from mitocode.core.agent.template.template_manager import AgentTemplateManager


def _template_json(name, description="A test template.", blocks=None):
    """Build minimal valid template.json text."""
    return json.dumps(
        {"name": name, "description": description, "session_autoload": blocks or []}
    )


def _write_template(root: Path, name: str, text: str) -> Path:
    """Create one template directory holding the given template.json text."""
    entry = root / name
    entry.mkdir(parents=True)
    (entry / "template.json").write_text(text, encoding="utf-8")
    return entry


def _discover(roots):
    """Run one discovery pass over *roots* and return the manager."""
    manager = AgentTemplateManager(roots)
    asyncio.run(manager.discover())
    return manager


# --- construction ----------------------------------------------------------


def test_bare_path_and_str_roots_become_unlabelled_roots(tmp_path):
    """Bare paths are accepted; the order they were given in is kept."""
    path_root = tmp_path / "a"
    labeled = TemplateRoot("workspace", tmp_path / "c")

    manager = AgentTemplateManager([path_root, str(tmp_path / "b"), labeled])

    assert manager.roots() == (
        TemplateRoot("", path_root),
        TemplateRoot("", tmp_path / "b"),
        TemplateRoot("workspace", tmp_path / "c"),
    )


def test_a_bare_string_root_is_rejected():
    """A string is a sequence of strings, so accepting it would scan characters."""
    with pytest.raises(TypeError, match="sequence"):
        AgentTemplateManager("/some/template/dir")


def test_construction_reads_no_disk(tmp_path):
    """Building the manager must not depend on the roots existing."""
    manager = AgentTemplateManager([tmp_path / "missing"])

    assert manager.list_templates() == ()
    assert manager.issues() == ()


# --- discovery -------------------------------------------------------------


def test_templates_are_discovered_under_every_root_with_their_source_label(tmp_path):
    """Each template carries the label of the root it was found under."""
    workspace = tmp_path / "workspace"
    global_root = tmp_path / "global"
    builtin = tmp_path / "builtin"
    _write_template(workspace, "alpha", _template_json("alpha"))
    _write_template(global_root, "beta", _template_json("beta"))
    _write_template(builtin, "gamma", _template_json("gamma"))

    manager = _discover(
        [
            TemplateRoot("workspace", workspace),
            TemplateRoot("global", global_root),
            TemplateRoot("builtin", builtin),
        ]
    )

    found = {template.name: template.source for template in manager.list_templates()}
    assert found == {"alpha": "workspace", "beta": "global", "gamma": "builtin"}


def test_the_catalogue_is_sorted_by_name(tmp_path):
    """The catalogue order is stable, whatever the filesystem returns."""
    root = tmp_path / "root"
    for name in ("zeta", "alpha", "mid"):
        _write_template(root, name, _template_json(name))

    manager = _discover([root])

    assert [template.name for template in manager.list_templates()] == [
        "alpha",
        "mid",
        "zeta",
    ]


def test_each_template_records_its_directory_and_file(tmp_path):
    """Callers need both paths: the directory for display, the file to re-read."""
    root = tmp_path / "root"
    entry = _write_template(root, "alpha", _template_json("alpha"))

    template = _discover([root]).get("alpha")

    assert template is not None
    assert template.path == entry
    assert template.template_path == entry / "template.json"


def test_a_root_that_does_not_exist_contributes_nothing_and_no_issue(tmp_path):
    """A missing optional root is normal: only the packaged one ships."""
    manager = _discover([tmp_path / "missing"])

    assert manager.list_templates() == ()
    assert manager.issues() == ()


def test_a_root_that_is_a_file_is_recorded_as_an_issue(tmp_path):
    """A file where a directory belongs is a configuration mistake worth naming."""
    not_a_dir = tmp_path / "file"
    not_a_dir.write_text("x", encoding="utf-8")

    manager = _discover([not_a_dir])

    assert manager.list_templates() == ()
    assert "not a directory" in manager.issues()[0].detail


def test_hidden_directories_and_plain_files_are_skipped_silently(tmp_path):
    """Editor droppings are not broken templates and must not be reported."""
    root = tmp_path / "root"
    root.mkdir()
    (root / ".hidden").mkdir()
    (root / "notes.txt").write_text("x", encoding="utf-8")
    _write_template(root, "alpha", _template_json("alpha"))

    manager = _discover([root])

    assert [template.name for template in manager.list_templates()] == ["alpha"]
    assert manager.issues() == ()


def test_a_directory_without_a_template_json_is_an_issue(tmp_path):
    """A directory that looks like a template but has no file is a typo."""
    root = tmp_path / "root"
    (root / "empty").mkdir(parents=True)

    manager = _discover([root])

    assert manager.list_templates() == ()
    assert "template.json" in manager.issues()[0].detail


def test_a_symlinked_template_directory_is_refused(tmp_path):
    """A link planted where a user saw a directory must not serve its target."""
    root = tmp_path / "root"
    root.mkdir()
    target = tmp_path / "elsewhere"
    _write_template(target, "real", _template_json("real"))
    (root / "linked").symlink_to(target / "real")

    manager = _discover([root])

    assert manager.list_templates() == ()
    assert "symlink" in manager.issues()[0].detail


def test_a_symlinked_template_json_is_refused(tmp_path):
    """The file itself gets the same treatment as the directory."""
    root = tmp_path / "root"
    entry = root / "linked"
    entry.mkdir(parents=True)
    outside = tmp_path / "outside.json"
    outside.write_text(_template_json("linked"), encoding="utf-8")
    (entry / "template.json").symlink_to(outside)

    manager = _discover([root])

    assert manager.list_templates() == ()
    assert "symlink" in manager.issues()[0].detail


def test_one_broken_template_does_not_cost_the_working_ones(tmp_path):
    """Scanning is strict per entry and forgiving overall."""
    root = tmp_path / "root"
    _write_template(root, "broken", "{not json")
    _write_template(root, "good", _template_json("good"))

    manager = _discover([root])

    assert [template.name for template in manager.list_templates()] == ["good"]
    assert len(manager.issues()) == 1


def test_a_name_that_differs_from_the_directory_is_an_issue(tmp_path):
    """The directory is the address, so the field must agree with it."""
    root = tmp_path / "root"
    _write_template(root, "alpha", _template_json("beta"))

    manager = _discover([root])

    assert manager.list_templates() == ()
    assert "directory name" in manager.issues()[0].detail


# --- precedence ------------------------------------------------------------


def test_an_earlier_root_shadows_a_later_one_and_says_so(tmp_path):
    """Workspace beats global beats builtin, and the loser stays visible."""
    workspace = tmp_path / "workspace"
    builtin = tmp_path / "builtin"
    _write_template(workspace, "shared", _template_json("shared", "workspace copy"))
    _write_template(builtin, "shared", _template_json("shared", "builtin copy"))

    manager = _discover(
        [TemplateRoot("workspace", workspace), TemplateRoot("builtin", builtin)]
    )

    templates = manager.list_templates()
    assert len(templates) == 1
    assert templates[0].source == "workspace"
    assert templates[0].description == "workspace copy"
    assert "shadowed" in manager.issues()[0].detail


# --- served catalogue ------------------------------------------------------


def test_get_returns_none_for_an_unknown_name(tmp_path):
    """An unknown name is a question, not an error: the tool words the refusal."""
    assert _discover([tmp_path / "missing"]).get("nope") is None


def test_a_second_discovery_replaces_the_catalogue_and_the_issues(tmp_path):
    """A problem that goes away stops being reported."""
    root = tmp_path / "root"
    _write_template(root, "broken", "{not json")
    manager = AgentTemplateManager([root])
    asyncio.run(manager.discover())
    assert manager.issues()

    (root / "broken" / "template.json").write_text(
        _template_json("broken"), encoding="utf-8"
    )
    asyncio.run(manager.discover())

    assert [template.name for template in manager.list_templates()] == ["broken"]
    assert manager.issues() == ()
