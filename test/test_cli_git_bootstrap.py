"""cli.git_bootstrap 模块单元测试。

覆盖四个分支：git 二进制缺失（提醒且不调 subprocess）、已在仓库内
（探测通过即返回，不重复 init）、无仓库（自动 git init）、init 失败
（警告不阻塞）。另覆盖 OSError 包装分支。
"""

import os
import subprocess
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from rich.console import Console

from mitocode.cli import git_bootstrap
from mitocode.cli.git_bootstrap import _run, ensure_git_repo


def _proc(code: int = 0, stderr: str = "") -> subprocess.CompletedProcess:
    """构造指定返回码的 CompletedProcess 替身."""
    return subprocess.CompletedProcess(
        args=[], returncode=code, stdout="", stderr=stderr
    )


def _record_console() -> Console:
    """构造带记录的 Console，便于断言本地化输出."""
    return Console(record=True, width=120)


def test_missing_git_warns_and_skips_subprocess(tmp_path, monkeypatch):
    """PATH 无 git → 本地化提醒、返回 False、不启动任何子进程."""
    monkeypatch.setenv("MITOCODE_LANG", "en")
    monkeypatch.setattr(git_bootstrap.shutil, "which", lambda name: None)
    runs = []
    monkeypatch.setattr(
        git_bootstrap.subprocess, "run", lambda *a, **k: runs.append(a) or _proc()
    )
    console = _record_console()

    assert ensure_git_repo(tmp_path, console) is False
    assert runs == []
    assert "git was not found" in console.export_text()


def test_existing_repo_returns_true_without_init(tmp_path, monkeypatch):
    """rev-parse 成功 → 已在仓库内，返回 True 且不执行 git init."""
    monkeypatch.setattr(git_bootstrap.shutil, "which", lambda name: "/usr/bin/git")
    calls = []

    def fake_run(cmd, **kwargs):
        calls.append(cmd)
        return _proc(0)

    monkeypatch.setattr(git_bootstrap.subprocess, "run", fake_run)

    assert ensure_git_repo(tmp_path, Console()) is True
    assert calls == [["git", "rev-parse", "--is-inside-work-tree"]]


def test_no_repo_auto_inits(tmp_path, monkeypatch):
    """rev-parse 失败 → 自动 git init，成功提示，返回 True."""
    monkeypatch.setenv("MITOCODE_LANG", "en")
    monkeypatch.setattr(git_bootstrap.shutil, "which", lambda name: "/usr/bin/git")
    calls = []

    def fake_run(cmd, **kwargs):
        calls.append(cmd)
        if cmd[1] == "init":
            return _proc(0)
        return _proc(1)

    monkeypatch.setattr(git_bootstrap.subprocess, "run", fake_run)
    console = _record_console()

    assert ensure_git_repo(tmp_path, console) is True
    assert calls == [
        ["git", "rev-parse", "--is-inside-work-tree"],
        ["git", "init"],
    ]
    assert "Initialized a new git repository" in console.export_text()


def test_init_failure_warns_without_raising(tmp_path, monkeypatch):
    """git init 失败 → 警告（含 stderr），返回 False，不抛异常."""
    monkeypatch.setenv("MITOCODE_LANG", "en")
    monkeypatch.setattr(git_bootstrap.shutil, "which", lambda name: "/usr/bin/git")

    def fake_run(cmd, **kwargs):
        if cmd[1] == "init":
            return _proc(1, stderr="boom")
        return _proc(1)

    monkeypatch.setattr(git_bootstrap.subprocess, "run", fake_run)
    console = _record_console()

    assert ensure_git_repo(tmp_path, console) is False
    output = console.export_text()
    assert "Failed to run 'git init'" in output
    assert "boom" in output


def test_run_returns_none_on_oserror(tmp_path, monkeypatch):
    """子进程启动抛 OSError → _run 返回 None 而非向上传播."""
    monkeypatch.setattr(
        git_bootstrap.subprocess,
        "run",
        lambda *a, **k: (_ for _ in ()).throw(OSError("gone")),
    )

    assert _run(["git", "init"], tmp_path) is None
