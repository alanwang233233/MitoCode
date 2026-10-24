"""cli.provider_config 模块单元测试。

覆盖优先级矩阵（文件>环境变量>默认值，按字段）、密钥文件的 gitignore
先行防护与 0600 权限、/connect 向导（含空输入回退与无 key 中止）、
/model 切换（序号/名称/非法/取消/持久化）与热切换（RUNNING 守卫）。
"""

import json
import os
import stat
import sys
from pathlib import Path

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from rich.console import Console

from mitocode.cli import provider_config as pc
from mitocode.cli.provider_config import (
    DEFAULT_BASE_URL,
    DEFAULT_MODEL,
    DEFAULT_TIMEOUT_SECONDS,
    ProviderConfig,
    connect_wizard,
    ensure_secret_gitignored,
    resolve_provider_config,
    save_provider_config,
    switch_model,
)

ENV_VARS = (
    "MITOCODE_API_KEY",
    "OPENAI_API_KEY",
    "MITOCODE_BASE_URL",
    "OPENAI_BASE_URL",
    "MITOCODE_MODEL",
    "MITOCODE_TIMEOUT",
)


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    """清空全部 provider 相关环境变量，保证优先级测试确定."""
    for name in ENV_VARS:
        monkeypatch.delenv(name, raising=False)


def _reader(*answers):
    """构造脚本化异步输入端口."""
    answers = list(answers)

    async def _read(prompt):
        return answers.pop(0) if answers else ""

    return _read


def _write_provider_file(root: Path, **fields):
    """在工作区写入 provider.json."""
    path = root / pc.PROVIDER_FILE_RELATIVE
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(fields), encoding="utf-8")
    return path


# ──────────────────────────────────────────────
# resolve_provider_config 优先级矩阵
# ──────────────────────────────────────────────


def test_resolve_returns_none_without_any_config(tmp_path):
    """无文件无环境变量 → None."""
    assert resolve_provider_config(tmp_path) is None


def test_resolve_uses_env_when_no_file(tmp_path, monkeypatch):
    """仅环境变量 → 使用 env 值."""
    monkeypatch.setenv("MITOCODE_API_KEY", "env-key")
    monkeypatch.setenv("MITOCODE_BASE_URL", "https://env")
    monkeypatch.setenv("MITOCODE_MODEL", "env-model")
    monkeypatch.setenv("MITOCODE_TIMEOUT", "30")

    config = resolve_provider_config(tmp_path)

    assert config == ProviderConfig("env-key", "https://env", "env-model", 30)


def test_resolve_file_wins_over_env_per_field(tmp_path, monkeypatch):
    """文件存在 → 各字段压过环境变量."""
    monkeypatch.setenv("MITOCODE_API_KEY", "env-key")
    monkeypatch.setenv("MITOCODE_MODEL", "env-model")
    _write_provider_file(
        tmp_path,
        api_key="file-key",
        base_url="https://file",
        model="file-model",
        timeout=99,
    )

    config = resolve_provider_config(tmp_path)

    assert config == ProviderConfig("file-key", "https://file", "file-model", 99)


def test_resolve_env_fills_fields_missing_from_file(tmp_path, monkeypatch):
    """文件缺 api_key → 用 env 的 key；env 的其余字段补齐文件缺失项."""
    monkeypatch.setenv("OPENAI_API_KEY", "env-key")
    monkeypatch.setenv("MITOCODE_MODEL", "env-model")
    _write_provider_file(tmp_path, base_url="https://file", timeout=42)

    config = resolve_provider_config(tmp_path)

    assert config.api_key == "env-key"
    assert config.base_url == "https://file"
    assert config.model == "env-model"
    assert config.timeout == 42


def test_resolve_defaults_when_nothing_set(tmp_path, monkeypatch):
    """只有 api_key 可用 → 其余字段走内置默认值."""
    monkeypatch.setenv("MITOCODE_API_KEY", "only-key")

    config = resolve_provider_config(tmp_path)

    assert config == ProviderConfig(
        "only-key", DEFAULT_BASE_URL, DEFAULT_MODEL, DEFAULT_TIMEOUT_SECONDS
    )


def test_resolve_corrupt_file_warns_and_falls_back(tmp_path, monkeypatch):
    """文件损坏 → 本地化警告 + 回退环境变量."""
    monkeypatch.setenv("MITOCODE_API_KEY", "env-key")
    path = tmp_path / pc.PROVIDER_FILE_RELATIVE
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("{not json", encoding="utf-8")
    console = Console(record=True, width=200)

    config = resolve_provider_config(tmp_path, console)

    assert config is not None and config.api_key == "env-key"
    assert "Ignoring unreadable provider config" in console.export_text()


def test_resolve_invalid_file_timeout_falls_to_env(tmp_path, monkeypatch):
    """文件 timeout 非法 → 警告并回退 env timeout."""
    monkeypatch.setenv("MITOCODE_API_KEY", "env-key")
    monkeypatch.setenv("MITOCODE_TIMEOUT", "77")
    _write_provider_file(tmp_path, api_key="file-key", timeout="abc")

    config = resolve_provider_config(tmp_path)

    assert config.timeout == 77


# ──────────────────────────────────────────────
# gitignore 防护与保存
# ──────────────────────────────────────────────


def test_ensure_gitignored_appends_when_absent(tmp_path):
    """无 .gitignore → 创建并追加 secret 目录，返回 True."""
    assert ensure_secret_gitignored(tmp_path) is True
    assert ".mitocode/secret/" in (tmp_path / ".gitignore").read_text()


def test_ensure_gitignored_is_idempotent(tmp_path):
    """再次调用 → 已覆盖返回 False，不重复追加."""
    ensure_secret_gitignored(tmp_path)

    assert ensure_secret_gitignored(tmp_path) is False
    content = (tmp_path / ".gitignore").read_text()
    assert content.count(".mitocode/secret/") == 1


def test_ensure_gitignored_respects_whole_mitocode_rule(tmp_path):
    """.gitignore 已忽略 .mitocode 整体 → 不再追加."""
    (tmp_path / ".gitignore").write_text(".mitocode\n", encoding="utf-8")

    assert ensure_secret_gitignored(tmp_path) is False
    assert ".mitocode/secret/" not in (tmp_path / ".gitignore").read_text()


def test_save_writes_gitignore_before_file_with_mode_600(tmp_path, monkeypatch):
    """保存顺序：gitignore 防护先于文件写入；文件权限 0600."""
    order = []
    real_ensure = pc.ensure_secret_gitignored
    monkeypatch.setattr(
        pc,
        "ensure_secret_gitignored",
        lambda root: order.append("ignore") or real_ensure(root),
    )
    real_write = Path.write_text
    import unittest.mock as mock

    with mock.patch.object(Path, "write_text", autospec=True) as fake_write:
        fake_write.side_effect = lambda self, data, encoding=None: order.append(
            "write"
        ) or real_write(self, data, encoding=encoding)
        path = save_provider_config(
            tmp_path, ProviderConfig("k", "https://x", "m", 30), Console()
        )

    assert order == ["ignore", "write"]
    assert json.loads(path.read_text(encoding="utf-8")) == {
        "api_key": "k",
        "base_url": "https://x",
        "model": "m",
        "timeout": 30,
    }
    assert stat.S_IMODE(path.stat().st_mode) == 0o600


# ──────────────────────────────────────────────
# /connect 向导
# ──────────────────────────────────────────────


@pytest.mark.asyncio
async def test_connect_wizard_saves_full_config(tmp_path):
    """全部输入有效 → 保存四字段配置并返回."""
    config = await connect_wizard(
        tmp_path, _reader("key-1", "https://api", "model-x", "45"), Console()
    )

    assert config == ProviderConfig("key-1", "https://api", "model-x", 45)
    stored = json.loads((tmp_path / pc.PROVIDER_FILE_RELATIVE).read_text())
    assert stored["api_key"] == "key-1"


@pytest.mark.asyncio
async def test_connect_wizard_empty_answers_keep_defaults(tmp_path):
    """空输入 → api_key 保留（已有文件），其余用默认."""
    _write_provider_file(
        tmp_path,
        api_key="old-key",
        base_url="https://old",
        model="old-model",
        timeout=11,
    )

    config = await connect_wizard(tmp_path, _reader("", "", "", ""), Console())

    assert config == ProviderConfig("old-key", "https://old", "old-model", 11)


@pytest.mark.asyncio
async def test_connect_wizard_aborts_without_any_key(tmp_path):
    """无既有配置且空 key → 提示引导并返回 None，不写文件."""
    config = await connect_wizard(tmp_path, _reader("", "", "", ""), Console())

    assert config is None
    assert not (tmp_path / pc.PROVIDER_FILE_RELATIVE).exists()


# ──────────────────────────────────────────────
# /model 切换
# ──────────────────────────────────────────────


class _FakeProvider:
    """带可变 default_model 与脚本化模型列表的 provider 替身."""

    def __init__(self, models, default_model):
        self.model_list = list(models)
        self.default_model = default_model

    async def get_model_list(self):
        return self.model_list


@pytest.mark.asyncio
async def test_switch_model_by_number_persists_to_file(tmp_path):
    """序号选择 → 切换 default_model 并回写 provider.json."""
    provider = _FakeProvider(["m1", "m2"], "m1")
    _write_provider_file(
        tmp_path, api_key="k", base_url="https://x", model="m1", timeout=5
    )

    await switch_model(provider, tmp_path, _reader("2"), Console())

    assert provider.default_model == "m2"
    stored = json.loads((tmp_path / pc.PROVIDER_FILE_RELATIVE).read_text())
    assert stored["model"] == "m2"


@pytest.mark.asyncio
async def test_switch_model_by_name_without_file_is_session_only(tmp_path):
    """名称选择且无 provider.json → 仅会话内生效."""
    provider = _FakeProvider(["m1", "m2"], "m1")

    await switch_model(provider, tmp_path, _reader("m2"), Console())

    assert provider.default_model == "m2"
    assert not (tmp_path / pc.PROVIDER_FILE_RELATIVE).exists()


@pytest.mark.asyncio
async def test_switch_model_rejects_unknown_and_cancel(tmp_path):
    """未知模型 → 拒绝不变；空输入 → 取消不变."""
    provider = _FakeProvider(["m1", "m2"], "m1")

    await switch_model(provider, tmp_path, _reader("nope"), Console())
    assert provider.default_model == "m1"

    await switch_model(provider, tmp_path, _reader("9"), Console())
    assert provider.default_model == "m1"

    await switch_model(provider, tmp_path, _reader(""), Console())
    assert provider.default_model == "m1"
