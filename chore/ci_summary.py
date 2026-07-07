#!/usr/bin/env python3
"""汇总 CI 各工具产物，生成 PR 评论 markdown 与状态 JSON。"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import xml.etree.ElementTree as ET
from pathlib import Path


def _read_text(path: Path) -> str:
    if not path.exists():
        return ""
    return path.read_text(encoding="utf-8")


def _run(cmd: list[str], cwd: Path | None = None) -> tuple[int, str, str]:
    proc = subprocess.run(
        cmd,
        cwd=cwd,
        capture_output=True,
        text=True,
        check=False,
    )
    return proc.returncode, proc.stdout, proc.stderr


def parse_pytest_report(xml_path: Path) -> dict:
    """解析 pytest 生成的 JUnit XML。"""
    text = _read_text(xml_path)
    if not text:
        return {"passed": False, "tests": 0, "errors": -1, "failures": -1, "skipped": 0}

    root = ET.fromstring(text)
    suite = root.find("testsuite")
    if suite is None:
        return {"passed": False, "tests": 0, "errors": -1, "failures": -1, "skipped": 0}

    tests = int(suite.get("tests", "0"))
    errors = int(suite.get("errors", "0"))
    failures = int(suite.get("failures", "0"))
    skipped = int(suite.get("skipped", "0"))
    time = suite.get("time", "0")

    return {
        "passed": errors == 0 and failures == 0,
        "tests": tests,
        "errors": errors,
        "failures": failures,
        "skipped": skipped,
        "time": time,
    }


def parse_coverage(xml_path: Path) -> dict:
    """解析 coverage.xml 获取总体覆盖率。"""
    text = _read_text(xml_path)
    if not text:
        return {"rate": 0.0, "covered": 0, "total": 0}

    root = ET.fromstring(text)
    rate_str = root.get("line-rate", "0")
    try:
        rate = float(rate_str) * 100
    except ValueError:
        rate = 0.0

    covered = 0
    total = 0
    for package in root.findall(".//package"):
        for cls in package.findall(".//class"):
            lines = cls.find("lines")
            if lines is None:
                continue
            for line in lines.findall("line"):
                total += 1
                if line.get("hits", "0") != "0":
                    covered += 1

    return {"rate": round(rate, 2), "covered": covered, "total": total}


def parse_gitleaks(json_path: Path) -> dict:
    """解析 gitleaks JSON 报告。"""
    text = _read_text(json_path)
    if not text:
        return {"leaks": [], "count": 0}

    try:
        leaks = json.loads(text)
    except json.JSONDecodeError:
        return {"leaks": [], "count": 0}

    if not isinstance(leaks, list):
        leaks = []

    return {
        "leaks": leaks,
        "count": len(leaks),
    }


def parse_pip_audit(output_path: Path) -> dict:
    """解析 pip-audit 原始输出。"""
    text = _read_text(output_path)
    # pip-audit 表格中漏洞行形如 "package  version ID        fix versions"
    # 通过统计 "CVE-" / "GHSA-" / "PYSEC-" 出现次数来估算漏洞数
    count = len(re.findall(r"\b(CVE-|GHSA-|PYSEC-)", text))
    return {"output": text, "vulnerability_count": count}


def parse_licenses(csv_path: Path) -> dict:
    """运行 check_licence.py 并解析结果。"""
    script = Path(__file__).with_name("check_licence.py")
    rc, stdout, _stderr = _run([sys.executable, str(script), str(csv_path)])
    risks = []
    for line in stdout.splitlines():
        if line.startswith("【发现"):
            risks.append(line)
    return {
        "passed": rc == 0,
        "rc": rc,
        "risks": risks,
        "count": len(risks),
    }


def black_changed(repo_root: Path) -> bool:
    """检查 black 是否修改了文件。"""
    rc, stdout, _stderr = _run(["git", "status", "--porcelain"], cwd=repo_root)
    return any(line.startswith((" M", "M ", "??")) for line in stdout.splitlines())


def parse_pylint(json_path: Path) -> dict:
    """解析 pylint JSON2 报告并计算评分。 无数据时返回 score=None。"""
    text = _read_text(json_path)
    if not text:
        return {"score": None, "issues": [], "count": 0}

    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        return {"score": None, "issues": [], "count": 0}

    # JSON2 格式顶层是 dict，包含 statistics / messages
    messages = data.get("messages", []) if isinstance(data, dict) else []
    stats = data.get("statistics", {}) if isinstance(data, dict) else {}

    raw_score = stats.get("score")
    if isinstance(raw_score, (int, float)):
        score = float(raw_score)
    else:
        # JSON2 的 statistics 中直接包含 score；兜底通过 pylint 默认公式计算
        counts = stats.get("messageTypeCount", {})
        error = counts.get("error", 0)
        warning = counts.get("warning", 0)
        refactor = counts.get("refactor", 0)
        convention = counts.get("convention", 0)
        statement = stats.get("statement", 1) or 1
        score = max(
            0.0,
            10.0
            - ((float(5 * error + warning + refactor + convention) / statement) * 10),
        )

    return {
        "score": round(score, 2),
        "issues": messages[:20],  # 仅展示前 20 条
        "count": len(messages),
    }


def parse_pyright(output_path: Path) -> dict:
    """解析 pyright 输出中的错误/警告数量。"""
    text = _read_text(output_path)
    errors = 0
    warnings = 0
    for line in text.splitlines():
        if "error:" in line.lower():
            errors += 1
        elif "warning:" in line.lower():
            warnings += 1
    return {"output": text, "errors": errors, "warnings": warnings}


def parse_eff_u_code(output_path: Path) -> dict:
    """解析 eff-u-code 输出。"""
    text = _read_text(output_path)
    return {"output": text}


def generate_markdown(
    pytest_result: dict,
    coverage_result: dict,
    gitleaks_result: dict,
    pip_audit_result: dict,
    license_result: dict,
    black_changed_flag: bool,
    pylint_result: dict,
    pyright_result: dict,
    eff_u_code_result: dict,
    commit_status: str,
    repo_root: Path,
) -> str:
    """生成 PR 评论 markdown。"""

    def status_badge(
        condition: bool, label_ok: str = "通过", label_fail: str = "失败"
    ) -> str:
        return f"{'✅' if condition else '❌'} {label_ok if condition else label_fail}"

    lines: list[str] = []
    lines.append("## CI 检查结果汇总\n")
    lines.append(
        f"- **pytest**: {status_badge(pytest_result['passed'])} — {pytest_result['tests']} 个测试，"
    )
    lines.append(
        f"  errors={pytest_result['errors']}, failures={pytest_result['failures']}, skipped={pytest_result['skipped']}"
    )
    lines.append(
        f"- **覆盖率**: {status_badge(coverage_result['rate'] >= 80)} — "
        f"{coverage_result['rate']:.2f}%（{coverage_result['covered']}/{coverage_result['total']} 行）"
    )
    lines.append(
        f"- **gitleaks**: {status_badge(gitleaks_result['count'] == 0)} — 发现 {gitleaks_result['count']} 条潜在泄露"
    )
    lines.append(
        f"- **black**: ✅ {'无变更' if not black_changed_flag else '已应用格式化'} — "
        f"{'代码已符合 black 规范' if not black_changed_flag else '已自动格式化代码'}"
    )
    if pylint_result["score"] is not None:
        lines.append(
            f"- **pylint**: {status_badge(pylint_result['score'] >= 8)} — "
            f"评分 {pylint_result['score']}/10，共 {pylint_result['count']} 条问题"
        )
    else:
        lines.append("- **pylint**: ⚪ 无数据/未运行")
    lines.append("")

    # 信息展示区
    lines.append("### 信息展示")
    if (repo_root / "pip-audit-output.txt").exists():
        lines.append("- **pip-audit**: 发现 {} 个已知漏洞".format(pip_audit_result["vulnerability_count"]))
    if (repo_root / "licenses.csv").exists():
        lines.append("- **pip-licenses**: {} — {} 个风险许可证".format(
            "✅ 通过" if license_result["passed"] else "⚠️ 发现风险", license_result["count"]
        ))
    lines.append("- **pyright**: {} 个 error, {} 个 warning".format(pyright_result["errors"], pyright_result["warnings"]))
    lines.append("")

    # 详细输出
    lines.append("<details>")
    lines.append("<summary>📋 pytest 详细</summary>")
    lines.append("")
    lines.append(f"```xml\n{_read_text(Path('pytest-cov-output.txt'))[:2000]}\n```")
    lines.append("</details>")
    lines.append("")

    lines.append("<details>")
    lines.append("<summary>🔒 gitleaks 详细</summary>")
    lines.append("")
    lines.append(
        f"```json\n{json.dumps(gitleaks_result['leaks'], indent=2, ensure_ascii=False)}\n```"
    )
    lines.append("</details>")
    lines.append("")

    if (repo_root / "pip-audit-output.txt").exists():
        lines.append("<details>")
        lines.append("<summary>🛡️ pip-audit 输出</summary>")
        lines.append("")
        lines.append(f"```text\n{pip_audit_result['output'][:3000]}\n```")
        lines.append("</details>")
        lines.append("")

    lines.append("<details>")
    lines.append("<summary>📝 pyright 输出</summary>")
    lines.append("")
    lines.append(f"```text\n{pyright_result['output'][:3000]}\n```")
    lines.append("</details>")
    lines.append("")

    if (repo_root / "eff-u-code-output.txt").exists():
        lines.append("<details>")
        lines.append("<summary>🎨 eff-u-code 输出</summary>")
        lines.append("")
        lines.append(f"```text\n{eff_u_code_result['output'][:3000]}\n```")
        lines.append("</details>")
        lines.append("")

    lines.append(f"**自动提交状态**: {commit_status}")
    lines.append("")
    lines.append("---")
    lines.append("*由 CI 自动生成*")

    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description="生成 CI 汇总报告")
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument(
        "--dry-run", action="store_true", help="只打印 markdown，不写入文件"
    )
    args = parser.parse_args()

    repo_root = args.repo_root.resolve()
    os.chdir(repo_root)

    pytest_result = parse_pytest_report(repo_root / "report.xml")
    coverage_result = parse_coverage(repo_root / "coverage.xml")
    gitleaks_result = parse_gitleaks(repo_root / "gitleak-result.json")
    pip_audit_result = parse_pip_audit(repo_root / "pip-audit-output.txt")
    license_result = parse_licenses(repo_root / "licenses.csv")
    pylint_result = parse_pylint(repo_root / "pylint-report.json")
    pyright_result = parse_pyright(repo_root / "pyright-output.txt")
    eff_u_code_result = parse_eff_u_code(repo_root / "eff-u-code-output.txt")

    # black 是否修改文件通过 git status 判断
    changed = black_changed(repo_root)

    # 提交状态由 CI 后续步骤注入环境变量，若不存在则显示未知
    commit_status = os.environ.get("CI_COMMIT_STATUS", "待执行")

    markdown = generate_markdown(
        pytest_result,
        coverage_result,
        gitleaks_result,
        pip_audit_result,
        license_result,
        changed,
        pylint_result,
        pyright_result,
        eff_u_code_result,
        commit_status,
        repo_root,
    )

    if args.dry_run:
        print(markdown)
        return 0

    (repo_root / "ci-summary.md").write_text(markdown, encoding="utf-8")

    # 状态 JSON 用于后续步骤判断
    summary = {
        "pytest": pytest_result,
        "coverage": coverage_result,
        "gitleaks": gitleaks_result,
        "pip_audit": pip_audit_result,
        "license": license_result,
        "black_changed": changed,
        "pylint": pylint_result,
        "pyright": pyright_result,
        "eff_u_code": eff_u_code_result,
    }
    (repo_root / "ci-summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    # 汇总脚本自身永远返回 0，是否阻塞由 CI 各步骤的 if/continue-on-error 控制
    return 0


if __name__ == "__main__":
    sys.exit(main())
