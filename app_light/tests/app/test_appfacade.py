"""AppFacade 翻译链路测试 — tests/data/part.ERB + jp_noval.json 规则,输出到 tests/output/appfacade/。

运行方式(项目根,两种均可)::

    python -m app.test_appfacade
    python app/test_appfacade.py

流程:
    1. 构建 AppFacade(注册 llama 服务 + TranslationTaskQueue)
    2. start_service("llama") 启动翻译服务(阻塞至模型就绪,可能 1-2 分钟)
    3. 提交 part.ERB 翻译任务(jp_noval.json 规则、无术语表)
    4. 轮询等待任务完成(期间捕获 stdout 以统计 _merge_chunks WARNING)
    5. 验证:completed / result 非空 / 输出行数与原文一致 / 空行数与原文一致(无每行后空行)
    6. 导出到 tests/output/appfacade/:part.ERB(翻译结果)+ appfacade_report.txt(验证报告)
    7. shutdown() 停止服务
"""

from __future__ import annotations

import contextlib
import io
import sys
import time
from pathlib import Path

# 直接运行(python app/test_appfacade.py)时 sys.path[0] 是脚本所在目录 app/,
# 把项目根加入 sys.path,保证 `from app.facade import ...` / `from core...` 均可解析。
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from app.facade import AppFacade
from core.contracts import TranslationRequest
from core.service import LlamaService
from core.task_que import TranslationTaskQueue

OUTPUT_DIR = _PROJECT_ROOT / "tests" / "output" / "appfacade"

_SOURCE_FILE = _PROJECT_ROOT / "tests" / "data" / "part.ERB"
_CONFIGS = {
    "translate_config": _PROJECT_ROOT / "configs/translate/args_llama/default.json",
    "prompts":          _PROJECT_ROOT / "configs/translate/prompts/default.json",
    "glossary":         None,   # 无术语表
    "rule":             _PROJECT_ROOT / "configs/translate/rules/jp_noval.json",
}


def build_facade() -> AppFacade:
    """构建 AppFacade:注册 llama 服务与翻译任务队列(与 ui/layout.py 同构)。"""
    return AppFacade(
        backend_dict={"llama": (LlamaService, TranslationTaskQueue)},
        config_dict={"llama": _PROJECT_ROOT / "configs/models/llama/default.json"},
    )


def wait_for_task(facade: AppFacade, tid: str, timeout: float = 300.0) -> dict:
    """轮询 _find_task 直到任务进入终态(completed/failed/cancelled)。"""
    t0 = time.time()
    while time.time() - t0 < timeout:
        t = facade._find_task(tid)
        if t is not None and t.get("status") in ("completed", "failed", "cancelled"):
            return t
        time.sleep(0.5)
    raise TimeoutError(f"任务 {tid} 在 {timeout}s 内未完成")


def verify(task: dict, source_text: str) -> list[str]:
    """校验翻译结果,返回问题列表(空列表 = 通过)。"""
    issues: list[str] = []
    if task.get("status") != "completed":
        issues.append(f"状态不是 completed: {task.get('status')}")
    result = task.get("result")
    if not isinstance(result, str) or not result.strip():
        issues.append(f"result 为空或非字符串: {result!r}")
        return issues

    src_lines = source_text.split("\n")
    out_lines = result.split("\n")
    src_blank = sum(1 for l in src_lines if l == "")
    out_blank = sum(1 for l in out_lines if l == "")

    if len(out_lines) != len(src_lines):
        issues.append(
            f"行数不一致: 原文 {len(src_lines)} 行, 输出 {len(out_lines)} 行"
        )
    if out_blank > src_blank:
        issues.append(
            f"输出空行({out_blank})多于原文空行({src_blank})——存在每行后空行问题"
        )
    return issues


def main() -> int:
    # Windows 控制台默认 GBK:切换 stdout 为 UTF-8 并容错,避免打印 ✓/✗ 等字符崩溃
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    source_text = _SOURCE_FILE.read_text(encoding="utf-8")
    print("[1/5] 构建 AppFacade…")
    facade = build_facade()

    print("[2/5] 启动 llama 翻译服务…(模型加载可能需要 1-2 分钟)")
    facade.start_service("llama")

    # ── 提交 + 等待:捕获期间 stdout(worker 线程的 chunk/WARNING 打印),回放后统计 ──
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        print(f"[3/5] 提交翻译任务: {_SOURCE_FILE.name} + {_CONFIGS['rule'].name} 规则")
        tid = facade.submit_task(TranslationRequest(
            task_type="llama",
            file_path=_SOURCE_FILE,
            file_name=_SOURCE_FILE.name,
            configs=dict(_CONFIGS),
        ))
        print(f"      task_id = {tid}")

        print("[4/5] 等待任务完成…")
        task = wait_for_task(facade, tid)
        print(f"      status = {task.get('status')} | progress = {task.get('progress')}")
        if task.get("error"):
            print(f"      error = {task.get('error')}")

    log = buf.getvalue()
    sys.stdout.write(log)
    sys.stdout.flush()
    warnings = [l for l in log.splitlines() if "行数不匹配" in l]

    result = task.get("result") if isinstance(task.get("result"), str) else ""
    issues = verify(task, source_text)

    # ── 导出到 tests/output/appfacade/ ──
    out_file = OUTPUT_DIR / _SOURCE_FILE.name
    out_file.write_text(result, encoding="utf-8")
    report = OUTPUT_DIR / "appfacade_report.txt"
    report_lines = [
        "AppFacade 翻译链路测试报告",
        "=" * 40,
        f"源文件: {_SOURCE_FILE}",
        f"规则:   {_CONFIGS['rule'].name}",
        f"任务 id: {tid}",
        f"状态:   {task.get('status')}",
        f"原文行数: {len(source_text.split(chr(10)))}",
        f"输出行数: {len(result.split(chr(10))) if result else 0}",
        f"原文空行: {sum(1 for l in source_text.split(chr(10)) if l == '')}",
        f"输出空行: {sum(1 for l in result.split(chr(10)) if l == '') if result else 0}",
        f"WARNING 计数(_merge_chunks 行数不匹配): {len(warnings)}",
        f"校验结果: {'PASS' if not issues else 'FAIL'}",
    ]
    if issues:
        report_lines.append("问题:")
        report_lines.extend(f"  - {i}" for i in issues)
    if warnings:
        report_lines.append("WARNING 明细:")
        report_lines.extend(f"  {w}" for w in warnings)
    report.write_text("\n".join(report_lines), encoding="utf-8")

    print(f"[5/5] 输出: {out_file} / {report}")
    print(f"      校验: {'PASS' if not issues else 'FAIL'}")
    print(f"      WARNING 计数: {len(warnings)}")
    for i in issues:
        print(f"        - {i}")

    facade.shutdown()
    print("[完成] 测试结束")
    return 0 if not issues else 1


if __name__ == "__main__":
    sys.exit(main())
