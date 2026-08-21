# Modular Translator — 模块化翻译器

面向 Windows 的**日文 → 简体中文翻译工作台**（Flet 桌面应用）：整合音频转写、本地/API 翻译、GPT-SoVITS 语音合成与字幕导出。

## 功能

- **翻译 Translate** — 双后端：本地 `llama-server`（llama.cpp）加载 Sakura GGUF 翻译模型；或 OpenAI 兼容 API。任务队列、规则式文本分割、词汇表与提示词模板。
- **转写 Transcribe** — MOSS 语音识别：说话人分离、时间戳、长音频分窗、静音边界/VAD、实时段预览。
- **语音合成 TTS** — 内嵌 GPT-SoVITS 推理引擎：角色预设（参考音频 + S1/S2 权重）、语速/种子等参数、并行推理。
- **字幕导出** — 完成页自动/手动导出 LRC 字幕（可带说话人前缀）。
- **日志 Log** — 环形缓冲实时日志，1s 轮询刷新，手动导出。
- **设置 Settings** — 基于 schema 的配置表单，统一管理 `configs/`。

> **状态说明**：核心后端（翻译/转写/合成）均已实现；翻译页的 UI 提交接线目前待重建（后端切换与文件选择已恢复）。

## 技术栈

Python 3.14 · Flet 0.86.2（桌面 UI） · llama.cpp `llama-server` · Sakura LLM（GGUF 翻译模型） · MOSS-Transcribe-Diarize · faster-whisper · GPT-SoVITS（内嵌 vendored 推理） · PyTorch · PyInstaller

## 快速开始

```bash
# 1. 准备外部依赖（不入库，约 13GB）
#    dependencies/  ：模型权重、llama-server 二进制、torch 运行时、MOSS 源码
#    characters/    ：角色语音资产（S1/S2 权重 + 参考音频）
#    app_light/dependencies 与 app_light/characters 是指向仓库根同名目录的 junction

# 2. 在已安装依赖的 venv 中启动
cd app_light
python APP.py
```

配置通过 `app_light/configs/` 管理：`models/llama|gsv|moss|API`、`system/default.ini`、`translate/`（提示词/词汇表/规则）、`transcribe/`、`tts/roles/`。API 密钥请复制 `configs/models/API/default.example.json` 为 `default.json` 填写（后者已 gitignore，不会入库）。

打包：`build/build_cpu.py` 与 `build/build_cuda.py`（PyInstaller onedir 绿色目录，须用项目 `.venv` 运行）。

## 目录结构

```
ModularTranslator/
├── app_light/            # 主程序（main 分支内容）
│   ├── APP.py            # 入口
│   ├── app/              # 应用门面、路径、日志、ffmpeg、torch 运行时
│   ├── core/             # 服务/执行器/任务队列/规则分割/MOSS/GSV 引擎
│   ├── ui/               # Flet 界面（translate/transcribe/tts/completed/log/settings）
│   ├── configs/          # 全部配置（含 API 模板）
│   └── build/            # 打包脚本
├── requirements.txt      # 依赖清单
├── dependencies/         # （不入库）外部权重/运行时
└── characters/           # （不入库）角色语音资产
```

## 分支

- `main` — 主开发分支（对应 `app_light/` 内容）
- `app_lite` / `app_workflow` — 版本分支；`tools/sync_version_dirs.py` 负责从分支导出物理快照

## 开发

测试用例位于 `app_light/tests/`（本地磁盘，未随仓库分发）；`pytest.ini` 已配置 `testpaths`。代码注释统一使用英文。

## 第三方许可与致谢

- [GPT-SoVITS](https://github.com/RVC-Boss/GPT-SoVITS)（MIT）— 内嵌语音合成引擎
- [BigVGAN](https://github.com/NVIDIA/bigvgan)（MIT，NVIDIA）及其内含许可证（HiFi-GAN 等）
- [MOSS-Transcribe-Diarize](https://github.com/Plachtaa/MOSS-Transcribe-Diarize) — 转写/说话人分离
- [faster-whisper](https://github.com/SYSTRAN/faster-whisper)（MIT）— VAD
- [llama.cpp](https://github.com/ggml-org/llama.cpp) — `llama-server` 本地推理
- [SakuraLLM](https://github.com/SakuraLLM/SakuraLLM) — 日→中翻译 GGUF 模型
- [Flet](https://flet.dev/)、[PyTorch](https://pytorch.org/)、[PyInstaller](https://pyinstaller.org/) 等

完整许可证文本见各上游项目仓库。本仓库内 vendored 代码（`app_light/core/gsv/vendor/`）保持与上游逐字节一致。

## 免责声明

本软件仅供学习与研究。模型权重与角色语音素材版权归各自权利人所有，请确保使用你拥有合法授权的数据。基于生成内容的使用后果由使用者自行承担。

---

# Modular Translator

A Windows desktop **Japanese → Simplified Chinese translation workstation** built with Flet: audio transcription, local/API translation, GPT-SoVITS speech synthesis, and subtitle export.

## Features

- **Translate** — dual backends: local `llama-server` (llama.cpp) with Sakura GGUF translation models, or OpenAI-compatible APIs. Task queue, rule-based text splitting, glossary and prompt templates.
- **Transcribe** — MOSS speech recognition with speaker diarization, timestamps, long-audio windowing, silence/VAD boundaries, and live segment preview.
- **TTS** — embedded GPT-SoVITS inference: character presets (reference audio + S1/S2 weights), speed/seed parameters, parallel synthesis.
- **Subtitle export** — auto/manual LRC export (optional speaker prefixes) from the Completed page.
- **Log** — ring-buffer live log with 1s polling refresh and manual export.
- **Settings** — schema-driven config form over `configs/`.

> **Status**: core backends (translate/transcribe/tts) are implemented; the translate page's UI submit wiring is pending rebuild (backend switch and file picker restored).

## Tech Stack

Python 3.14 · Flet 0.86.2 (desktop UI) · llama.cpp `llama-server` · Sakura LLM (GGUF) · MOSS-Transcribe-Diarize · faster-whisper · GPT-SoVITS (vendored inference) · PyTorch · PyInstaller

## Quick Start

```bash
# 1. Prepare external dependencies (not in git, ~13GB)
#    dependencies/ : model weights, llama-server binaries, torch runtime, MOSS source
#    characters/   : role voice assets (S1/S2 weights + reference audio)
#    app_light/dependencies and app_light/characters are junctions to the repo-root dirs

# 2. Run with a venv that has the dependencies installed
cd app_light
python APP.py
```

Configuration lives under `app_light/configs/`: `models/llama|gsv|moss|API`, `system/default.ini`, `translate/` (prompts/glossary/rules), `transcribe/`, `tts/roles/`. For the API key, copy `configs/models/API/default.example.json` to `default.json` (gitignored).

Packaging: `build/build_cpu.py` and `build/build_cuda.py` (PyInstaller onedir; run with the project `.venv`).

## Directory Layout

```
ModularTranslator/
├── app_light/            # main app (contents of the `main` branch)
│   ├── APP.py            # entry point
│   ├── app/              # app facade, paths, logging, ffmpeg, torch runtime
│   ├── core/             # services, executors, task queue, rule splitter, MOSS/GSV engines
│   ├── ui/               # Flet UI (translate/transcribe/tts/completed/log/settings)
│   ├── configs/          # all configuration (incl. API template)
│   └── build/            # packaging scripts
├── requirements.txt      # dependency manifest
├── dependencies/         # (not in git) external weights/runtime
└── characters/           # (not in git) role voice assets
```

## Branches

- `main` — primary development branch (content of `app_light/`)
- `app_lite` / `app_workflow` — version branches; `tools/sync_version_dirs.py` refreshes physical snapshots from branches

## Development

Tests live in `app_light/tests/` (local disk, not distributed with the repo); `pytest.ini` configures `testpaths`. Comments are written in English.

## Third-party Licenses & Acknowledgements

- [GPT-SoVITS](https://github.com/RVC-Boss/GPT-SoVITS) (MIT) — embedded TTS engine
- [BigVGAN](https://github.com/NVIDIA/bigvgan) (MIT, NVIDIA) and its bundled licenses (HiFi-GAN et al.)
- [MOSS-Transcribe-Diarize](https://github.com/Plachtaa/MOSS-Transcribe-Diarize) — transcription/diarization
- [faster-whisper](https://github.com/SYSTRAN/faster-whisper) (MIT) — VAD
- [llama.cpp](https://github.com/ggml-org/llama.cpp) — local `llama-server` inference
- [SakuraLLM](https://github.com/SakuraLLM/SakuraLLM) — JP→ZH translation GGUF models
- [Flet](https://flet.dev/), [PyTorch](https://pytorch.org/), [PyInstaller](https://pyinstaller.org/) et al.

See each upstream repository for the full license texts. Vendored code in this repo (`app_light/core/gsv/vendor/`) is kept byte-identical to upstream.

## Disclaimer

This software is provided for learning and research purposes. Model weights and character voice assets belong to their respective rights holders — ensure you have legal authorization for the data you use. Use at your own risk.
