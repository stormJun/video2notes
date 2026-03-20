# Resumable Single Video Pipeline Design

## Goal

让长视频主链路具备阶段级断点续跑能力。在默认模式下，如果某阶段的产物已经完整存在，则跳过该阶段；只有显式传入对应 `--force-*` 参数时才重跑该阶段，避免中断后从 Whisper 或更早步骤重新开始。

## Non-Goals

- 本次不引入新的全局状态文件，如 `pipeline_state.json`
- 不改变现有输出目录结构
- 不降低现有 Codex 审阅质量门槛

## Current Problem

`video_to_notes.py` 当前在 `main()` 中无条件线性执行：

1. `ffprobe`
2. 抽取 `audio.wav`
3. Whisper 转写
4. 抽帧
5. OCR
6. 构建审阅材料
7. Codex 审阅
8. 生成 `note.md`
9. 写入 `metadata.json`

这意味着长跑在审阅或笔记生成阶段中断后，再次执行会重复跑 Whisper、抽帧、OCR，既慢也增加额外失败面。

## Proposed Approach

采用“阶段产物存在即跳过”的恢复模型，并提供显式阶段覆盖开关。

### Stages

1. `audio`
2. `transcribe`
3. `frames`
4. `ocr`
5. `review_artifacts`
6. `codex_review`
7. `note`
8. `metadata`

### Readiness Rules

每个阶段都通过独立的 readiness helper 判定是否可跳过：

- `audio`: `audio.wav` 存在且非空
- `transcribe`: `transcript.txt/.json/.srt/.vtt/.tsv` 全部存在；`transcript.txt` 非空；`transcript.json` 为非空 JSON，且优先要求包含可解析 `segments`
- `frames`: `assets/frame_*.jpg` 数量达到按当前 `duration_seconds`、`frame_interval`、`max_frames` 计算出的期望值
- `ocr`: `ocr.json` 存在，内容为列表，条目数与当前帧数一致
- `review_artifacts`: `transcript.cleaned.txt`、`transcript.corrections.json`、`codex_review_prompt.md`、`review_segments.json` 存在；`transcript.corrections.json` 满足固定 schema 最低要求
- `codex_review`: 在 `review_artifacts` 基础上，`review_report.json` 存在且 `status=passed`，同时 `review_artifacts_completed(...)` 返回真
- `note`: `note.md` 存在且非空
- `metadata`: 始终重写，不做跳过

### Force Flags

CLI 新增：

- `--force-audio`
- `--force-transcribe`
- `--force-frames`
- `--force-ocr`
- `--force-review-artifacts`
- `--force-codex-review`
- `--force-note`

语义：

- 被 force 的阶段一定重跑
- 依赖该阶段产物的下游阶段也必须重跑或至少重建
- `metadata` 始终最后刷新

示例：

- `--force-transcribe` 会连带重跑 `review_artifacts`、`codex_review`、`note`
- `--force-frames` 会连带重跑 `ocr`、`review_artifacts`、`codex_review`、`note`
- `--force-codex-review` 只重跑审阅和 `note`

## Module Split

本次不继续把新逻辑堆进单文件，按职责拆成少量模块：

- `video_to_notes.py`: CLI 入口，仅负责参数解析和调度
- `video_to_notes_pipeline.py`: 阶段计划与主流程 orchestration
- `video_to_notes_artifacts.py`: readiness 检查、输出路径、元数据和阶段计划
- 保留现有审阅相关核心逻辑在原位置，后续若改动过大再单独拆分

拆分原则是“小步重组”：

- 先抽离纯函数和 orchestration
- 不在本次同时重写所有媒体/OCR/Codex helper 的归属
- 保持现有测试可逐步迁移

## Data Flow

1. CLI 解析输入参数和 force flags
2. pipeline 计算 `output_paths`
3. pipeline 读取现有产物，生成阶段执行计划
4. 对每个阶段输出 `run` 或 `skip` 决策及原因
5. 执行必要阶段
6. 仅在相关上游产物变化时重建下游产物
7. 最后刷新 `metadata.json`

## Error Handling

- readiness 检查失败不报错，视为“该阶段未完成”，转为执行该阶段
- 真正执行阶段时沿用现有异常行为
- 若 `codex_review` 失败，仍保持当前质量门槛和失败退出
- 若某阶段被跳过，打印明确日志说明原因，避免误判为未执行

## Testing Strategy

新增/调整单测覆盖：

- 各阶段 readiness helper 的完整与不完整场景
- force flag 对阶段计划的级联影响
- 默认续跑时，已完成上游阶段不会再次调用相应命令
- 指定 force 时，仅期望阶段及其下游会重跑
- 现有 `run_codex_review(...)` 逻辑保持兼容

## Constraints

- 不删除已有中间产物
- 不改变 `transcript.txt` / `transcript.cleaned.txt` / `transcript.corrections.json` / `review_segments.json` 的既有语义
- 在真实样本上必须再次验证：
  - 全新输出目录首跑可成功
  - 中断后重跑可跳过 Whisper 之前已完成阶段
  - `note.md` 与 `review_report.json` 仍能生成

## Risks

- readiness 判定过宽会跳过坏产物
- readiness 判定过严会降低续跑收益
- 模块拆分若处理不当，可能破坏现有测试导入路径

缓解方式：

- 先以“最小但明确”的完整性判定落地
- 保持旧接口名可导入
- 用 monkeypatch 测试验证阶段是否真的被跳过
