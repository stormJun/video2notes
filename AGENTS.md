# AGENTS.md

## Required Entry Rule

在此目录工作前，必须先阅读 `PROJECT_SPEC.md`。

## Required Working Order

1. 先阅读 `PROJECT_SPEC.md`
2. 先检查当前关键文件是否存在：
   - `video_to_notes.py`
   - `video_to_notes_pipeline.py`
   - `video_to_notes_artifacts.py`
   - `video_to_notes_note.py`
   - `video_to_notes_schema.py`
   - `video_to_notes_visual.py`
   - `tests/test_video_to_notes.py`
   - `notes/`

3. 默认主流程不得停在“只生成中间产物”这一步。
   除非用户明确要求，否则应继续执行完整审阅阶段和 note-generation stage，直到产出最终 `note.md` 与对应报告文件。

## Constraints

- 不要在未验证现有流程前直接重构。
- 不要删除已有中间产物，除非用户明确要求。
- 修改脚本前，优先保持现有最小闭环可运行。
- 不要把人工参与当作默认步骤。
- 不要只做全局粗略改写；应保留分段审阅和留痕。
- 不要停在生成 `codex_review_prompt.md` 或 `note_generation_prompt.md`。
- 当用户要求“把某个视频转成图文文档”时，不要擅自退化成“复用已有 transcript / OCR / 分段材料，只重跑 note 阶段”；除非用户明确允许，否则应优先完成完整主链路。
- 阶段级重跑只应在显式 `--force-*` 或排障需要时进行。
- 每次声称“完成”前，必须给出实际验证结果。
