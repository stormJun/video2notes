# PROJECT_SPEC

## Project Goal

将 `transformer_downloader` 目录中的课程视频整理为适合学习和复习的图文笔记，而不是只保留原始音视频文件。

最终目标不是单纯转录字幕，而是生成结构化、可阅读、可检索、可继续加工的学习材料。

## Current Inputs

输入目录：

- `/Users/songxijun/workspace/transformer_downloader`

当前已知输入文件包括：

- 课程资料现在统一放在 `materials/` 下：
  - `materials/videos/`：多个课程视频 `.mp4`
  - `materials/slides/`：原始课件源文件，例如 `.pptx`
  - `materials/docs/`：课件导出的 `.pdf` 和其他参考资料
  - `materials/archives/`：代码压缩包 `.rar`
  - `materials/code/`：解压后的代码或代码图片等资料

## Current Layout

当前目录已经按“项目文件”和“课程资料”分开：

```text
transformer_downloader/
  AGENTS.md
  PROJECT_SPEC.md
  video_to_notes.py
  video_to_notes_artifacts.py
  video_to_notes_note.py
  video_to_notes_pipeline.py
  video_to_notes_schema.py
  video_to_notes_visual.py
  download_wqyunpan_videos.sh
  docs/
  tests/
  notes/
  work/
  materials/
    videos/
    slides/
    archives/
    docs/
    code/
```

## Example Working Command

当前有一个已验证过的单视频命令示例，可在需要验证完整长跑时使用：

```bash
cd /Users/songxijun/workspace/transformer_downloader
python3 video_to_notes.py \
  --input 'materials/videos/4.1BERT-2训练.mp4' \
  --output-root notes \
  --frame-interval 60 \
  --max-frames 4 \
  --whisper-model small \
  --language zh
```

这条命令当前可以生成：

- `pipeline/01_media/audio/audio.wav`
- `pipeline/01_media/audio/transcript.txt/.srt/.json/.vtt/.tsv`
- `pipeline/02_review/transcript.cleaned.txt`
- `pipeline/02_review/transcript.corrections.json`
- `pipeline/02_review/codex_review_prompt.md`
- `pipeline/02_review/review_report.json`
- `pipeline/02_review/review_segments.json`
- `pipeline/01_media/visual/visual_candidates/frame_*.jpg`
- `pipeline/01_media/visual/visual_candidates.ocr.json`
- `pipeline/01_media/visual/visual_units.json`
- `pipeline/03_alignment/visual_alignment.json`
- `pipeline/03_alignment/slides_index.raw.json`
- `pipeline/04_structure/note_outline.json`
- `pipeline/04_structure/note_blocks.json`
- `pipeline/03_alignment/slides_cleanup_prompt.md`
- `pipeline/03_alignment/slides_cleanup_report.json`
- `pipeline/05_note/note_generation_prompt.md`
- `pipeline/05_note/note_generation_report.json`
- `pipeline/00_run/metadata.json`
- `pipeline/05_note/note.md`
- `pipeline/00_run/`
- `pipeline/01_media/audio/`
- `pipeline/01_media/visual/`
- `pipeline/02_review/`
- `pipeline/03_alignment/`
- `pipeline/04_structure/`
- `pipeline/05_note/`

## Visual Source Modes

当前主流程明确区分两条图文来源路径：

1. `slides-first`
   - 前提：显式提供可用的课程课件 PDF（`--slides`）
   - 行为：优先将 `review_segments` / `transcript.cleaned.txt` 与 PDF 页标题、页文本对齐
   - 最终配图：优先使用 `slides_preview/rendered/slide_XXX.png`
   - 视频抽帧角色：辅助定位、OCR 兜底、无匹配时回退

2. `video-first`
   - 前提：没有可用课件 PDF，或课件与当前视频不匹配
   - 行为：以视频中提取的 `visual_units` 为主，与 transcript / OCR 对齐
   - 最终配图：来自视频抽帧、候选页聚类后的视觉单元

CLI 中通过 `--visual-source-mode auto|slides-first|video-first` 控制。

- `auto`：默认模式；只有显式传入可用 `--slides` 时才走 `slides-first`，否则走 `video-first`
- `slides-first`：优先走课件 PDF 主路径；若未显式提供或当前课件不可用，自动回退到 `video-first`
- `video-first`：忽略课件 PDF，始终以视频视觉单元为主

默认情况下，脚本在生成上述基础产物后，会继续调用本机 `codex exec` 自动完成：

- 读取 `codex_review_prompt.md`
- 读取 `review_segments.json`
- 逐段并发审阅转写内容
- 修正 `transcript.cleaned.txt`
- 填写 `transcript.corrections.json`
- 写入 `review_report.json`
- 生成 `visual_alignment.json`、`note_outline.json`、`note_blocks.json`
- 生成 `note_generation_prompt.md`
- 再继续执行独立的 note-generation stage，产出最终 `note.md`
- 写入 `note_generation_report.json`

## Current Proven Workflow

截至当前，这条最小闭环已经被实际验证通过：

1. 读取单个 `mp4`
2. 提取 `pipeline/01_media/audio/audio.wav`
3. 使用本地 Whisper 生成：
   - `pipeline/01_media/audio/transcript.txt`
   - `pipeline/01_media/audio/transcript.srt`
   - `pipeline/01_media/audio/transcript.json`
   - `pipeline/01_media/audio/transcript.vtt`
   - `pipeline/01_media/audio/transcript.tsv`
4. 初始化 Codex 审阅材料：
   - `pipeline/02_review/transcript.cleaned.txt`
   - `pipeline/02_review/transcript.corrections.json`
   - `pipeline/02_review/codex_review_prompt.md`
   - `pipeline/02_review/review_segments.json`
5. 抽取更密的候选视频画面到 `pipeline/01_media/visual/visual_candidates/`
6. 使用当前 OCR 流程对候选画面执行识别，输出 `pipeline/01_media/visual/visual_candidates.ocr.json`
7. 聚类和筛选候选视觉页，生成 `pipeline/01_media/visual/visual_units.json`
8. 主流程会继续调用本机 `codex exec`，让 Codex 读取 `codex_review_prompt.md` 与 `review_segments.json` 并自动完成：
   - 按分段顺序审阅原始转写
   - 修正 `transcript.cleaned.txt`
   - 填写 `transcript.corrections.json.segment_reviews`
   - 填写 `transcript.corrections.json.corrections`
9. 生成 `pipeline/03_alignment/visual_alignment.json`、`pipeline/04_structure/note_outline.json`、`pipeline/04_structure/note_blocks.json`
10. 生成 `pipeline/05_note/note_generation_prompt.md`
11. 继续调用本机 `codex exec` 完成独立 note-generation stage，生成或刷新最终 `pipeline/05_note/note.md`
12. 写入 `pipeline/05_note/note_generation_report.json`
13. 生成 `pipeline/00_run/metadata.json`

说明：

- `pipeline/01_media/audio/transcript.txt` 始终保留原始 Whisper 输出
- `pipeline/02_review/transcript.cleaned.txt` 供 Codex 在上下文中自动修正术语
- `pipeline/05_note/note.md` 默认优先读取 `pipeline/02_review/transcript.cleaned.txt`，若其不存在或为空，再回退到 `pipeline/01_media/audio/transcript.txt`
- `pipeline/02_review/codex_review_prompt.md` 应被视为给 Codex 的可执行任务单，包含输入、输出、步骤、验收标准
- `pipeline/02_review/review_segments.json` 应保存主流程自动切分出的审阅分段，每段带时间范围、文本和 OCR 线索
- `pipeline/02_review/transcript.corrections.json` 应保持固定结构，至少包含 `review_status`、`last_updated`、`segment_reviews[]`、`corrections[].raw/cleaned/reason/evidence`
- `pipeline/02_review/review_report.json` 应记录审阅尝试次数、每次失败原因、最终是否通过质量门槛
- `pipeline/01_media/visual/visual_units.json` 应表示去重后的视觉单元，而不是直接暴露原始帧
- `pipeline/03_alignment/visual_alignment.json` 应显式记录当前 `visual_source_mode`，以及语义段与视觉单元 / 课件 PDF 页的选择结果、候选和原因
- `pipeline/03_alignment/slides_index.raw.json` 保存 PDF 直接抽取的原始页文本索引，供后续清洗
- `pipeline/03_alignment/slides_index.json` 应记录当前课件 PDF 是否 `usable`、匹配分数、命中 token、最终 `effective_visual_source_mode`
- `pipeline/03_alignment/slides_cleanup_prompt.md` 是给 Codex 的课件页文本清洗提示
- `pipeline/03_alignment/slides_cleanup_report.json` 应记录课件页文本清洗是否成功、使用的是 codex 还是规则兜底
- `pipeline/04_structure/note_outline.json` 和 `pipeline/04_structure/note_blocks.json` 是最终笔记的结构化骨架，`pipeline/05_note/note.md` 只负责展示
- `pipeline/05_note/note_generation_prompt.md` 应被视为 note-generation stage 的可执行任务单
- `pipeline/05_note/note_generation_report.json` 应记录 note stage 的尝试次数、失败原因、质量门结果和是否实际改写了 `pipeline/05_note/note.md`
- 生成 `codex_review_prompt.md` 不是终点，Codex 必须在同一次任务里继续读取并执行它
- 生成 `note_generation_prompt.md` 也不是终点，默认流程必须继续执行 note-generation stage
- 不应把“人工补改清洗稿”当作默认流程，默认流程应是 Codex 自动纠正后继续产出最终笔记
- 审阅完成不能只看 `review_status=done`，还要检查：
  - `transcript.cleaned.txt` 不能与 `transcript.txt` 完全相同
  - `segment_reviews` 必须覆盖 `review_segments.json` 中的全部分段，且每段有非空总结
  - 长转写不能只有 1 条纠错记录
  - 每条有效纠错应至少包含 `raw`、`cleaned`、`reason`、`evidence`
- 如果上述质量校验不通过，主流程应把失败原因反馈给 Codex，并自动重试一轮审阅；只有重试后仍不达标，才判定失败
- note-generation stage 也有独立质量门，至少检查：
  - `note.md` 包含知识小结
  - 包含核心定义卡片
  - 包含知识框架
  - 包含精确时间戳引用
- 若 note-generation stage 未通过，应写入 `note_generation_report.json` 并可重试
- 如需跳过这一步，仅在调试或排障时显式使用 `--skip-codex-review`
- 如需跳过 note-generation 的 Codex 阶段，仅在调试或排障时显式使用 `--skip-codex-note`
- 默认主流程支持阶段级断点续跑；若对应产物已完整存在则跳过，可通过 `--force-audio`、`--force-transcribe`、`--force-frames`、`--force-ocr`、`--force-review-artifacts`、`--force-codex-review`、`--force-note` 强制重跑指定阶段
- `--codex-review-parallelism` 当前默认值为 `5`
- `metadata.json` 应记录本次运行的 `visual_source_mode`、`slides_path` 与 `slides_usable`
- 当用户明确要求“把某个视频转成图文文档”时，默认应优先完成完整主链路，不要擅自退化成“只复用已有 transcript / OCR / review 产物重跑 note 阶段”；只有在用户明确允许或仅为排障时，才可这样做

## Pipeline Architecture

当前代码中的主流程已经按 5 个大阶段组织：

1. `media_pipeline`
   - `video -> audio -> transcript`
   - `video -> visual_candidates -> visual_candidates.ocr -> visual_units`
   - 媒体层内部已拆成音频支线和视觉支线，允许独立演进

2. `review_pipeline`
   - `transcript + ocr -> review_segments -> transcript.cleaned / corrections / review_report`

3. `alignment_pipeline`
   - 在这一层内决定 `slides-first / video-first`
   - `review + visuals -> visual_alignment`
   - 必要时输出 `ppt_alignment_debug.md` 与课件页索引

4. `structure_pipeline`
   - `visual_alignment -> note_outline / note_blocks`

5. `note_pipeline`
   - `note_outline + note_blocks -> note_generation_prompt / note.md`

说明：

- 这 5 个阶段是逻辑 DAG，不是要求所有步骤都串行。
- 当前实现中，媒体层已经拆成两条分支：
  - 音频支线：`audio -> transcript`
  - 视觉支线：`frames -> ocr -> visual_units`
- 需要全局一致性的阶段（例如 `alignment`、最终 `note` 编排）仍在汇合点统一收口。

已验证样本：

- 输入视频：`materials/videos/4.1BERT-2训练.mp4`
- 输出目录：`notes/4-1bert-2训练/`

## Pipeline Artifact View

为了解决中间产物平铺过多、难以看清阶段边界的问题，输出目录现在额外维护一套 `pipeline/` 视图：

```text
notes/<slug>/
  pipeline/
    README.md
    00_run/
    01_media/
      audio/
      visual/
    02_review/
    03_alignment/
    04_structure/
    05_note/
```

说明：

- `pipeline/` 里的文件现在就是主流程的 canonical 真实产物，不再只是链接视图
- `00_run/`：运行级汇总，例如 `metadata.json`
- `01_media/audio/`：`audio.wav` 与各类 `transcript.*`
- `01_media/visual/`：`visual_candidates/`、`visual_candidates.ocr.json`、`visual_units.json`
- `02_review/`：`review_segments.json`、`codex_review_prompt.md`、`transcript.cleaned.txt`、`transcript.corrections.json`、`review_report.json`
- `03_alignment/`：`visual_alignment.json`、`ppt_alignment_debug.md`、`slides_preview/`、`slides_index*.json`、`slides_cleanup_*`
- `04_structure/`：`note_outline.json`、`note_blocks.json`
- `05_note/`：`note_generation_prompt.md`、`note_generation_report.json`、`note.md`

## Current Scripts And Files

当前与这条流程直接相关的文件：

- 主脚本：`video_to_notes.py`
- 流水线编排：`video_to_notes_pipeline.py`
- 产物 readiness / stage plan：`video_to_notes_artifacts.py`
- 笔记结构与渲染：`video_to_notes_note.py`
- 输出路径与时间格式：`video_to_notes_schema.py`
- 视觉预处理与 OCR 辅助：`video_to_notes_visual.py`
- 测试文件：`tests/test_video_to_notes.py`
- 项目说明：`PROJECT_SPEC.md`
- 历史样本试跑：`work/_test_run/`

## What Codex Should Do Next Time

后续如果继续在这个目录工作，Codex 默认应该按下面顺序行动：

1. 先阅读本文件 `PROJECT_SPEC.md`
2. 查看当前目录是否已有：
   - `video_to_notes.py`
   - `tests/test_video_to_notes.py`
   - `notes/`
3. 先运行测试，确认当前脚本没有退化：

```bash
cd /Users/songxijun/workspace/transformer_downloader
pytest tests/test_video_to_notes.py -q
```

4. 测试通过后，如任务本身需要验证长跑、排查真实样本问题、或修改了主流程行为，再按需运行单视频样本命令确认完整链路；不要把长跑样本验证当成每次会话的必做前置步骤。

5. 默认主流程不得停在“生成审阅材料”或“生成 note prompt”这一步。
   Codex 必须继续读取 `codex_review_prompt.md` 与 `review_segments.json`，按分段完成清洗稿纠正和纠错记录填写；随后还应继续执行 note-generation stage，产出最终笔记并写入 `note_generation_report.json`。

6. 在确认现有流程可用后，再继续做增强工作，例如：
   - 更智能的关键帧抽取
   - 更强的 OCR
   - 字幕与图片自动对齐
   - 更接近教材风格的笔记整理

## Recommended Prompt For Future Sessions

如果下次你想让 Codex 继续推进，可以直接用类似下面的话：

```text
请先阅读 /Users/songxijun/workspace/transformer_downloader/PROJECT_SPEC.md。
这个项目的目标是把 mp4 课程视频转成图文学习笔记。
先检查现有脚本 video_to_notes.py 和测试 tests/test_video_to_notes.py 是否还能跑通。
在不破坏现有最小闭环的前提下，继续执行完整主流程：
先生成基础产物，再自动读取每个视频目录里的 codex_review_prompt.md 和 review_segments.json，
按分段自动修正 transcript.cleaned.txt、自动填写 transcript.corrections.json，
然后继续执行 note-generation stage，产出最终 note.md 和 note_generation_report.json，不要停在生成 prompt 这一步。
```

如果你只想让 Codex 先验证现状，可以这样说：

```text
请先阅读 /Users/songxijun/workspace/transformer_downloader/PROJECT_SPEC.md，
然后验证当前单视频流程是否还能跑通，不要先重构。
```

如果你想让 Codex 继续做增强开发，可以这样说：

```text
请先阅读 /Users/songxijun/workspace/transformer_downloader/PROJECT_SPEC.md。
当前最小闭环已经跑通，请在保持现有脚本可运行的前提下，
先执行完整自动流程：生成基础产物后继续读取 codex_review_prompt.md 和 review_segments.json，
按分段自动修正 transcript.cleaned.txt、自动填写 transcript.corrections.json，
再继续优化关键帧抽取和图文对齐。
```

## Target Outputs

后续处理应为每个视频生成一套学习资料，建议目录结构如下：

```text
transformer_downloader/
  materials/
  notes/
    <video-name>/
      note.md
      transcript.txt
      transcript.cleaned.txt
      transcript.corrections.json
      codex_review_prompt.md
      review_report.json
      review_segments.json
      transcript.json
      visual_candidates/
      visual_candidates.ocr.json
      visual_units.json
      visual_alignment.json
      note_outline.json
      note_blocks.json
      note_generation_prompt.md
      note_generation_report.json
      metadata.json
```

其中：

- `note.md`：最终学习笔记，图文并茂
- `transcript.txt`：原始 Whisper 转写
- `transcript.cleaned.txt`：Codex 自动校正后的学习版文本
- `transcript.corrections.json`：分段审阅记录和关键术语修正记录，固定 schema
- `codex_review_prompt.md`：给 Codex 的审阅提示词、执行步骤和验收标准，且必须在同一次任务中被继续执行
- `review_report.json`：审阅质量报告，记录尝试次数、失败原因和最终状态
- `review_segments.json`：主流程自动切分的审阅段落，供 Codex 逐段校对
- `transcript.json`：语音转写结果，带时间戳
- `visual_candidates/` 与 `visual_candidates.ocr.json`：更密的候选视觉页及其 OCR 结果
- `visual_units.json`：去重后的视觉单元
- `visual_alignment.json`：语义段与视觉单元的对齐结果
- `note_outline.json` 与 `note_blocks.json`：最终讲义骨架
- `note_generation_prompt.md`：给 note-generation stage 的执行提示
- `note_generation_report.json`：note-generation 阶段的质量报告
- `metadata.json`：处理参数、输入文件名、时长、处理状态等

## Working Principles

1. 目标是“学习材料”，不是“原始字幕”。
2. 图文必须对齐，图片不能随意插入。
3. 每一步都应保留中间产物，便于重跑和排错。
4. 优先使用本地工具完成处理。
5. 允许批量运行，也允许针对单个视频重跑。
6. 已完成结果默认不覆盖，除非显式指定。

## Standard Workflow

### Step 1. Discover Inputs

识别当前目录中的：

- `materials/videos/` 中的视频文件
- `materials/slides/` 中的课件文件
- `materials/code/` 和 `materials/archives/` 中的代码资料
- `materials/docs/` 中的其他参考资料

要求：

- 建立输入清单
- 记录文件名、大小、类型
- 为每个视频分配一个稳定的输出目录名

### Step 2. Extract Audio

从每个 `mp4` 中提取音频，供后续转写。

要求：

- 保留原视频不变
- 生成单独音频中间文件
- 记录处理日志

### Step 3. Transcribe Speech

将音频转换为带时间戳的文本。

要求：

- 结果按时间片组织
- 能定位到每段讲解对应的视频时间
- 尽可能保留术语、模型名、代码关键词

### Step 3.5. Prepare Review Artifacts

生成供 Codex 自动校正的中间文件。

要求：

- 永远保留 `transcript.txt` 原始版本
- 首次运行时创建 `transcript.cleaned.txt` 作为 Codex 自动修订副本
- 创建 `transcript.corrections.json` 记录关键修正
- 创建 `codex_review_prompt.md`，把任务目标、输入输出、步骤、OCR 提示、验收标准写清楚
- 不依赖静态术语字典，优先结合上下文纠正术语
- 生成这些文件后，Codex 必须继续读取 `codex_review_prompt.md`，自动完成清洗稿和纠错记录
- 默认流程不等待人工参与，除非用户明确要求人工审阅模式

建议 `transcript.corrections.json` 结构：

```json
{
  "source_transcript": "transcript.txt",
  "cleaned_transcript": "transcript.cleaned.txt",
  "review_status": "pending",
  "last_updated": null,
  "corrections": [
    {
      "raw": "原错误词",
      "cleaned": "修正后词语",
      "reason": "为什么这样改",
      "evidence": ["OCR 线索", "上下文线索"]
    }
  ]
}
```

### Step 4. Extract Key Frames

从视频中提取关键画面。

优先抓取：

- PPT 翻页时刻
- 代码页切换时刻
- 图表、公式、结构图出现时刻
- 明显场景变化时刻

要求：

- 不能每秒盲目截帧导致冗余过多
- 每张截图应保留时间点
- 文件名稳定、可追溯

### Step 5. OCR and Visual Parsing

对关键帧执行 OCR，提取其中的文字内容。

重点提取：

- 标题
- 章节编号
- 列表要点
- 代码片段
- 公式文本
- 图表标签

要求：

- OCR 结果要和截图一一对应
- 识别失败的图片要能单独标记

### Step 6. Align Text and Images

把字幕、截图、OCR 按时间顺序对齐。

目标：

- 知道“某段讲解”对应“哪几张关键图”
- 知道“某张图”出现时讲者正在讲什么

要求：

- 对齐结果必须可复用
- 不是临时内存拼接，而是写入结构化文件

### Step 7. Generate Learning Notes

基于转写文本、关键帧和 OCR，生成结构化学习笔记。

每篇笔记至少应包含：

- 标题
- 本节主题
- 分章节内容
- 关键配图
- 重点概念解释
- 代码/公式说明
- 小结

更理想的增强内容：

- 术语表
- 易混淆点
- 复习问题
- 实践建议

### Step 8. Save and Verify

输出最终学习笔记并检查质量。

要求：

- `note.md` 引用的图片路径必须有效
- 各中间文件完整存在
- 失败任务必须有日志

## Quality Bar

一份合格的学习笔记应满足：

1. 不是逐字逐句堆字幕。
2. 有明确章节结构。
3. 有和内容匹配的图片。
4. 能读出视频主题、核心概念和结论。
5. 如果视频中包含代码、公式或图表，应尽量保留这些信息。
6. 输出适合后续继续编辑，而不是一次性废稿。

## Quality Improvement Priority

如果当前生成的图文文档质量不高，后续优化应按以下顺序推进：

### Priority 1. Improve Transcription Quality

优先提升语音转写质量，而不是先改文风。

建议方向：

- 将 Whisper 模型从 `tiny` 升级到 `small` 或 `medium`
- 在转写前做音频预处理，例如：
  - 单声道
  - 16k 采样
  - 音量归一化
  - 降噪
- 建立术语修正规则，统一以下词汇：
  - Transformer
  - BERT
  - RoBERTa
  - ALBERT
  - embedding
  - QKV

### Priority 2. Replace Fixed-Interval Frame Sampling

当前固定间隔抽帧只能用于最小闭环验证，不适合作为最终方案。

后续应优先改成内容驱动抽帧，例如：

- PPT 翻页时抽帧
- 代码页切换时抽帧
- 图表、公式、结构图首次出现时抽帧
- 明显场景变化时抽帧

### Priority 3. Improve OCR Quality

OCR 不应直接对原始截图裸跑。

建议先增加图像预处理：

- 裁边
- 放大
- 对比度增强
- 二值化
- 视情况只裁取标题区、正文区、代码区

低质量 OCR 不应直接写进最终笔记。

### Priority 4. Build Transcript-Frame Alignment

质量提升的关键不是“有字幕”和“有图片”，而是两者要绑定。

后续必须建立时间窗口对齐关系：

- 某张关键帧对应哪一段字幕
- 某段讲解对应哪几张图

这个对齐结果应保存为结构化文件，而不是临时拼接。

### Priority 5. Improve Note Generation

在前面四项原始素材质量提升之后，再优化笔记生成逻辑。

建议把内容拆成三层：

1. 事实层：
   - transcript
   - OCR
   - 时间点
2. 结构层：
   - 章节
   - 小节
   - 图文对应关系
3. 学习层：
   - 概念解释
   - 小结
   - 易错点
   - 复习问题

### Priority 6. Use Supporting Materials

现有项目不只有视频，还包括：

- `materials/slides/`
- `materials/archives/`
- `materials/code/`
- `materials/docs/`

这些资料应在后续质量提升中用于：

- 补全章节结构
- 校正术语
- 提高代码和公式说明质量
- 作为 OCR 低质量时的补充信息源

## Optimization Rule

后续优化时，默认遵守以下原则：

1. 先提升输入质量，再提升输出文风。
2. 先解决转写、抽帧、OCR、对齐，再优化总结语气。
3. 不要用“更像文章”掩盖“识别不准”。

## What Counts As Failure

以下情况不能视为完成：

- 只有纯字幕文本，没有结构整理
- 只有截图，没有文字说明
- 图片和文本没有对应关系
- 输出文件缺失或路径混乱
- 无法定位处理失败的视频

## Execution Rules For Future Work

后续 Codex 在这个目录中工作时，应默认遵守以下规则：

1. 先理解当前输入与已有产物，再决定下一步。
2. 优先保留中间产物，避免黑盒式一次处理到底。
3. 每次新增脚本都要明确：
   - 输入是什么
   - 输出到哪里
   - 是否支持批量
   - 如何断点续跑
4. 每次声称“完成”前，都要给出实际验证方式。
5. 如果要引入新工具，先说明依赖、安装方式和目的。

## Recommended Milestones

建议按以下顺序推进，而不是一次做完全部：

### Milestone 1

实现单个 `mp4 -> transcript + keyframes + markdown skeleton`

### Milestone 2

补充 OCR 和图文对齐

### Milestone 3

补充结构化整理，使内容更接近学习笔记

### Milestone 4

支持批量处理整个目录

### Milestone 5

支持增量重跑、失败恢复、质量检查

## Immediate Next Task

当前项目的下一步建议是：

1. 先选择一个体量适中的视频作为样本
2. 做通 `单视频 -> 图文 Markdown` 的最小闭环
3. 验证输出质量后，再扩展到整个目录批量处理

建议样本可从以下文件中选择一个：

- `materials/videos/4.1BERT-2训练.mp4`
- `materials/videos/4.3.1ALBERT.mp4`
- `materials/videos/1.1.1RNN基本原理.mp4`
