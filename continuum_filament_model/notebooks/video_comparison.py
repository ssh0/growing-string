"""Marimo review page for observation/model comparison artifacts.

Start from the repository root:

    marimo edit continuum_filament_model/notebooks/video_comparison.py

The page is read-only with respect to artifacts.  The pipeline is rerun from
``continuum_filament_model/video_compare.py`` so expensive video processing is
explicit and reproducible.
"""

import marimo

__generated_with = "0.23.6"
app = marimo.App(width="full")


@app.cell
def _():
    import marimo as mo

    mo.md(
        """
        # 動画・連続フィラメント比較レビュー

        このページは生成済みartifactを読み、動画フレーム、中心線候補、モデル中心線、
        品質フラグ、censor、比較metricを同じ時刻で確認します。観測はpixel座標、モデルは
        model-unitで別々に表示します。`pixel_per_model_unit` を指定した登録だけが定量metric
        を有効にします。候補が複数あるフレームや追跡飛びは自動的にcensor対象です。

        処理の再実行はCLIから行います:

        ```bash
        PYTHONPATH=continuum_filament_model/src \\
        python continuum_filament_model/video_compare.py all \\
          --video img/gray5.mp4 --output /tmp/growing-string-gray5 \\
          --model /tmp/model/trajectory.npz
        ```
        """
    )
    return (mo,)


@app.cell
def _(mo):
    import numpy as np

    output_input = mo.ui.text(value="/tmp/growing-string-gray5", label="artifact directory")
    frame_input = mo.ui.slider(0, 10000, value=0, step=1, label="frame")
    roi_input = mo.ui.text(value="", label="ROI x0,y0,x1,y1 (再実行設定の記録)")
    scale_input = mo.ui.text(value="", label="pixel_per_model_unit (空欄=未校正)")
    controls = mo.vstack([output_input, frame_input, roi_input, scale_input])
    return controls, frame_input, np, output_input, roi_input, scale_input


@app.cell
def _(controls, frame_input, mo, np, output_input, roi_input, scale_input):
    import csv
    import json
    import sys
    from pathlib import Path

    import matplotlib.pyplot as plt

    source_dir = Path.cwd() / "continuum_filament_model" / "src"
    if str(source_dir) not in sys.path:
        sys.path.insert(0, str(source_dir))
    from growing_filament.video_comparison import SegmentationConfig, iter_video_frames, probe_video, segment_mask

    output = Path(output_input.value).expanduser()
    metadata_path = output / "metadata.json"
    summary_path = output / "observation_summary.csv"
    centerline_path = output / "centerline.csv"
    if not metadata_path.exists() or not summary_path.exists() or not centerline_path.exists():
        result = mo.vstack([controls, mo.md(f"artifactが見つかりません: `{output}`。先にCLIでextract/allを実行してください。")])
    else:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        with summary_path.open(encoding="utf-8", newline="") as handle:
            summary = list(csv.DictReader(handle))
        with centerline_path.open(encoding="utf-8", newline="") as handle:
            centerline = list(csv.DictReader(handle))
        if not summary:
            result = mo.vstack([controls, mo.md("観測候補がありません。threshold、polarity、ROI、min component sizeを見直してください。")])
        else:
            available_frames = sorted({int(row["frame"]) for row in summary})
            selected = min(available_frames, key=lambda value: abs(value - int(frame_input.value)))
            selected_summary = [row for row in summary if int(row["frame"]) == selected]
            points = {}
            for row in centerline:
                if int(row["frame"]) == selected:
                    points.setdefault(row["filament_id"], []).append((int(row["point_id"]), float(row["x"]), float(row["y"])))
            for values in points.values():
                values.sort()

            video_path = Path(metadata["video"]["path"])
            frame_image = None
            try:
                stride = int(metadata["segmentation"].get("frame_stride", 1))
                for index, _, image in iter_video_frames(video_path, probe_video(video_path), stride, max_frames=1_000_000):
                    if index == selected:
                        frame_image = image
                        break
            except Exception as exc:
                frame_image = None
                read_error = f"フレーム読出し失敗: `{type(exc).__name__}: {exc}`"
            else:
                read_error = None

            if frame_image is None:
                message = read_error or f"frame {selected} は入力から読めません。"
                result = mo.vstack([controls, mo.md(message)])
            else:
                observation_config = SegmentationConfig.from_mapping(metadata["segmentation"])
                mask, _ = segment_mask(frame_image, observation_config)
                figure, axes = plt.subplots(1, 3, figsize=(18, 5), constrained_layout=True)
                axes[0].imshow(frame_image, cmap="gray")
                axes[0].set_title(f"observed pixel / frame={selected}")
                for filament_id, values in points.items():
                    xy = np.asarray([[x, y] for _, x, y in values])
                    axes[0].plot(xy[:, 0], xy[:, 1], linewidth=2, label=filament_id)
                if points:
                    axes[0].legend(loc="best", fontsize=7)
                axes[0].set_xlim(0, frame_image.shape[1])
                axes[0].set_ylim(frame_image.shape[0], 0)
                axes[1].imshow(mask, cmap="gray")
                axes[1].set_title("segmentation mask")
                for filament_id, values in points.items():
                    xy = np.asarray([[x, y] for _, x, y in values])
                    axes[1].plot(xy[:, 0], xy[:, 1], linewidth=2, label=filament_id)
                axes[1].set_xlim(0, frame_image.shape[1])
                axes[1].set_ylim(frame_image.shape[0], 0)

                comparison_path = output / "comparison.csv"
                if comparison_path.exists():
                    with comparison_path.open(encoding="utf-8", newline="") as handle:
                        comparison_rows = list(csv.DictReader(handle))
                else:
                    comparison_rows = []
                axes[2].axis("off")
                selected_row = selected_summary[0]
                lines = [
                    f"frame={selected}",
                    f"time={selected_row['time']} s",
                    f"candidates={len(selected_summary)}",
                    f"quality={selected_row['quality']}",
                    f"flags={selected_row['quality_flags']}",
                    f"censor={selected_row['censor']}",
                    "",
                    "model / comparison",
                ]
                matching = [row for row in comparison_rows if int(row["frame"]) == selected]
                if matching:
                    row = matching[0]
                    lines.extend([
                        f"metric_status={row['metric_status']}",
                        f"reason={row['metric_reason']}",
                        f"endpoint_distance_px={row.get('endpoint_distance_px', '')}",
                        f"shape_rmse_px={row.get('shape_rmse_px', '')}",
                    ])
                else:
                    lines.append("comparison.csv not found; model panel is unavailable")
                if scale_input.value.strip():
                    lines.append(f"UI scale={scale_input.value.strip()} (rerun compare to apply)")
                if roi_input.value.strip():
                    lines.append(f"UI ROI={roi_input.value.strip()} (rerun extract to apply)")
                axes[2].text(0.02, 0.98, "\n".join(lines), va="top", family="monospace")
                result = mo.vstack([
                    controls,
                    mo.md(f"**selected frame:** `{selected}` / available `{available_frames[0]}..{available_frames[-1]}`"),
                    figure,
                ])
    result


if __name__ == "__main__":
    app.run()
