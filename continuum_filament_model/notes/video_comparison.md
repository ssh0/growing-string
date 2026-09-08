# 実観察動画と連続フィラメント出力の比較

## 方針

`src/growing_filament/video_comparison.py` は、コアの物理solverとは独立した観測・可視化層です。
動画から得た中心線はデータ契約

```text
time, filament_id, point_id, x, y, quality
```

に従い、`x,y` の座標系は `pixel` とmetadataに明記します。モデルの座標は `model-unit` で、
scaleを推測して変換しません。OpenCVはimportせず、ffmpegのraw frame読出しと、skimageが利用できない
場合のNumPy-only connected-component/skeleton fallbackを使います。

これは「完全自動で単一フィラメントを追跡する」機能ではありません。componentの分岐・複数輪郭・
追跡飛びは候補、`quality_flags`、`censor`、`events.csv` に残し、比較可能区間を人が確認します。

## 再実行

動画metadataだけを確認するには:

```bash
PYTHONPATH=continuum_filament_model/src \
python - <<'PY'
from growing_filament.video_comparison import probe_video
print(probe_video("img/gray5.mp4").to_dict())
PY
```

中心線候補を一時ディレクトリへ出力する例:

```bash
PYTHONPATH=continuum_filament_model/src \
OBS_VIDEO=${OBS_VIDEO:?set input video path} \
python continuum_filament_model/video_compare.py extract \
  --video "$OBS_VIDEO" \
  --output "${OBS_OUTPUT:-/tmp/growing-string-gray5}" \
  --polarity dark --background median --threshold otsu \
  --frame-stride 5 --min-component-size 12
```

原動画も同じ設定で実行できます。色の違いはffmpegのgray変換後に評価するため、原動画で閾値が
弱い場合は `background=local_median`（scipyがない場合はglobal medianへfallback）、ROI、
`threshold=percentile`、`threshold_value` を設定します。
設定はJSONでも渡せます。ROIは `[x0,y0,x1,y1]`、thresholdのabsolute値はcontrast後の`[0,1]`です。

model trajectory（既存 `save_trajectory()` の`.npz`）、model centerline CSV、または簡易JSONを読みます。
時間と空間の登録を明示する場合:

```json
{
  "pixel_per_model_unit": 12.0,
  "x_offset_px": 80.0,
  "y_offset_px": 40.0,
  "rotation_deg": 0.0,
  "time_scale": 1.0,
  "time_offset": 0.0,
  "max_time_error_s": 0.2
}
```

`model_time = (observed_time - time_offset) / time_scale` とし、空間は回転後にscale・offsetを
適用します。`pixel_per_model_unit` が空欄の場合、pixelとmodel-unitを左右に表示するだけで、
endpoint distance、shape RMSE、length difference等の定量metricを計算しません。

```bash
PYTHONPATH=continuum_filament_model/src \
OBS_VIDEO=${OBS_VIDEO:?set input video path} \
MODEL_OUTPUT=${MODEL_OUTPUT:?set model trajectory path} \
python continuum_filament_model/video_compare.py all \
  --video "$OBS_VIDEO" \
  --output "${OBS_OUTPUT:-/tmp/growing-string-gray5}" \
  --model "$MODEL_OUTPUT"

# 校正値を使用する場合だけ --registration を追加する
# --registration /tmp/registration.json
```

比較確認ページ:

```bash
marimo edit continuum_filament_model/notebooks/video_comparison.py
```

## 設定と出力

`SegmentationConfig` の主要設定は次のとおりです。

- `polarity`: `dark` / `bright`
- `background`: `none` / `median` / `local_median` / `scalar`
- `contrast`: `none` / `percentile`
- `threshold`: `otsu` / `absolute` / `percentile`
- `skeleton_backend`: `auto` / `skimage` / `numpy`。topology判定はNumPy Zhang–Suen backendをcanonicalに使い、flags/censor/exportをbackend間で一致させる。topology graphは4近傍edge、centerline順序は8近傍trace
- `roi`, `frame_stride`, `min_component_size`, `max_components
- `max_centerline_points`, `max_jump_px`, `min_quality`

一時outputには次が作られます。

- `centerline.csv`: 必須6列に加え、frame、pixel座標、flags、censor
- `observation_summary.csv`: length、endpoint distance、curvature summary、quality/censor
- `events.csv`: missing、ambiguous、large jump、skeleton loss等
- `metadata.json`: video metadata、設定、座標系、validation、限界
- `manifest.json`: 入力logical ID/SHA-256/size、設定・command SHA-256、runtime、frame coverage、lineage、artifact hash/size、budget
- `lineage.csv`: matched/new/reconnected/missingを含む追跡lineage（欠損区間を削除しない）
- `comparison.csv/json`: sampled frameを母集団として保持し、model time/error、nearest-frame matching、両endpoint対応、model-unit metric、metric status/reason
- `comparison.json` / `comparison_manifest.json`: eligible denominator、excluded rows/reasons、selected/not-selected lineage ID、CSV/JSON hash/size
- `comparison_manifest.json`: comparison CSV/JSONのSHA-256とbyte size、eligible/excluded denominator
- `comparison.mp4`, `frames/`: 左=観測pixel、右=model-unit。大容量のためGit管理しない
- `results/video_comparison/{gray5,original}_manifest.json`: 動画本体を含めず、入力hash/size/ffprobe、short smoke/full-period run、comparison artifact hash/sizeだけをcompactに保存

同じ入力・同じ設定では、生成時刻をmetadataへ入れず、CSV/JSONの順序・数値書式を固定しているため、
manifestとcompact comparisonの再現性を検査できます。Python/NumPy/ffmpegの実装差は別途考慮してください。

## 品質判定と限界

- `quality` は面積と抽出中心線点数からなる決定的な候補スコアで、物理的な確率ではありません。
- `ambiguous_components`、`components_truncated`、`branched_component`、`loop_component`、`large_jump`、`short_centerline`、`skeleton_loss`、`out_of_view`、`roi_clipped`、`new_lineage`、`reconnected_after_missing`、`low_quality` は通常censorです。loopはcenterline.csvへ開曲線として出力しません。
- ROI外、欠損、時間対応不能も比較をcensorします。
- 入力動画に複数輪郭・接触・折りたたみがある場合、候補を黙って一本へ結合しません。
- pixel→physical calibration、実験真値、パラメータ同定、接触/摩擦/有限径の物理則、普遍性・臨界指数は、
  このpipelineから主張しません。
- `model summary`だけで中心線がない場合は、geometry metricを作らず理由を出します。

## 依存関係・出力budget

必須はPython >=3.10、NumPy >=1.23、ffmpeg/ffprobe >=4.4です。`imageio >=2.25`、Pillow >=9、
scikit-image >=0.19、SciPy >=1.8、marimo >=0.23はoptionalです。OpenCVは使用しません。
scikit-imageがない場合はconnected components/skeletonにNumPy fallbackを使います。比較動画の生成には
ffmpegの`libx264` encoderが必要です。今回の実行環境はPython 3.11.5、NumPy 2.2.6、ffmpeg/ffprobe 8.1.2、
imageio 2.36.0、Pillow 10.4.0、scikit-image 0.24.0、marimo 0.23.6でした。

```bash
python --version
ffmpeg -version
ffprobe -version
ffmpeg -hide_banner -encoders | grep 264
```

`--output-budget-mb N` をextract/allへ指定すると、metadata/CSV/events/lineageの生成物budgetをmanifestへ
記録します。render manifestはcomparison動画、代表frame、comparison CSV/JSONのhash/sizeとfull/partial statusを記録します。
`--max-frames` はpartial runとなり、指定しないrunだけをfull-periodと記録します。

入力動画はローカルに提供されたDropbox由来ファイルでしたが、リポジトリ内にライセンス・再配布条件を確認できる
出典資料がないため、サイズ（gray5 3.9 MB、原動画 13 MB）とSHA-256だけをmanifest/noteへ記録し、動画本体はGitへ
含めていません。これは再配布許諾や実験データの公開を主張するものではありません。

## 検証記録（実装時点）

入力2本を一時ディレクトリへ、`local_median`、dark polarity、ROI `[280,150,420,320]`、
`threshold=absolute`、`threshold_value=0.8`、`frame_stride=15`、`min_component_size=30`、
`max_components=1` で通しました。ffprobe metadataは gray5 が 680x512、15 fps、349 frames、
23.266667 s、原動画が 680x512、30 fps、698 frames、23.266 s でした。どちらも metadata、centerline、
events、代表frame、side-by-side comparison videoを生成できました。segment候補は動画全体で安定した
単一lineageにはならず、frame_stride=15のfull-period runではgray5が24 processed frames・20候補・
3つのcandidate ID（20候補がcensor、branched/loop flagsを含む）、原動画が47 processed frames・41候補・
3つのcandidate ID（41候補がcensor）となりました。比較CSVの母集団はlineageのsampled frameを保持し、
gray5は24行、原動画は47行です。これは完全自動追跡の成功とは扱わず、large jump、missing、topology、
censorを比較可能区間の境界として残しています。

同じ gray5 設定を2回実行した `centerline.csv`、`observation_summary.csv`、`events.csv`、`lineage.csv`、
`metadata.json`、`manifest.json` のSHA-256は一致しました。短いsmoke（max_frames=3）はpartial、full-period
runはmax_frames未指定としてmanifestへ記録しました。WMVはCFR decodeでffprobe frame_count=698を再現し、
stride=15で47 frames（last=690）となることを機械検査しました。P1B runnerの一時trajectoryを読み、
scale未指定で比較したケースでは `calibration_status=not_calibrated_metrics_suppressed` となり、
model時間が未対応の行も `model_time_unmatched_or_centerline_unavailable` としてcensorされました。
これらは観察の品質・自動追跡・モデルの物理整合を証明する結果ではありません。marimoページのsliderはlineageのmissing frameも含むprocessed rangeを母集団とし、selected frameのlineage/events、censor、eligible/excluded denominatorを表示します。

## 検証

合成fixture（直線、円弧、sinusoidal、成長、欠損、noise、ambiguous、real-mask T branch、矩形loop、
NumPy/skimage topology parity、ROI truncation、endpoint order、leading missing population、event重複）は
`tests/test_video_comparison.py` にあります。先頭missing frameもprocessed frame rangeから比較母集団へ残し、
`unknown` lineage、missing reason、censor、eligible/excluded denominatorを検査します。既存モデルを変更せず、次で実行します。

```bash
PYTHONPATH=continuum_filament_model/src \
python -m unittest discover -s continuum_filament_model/tests -v
python -m compileall -q continuum_filament_model
python -m py_compile continuum_filament_model/video_compare.py
marimo check continuum_filament_model/notebooks/video_comparison.py
```

実動画を検証するときは `/tmp` などへ出力し、`metadata.json`、代表frame、`centerline.csv`、
`events.csv`、`comparison.mp4` を確認します。今回のfull-period runはframe_stride=15で、gray5は24 sampled frames、
原動画はffmpegのCFR decodeにより47 sampled frames（last=690）です。動画本体、full mask、per-frame大量データはcommitしません。
compact run manifestだけをGit管理します。
