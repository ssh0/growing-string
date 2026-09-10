# Scale-free morphology comparison

## 目的と範囲

`growing_filament.scale_free_comparison` は、実動画の pixel/model-unit
登録値や video/model time 登録値を必要としない形態比較モードである。対象は
`stage2_free_free_growth_relaxation_buckling` の metadata が付いた free/free・一様成長・
非接触の Stage 2 NPZ 出力と、`run_pipeline()` が抽出した動画中心線である。scope metadata
のない CSV/JSON は対象モデルとして扱わず比較不能とする。solver の式、接触物理、物性 fit
は変更・実行しない。

観測中心線とモデル中心線は、それぞれの現在の輪郭長 `L` で独立に正規化する。時間の
照合は行わず、各系列について

```text
q = (L - L_initial) / (L_final - L_initial)
```

を計算し、最近傍の `q` のみを対応付ける。補間、pixel/model scale の推定、time scale・
time offset の推定は行わない。frame/time は coverage と追跡用の provenance として保存するが、
physical time とは扱わない。

## 保存する量と入力品質

各行は、観測の frame/time、長さ、`q`、quality、quality flags、lineage status、censor と、
対応モデル frame/time、`q`、progress error を保存する。valid な中心線について次を観測・
モデルの双方で保存する。

- normalized endpoint distance (`endpoint_distance/L`)
- normalized radius of gyration (`R_g/L`, 輪郭長重み)
- normalized peak deflection（端点 chord からの最大偏位 `/L`）
- curvature RMS times length
- 1--6 mode fractions
- 回転・平行移動・端点方向を除いた normalized shape distance（valid な matched 行のみ）

`new_lineage`、`reconnected_after_missing`、missing、branch/loop、low quality、ROI/image
boundary 等の既存 censor はそのまま保持し、shape distance の分母から除外する。モデルは free/free・一様成長・非接触 Stage 2 の deterministic baseline、または
明示的な `initial_condition_sensitivity` protocol（perturbation range、metrics、acceptance
criteria付き）に限定する。`exploratory_replicate` は deterministic として受理しない。
video alignment は translation/rotation/endpoint orientation の shape alignment として記録し、
initial-condition sensitivity や物性 fitとは分離する。`model_inadequacy`、parameter
identification、calibrated dynamics は、このモードの成果物に作成しない。

次の状態は推測で補わず、`status` と `input_quality.reasons` に明示する。manifest の `validation.valid`、中心線の厳密な frame/time/point/censor/quality contract、または lineage の整合性が invalid の場合は、中心線の行・lineage・frame coverage を保持したまま比較不能とする。成長進行度の端点と `q` は、有効・非censor・許可された lineage の中心線だけから計算し、除外されたフレームへ補間しない。曲率 RMS は固定弧長サンプリング後に計算し、入力点の細分割に依存させない。

- manifest、観測中心線 contract、または観測 frame key が invalid：`input_quality_invalid_observation_contract`
- provenance-bearing Stage 2 NPZ が invalid、または unsupported model format：`input_quality_invalid_model_contract`（フレームがない場合は `model_centerline_unavailable`）
- lineage artifact が欠落または summary/centerline と整合しない：`input_quality_invalid_observation_contract`
- 有効な長さが2点未満：`insufficient_length_observations`
- 初期・終端長の差がほぼ0：`zero_growth_span`
- 長さが減少する系列：`non_monotonic_lengths`
- valid centerline が0件、または censor で全件除外：`input_quality_no_eligible_centerline`
- model centerline がない：`model_centerline_unavailable`

zero-span / non-monotonic の場合は `q` 対応と shape distance を抑制する。入力品質で比較
不能でも、missing/censor/lineage、frame coverage、入力・モデル・設定・revision hash と
外部 artifact identifier を compact diagnostic に残す。

## 成果物と再実行

単一比較は次で実行できる。まず `extract` で既存観測 pipeline を実行し、その出力に対して
`scale-free` を呼ぶ。登録 JSON は渡さない。

```bash
PYTHONPATH=continuum_filament_model/src \
python continuum_filament_model/video_compare.py extract \
  --video img/gray5.mp4 --output /tmp/growing-string-gray5-scale-free \
  --polarity dark --background local_median --threshold absolute \
  --threshold-value 0.8 --frame-stride 15 --min-component-size 30 \
  --max-components 1 --roi 280 150 420 320

PYTHONPATH=continuum_filament_model/src \
python continuum_filament_model/video_compare.py scale-free \
  --output /tmp/growing-string-gray5-scale-free \
  --model /tmp/stage2/_runs/fast_growth_low_bend/trajectory.npz
```

実際の Stage 2 run と gray5 を一度に比較する bounded runner は次である。既定で
`fast_growth_low_bend` と `fast_growth_high_bend` の deterministic run を実行する。
trajectory、raw centerline、per-case CSV は一時 artifact とし、root の `compact_summary.json`、
`summary.csv`、`compact_manifest.json` のみをレビュー対象にする。

```bash
TMP_DIR=$(mktemp -d /tmp/growing-string-scale-free.XXXXXX)
PYTHONPATH=continuum_filament_model/src \
python continuum_filament_model/benchmarks/scale_free_shape_comparison.py \
  --config continuum_filament_model/benchmarks/configs/stage2_free_free.json \
  --video img/gray5.mp4 --output "$TMP_DIR"
```

### gray5 bounded 実行記録

コミット済み `gray5.mp4` を使い、source revision
`ad461ca43e47aa784755f8293ed5991689e2c5d9` で上記の2 deterministic fixtureを
比較した。入力は `extracted`、24 sampled frames、20 candidates が censor となった。
両ケースとも有効中心線0件と観測長の `insufficient_length_observations` により
`input_quality_no_eligible_centerline`、`compared_rows=0` となった。これは scale-free
agreement の証拠ではなく、中心線品質による比較不能を明示した結果である。動画の
SHA-256、設定・shape設定 SHA-256、model runのSHA-256、external artifact IDは
`results/scale_free_shape_comparison/compact_summary.json` と `compact_manifest.json` に保存した。

成果物の schema version は `continuum-filament-scale-free-shape-0.1`、runner は
`continuum-filament-scale-free-runner-0.1`。manifest の `registration.status` は常に
`not_required_not_inferred` であり、null 以外の絶対登録値を成果物へ追加しない。

## 解釈の境界

scale-free の shape agreement は、成長進行度に沿った無次元形態の一致を示し得るが、
calibrated dynamics、成長速度、物性値、時刻同期、pixel/model scale、parameter
identification の証拠ではない。censor が比較を妨げる場合は「比較不能」という入力品質結果
のみを報告する。形状差を `model_inadequacy` や接触の必要性へ変換しない。
