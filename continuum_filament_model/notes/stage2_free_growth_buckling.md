# Stage 2：free/free 成長誘起座屈の探索

## 目的と範囲

Stage 2 は、マージ済みの free/free 端点診断を使い、非接触のまま

- 一様な参照長成長
- 伸長剛性と曲げ剛性の比
- 初期非対称を含む初期 imperfection
- 時間刻みと空間解像度
- 基板 drag

の因果関係を探索する bounded experiment である。伸長剛性と基板 drag は、
`fast_growth_low_bend` を基準にした独立の `parameter_contrast` として明示する。
入口は
`benchmarks/free_growth_buckling_stage2.py`、既定設定は
`benchmarks/configs/stage2_free_free.json` である。既存の solver 式や legacy
model directory は変更しない。

この段階では `contact_stiffness=0`、`diameter=0`、free/free とし、接触、摩擦、
接着、折りたたみ、局所成長、パラメータ同定は行わない。`buckling-candidate` は定義した観測閾値を超えた形態ラベルであり、臨界値・相境界・
相転移を意味しない。

## 因果 contrast の設計

既定設定では `fast_growth_low_bend` の `growth_rate=0.20`、
`bending_stiffness=0.02`、`amplitude=0.02`、`n_nodes=9`、`dt=0.002` を固定し、
次の4条件だけを追加する。

| 条件 | 変更因子 | 値 | 基準値 |
| --- | --- | ---: | ---: |
| `fast_growth_low_bend_soft_axial` | `axial_stiffness` | 2.5 | 5.0 |
| `fast_growth_low_bend_stiff_axial` | `axial_stiffness` | 10.0 | 5.0 |
| `fast_growth_low_bend_low_drag` | `drag_density` | 0.5 | 1.0 |
| `fast_growth_low_bend_high_drag` | `drag_density` | 2.0 | 1.0 |

runner は各条件を `run_kind=parameter_contrast` として基準fixture・refinement・
seed付きreplicateから分離し、条件名、基準fixture、変更因子・値、入力hash、Git revision、
初期・終状態hash、イベント列hashを保存する。接触剛性・径は全条件で0、境界はfree/free
のままであり、接触物理は追加しない。

各runには `tau_b=zeta L^4/(EI pi^4)`、`tau_s=zeta L^2/EA`、
`G_b=growth_rate*tau_b`、`G_s=growth_rate*tau_s`、`chi=EI/(EA L^2)`、
`dt/tau_b`、`dt/tau_s`、初期mesh比を保存する。これらは次元の異なる入力値を
比較するための記録量であり、臨界値や同定結果ではない。

## 再現コマンド

リポジトリルートで実行する。通常実行では各runのmetrics.csv・events・manifestとcompact結果を指定した
出力ディレクトリに置き、`--video` 指定時だけ選択した `video_model_case` の節点軌跡を保存する。
動画中間生成物と軌跡は大きいため、通常は `/tmp` を使う。

```bash
TMP_DIR=$(mktemp -d /tmp/growing-string-stage2.XXXXXX)
PYTHONPATH=continuum_filament_model/src \
python continuum_filament_model/benchmarks/free_growth_buckling_stage2.py \
  --config continuum_filament_model/benchmarks/configs/stage2_free_free.json \
  --output "$TMP_DIR"
```

既定デザインは、決定論的 fixture、独立した parameter contrast、`n_nodes` と `dt` の
refinement、seed付きの exploratory replicate を別々に実行する。`compact_summary.json` には endpoint
trajectory と sampled time-series が含まれ、`summary.csv` は比較しやすい1行/実験で
ある。`compact_manifest.json` は config hash、source revision、各 run の provenance
を保持する。deterministic fixture/refinement と seeded replicate を同じ統計へ混ぜない。

## 記録する観測量

各 run は、少なくとも以下を記録する。

- endpoint の位置と endpoint distance、輪郭長、参照長
- 線分の `EA*(l-a)/a` による signed axial-force proxy と圧縮 proxy
- free endpoint force residual、曲げ moment residual、shear-equivalent residual
- `max_transverse_amplitude`、曲率 RMS、1--6 mode の分率と dominant mode
- stretch/bend/contact energy、成長 work、Euler trajectory による dissipation 推定、力学収支残差
- accepted/rejected `dt`、棄却理由、event sequence hash、接触イベント数
- Git revision、Python/NumPy、入力・初期・終状態 hash、seed/trial lineage

成長 work は同じ幾何で参照長だけを1ステップ成長させた離散 energy 差であり、完全な
連続体成長仕事の導出ではない。dissipation は accepted Euler trajectory の診断推定値で
ある。端点残差は過渡状態ではゼロである必要はなく、free/free の境界診断として保存する。

座屈 onset は、`max_transverse_amplitude > max(3*initial_amplitude, 0.005*length)`
を初めて満たす accepted observation と定義する。ノイズ付き replicate の結果は、
fixture と独立の exploratory distribution として報告し、この定義から相境界を外挿しない。

## 実動画との比較

実入力が存在する場合は、既存の `growing_filament.video_comparison` を読み取り専用で
呼び出す。たとえばローカルの gray5 は次のように実行する。

```bash
TMP_DIR=$(mktemp -d /tmp/growing-string-stage2-gray5.XXXXXX)
VIDEO_PATH=/path/to/video.mp4
PYTHONPATH=continuum_filament_model/src \
python continuum_filament_model/benchmarks/free_growth_buckling_stage2.py \
  --config continuum_filament_model/benchmarks/configs/stage2_free_free.json \
  --output "$TMP_DIR" \
  --video "$VIDEO_PATH"
```

動画本体、raw centerline、comparison CSV、trajectory は `_video_artifacts/` と `_runs/`
に出力され、Gitへ追加しない。`video_comparison_manifest.json` は入力 hash、ffprobe
metadata、抽出候補数、lineage/censor、calibration/holdout の状態を compact に記録する。
入力が存在しない・読めない・中心線契約を満たさない場合は、`input_missing` または
`unusable` とし、legacy triangular-lattice videoで代用しない。

pixel per model unit と time registration は推測しない。定量比較には
`pixel_per_model_unit`、`time_scale`、`time_offset` の3値を明示する。いずれかが無い場合は
モデル時刻照合を実行せず、動画が支持する pixel の `L(t)`、endpoint distance、curvature、
frame coverage を観測 QC として保持するが、model/pixel の定量 overlay と parameter fitting は
抑制する。登録値があっても
品質 censor、new/reconnected lineage、missing、branch/loop は比較母集団から除外し、
calibration と holdout の records を混同しない。入力中心線が censored のとき、見た目の
overlayからモデル不足・入力品質・数値未収束を区別せずに結論を出してはならない。
校正済みで `metric_status=computed` の行だけを対象に、`shape_rmse_px > 5.0` または
`abs(length_difference_px) / model_length_px > 0.25` を `model_inadequacy` 候補として記録する。
これは候補条件であり、接触物理やモデル不足の確定診断ではない。候補は入力品質/censorと
別フィールドに保存し、候補行がない・未校正・eligible行がない状態も区別する。動画処理の
失敗は `missing_ffmpeg`、`decode_or_corrupt_input`、`invalid_format`、`analysis_failure`
の安定したカテゴリで保存し、絶対パスや例外詳細はcompact成果物へ出さない。選択した
モデルrunのfailure・numerically-unresolved分類、または refinement 間の onset/peak/分類の
不一致は `numerical_nonconvergence` を優先し、動画の観測差を `model_inadequacy` として
評価しない。

## 結果の解釈

- `input_quality` / `censor`：動画抽出、lineage、ROI、branch、missing、品質フラグの問題。
- `model_inadequacy`：非接触 free/free の仮定で観測差が残る場合。接触が必要だと推測して
  接触 solver をこの Stage 2 に追加せず、観測差を証拠として follow-up に残す。
- `numerical_nonconvergence`：accepted/rejected dt、refinement、event、run failure の問題。
- fixture と replicate の形態差：seed付き初期 imperfection の分布として保存し、実験ノイズ
  の同定や相境界の証拠とはしない。
- parameter contrast の形態差：成長率、曲げ剛性、初期 imperfection、`dt`、空間解像度を
  固定した限定的な因果対照として保存し、他の未探索要因へ一般化しない。

## 実行済み探索の要約

`results/stage2_free_free/` は、source revision
`b4c1b56fc168bdb5542157dfd3e2d38333864d3b`、`t_end=4.0` の22 run（決定論的 fixture 5、
refinement 4、parameter contrast 4、seed付き replicate 9）をcompactに保存する。全runは
`contact_enabled=false`、失敗0、拒否0であった。決定論的な fast-growth/low-bend 条件は
`t≈3.30` に定義した onset を通過し、`n_nodes=9,13` と `dt=0.002,0.001` の4 refinement
でも onset は `3.298--3.316`、peak transverse amplitude は約 `0.0876--0.0894` であった。
追加したcontrastでは、soft axialのonsetは約3.634、stiff axialは約3.132、high dragは
約2.842で、low dragは閾値未満だった。これらは成長率、曲げ剛性、初期 imperfection、
`dt`、空間解像度を固定した限定的な対照であり、相境界・臨界値・実験パラメータ同定ではない。
fast-growth/low-bend replicate は3本中2本が candidate、1本が閾値未満であり、これは
初期 imperfection の探索的な感度としてのみ扱う。

現worktreeには実動画入力が存在しないため、現行HEADでの動画pipelineは
`input_missing`、`quantitative_fitting=suppressed`、`model_inadequacy=not_assessed_input_missing`
として記録した。動画入力が利用可能になった場合は、`VIDEO_PATH` に実動画を指定して再実行し、
pixel/model scaleとtime registrationを明示した場合だけ定量比較へ進む。実動画がない状態を、
過去の抽出結果やlegacy videoによる比較結果として解釈しない。

この段階の compact result だけから、実動画のパラメータ同定、接触・摩擦・折りたたみの
再現、臨界値、普遍性を主張しない。
