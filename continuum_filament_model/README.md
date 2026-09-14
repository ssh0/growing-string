# Continuum filament model

## 目的

このディレクトリは、既存のシミュレーションコードを変更せずに、研究の主モデルを再設計・検証するための新しい作業領域です。

方針は次のとおりです。

> 一貫した連続モデルを発展させ、実験データとの整合を検証する。三角格子モデルは、その連続モデルの粗視化系・統計的な比較対象として位置づける。

既存の次のディレクトリは変更しません。

- `growing_natural_length_model/`
- `constant_length_model/`
- `triangular_lattice/`

これらは過去の研究資産・比較対象として保存します。新しい物理モデル、検証コード、研究メモ、論文草稿は本ディレクトリに追加します。

## 入口

作業を始めるときは、次の順で読みます。

1. `notes/research_direction.md` — 研究課題、物理との接合、三角格子との関係。
2. `notes/model_spec.md` — 現在のプロトタイプの状態変数、エネルギー、成長、境界条件。
3. `notes/validation_plan.md` — 数値検証、物理検証、実験比較の受入条件。
4. `paper_draft.md` — 最終論文の初稿。結果未取得の部分は未実施・計画として記載。
5. `src/growing_filament/` — 新しい連続モデルの最小実装。
6. `tests/` — 実装の不変条件と数値検証。

## 現在のプロトタイプ

初期版は、次の範囲に限定しています。

- 2次元・開曲線・単一フィラメント
- 慣性を無視した過減衰ダイナミクス
- 伸長エネルギーと離散曲げエネルギー
- 基板ドラッグ
- 有限径の非局所線分ペナルティ接触（摩擦なし）
- 既存互換の非隣接ノード間軟接触ペナルティ
- 非隣接線分の最近接距離・最近接点・有限径 gap・交差の幾何診断
- 線形試行中の swept crossing 検出とステップ棄却
- 局所参照長の指数成長
- 参照長が長くなった場合の中点再メッシュ

これは実験を再現済みのモデルではありません。特に、以下は未確定または未実装です。

- 実験に対応した成長分布（全体成長か先端成長か）
- フィラメントの径、断面積、線密度の変化
- 接着・摩擦・ヒステリシスを含む接触則
- 摩擦・接着・履歴を含む高度な接触則と有限時間刻みでの非貫入保証
- 異方的な基板抵抗
- 実験画像からの中心線抽出・同定（観測層として実装済み。入力動画の提供とlineage確認は別途必要）
- 妥当な物理単位への較正

## 実行方法

外部パッケージを必要としない最小構成です。Python 3.10以上とNumPyを想定します。

リポジトリルートから、次を実行します。

```bash
PYTHONPATH=continuum_filament_model/src \
python continuum_filament_model/src/run_smoke.py
```

テストは次のとおりです。

```bash
PYTHONPATH=continuum_filament_model/src \
python -m unittest discover -s continuum_filament_model/tests -v
```

テストが通ることは、物理モデルが実験を再現することを意味しません。数値ベンチマークと実験比較は `notes/validation_plan.md` に従って別々に確認します。

### CI

`.github/workflows/continuum-filament-tests.yml` が、`continuum_filament_model/**` または同workflow自身の変更を含む `master` 向けPull Requestおよび手動実行で、上記のテストコマンドを実行します。既存テストがimportする外部Pythonパッケージとして、CIでは、既存の検証記録に合わせて `numpy==2.2.6` と、ベンチマークのプロット出力に必要な `matplotlib==3.10.9` をインストールします。動画fixtureのテストには `ffmpeg` が必要なため、Ubuntu runnerのaptパッケージを追加でインストールします。

CIの実行環境は `ubuntu-24.04` とPython `3.11.5` に固定しています。これはPython `3.10以上`というモデルの想定範囲から選んだ代表環境であり、CIの成功はこのLinux環境での回帰スイートの成功だけを示します。macOS・Windowsや、他のPython・NumPy・ffmpegの組合せの互換性、物理モデルの実験再現性は、このworkflowでは保証しません。

Gate 3 の幾何診断・イベント・再現性回帰を含むテストは、同じテストコマンドで実行します。
各 `OverdampedGrowingFilament` は、初期形状診断、各試行の要求/試行/受理 `dt`、状態・エネルギー要約、
理由別の棄却（`crossing_rejection`、`nonfinite`、`displacement_exceeded` など）を `events` に記録します。
`run_manifest()` または `growing_filament.reproducibility.build_manifest()` は、Git revision、Python/NumPy、
入力hash、初期状態hash、canonical終状態hash、イベント列、受理/棄却数をJSON互換形式で返します。
`save_trajectory()` は、軌跡とともにこのmanifest/event列を保存できます。

線分の最近接距離・最近接点・パラメータ・有限径 `gap` / `penetration` は
`growing_filament.geometry.segment_contact_geometry()` または
`nonlocal_segment_contacts()` で計算します。中心線交差と交差しない有限径 gap 接触は
診断種別を分け、`d=0` では法線を捏造せず `normal=None` とします。配列添字、再メッシュ、
将来のlineage IDの制約と幾何契約は `notes/segment_contact_geometry.md` に記録しています。
この幾何契約を使う線分 penalty の定式化・scatter・U字 fixture・力学検証は
`notes/segment_penalty_contact.md` に記録しています。

再メッシュの小規模ベンチマークは、次で実行できます。

```bash
PYTHONPATH=continuum_filament_model/src \
python continuum_filament_model/benchmarks/remesh_convergence.py
```

このベンチマークは、解析的な滑らかな合成形状を `a_max` ごとに弧長サンプルする
解像度試験と、固定条件の短い成長 run を含みます。角点を含む同一折れ線では
曲率特異性により曲げエネルギー・節点力のメッシュ独立性を仮定せず、形状比較には
`growing_filament.observables.arc_length_weighted_radius_of_gyration` を使います。

Gate 2の時間積分・散逸ベンチマークは、次で実行できます。

```bash
PYTHONPATH=continuum_filament_model/src \
python continuum_filament_model/benchmarks/time_integration.py
```

3節点の固定メッシュ製造解について `dt=0.02, 0.01, 0.005` の一次収束誤差を出力し、
`dt=0.01` と `dt=0.005` の4節点・成長なし診断について、エネルギー時系列、受理した
`dt`、棄却理由を出力します。成長なしの試行はエネルギー非増加を受理条件としますが、
成長ありでは成長がエネルギーを注入し得るため、単調減少を要求しません。

## Free/free 端点力学ゲート

実験整合を主目的とする境界条件は `free/free` です。端点を固定しない自然境界の
保存力残差、離散曲げモーメント、shear-equivalent residual は
`OverdampedGrowingFilament.endpoint_diagnostics()` で取得できます。固定端は既存
ベンチマークの比較controlとしてのみ扱います。符号規約と受入条件は
`notes/free_end_dynamics.md` に記録しています。

focused test と bounded benchmark は次で実行します。benchmark は全節点軌跡を保存せず、
endpoint trajectory、work/dissipation、residual、時間・空間 refinement のcompact summary
だけを指定した一時ディレクトリへ出力します。接触はこの段階では無効です。

```bash
PYTHONPATH=continuum_filament_model/src \
python -m unittest continuum_filament_model.tests.test_free_end_dynamics -v

TMP_DIR=$(mktemp -d /tmp/growing-string-free-end.XXXXXX)
PYTHONPATH=continuum_filament_model/src \
python continuum_filament_model/benchmarks/free_end_benchmark.py \
  --output "$TMP_DIR"
```

## Stage 2：free/free 成長–緩和–座屈探索

`benchmarks/free_growth_buckling_stage2.py` は、既存の非接触 free/free solver を使った
探索的な bounded experiment です。既定設定は
`benchmarks/configs/stage2_free_free.json`、仕様と解釈の制約は
`notes/stage2_free_growth_buckling.md` に記録します。成長率、曲げ/伸長比、基板drag、初期
imperfection、時間刻み、空間解像度を探索し、axial stiffness と drag density の独立した
parameter contrast を deterministic fixture/refinement、seed付き exploratory replicate と
分けて保存します。endpoint trajectory、参照長・輪郭長、
axial-force proxy、endpoint force/moment residual、transverse amplitude、曲率 RMS、
mode fraction、energy/work/dissipation、accepted/rejected `dt`、無次元量、events、
provenance を compact summary へ出力します。`buckling-candidate` は観測ラベルであり、
相境界ではありません。
接触、摩擦、接着、折りたたみは無効のまま維持します。

```bash
TMP_DIR=$(mktemp -d /tmp/growing-string-stage2.XXXXXX)
PYTHONPATH=continuum_filament_model/src \
python continuum_filament_model/benchmarks/free_growth_buckling_stage2.py \
  --config continuum_filament_model/benchmarks/configs/stage2_free_free.json \
  --output "$TMP_DIR"
```

動画入力が存在する場合は、同runnerに `--video /path/to/video.mp4` を渡します。既存の
video-comparison pipelineを読み取り専用で使い、入力hash、lineage/censor、品質、calibration/
holdoutの状態をcompactに記録します。scaleやtime registrationを推測せず、未校正または
censoredな場合の定量overlay・parameter fittingは抑制します。動画本体、中心線、軌跡などの
大きな成果物は `_video_artifacts/`・`_runs/` の一時出力に残し、Gitへ追加しません。
入力が欠ける場合は明示的な `input_missing` とし、legacy動画へ置換しません。

### 登録に依存しない形態比較

`growing_filament.scale_free_comparison` と
`benchmarks/scale_free_shape_comparison.py` は、`img/gray5.mp4` から
`run_pipeline()` が抽出した中心線と Stage 2 free/free 軌跡を、pixel/model-unit および
video/model-time 登録なしで比較します。輪郭長で各系列を独立に正規化し、
`q=(L-L_initial)/(L_final-L_initial)` と `s/L` を使って形態遷移を対応付けます。
quality、censor、lineage、frame/time coverage、入力・モデル・設定・revision hashを保存し、
物性fit・parameter identification・`model_inadequacy`判定は抑制します。通常の
validated-centerline モードでは zero-growth、non-monotonic、中心線不適格時に比較不能の
診断を残します。bounded runner は、有限な censor 済み中心線を再分類せず探索的に比較する
candidate-input モードと、deterministic fixture とは分離した検証済み初期値感度 population を
使用します。補間や絶対登録の推測はしません。モード別の契約と解釈は
`notes/scale_free_shape_comparison.md` を参照してください。

## focused free/free 力学収束ゲート

形態ラベルだけでなく力学的な収束を確認する bounded runner は
`benchmarks/free_free_convergence_gate.py` です。対象は free/free・一様成長・伸長・曲げ・等方基板dragの非接触モデルだけで、接触・摩擦・接着・折りたたみ・gray5の物性fitは含めません。

```bash
TMP_DIR=$(mktemp -d /tmp/growing-string-free-free-gate.XXXXXX)
PYTHONPATH=continuum_filament_model/src \
python continuum_filament_model/benchmarks/free_free_convergence_gate.py \
  --config continuum_filament_model/benchmarks/configs/free_free_convergence_gate.json \
  --output "$TMP_DIR"
```

既定suiteは straight、boundary-near、buckled-candidate の deterministic fixture を、時間3水準・空間3水準で別々に比較します。accepted/requested `dt`、reject/event、energy、参照長成長work、散逸推定、端点force/moment残差、onset、peak transverse、曲率RMS、mode spectrum/fractions、total length、初期条件摂動と provenance を記録します。`morphology_status` と `mechanics_status` は分離し、どちらかが未収束なら `numerically-unresolved` を維持します。parameter contrast と seed付き初期条件感度 replicate は deterministic population と混ぜません。

無次元量 `G_b`、`G_s`、`chi` の定義は `benchmarks/buckling_benchmark.py:dimensionless_groups` に一元化しています。詳細な受入条件、reason code、gray5観測契約は `notes/free_free_convergence_gate.md` を参照してください。

## P0-B：線形mode・時間／空間数値ゲート

現行の非接触モデルについて、端点の**位置だけを固定し、接線は自由**とした離散線形化を検証します。これはクランプ端（位置と接線を固定）ではありません。

```bash
TMP_DIR=$(mktemp -d /tmp/growing-string-p0b.XXXXXX)
PYTHONPATH=continuum_filament_model/src \
python continuum_filament_model/benchmarks/linear_mode_convergence.py \
  --config continuum_filament_model/benchmarks/configs/p0b_linear_mode.json \
  --output "$TMP_DIR"
```

線形mode部では、既存のエネルギーgradient（`forces()`）を有限差分したHessian、離散曲げHessian `K=(EI/h^3)D^T D`、参照長重みのdrag `Gamma=zeta*h I` の固有値・mode shapeを比較します。小振幅・成長なし・`contact_stiffness=0`・`diameter=0`で、`n_nodes=[5,9,17]`、複数の`dt`について減衰率も測定します。`tau_b=zeta*L^4/(EI*pi^4)`は第一正弦modeの**定義値**として保存するだけで、固定位置・自由接線の離散mode時間の真値とは仮定しません。実測`tau_mode=1/decay_rate_measured`を別列で記録します。

成長部のgate scopeは`morphology-only`です。`straight`、`boundary-near`、`buckled-candidate`と成長なしcontrolを、時間方向は`n_nodes=9`で`dt, dt/2, dt/4`、空間方向は`n_nodes=[5,7,9]`で比較します。`A1/L`、mode spectrum、onset、peak transverse、curvature RMS、energy、accepted/rejected/eventを保存し、次の量を分離します。`morphology-converged`は成長–緩和全体やenergy/workの収束を意味せず、実accepted `dt`、reject数、energy・散逸の変動を監査情報として併記します。

- `growth_reference_energy_change`：同じ幾何で参照長を更新した離散energy差。完全な連続体growth workの導出・主張ではありません。
- `dissipation_euler_estimate`：受理Euler試行の`dt*sum Gamma_i|v_i|^2`という診断値。
- `remesh_energy_jump`：再mesh前後のenergy差（固定mesh gateでは0であることを確認）。
- `rejected_trials`と`event_count`：solverの試行・イベント数であり、energy項ではありません。

`g=0`ではenergy非増加を確認し、`g>0`では成長によるenergy変化を単純な散逸や再mesh jumpと混同しません。比較条件、許容値、分類規則、未解決判定は `notes/p0b_linear_mode_convergence.md` に固定します。P1B.2で`numerically-unresolved`だった境界近傍・座屈候補は、収束不一致や許容値超過があれば同じラベルを維持します。線形減衰率には`decay_rate_relative_tolerance=0.01`の診断thresholdを適用します。このgateから座屈境界・臨界値・普遍性・相転移を主張しません。

compactな実行結果だけを残す場合は、`$TMP_DIR/compact_summary.json` と`compact_summary.csv`を確認し、per-runのtrajectoryやmetricsをリポジトリへコピーしません。

## P1B：成長–緩和競合と座屈ベンチマーク

非接触・両端固定・決定論的な `y(x)=A sin(pi*x/L)` 摂動を用いた小規模ベンチマークを、`benchmarks/buckling_benchmark.py` で実行できます。既存の `growing_filament` APIを呼び出し、成長・力・再メッシュの式を複製しません。`G_b = growth_rate * tau_b`（`tau_b = drag_density * L**4 / (bending_stiffness * pi**4)`）と `G_s`、`EI/(EA*L**2)` を保存し、`straight` / `buckled-single` / `unresolved` の機械的regimeとして整理します。これは相転移の主張ではありません。

```bash
PYTHONPATH=continuum_filament_model/src \
python continuum_filament_model/benchmarks/buckling_benchmark.py \
  --config continuum_filament_model/benchmarks/configs/p1b_noncontact.json \
  --output /tmp/growing-string-p1b
```

各ケースには実効設定、無次元量、Git revision、manifest、イベント、主要観測量のCSV/JSON、図を保存します。既定suiteには決定論的比較に加え、3 seedのslow/fast/high-EI trialを含め、`trial_summary.json` と問い別の `question_comparison.csv/json/png` に代表値・標準偏差を出力します。`contact_stiffness=0`、`diameter=0`、初期非交差を強制し、接触・折りたたみ・実験fit・三角格子比較は未実装です。詳細なfixture、問いごとの比較結果、分類規則、固定端反力proxyの限界、未解決事項は `notes/p1b_buckling_benchmark.md` を参照してください。

## P1B.2：収束・regime map・trial頑健性

P1B.2は、P1Bの既存runner/APIを使った小規模な非接触追加実験です。コア物理式、接触モデル、時間積分器は変更しません。決定論的fixture（`seed=null, trial=0`）と、初期imperfectionだけを乱数化したseed付きtrialを別集計します。trialは確率的な力学則や実験ノイズモデルではありません。

設定と実行入口は次のとおりです。

```bash
TMP_DIR=$(mktemp -d /tmp/growing-string-p1b2.XXXXXX)
PYTHONPATH=continuum_filament_model/src \
python continuum_filament_model/benchmarks/p1b2_experiments.py \
  --config continuum_filament_model/benchmarks/configs/p1b2_noncontact.json \
  --output "$TMP_DIR"
```

per-runのconfig、metrics、events、manifest、summaryは再現性確認用に一時出力へ保存します。これらをGit管理下へコピー・commitしません。commitするのは、実験結果を要約した `results/p1b2/` 直下のcompact CSV/JSON/PNGと `compact_manifest.json` だけです。compact manifestにはsource/execution revision、configのSHA-256、seed一覧、run数、出力上限、主要summaryのSHA-256、各compactファイルのSHA-256を記録しています。

`--mode pilot`、`--mode convergence`、`--mode grid` で段階実行もできます。既定設定は、(1) straight / 境界近傍 / buckled-single のpilot、(2) 各代表点の3空間解像度×2刻み幅（半減を含む）、(3) 最低3×3の `G_b × chi` grid、各cellの決定論的fixture＋5 seed trial、(4) `L=2.0` と `L=1.5` の小規模比較を実行します。`G_b=g tau_b`、`chi=EI/(EA L^2)` の軸値から `EI` と `g` を計算し、実効値も各manifestへ保存します。

分類は相転移・臨界曲線の主張ではありません。gridのtrial分類は次の4種類です。

- `resolved-straight`: 5/5 trialが `straight`
- `resolved-buckled`: 5/5 trialが `buckled-single`
- `trial-mixed`: `straight` と `buckled-single` が混在し、`unresolved` がない
- `numerically-unresolved`: 少なくとも1 trialが `unresolved`、失敗、または分母不足

各cellのCSV/JSONには分母、分類件数・割合、座屈開始時刻・最大横変位・第一モード分率・曲率・エネルギーの中央値、IQR、平均、標準偏差、欠測数、未解決理由を保存します。決定論的fixtureの分類・開始時刻・最大変位は別列で保持し、trial統計へ混ぜません。

収束判定の許容値は設定ファイルと `convergence_summary.json` に固定保存します。既定値は、全解像度・刻み幅で分類が一致し `unresolved` でないこと、座屈開始時刻の基準 run との差が相対10%以内、最大横変位の相対15%以内（長さの0.2%を下限）です。満たさない代表点は成功扱いにせず `numerically-unresolved` と記録します。

成果物の構造は次のとおりです。per-run artifactは一時出力にのみ置き、Gitにはcompact集計を残します。

```text
results/p1b2/
├── compact_manifest.json
├── compact_summary.json
├── effective_config.json
├── experiment_summary.json
├── pilot_summary.csv/json/png
├── convergence_runs.csv
├── convergence_summary.csv/json/png
├── regime_map.csv/json/png
├── trial_summary.csv/json
└── size_comparison.csv/json/png
```

既定configの実行計画は87 run、per-run trajectory・plotなし、総出力上限120 MBです。実際の実行時間と一時出力バイト数は `experiment_summary.json` に記録します。per-run manifestにはGit revision、実効設定、`seed`、trial番号、初期状態hash、終状態hash、イベント列を含みますが、commit対象ではありません。同一seed・同一設定では一時出力上で設定・初期状態hash・主要結果の一致を検査できますが、異なるseedは一致させずtrial分布として扱います。完全一致の範囲は同じPython/NumPy/Git環境に限定されます。

今回のcompact結果は、最終コードrevision `6eb930f902989107b2b3718d26e821524abb44b2` 上で87 runを再実行して生成しました。config SHA-256は `834d5beed66d0407a16c5efe32dac3332b3be690cf67b3b92d038ed8d680dec9`、seed setは `[101, 202, 303, 404, 505]`、一時出力の実測値は3,406,517 bytes、主要summary（`compact_summary.json`）のSHA-256は `f1cd28441b9f0c684977254293ed66759fbf24c364879325c2be67a7f001f5dd` です。旧revision `d105884595cd72cf8a11d87c9cc6e325240458fc` と比較して、分類・収束status・未解決領域に変更はなく、差分はprovenanceと実行メタデータに限定されます。

結果の限定は明確です。数値収束が確認できたのはstraight代表点だけで、boundary近傍とbuckled代表点は `numerically-unresolved` のままです。この小規模・非接触suiteから臨界値、臨界曲線、普遍性、実験適合、接触・有限径の結論は導きません。

この追加実験が扱わないものは、接触・有限径・折りたたみ、実験fit、三角格子比較、普遍性、臨界指数です。境界域や解像度依存が残る場合は、未解決として報告します。

## 実観察動画との比較

動画から中心線候補を抽出し、pixel座標の観測データとmodel-unitのtrajectoryを、品質・censor付きで比較する独立パイプラインを追加しています。コア物理solverは変更しません。入口は `video_compare.py`、実装は `src/growing_filament/video_comparison.py`、レビュー用marimoページは `notebooks/video_comparison.py` です。

```bash
PYTHONPATH=continuum_filament_model/src \
python continuum_filament_model/video_compare.py extract \
  --video /path/to/video.mp4 --output /tmp/filament-observation \
  --polarity dark --background median --threshold otsu --frame-stride 5
```

`centerline.csv` は `time, filament_id, point_id, x, y, quality` を含み、`metadata.json` と `manifest.json` に座標系、設定、品質検査、入力hashを保存します。scale/time calibrationが未指定ならpixel/model-unitを別表示し、定量metricは抑制します。ROI、複数component、飛び、欠損、skeleton不良は `events.csv`、`quality_flags`、`censor` に残します。入力動画、full mask、per-frame大量データ、比較動画は `/tmp` 等へ出力し、Gitへ追加しません。

動画CLIの必須依存はPython >=3.10、NumPy >=1.23、ffmpeg/ffprobe >=4.4です。imageio、Pillow、scikit-image、SciPyはCLI pipelineではoptionalで、scikit-imageがない場合はNumPy fallbackを使います。marimo review page（`notebooks/video_comparison.py`）は`marimo >=0.23`と`matplotlib`を必須とし、`matplotlib.pyplot`をimportして描画します。したがってmarimo/matplotlibはCLIだけを使う場合に限りoptionalです。生成manifestの`runtime`にはmarimoとmatplotlibのversion（import不能時は`unavailable:<ExceptionName>`）を記録します。今回の検証時versionはmarimo 0.23.6、matplotlib 3.10.9です。

詳細なschema、登録式、lineage/censor、依存version、output budget、限界、再実行コマンドは `notes/video_comparison.md` を参照してください。入力動画本体はライセンス・再配布条件を確認できないためGitへ含めず、`results/video_comparison/` にcompact run manifestだけを置きます。

## 発表用の追加数値・観測データ

中間発表向けの粗い3x3相図と未解決セルの物理的内訳を補うため、決定論的な `G_b × chi` 7x8（56条件）密度座屈マップを追加しました。既存solverを変更せず、`A_max/L`、第一モード分率、曲率RMS、onset時刻、受理Eulerの散逸エネルギー推定値、支配モードを測定します。`dominant_mode` と波形分類（高次モード波、局所座屈、mixed mode、sub-threshold transient）を同じセルに保存し、判定閾値を `notes/dense_buckling_heatmap.md` に固定しています。代表4条件の `t0/t_mid/t_end` 座標だけを出力します。

```bash
PYTHONPATH="$PWD:$PWD/continuum_filament_model/src" \
python continuum_filament_model/benchmarks/dense_buckling_heatmap.py \
  --config continuum_filament_model/benchmarks/configs/p1b2_dense_heatmap.json \
  --output continuum_filament_model/results/presentation_data/dense_buckling
```

有限径接触については、U字自己接触とS字接触・折りたたみの代表ケースを `t0/t_mid/t_end` で抽出し、接触点、最近接点、法線、貫入量、ペナルティ法線力を `results/presentation_data/contact_snapshots/` に保存します。全ステップ座標や動画は保存しません。

実観察動画の抽出入口は `benchmarks/video_presentation_export.py` です。`img/gray5.mp4` が存在する環境では、既存のPIL/imageio/skimage互換パイプラインを使って生中心線CSV、代表中心線JSON、`L(t)`、曲率プロファイルを生成できます。`--model` と任意の `--registration` を追加すると、同じ出力先に既存の比較CSV/JSONも生成します。動画本体はこのworktreeには含めず、承認済みのローカル入力から抽出しました。今回のcompact成果物は論理ID `img/gray5.mp4`、`status=extracted` として記録し、代表6フレームの中心線、24サンプルの `L(t)`、1,089点の曲率プロファイルを保存しています。入力動画が別環境にある場合も同じコマンドで再生成できます。

## 三角格子モデルとの関係

`triangular_lattice/` は、新モデルのコードへ直接importしません。接続は観測量と無次元パラメータを介して行います。

共通化を目指す観測量は、少なくとも次のものです。

- 全輪郭長
- 端点間距離
- 慣性半径
- 曲率・折れ角分布
- 接触率・接触長
- roughness
- 成長ステップまたは時間に対する形態変化

三角格子側の `beta` は、現時点では物理温度や曲げ剛性ではなく、局所成長の曲げバイアスを表す無次元パラメータとして扱います。連続モデルの曲げ剛性との対応は、接線相関・持続長・曲率分布などを用いて別途較正します。

## データと成果物の扱い

新しい実験比較データ・シミュレーション結果は、既存の `triangular_lattice/results/` へ直接保存しません。まず本ディレクトリ下に、run ID、設定、seed、Git revision、入力データの識別子を含む実験単位の成果物を保存する設計にします。

既存の結果を再生成・上書き・移動しないでください。既存結果を使う場合は、由来が確認できたものと未確認のlegacyデータを分けて記録します。

## 変更方針

- 既存モデルのバグ修正をこのディレクトリの変更で済ませない。比較対象を変更する必要がある場合は、別作業として明示する。
- 新しい物理仮定は、コードだけでなく `notes/model_spec.md` に追記する。
- 新しい検証条件は、実装と `notes/validation_plan.md` の両方に反映する。
- `paper_draft.md` には、実施済みの結果と計画中の結果を混在させない。
- 実験データに適合させる前に、合成データと解析解で数値モデルを検証する。
