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
- ノード間の軟接触ペナルティ
- 非隣接線分の最近接距離・最近接点・有限径 gap・交差の幾何診断
- 線形試行中の swept crossing 検出とステップ棄却
- 局所参照長の指数成長
- 参照長が長くなった場合の中点再メッシュ

これは実験を再現済みのモデルではありません。特に、以下は未確定または未実装です。

- 実験に対応した成長分布（全体成長か先端成長か）
- フィラメントの径、断面積、線密度の変化
- 接着・摩擦・ヒステリシスを含む接触則
- 線分接触の反発力（線分距離は現在は診断専用）
- 異方的な基板抵抗
- 実験画像からの中心線抽出・同定
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
将来のlineage IDの制約と、後続の接触方式へ渡す契約は
`notes/segment_contact_geometry.md` に記録しています。これらは節点反発の力則とは分離されており、
線分反発・摩擦・接着・動的CCDを意味しません。

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
