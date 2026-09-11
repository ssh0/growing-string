# free/free 非接触 力学収束ゲート

## 目的と範囲

`benchmarks/free_free_convergence_gate.py` は、自由端・一様な参照長成長・伸長・離散曲げ・等方基板 drag の競合を、**形態ラベルだけでなく力学的な収束**として監査する bounded runner である。solver のエネルギー式や時間積分を複製せず、既存の `OverdampedGrowingFilament` を呼び出す。

このゲートの境界条件は常に `free/free` で、次を有効化・追加しない。

- `contact_stiffness=0`, `diameter=0`
- 接触、摩擦、接着、折りたたみ、局所成長
- gray5 の物性 fit、parameter identification、形態差からの physics claim

contact または fixed boundary を設定した config は、runner が無視して進めず、設定エラーとして拒否する。legacy の `growing_natural_length_model/`、`constant_length_model/`、`triangular_lattice/` は変更しない。

## 実行

軌跡・metrics・event・manifest は指定した一時ディレクトリの `_runs/` にだけ保存する。Gitへ追加するのは、必要に応じて compact summary と provenance のみとする。

```bash
TMP_DIR=$(mktemp -d /tmp/growing-string-free-free-gate.XXXXXX)
PYTHONPATH=continuum_filament_model/src \
python continuum_filament_model/benchmarks/free_free_convergence_gate.py \
  --config continuum_filament_model/benchmarks/configs/free_free_convergence_gate.json \
  --output "$TMP_DIR"
```

出力の主な契約は次のとおり。

- `convergence_summary.json/csv`: 代表点ごとの temporal/spatial 判定
- `temporal_runs.csv`, `spatial_runs.csv`: 決定論的 fixture の compact 行
- `controls.csv`: 成長なし等の control。fixture の収束統計へ混ぜない
- `contrasts.csv`: growth rate、`EA`、`EI`、drag density、initial imperfection の明示 contrast
- `sensitivity_replicates.csv`: seed と perturbation を持つ初期条件感度 population
- `compact_summary.json`: suite-level の境界、除外範囲、population、観測量、判定
- `compact_manifest.json`: config hash、revision、件数、compact artifact hash
- `_runs/<run>/metrics.csv`, `events.json`, `manifest.json`: 大きい一時成果物

既定設定は straight、boundary-near、buckled-candidate の3 deterministic representative を持つ。各 representative は temporal `dt=[dt, dt/2, dt/4]` の3水準と spatial `n_nodes=[5,9,13]` の3水準を実行する。temporal の各水準は少なくとも最大 `dt` まで走り、accepted `dt` が重複する設定は未解決または設定エラーとして扱う。spatial run は temporal finest `dt` を使い、`spatial_refinement.dt` が一致しない config は拒否する。buckled candidate の `t_end` は onset と post-onset の両方を含めるように設定している。設定を短縮した smoke run は、物理的な onset や収束結論ではなく、schema・分類・失敗保持の確認に限る。

## 観測量と診断

各 run の compact summary と per-step metrics は、次を分離して保存する。

### 数値監査

- requested `dt`、accepted `dt` の min/max/mean/値集合
- rejected trials、理由別件数、event count、event sequence hash
- accepted `dt` の集合・mean、`mechanical_balance_residual_cumulative`、remesh 発生・remesh energy jump
- source/config/initial state/final state/event hash、Python/NumPy、seed、perturbation。compact row と refinement audit に同じ値・hashを保持する

### 形態・mode

- morphology label、onset time、peak transverse amplitude
- curvature RMS
- mode spectrum（1--6 の係数）と mode fractions、dominant mode
- initial-condition amplitude、seed、noise fraction、perturbation mode
- total contour length と reference length

### 力学・仕事

- total/stretch/bend energy と初期・最終・span
- `growth_work`: 同じ幾何で参照長だけを更新した離散 energy change を、成長仕事の診断値として `growth_work_step` / `growth_work_cumulative` に記録する。完全な連続体 growth-work 導出とは主張しない
- `dissipation_estimate`: accepted Euler 区間の `dt * sum_i Gamma_i |v_i|^2`
- remesh energy jump、mechanical energy change、mechanical balance residual
- endpoint force residual、bending moment residual、shear-equivalent residual。compact row と refinement audit にも `endpoint_shear_residual_final` を保存する

端点残差は過渡状態の自然境界診断であり、過渡 run の各時刻でゼロであることを要求しない。`free/free` の符号規約は `notes/free_end_dynamics.md` に従う。

## 判定規則

temporal と spatial を別々に比較し、それぞれに次の3つの status を持つ。

- `morphology_status`: onset、peak transverse、curvature RMS、mode fractions、形態 label の比較だけ
- `mechanics_status`: energy、growth work、dissipation、endpoint force/moment/shear residual、balance residual の cumulative/max、total length の比較
- `status`: 上記の両方が通った場合だけ `resolved`

いずれかの run が失敗・未解決、remesh が発生、形態 label が不一致、または設定した許容値を外れた場合は、対応する reason code を保存し、`status=numerically-unresolved` とする。mechanical balance residual は `1e-12` の絶対 floor を除き、残差自身を分母とするため、時間刻み依存の残差減少を O(1) floor で隠さない。`morphology-converged` だけから `mechanics-converged` や `resolved` へ再分類しない。これは「座屈がない」という意味ではなく、指定した時間・空間解像度と力学監査で結果を確定できないという意味である。

時間刻みの reject や accepted/requested の差は、必ず audit として残す。reject があることだけで形態を都合よく resolved にせず、run failure、remesh、収束指標の不一致を unresolved reason として保持する。

## 無次元量

`G_b`、`G_s`、`chi` は `benchmarks/buckling_benchmark.py:dimensionless_groups` を唯一の定義元として import する。

```text
 tau_b = zeta * L**4 / (EI * pi**4)
 tau_s = zeta * L**2 / EA
 G_b   = growth_rate * tau_b
 G_s   = growth_rate * tau_s
 chi   = EI / (EA * L**2)
```

各 run の effective `growth_rate`, `EA`, `EI`, `zeta`, `L`, `G_b`, `G_s`, `chi` を保存する。異なる入力値の単純な見た目比較や、gray5 の未確定観測をこの無次元量へ逆同定することは行わない。

## deterministic と sensitivity の契約

- deterministic fixture/refinement: `seed=null`, deterministic sine imperfection。temporal/spatial convergence の母集団。初期参照長は常に configured `L/(n_nodes-1)` の一様値とし、摂動振幅で総参照長・物質量を変えない。
- control: 成長なし等。deterministic refinement の判定とは別。
- parameter contrast: 基準 representative から一因子だけを変更し、growth rate、EA、EI、drag density、amplitude を明示する。初期参照長は固定し、変更因子以外の物質量条件を保つ。contrast は temporal/spatial refinement を実施しない単一解像度の探索的入力対照であり、各 run を `numerical_status=numerically-unresolved`、`numerical_reason_codes=[not_refined_across_time_or_space]` として保持する。
- sensitivity replicate: seed と amplitude factor/noise fraction を記録する。初期参照長は固定し、摂動振幅・seed noiseによる初期形状だけを変える。確率的な力学則・実験ノイズモデルではなく、初期条件感度の母集団である。各 member も同じ理由で `numerically-unresolved` とし、形態差を物理的な差として解釈しない。

`gray5` は入力品質と観測契約が未確定のため、このゲートでは読み込まず、物性 fit や physics claim を行わない。将来比較する場合に必要な契約は、pixel-to-length、撮影間隔/time registration、centerline quality/censor、filament lineage、calibration/holdout 分離、入力 hash と provenance である。

## 実行済み compact 結果

`results/free_free_convergence_gate/` は schema version 6、commit `dec712850cd6da2cd3110e7918896ab72423d214` 上で既定 config を実行した compact 結果である。deterministic fixture/refinement 18 run、control 1 run、parameter contrast 5 run、初期条件 sensitivity 9 runを含む。時間 refinement は3代表点すべてで morphology は `morphology-converged` だが、mechanics は `mechanical_balance_residual_cumulative_out_of_tolerance` により `numerically-unresolved` である。一方、空間 refinement は3代表点すべてで少なくとも morphology または mechanics の不一致があり、`numerically-unresolved` のまま保持した。parameter contrast と sensitivity replicate は各 run/member に `not_refined_across_time_or_space` を付与し、探索的な形態差を数値収束済みの物理差と解釈しない。したがって、この結果は形態の bounded consistency と力学・空間方向・探索 population の未解決を示す監査記録であり、座屈境界・臨界値・物性 fit の根拠ではない。

per-run の `metrics.csv`・`events.json`・`manifest.json` は実行時の一時ディレクトリに残し、`results/` には compact provenance-bearing summary だけを保存する。
