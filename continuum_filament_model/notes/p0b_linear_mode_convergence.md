# P0-B：線形mode・時間／空間数値ゲート

## 目的と結論の範囲

P0-Bは、PR #12/#13/#14 merge後の現行非接触モデルを対象に、次を数値ゲートとして固定する。

1. 小振幅・成長なし・非接触fixtureで、離散線形化のHessian、drag、固有mode、減衰率を比較する。
2. `straight`、`boundary-near`、`buckled-candidate`および成長なしcontrolについて、成長ありの時間・空間比較を行う。
3. `FilamentState`の`time`有限性と`step`整数性を既存の状態契約へ補完する。

これは座屈境界、臨界値、普遍性、相転移、実験適合を決める実験ではない。P1B.2の未解決領域を解消したとみなす条件ではなく、むしろ時間・空間依存を再び検出した場合に`numerically-unresolved`を維持するためのgateである。

## 境界条件と非接触条件

現行fixtureの境界条件は、両端の**位置を初期位置へ固定し、端点接線は固定しない**条件である。したがって、クランプ端（接線固定）ではない。外部押し込み、ピン端、周期境界、閉曲線は扱わない。

全runで次を固定する。

- `contact_stiffness=0`
- `diameter=0`
- 初期非交差
- 両端位置固定、端点接線自由
- 明示的な接触force、摩擦、接着、solverなし

## 線形mode部

一様meshの間隔を `h=L/(N-1)`、自由な内部横変位を `y=(y_1,...,y_{N-2})` とする。小振幅の離散曲げエネルギーは、現行の単位接線・局所参照長の式から

```text
E_b^(2) = EI/(2 h^3) ||D y||^2
K = d^2 E/dy^2 = (EI/h^3) D^T D
Gamma = zeta*h*I
Gamma v = -K y
```

となる。`D`は内部nodeの2階差分行列で、`y_0=y_{N-1}=0`だけを課す。この`K`が、端点位置固定・接線自由の離散参照Hessianである。伸長項は直線まわりの横方向一次線形化へ寄与しないため、`EA`はfixtureの非線形energy確認には残すが、横modeの一次Hessianには現れない。

runnerは次を比較する。

- `eigenvalue_analytic`：上記の離散Hessianとdragの一般化固有値。
- `eigenvalue_numerical_hessian`：公開`forces()`の有限差分gradientから得たHessianの固有値。
- `mode_shape_l2_error`、`mode_shape_mass_overlap`：固有vectorの形状比較。
- `decay_rate_analytic`：離散Hessian/dragの第一mode率。
- `decay_rate_measured`、`tau_mode_measured=1/decay_rate_measured`：小振幅runのlog振幅fit。
- `tau_b_definition=zeta*L^4/(EI*pi^4)`：第一正弦modeに基づく定義値。端点接線自由の離散mode時間の真値とは仮定しない。

標準configは`n_nodes=[5,9,17]`、`dt=[2e-4,1e-4,5e-5]`、`contact_stiffness=diameter=0`である。固有値とmode shapeの比較は`dt`に依存しないが、実測減衰率は時間積分誤差を含むため、別列で保持する。

## 成長部とenergyの分解

各代表点について、まず`n_nodes=13`を固定して`dt,dt/2,dt/4`を比較し、その後`n_nodes=[9,13,17]`を同じ基準`dt`で比較する。`a_max=2h`かつ本configの短時間runではmesh変更が起きないことを固定mesh条件として確認する。もし再meshが起きれば、そのrunのgate statusは成功にせず`numerically-unresolved`とする。

各受理stepで、次の順序を記録する。`r`はstep前の幾何、`a`はstep前の参照長、`a_grown`は成長後、`r_remesh`と`a_remesh`は再mesh後、`r_trial`は受理後の幾何である。

```text
E_before = E(r, a)
E_growth = E(r, a_grown) - E(r, a)
E_remesh = E(r_remesh, a_remesh) - E(r, a_grown)
E_mechanical = E(r_trial, a_remesh) - E(r_remesh, a_remesh)
Delta E = E_growth + E_remesh + E_mechanical
D_Euler = dt * sum_i Gamma_i |v_i|^2
```

ここで`E_growth`は完全な連続体growth workを導出した値ではなく、同じ幾何で参照長を更新した**離散energy差の診断**である。`D_Euler`も受理Euler試行で評価した散逸の診断値で、`E_mechanical=-D_Euler`を恒等式として仮定しない。`E_remesh`、rejected trial、event countを別の列に置き、成長・散逸・再mesh・棄却を混同しない。

- `g=0`：受理energy列が許容値内で非増加であることを確認する。
- `g>0`：全energyの単調減少を要求しない。growth reference energy changeと散逸推定を併記する。
- 完全なgrowth work導出、成長分布の物理同定、接触後のworkは本taskの対象外である。

## 判定と許容値

成長runの機械的分類は既存P1B/P1B.2と同じである。

- 初期第一mode振幅の3倍または`0.005L`を超えない：`straight`
- 閾値を超え、peakで第一modeが支配的かつ第一mode分率が`>=0.70`：`buckled-single`
- その他、失敗、非有限値：`unresolved`

時間・空間gateの既定許容値は次のとおりである。

- 全比較runで分類が一致し、`unresolved`でない。
- onsetの有無が一致し、存在する場合の相対差が`10%`以内。
- peak transverseの相対差が`15%`以内。分母の下限は`0.002L`。
- peak `A1/L`の相対差が`15%`以内。
- fixed-mesh runでremesh countが0。
- `g=0` runでenergy非増加。

どれか一つでも満たさない代表点は`numerically-unresolved`とする。これは「座屈が存在しない」という意味ではなく、現行の時間刻み・空間解像度・判定規則で確定できないという意味である。P1B.2で未解決だったboundary近傍・buckled候補を、単一の結果や見た目だけで`resolved`へ再分類しない。

## 実行と成果物

```bash
TMP_DIR=$(mktemp -d /tmp/growing-string-p0b.XXXXXX)
PYTHONPATH=continuum_filament_model/src \
python continuum_filament_model/benchmarks/linear_mode_convergence.py \
  --config continuum_filament_model/benchmarks/configs/p0b_linear_mode.json \
  --output "$TMP_DIR"
```

一時出力にはper-runの`growth/*/metrics.csv`とsummaryが作られる。リポジトリへ残すのは、レビュー可能なcompact集計だけとする。

```text
continuum_filament_model/results/p0b/
├── compact_summary.json
└── compact_summary.csv
```

compact summaryには、実行revision、config hash、境界条件、非接触条件、線形mode比較、成長energy分解、accepted/rejected/event、判定status、P1B.2との関係、未主張事項を保存する。trajectory、per-run event、plotはcommitしない。

## 未解決・対象外

- 座屈境界、臨界値、臨界曲線、臨界指数、普遍性。
- 線分接触force、friction、adhesion、折りたたみ。
- 実験fit、物理単位較正、CI、外部依存。
- 完全なgrowth workの連続体導出。
- 端点反力の独立solver。

これらはP0-Bのpassやcompact summaryの`converged`を根拠に確定しない。
