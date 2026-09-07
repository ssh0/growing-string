# P1B.2：数値収束・成長–緩和regime map・trial頑健性

## 目的と範囲

P1B PR #10 merge後の `benchmarks/buckling_benchmark.py` と `src/growing_filament` の既存APIを使い、P1Bで未回答だった次の範囲を小規模な非接触実験で確認した。

1. `G_b = g tau_b`、`chi = EI/(EA L^2)`で整理した傾向の、`dt`・空間解像度への依存。
2. straight / buckled-single / trial混在 / 数値未解決の領域。
3. 決定論的初期摂動とseed付き初期imperfectionの差、座屈開始時刻と最大横変位。
4. `L=2.0` と `L=1.5` の小規模な系サイズ比較。

接触、有限径、折りたたみ、実験fit、三角格子比較、普遍性、臨界曲線・臨界指数は対象外である。分類は機械的なregimeラベルであり、相転移の証拠ではない。

## 実験条件と再実行

```bash
PYTHONPATH=continuum_filament_model/src \
python continuum_filament_model/benchmarks/p1b2_experiments.py \
  --config continuum_filament_model/benchmarks/configs/p1b2_noncontact.json \
  --output continuum_filament_model/results/p1b2
```

実行したrevisionは `experiment_summary.json` と各manifestの `git_revision` に保存した。実験は87 run（pilot 3、収束18、grid 54、サイズ比較12）、所要約135.8秒、出力約3.4 MBで、trajectoryとrunごとのplotは保存しないcompact policyである。出力上限は120 MB、計画run数上限は100である。

全runで `contact_stiffness=0`、`diameter=0`、両端固定、初期非交差、決定論的Euler積分・既存再メッシュを使用した。seed付きtrialは `numpy.default_rng(seed)` の標準正規乱数を内部節点へ加え、標本標準偏差で規格化して `amplitude * noise_fraction`（既定5%）を振幅とした。乱数化したのは初期imperfectionだけであり、確率的力学則や実験ノイズモデルではない。

- 決定論的fixture：`seed=null, trial=0`
- grid trial：各cell 5 seed（101, 202, 303, 404, 505）
- 同一seed・同一設定の初期状態hash、終状態hash、主要結果、compact event summaryは一致した（pilot 3点および `G_b=0.19, chi=0.00025, seed=303` の再実行比較）。異なるseedは一致を要求せず、trial統計として扱う。

各runの `summary.json` に分類、開始時刻、最大変位、第一モード分率、曲率、エネルギー、reject件数・理由、実効設定を保存し、`manifest.json` に入力hash、初期状態hash、canonical終状態hash、trial識別子、compact event summaryを保存した。全manifestの索引は `results/p1b2/manifest_index.csv/json` にある。compact eventは全時刻の状態を保存せず、初期化、受理/棄却数、棄却理由別件数を保存する。時系列指標のサンプルは各runの `metrics.csv` に最大64行、全行を使った分類結果はsummaryに保存した。

## 固定した分類・収束許容値

分類は既存P1B runnerの規則をそのまま使った。

- 初期第一モード振幅の3倍または `0.005 L` を超えない：`straight`
- 閾値を超え、ピーク時に支配モード1かつ第一モード分率 `>=0.70`：`buckled-single`
- それ以外、実行失敗、非有限値：`unresolved`

収束の基準runは、同一代表点で最大節点数かつ最小`dt`のrunとした。既定許容値は次のとおりで、`convergence_summary.json` にも保存した。

- 全3空間解像度×2`dt`の分類が一致し、`unresolved`でない。
- 座屈開始時刻の基準runとの差が相対10%以内。全runで開始なしの場合は一致とする。
- 最大横変位の基準runとの差が相対15%以内。相対差の分母は`max(|reference|, 0.002 L)`。

いずれかを満たさない代表点は、成功・収束済みとせず `numerically-unresolved` とした。

## Pilotと数値収束

pilotは `n_nodes=9, dt=0.001, L=2, EA=100, EI=0.1` で、成長率を変えて選んだ。

| 役割 | `g` | `G_b` | pilot分類 | onset | peak max \|y\| |
|---|---:|---:|---|---:|---:|
| straight | 0.05 | 0.08213 | straight | — | 0.02437 |
| boundary-near | 0.15 | 0.24638 | straight | — | 0.04437 |
| buckled-single | 0.20 | 0.32851 | buckled-single | 0.192125 | 0.06930 |

pilotの役割名は事後に分類を置き換えるためのものではなく、収束比較の代表点の選び方を記録するためのもの。出力は `pilot_summary.csv/json/png`。

収束結果は次のとおり。

| 代表点 | 全runの分類 | onset相対差最大 | peak相対差最大 | 判定 |
|---|---|---:|---:|---|
| straight | straight × 6 | 0 | 0.1297 | **converged** |
| boundary-near | straight 5、buckled-single 1 | —（開始有無が不一致） | 1.0021 | **numerically-unresolved** |
| buckled-single | straight 3、buckled-single 2、unresolved 1 | 0.1505 | 0.6783 | **numerically-unresolved** |

したがって、straight代表点のこの範囲では分類と指標が許容値内で安定した。一方、境界近傍と座屈代表点は空間解像度・`dt`により分類または指標が変わる。特に `n_nodes=7, dt=0.001` 付近を、座屈の確定結果として扱わない。これは「数値的に座屈が存在しない」という意味ではなく、この設定・判定規則で解像できていないという意味である。

## 3×3 `G_b × chi` map

gridの軸は次のとおりである。

- `G_b = [0.1642557161, 0.19, 0.3285114321]`
- `chi = [0.000125, 0.00025, 0.0005]`
- `L=2, EA=100, zeta=1, n_nodes=7, dt=0.001, t_end=0.2`
- 各cellは決定論的fixture1件＋5 seed trial

seed trial分類（`straight / buckled-single / unresolved` の件数）は次のとおり。

| `G_b` | `chi=0.000125` | `chi=0.00025` | `chi=0.0005` |
|---:|---|---|---|
| 0.1643 | 0/0/5、numerically-unresolved | 5/0/0、resolved-straight | 5/0/0、resolved-straight |
| 0.19 | 0/1/4、numerically-unresolved | 3/2/0、**trial-mixed** | 0/5/0、resolved-buckled |
| 0.3285 | 0/0/5、numerically-unresolved | 0/0/5、numerically-unresolved | 0/0/5、numerically-unresolved |

この小規模gridで記述できる範囲は次のとおり。

- 中央付近の`G_b=0.19, chi=0.00025`は、決定論的fixture単独では`straight`だが、seed trialでは3/5 straight・2/5 buckled-singleとなり、trial-mixedとして扱うべき領域である。
- `G_b=0.1643`の中・高`chi`では5/5 straight、`G_b=0.19, chi=0.0005`では5/5 buckled-singleで、同じ小規模fixture内の傾向は`G_b`増加・曲げ剛性条件に整合する。
- 低`chi`や高`G_b`では第一モード分率が閾値に届かず、`unresolved`が多い。これはstraightまたはbuckledの証拠へ読み替えない。
- したがって図は、`resolved-straight`、`resolved-buckled`、`trial-mixed`、`numerically-unresolved`を区別した観測mapであり、臨界曲線・臨界値をフィットしていない。

各cellの中央値・IQR・平均・標準偏差・欠測数・未解決理由は `regime_map.csv/json` と `trial_summary.csv/json` にある。例えばtrial-mixed cellの開始時刻は有効2/5、中央値0.1960、IQR 0.0005、平均0.1960、標準偏差0.00071であり、straight trialの開始時刻欠測3件も分母5のまま保存される。

## 系サイズ比較

`G_b=0.1643, chi=0.00025` と `G_b=0.19, chi=0.00025`を、`L=2.0`から`L=1.5`へ変更し、`G_b`と`chi`が同じになるよう`EI`と`g`を再計算した。`L=1.5`では両点とも決定論的・5 trialともbuckled-single（resolved-buckled）になった。一方、`L=2.0`では前者がresolved-straight、後者がtrial-mixedである。

従って、この小規模な比較では無次元軸だけで分類傾向が完全には保たれない。`n_nodes`、`dt`、有限時間、初期振幅、境界の離散化も同時に影響し得るため、無次元整理の普遍性を主張しない。出力は `size_comparison.csv/json/png` に保存した。

## 研究上の結論と未回答

### 回答できた範囲

- straightの代表点について、3空間解像度×2`dt`で分類は一致し、peak変位は固定許容値内だった。
- 境界近傍とbuckled代表点は、解像度・`dt`に対して未解決であることを実験結果として示せた。
- 3×3 grid・各cell 5 seedで、決定論的fixtureとは別にtrial混在域と数値未解決域を区別できた。
- 同一seed・同一設定の決定論性（hash・主要結果一致）と、異なるseedのtrialばらつき（件数、割合、中央値、IQR、平均、標準偏差、欠測）を分けて保存した。
- `L=1.5`の小規模比較では同じ`G_b, chi`でも分類が変わり、サイズ変更で傾向が自動的に維持されるとは言えないことが分かった。

### 未回答・次段階

- 接触、有限径、線分反発、摩擦、接着、折りたたみ。
- 実験中心線とのfit、実験ノイズ、物理単位の較正。
- 三角格子モデルとの比較、普遍性、臨界曲線、臨界指数。
- さらに広い`G_b`・`chi`範囲での統計的境界推定。
- 境界近傍・高`G_b`の数値未解決を解消する高解像度・より安定な積分器の検討。ただし本作業では時間積分器・コア物理式を変更していない。

未解決域をresolved結果へ再分類せず、次の接触・実験比較作業とは別に扱う。
