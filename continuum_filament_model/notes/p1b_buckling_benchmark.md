# P1B：成長–緩和競合と座屈ベンチマーク

## 目的と範囲

P1Bでは、成長率、伸長剛性 `EA`、曲げ剛性 `EI`、基板ドラッグ密度 `zeta` の競合を、まず接触を切った小規模な決定論的ベンチマークで調べる。問いは次の3点である。

1. 曲げ緩和時間で無次元化した成長率で、直線維持から単一モード座屈への条件を整理できるか。
2. 座屈の開始時刻と支配モードは、`EI`、初期摂動振幅、解像度、`dt` に対してどの程度頑健か。
3. 既存APIの再メッシュ、時間積分、イベント記録、manifestを用いて、同じ設定を再実行できるか。

ここでの分類は機械的なregime mapであり、「相転移」や普遍性を主張するものではない。実験データ、大規模sweep、三角格子比較、接触・折りたたみはこの段階の対象外である。

## Fixtureと非接触条件

`benchmarks/buckling_benchmark.py` は `src/growing_filament` の `OverdampedGrowingFilament`、`ModelParameters`、既存の観測量、イベント、manifest APIを呼び出す。エネルギー、力、成長、再メッシュの式をbenchmark側で複製しない。

初期形状は、端点間距離を `L` として、等間隔の節点に

```text
y(x) = A sin(pi x / L)
```

を与える。`rest_lengths` はこの初期折れ線の幾何学的線分長とし、初期の伸長エネルギーを除く。両端の位置を固定し、`A > 0` の決定論的摂動を与えるのは、完全な直線が対称性を保ったまま座屈しないためである。これは座屈の向きを選ぶ物理的ノイズではなく、再現可能なモード励起である。

全ケースで次を強制する。

- `contact_stiffness = 0`
- `diameter = 0`
- 両端位置固定
- `reject_crossing = true`
- 初期形状は既存ライブラリの初期非交差診断を通過

したがって、出力に接触診断が現れないことは、線分接触力が正しいことを意味しない。線分接触、反発、接着、摩擦、ヒステリシス、折りたたみは次段階で別途設計する。

## 無次元量

`length` は初期の端点間距離を代表長 `L` とする。第一正弦モードの線形曲げ緩和を基準に、次を保存する。

```text
tau_b = zeta L^4 / (EI pi^4)
tau_s = zeta L^2 / EA
G_b   = growth_rate * tau_b
G_s   = growth_rate * tau_s
chi   = EI / (EA L^2)
dt/tau_b
(dx/L)
```

`G_b` は「成長が曲げ緩和より速いか」を見る主軸であり、`G_s` は伸長緩和との競合を記録する。`tau_b` は第一正弦モードの定義であり、クランプされた接線条件の固有値や実験で同定された時間ではない。`diameter/L=0` も明示的に保存する。

## 評価指標と分類

時系列CSVには、次を保存する。

- 最大横変位
- 第一モード振幅と第一モード分率、支配モード
- 最大・RMS離散曲率
- 端点間距離、参照長、輪郭長
- 既存の節点平均半径・弧長重み付き慣性半径
- 伸長、曲げ、接触、合計エネルギー
- 受理・棄却数、棄却理由
- 固定端力の診断値

分類の規則は実装に固定する。最大横変位が初期第一モード振幅の3倍（または `0.005 L`）を超えない場合を `straight`、超えて第一モードが支配的（支配モード1、第一モード分率 `>= 0.70`）な場合を `buckled-single` とする。それ以外、失敗、非有限値、交差棄却で終了したケースは `unresolved` とする。座屈開始時刻はこの閾値を最初に超えた受理状態の時刻である。閾値は物理的な臨界値ではなく、このベンチマークの機械的な比較規則である。

固定端反力について、現行ライブラリには独立した反力APIがない。そのため出力する値は、現在の状態で `forces()` が返す拘束前の端点力の負値であり、拘束反力の診断proxyに限定する。新しい物理量や反力則を追加したものではない。

## 実行方法

リポジトリルートで、設定ファイルを使って実行する。

```bash
PYTHONPATH=continuum_filament_model/src \
python continuum_filament_model/benchmarks/buckling_benchmark.py \
  --config continuum_filament_model/benchmarks/configs/p1b_noncontact.json \
  --output /tmp/growing-string-p1b
```

大規模sweepではなく、デフォルト設定は成長なし校正、遅い成長、速い成長、`EI`変更、摂動振幅変更の小規模ケースと、`dt`半減・空間解像度変更・振幅変更の感度ケースだけを含む。単一ケースは次で実行できる。

```bash
PYTHONPATH=continuum_filament_model/src \
python continuum_filament_model/benchmarks/buckling_benchmark.py \
  --config continuum_filament_model/benchmarks/configs/p1b_noncontact.json \
  --case fast_growth --no-sensitivity \
  --output /tmp/growing-string-p1b-fast
```

`--growth-rate`、`--bending-stiffness`、`--dt`、`--amplitude`、`--n-nodes` はCLIからも上書きできる。各ケースの実効設定、`dimensionless.json`、`metrics.csv`、`events.json`、`manifest.json`、`trajectory.npz`、`summary.json`、`overview.png`を保存し、ルートにはケース比較の `summary.csv` と `suite.json`を保存する。manifestにはGit revision、Python/NumPy、入力hash、初期・終状態hash、イベント列、受理・棄却数を含める。

同一設定を別の出力先へ2回実行し、`config.json`、`manifest.json`、`metrics.csv`、`summary.json`を比較する。数値環境が異なる場合はPython/NumPyとGit revisionを比較し、完全一致を仮定しない。

## 結果の読み方と未解決事項

`G_b`が大きいケースで横変位が増えても、それだけで力学的な相転移とは言わない。`dt`、空間解像度、振幅の感度、イベントの棄却理由、初期非交差条件を併記する。`unresolved` は「現行の判定規則で単一モードと確定できない」ことを表し、失敗を座屈の証拠として扱わない。

未実装・未解決の事項は次のとおりである。

- 線分間の接触反発、接着、摩擦、接触後の折りたたみ
- 有限径の物理則と径の実験較正
- 端点反力の独立API
- 実験中心線データとのfit・予測評価
- 三角格子モデルとの写像
- `G_b`をまたぐ統計的な再現性、系サイズ依存性、普遍性
- 「相転移」や臨界指数の主張

このベンチマークの出力は、次の接触・実験比較段階へ進む前の、数値条件と機械的regime分類を追跡するための資料である。
