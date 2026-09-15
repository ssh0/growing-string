# 有限径線分ペナルティ接触力

## 目的と適用範囲

`growing_filament.model.OverdampedGrowingFilament` に、中心線の非隣接線分同士が有限径 `D` の排除距離へ侵入したときの、摩擦なし・保守的な反発力を追加した。接触幾何は既存の
`growing_filament.geometry.nonlocal_segment_contacts()` を利用する。新しい依存関係、摩擦、接着、履歴変数、連続時間の非貫入保証（CCD）は導入していない。

これは最小の penalty prototype であり、有限径フィラメントの実験妥当性や大きな時間刻みでの非貫入を保証する完成 solver ではない。C1 の成長–接触–折りたたみ検証は `notes/contact_folding_validation.md` と `benchmarks/contact_folding_validation.py` に分離している。

## 定式化

線分 `i=(r_i,r_{i+1})` と線分 `j=(r_j,r_{j+1})` の最近接点を

```text
x(s) = (1-s) r_i + s r_{i+1}
y(u) = (1-u) r_j + u r_{j+1}
```

と書く。ここで `s,u ∈ [0,1]` は既存の最近接探索が返すパラメータ、`d = |x(s)-y(u)|` は中心線間距離、`D` は `ModelParameters.diameter` である。貫入量と接触エネルギーは

```text
δ = [D-d]_+ = max(0, D-d)
E_c,ij = 1/2 k_c δ^2
```

とする。`d > 0` では、法線を最近接点から

```text
n = (x(s)-y(u))/d
```

と定義する。`n` は線分 `j` から線分 `i` へ向くため、線分 `i` へ加える対力は

```text
f_ij = k_c δ n
```

であり、線分 `j` には `-f_ij` を加える。4端点への scatter は双線形形状関数

```text
F_i     += (1-s) f_ij
F_{i+1} += s     f_ij
F_j     -= (1-u) f_ij
F_{j+1} -= u     f_ij
```

で行う。このため各接触対の内力の総和はゼロであり、`d>0` の有効な非交差状態ではエネルギーの負の有限差分勾配と一致する。

2次元の平行線分の投影が重なる場合は、既存 geometry 契約の決定的な代表値（重なり区間の中点）を使う。`d=0` の交差・共線重なりでは法線が一意でないため、幾何診断と同じく任意の法線を捏造しない。エネルギーは記録するが、その対の力は加えない。この `d=0` 形状は力学的に無効であり、力とエネルギー勾配の有限差分一致の受入範囲外である。初期中心線交差は既存のモデル検証で拒否されるため、通常の動力学 fixture は `d>0` から開始する。

## 既存節点接触との互換性

`contact_stiffness > 0` かつ `diameter > 0` のとき、線分ペナルティを標準の非局所接触応答として計算する。既存の非隣接節点間ペナルティは、過去の3節点 fixture・パラメータ利用者との互換性のため `ModelParameters.enable_legacy_node_contact=True` の既定値で保持する。したがって、節点と線分の両方が同じ設定で重なる構成では、互換モードのエネルギーと力は両項から寄与する。C1 検証ではこのフラグを `False` にし、`contact_energy_components()` の `segment_c1` と `node_legacy`、manifestの provenance で別ラベルにする。新しい接触試験では、原則として node 項を混ぜず線分項を単独検証する。

接触エネルギーは `energy_components()["contact"]` に含まれ、`forces()` は伸長・曲げ・節点接触・線分接触を合算した `-dE/dr` を返す。過減衰ステップでは、この合算力を既存の参照長重み付き drag で速度へ変換する。

## 検証 fixture と境界条件

`tests/test_segment_contact_forces.py` の U 字 fixture は次の形状である。

```text
r = [(-0.5, 2.0), (-0.5, 0.0), (0.5, 0.0), (0.5, 1.0)]
D = 1.2,  k_c = 20
```

非隣接な線分 `(0,2)` の距離は `1.0`、貫入は `0.2`、最近接パラメータは `(s,u)=(0.75,0.5)` である。端点間の非隣接節点距離は `D` より大きいため、期待する線分力は

```text
[[-1, 0], [-3, 0], [2, 0], [2, 0]]
```

となる。次の境界・不変性をテストする。

- `d >= D` で線分エネルギー・線分力がゼロ（閾値上を含む）。
- `s,u` は `[0,1]` 内で、scatter が各端点へ分配される。
- 有限差分 `-∂E_c/∂r` と解析的 scatter 力が一致する。
- 内力の総和がゼロ（作用反作用）。
- 剛体平行移動でエネルギー・力が変わらず、剛体回転で力だけ同じ角度だけ回転する。
- 固定端 U 字を小さい `dt` で1ステップ進めると、線分距離が増え、貫入が減る。fixture ではステップ棄却なし、過剰な移動なしを確認する。

交差、共線重なり、最近接法線が未定義な `d=0` の分類は `test_segment_contact_geometry.py` で、丸め誤差による誤交差と真の交差・重なり・端点接触の回帰は `test_geometry_intersections.py` で検証する。これは交差状態を penalty force だけで解決する仕様ではない。

## 実行した検証

実装後に、プロジェクト指定のコマンドで既存テストと新規テストを一括実行する。

```bash
PYTHONPATH="$PWD:$PWD/continuum_filament_model/src" \
python -m unittest discover -s continuum_filament_model/tests -v
```

受入条件は、上記のテストスイートが成功することである。テストが成功しても、有限の `dt`、複数接触、再メッシュ、動的な交差回避について非貫入が証明されたことを意味しない。
