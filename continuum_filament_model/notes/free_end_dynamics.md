# Free/free 端点力学の境界条件と検証

## 目的と範囲

この文書は、`continuum_filament_model/src/growing_filament/model.py` の開曲線を
**free/free** 境界条件で検証するための規約を固定する。固定端は既存ベンチマークの
比較 control であり、実験再現の根拠ではない。この段階では接触を無効化し、摩擦、接着、
異方的ドラッグ、成長局在、接触ソルバーを追加しない。

主モデルの候補は、端点を固定しない開曲線、伸長、離散曲げ、等方的な基板ドラッグ、
一様な参照長成長である。有限径接触・折りたたみは後続段階で検証する。

## 物理・離散規約

状態は節点列 `r_0, ..., r_{N-1}` と参照長 `a_i` で表す。エネルギーと力は
`notes/model_spec.md` の式を正本とし、シミュレータの保存力は

```text
F_i = -dE/dr_i
```

とする。したがって `endpoint_diagnostics()` の `force_residual` は、端点に作用する
保存力そのものであり、自然な free 端点の受入条件は `force_residual = 0` である。
固定端では端点を初期位置へ戻す制約を適用し、`constraint_reaction` は
`-force_residual` とする。これは固定端を実験条件と解釈するための量ではない。

端点の向きは次のとおり定義する。

- material tangent: `t_0` または `t_{N-2}`（左から右向き）
- outward tangent: 左端では `-t_0`、右端では `t_{N-2}`
- outward normal: outward tangent を反時計回りに90度回転したもの
- `axial_force_residual`: `force_residual` の outward tangent 成分
- `shear_equivalent_residual`: `force_residual` の outward normal 成分

曲げエネルギーは内点の接線差分
`(EI/(2 h_i)) |t_i - t_{i-1}|^2` のみを持つため、端点の曲げ診断は
最初・最後の接線差分から得られる。`bending_moment` は、端点接線に関する
一般化保存力 `-dE/dtheta`（左端は `+(EI/h) Delta_t`、右端は
`-(EI/h) Delta_t`）の material tangent に垂直な符号付き成分である。
`bending_shear_equivalent` は曲げ力だけを outward normal に射影した値である。
これらは実装された離散エネルギーの自然境界残差を可視化する診断であり、新しい力や
境界拘束を追加しない。

端点の全力は伸長、曲げ、接触の和であり、`contact_enabled` が false の場合は
接触力がゼロであることを診断で確認する。全節点の力の合計と左端まわりのトルクも
剛体並進・回転不変性の監査用に記録する。

## 検証と受入条件

`tests/test_free_end_dynamics.py` が次を検証する。

1. 直線・無伸長・接触なしの free/free は、両端の力、曲げモーメント、shear-equivalent
   residual が丸め誤差内でゼロであり、時間発展で動かない。
2. 形状を平行移動しても端点残差は不変であり、剛体回転では力ベクトルが同じ回転を
   受け、ノルムと符号付き曲げモーメントが保存される。
3. 曲がった成長なしの free/free は、固定端を暗黙に復元せず、端点が移動しながら
   エネルギーを減少させる。同じ初期状態の fixed/fixed control は端点を動かさない。
4. 3節点の一様成長では、成長後の参照長に対する離散伸長力から予測される最初の
   free-end 変位と一致し、接触無効時の端点接触力はゼロである。
5. `benchmarks/free_end_benchmark.py` は free/free を主系列、fixed/fixed を control として、
   成長応答と曲げ緩和について時間・空間解像度、端点時系列、力・モーメント残差、
   成長 work、散逸 work、力学収支残差をコンパクトに出力する。全節点軌跡は保存しない。

この段階の数値報告は、実験動画との一致、座屈臨界値、有限径接触・折りたたみの妥当性を
主張しない。成長中は参照長更新がエネルギーを注入するため、全エネルギー単調減少は
受入条件にしない。成長なしの曲げ緩和では、受理されたステップのエネルギー非増加と
時間・空間 refinement の別々の報告を要求する。

## 再現コマンド

軽量な focused test:

```bash
PYTHONPATH=continuum_filament_model/src \
python -m unittest continuum_filament_model.tests.test_free_end_dynamics -v
```

bounded benchmark（`/tmp` の一時ディレクトリに compact summary のみを出力）:

```bash
TMP_DIR=$(mktemp -d /tmp/growing-string-free-end.XXXXXX)
PYTHONPATH=continuum_filament_model/src \
python continuum_filament_model/benchmarks/free_end_benchmark.py \
  --output "$TMP_DIR"
```

`summary.json` の `scope.contact_enabled` は false でなければならず、`records` は
`endpoint_trajectory` だけを含む。大きな node trajectory、動画、接触結果はこの段階の
Git成果物へ追加しない。
