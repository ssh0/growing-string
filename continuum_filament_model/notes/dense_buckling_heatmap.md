# Dense buckling heatmap

`benchmarks/dense_buckling_heatmap.py` は、既存の非接触フィラメントsolverを変更せず、`G_b × chi` の決定論的な密度マップを測定する実験ハーネスである。各セルの初期形状は同じ正弦摂動で、乱数trialや実験ノイズを含めない。

## セルの測定とピーク状態

各 accepted state のモード係数を、固定端を結ぶ弦に垂直な変位 `y(s)` として、`n=1..6` の正弦基底へ投影する。ピーク状態は全軌道で `max(abs(transverse displacement))` が最大の accepted state とする。そこで次を保存する。

- `A_max_over_L`：ピーク最大横変位を設定長 `L` で割った値。
- `first_mode_fraction`：`|A_1| / sqrt(sum_n |A_n|^2)`。
- `mode_fractions`：上記の分母で正規化した `n=1..6` の絶対モード係数。
- `dominant_mode`：`|A_n|` が最大のモード次数。
- `curvature_rms`：既存の離散曲率のRMS。
- `onset_time`：`max(3|A_1(t0)|, 0.005L)` を初めて超えた accepted time。
- `dissipation_energy`：同一節点数のaccepted区間について `dt * sum_i Gamma_i |v_i|^2` を累積したEuler推定値。再メッシュ区間は節点対応を推測せず、その区間の増分を除外する。

## unresolvedの判定根拠

既存の機械的分類（`straight`、`buckled-single`、`unresolved`）と、形態の解体分類を分離する。`unresolved` は `onset` が検出され、第一モード支配が70%未満、または既存分類規則が単一座屈を確定できないセルである。形態分類はピーク状態に対して次の固定ルールを適用する。

1. `sub_threshold_transient`：ピーク横変位が onset 閾値以下。座屈形態を確定せず、単に `straight` と呼ぶ場合もこの理由を保持する。
2. `higher_mode_wave`：`dominant_mode >= 2` かつ `sqrt(sum_{n>=2} f_n^2) >= 0.30`。高次モードの波が第一モード分率を下げたと判定する。
3. `localized_buckling`：`max_curvature / curvature_rms >= 3.0`。曲率が局所集中した変形を別分類する。
4. `higher_mode_wave_and_localized_buckling`：2と3を同時に満たす場合。
5. `mixed_mode`：onset後に `first_mode_fraction < 0.70` だが、2と3のどちらも単独では満たさない場合。
6. `single_mode`：onset後に第一モード分率が70%以上で、上記の局所集中条件を満たさない場合。

この分類は数値データに対する定義済みの形態ラベルであり、物理的な相転移・臨界値の主張ではない。`heatmap.json` の `unresolved_decomposition` と各recordの `waveform_classification.detail` に、判定閾値とピーク値を併記する。

## 代表スナップショット

`snapshots.json` は、`straight`、`single_buckling`、`higher_mode`、`boundary_near` の役割ごとに、`t0/t_mid/t_end` の3状態だけを保存する。`boundary_near` は、可能な場合は `higher_mode` と異なるcell、かつ高次モード以外の `unresolved` cellを優先する。したがって、同一セルの重複によって異なる波形を隠さない。対象分類が軸範囲に存在しない場合だけ、決定的なfallbackを使用し、役割の実体をJSONに記録する。

## provenance

各実行の `source_revision` は、成果物生成時に `git rev-parse HEAD` で得た参照HEAD SHAである。生成物を含むcommit自身を自己参照する値ではない。動画の入力識別子は論理ID `img/gray5.mp4` とし、一時worktreeやローカル絶対パスをcompact成果物へ保存しない。
