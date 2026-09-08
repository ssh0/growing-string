# 有限径フィラメントの接触座屈・折りたたみベンチマーク

## 目的と適用範囲

`benchmarks/contact_buckling_benchmark.py` は、有限径 `D` と線分ペナルティ接触剛性 `k_c` を持つ成長フィラメントについて、自己接触・二次的な折りたたみ・接触時のエネルギー収支を同じ条件で比較するための実験ハーネスである。力、成長、再メッシュ、交差拒否は `growing_filament.model.OverdampedGrowingFilament` に委譲し、ベンチマーク側で物理則を複製しない。

標準 fixture は、両端の位置を固定した半円状の U 字形である。S 字形は `initial_shape: "s"` で選択できる。これらは代表的な検証用初期条件であり、実験形状の同定結果ではない。設定済みの標準25ケースは、初期U字接触と penalty 感度を中心にした診断スイートであり、動的な長時間 coiling、複数巻きの形成、相転移の実証ではない。出力は相図の候補と数値感度を整理するためのもので、臨界曲線、普遍性、実験再現性を主張するものではない。

## 数理仕様と無次元軸

モデルの有限径接触は、非隣接線分 `i,j` の最近接距離 `d_ij` に対して

```text
δ_ij = max(0, D - d_ij)
E_c,ij = 1/2 k_c δ_ij^2
```

で定義される。`d_ij > 0` のときの法線反発力と端点への双線形 scatter は `notes/segment_penalty_contact.md` の定義に従う。中心線交差では法線を任意に選ばず、モデルのステップ受理条件で交差を拒否する。

ベンチマークが記録する主な軸は次のとおりである。

```text
chi = EI / (EA L^2)
Pi_c = k_c D^2 / EI
G_b = g * zeta L^4 / (EI pi^4)
G_s = g * zeta L^2 / EA
```

ここで `EA` は `axial_stiffness`、`EI` は `bending_stiffness`、`zeta` は `drag_density`、`L` は `length`、`g` は `growth_rate` である。`Pi_c` は接触剛性を曲げ剛性と径で無次元化した軸であり、penalty 法の剛性依存性を比較するための指標である。`D/L`、`dt/tau_b` もマニフェストと集計 CSV に保存する。

P1B との出力互換性のため、無次元量のJSONには `G_b`、`G_s`、`chi` をそのまま保存し、説明的な別名として `growth_number_G_b`、`growth_number_G_s`、`bending_to_stretching` 相当の `chi` を扱う。summary CSV/JSON の `G_b`、`G_s`、`chi` は P1B と同じ定義であり、P2固有の `Pi_c`、`contact_stiffness`、`diameter` と併記する。したがって、P1Bの非接触結果とP2の有限径結果を、同じ `G_b`–`chi` 軸で比較できるが、接触状態を無視した同一相図とは解釈しない。

## 計測量の定義

各 accepted state について、次の時系列を `metrics.csv` に保存する。

- `buckling_time`：最大横変位が初期値の3倍、かつ `buckling_threshold_fraction * L` 以上になった最初の accepted time。初期 fixture の形状が既に大きく変形している場合、未検出になる。座屈の物理的臨界時刻を意味しない。
- `contact_time`：`d_ij <= D` を満たす非隣接線分対が初めて現れた accepted time。初期 fixture が有限径接触を含む場合は `0` になる。
- `active_contact_pairs`：`d_ij <= D` の非隣接線分対数。閾値上を含む。
- `contact_length`：有限径 gap contact の各対について、対応する2線分の幾何長の小さい方を足し上げた active-pair support estimator。最近接点1個だけから連続接触区間を推定しないための保守的な集計であり、線積分による真の接触長ではない。異なる対が同一線分を共有する場合は重複計上される。
- `max_penetration`：`max(0, D - d_ij)` の時系列最大値。
- `penetration_ratio`：`max_penetration / D`。`D=0` では `0` とする。
- `radius_of_gyration`：節点平均の従来指標。`arc_length_weighted_radius_of_gyration` は再メッシュの影響を抑えた補助指標である。
- `max_curvature`、`rms_curvature`：既存の離散曲率と、符号付き曲率から算出した RMS。
- `fold_count`：符号付き曲率の非零符号列における符号反転数。折りたたみの候補数であり、自己交差を許すループ数ではない。
- `self_loop_count`：active finite-radius contact pair graph の連結成分数。接触で閉じ込められた折りたたみ領域の候補を数えるトポロジー proxy であり、幾何学的な単純閉曲線の検出ではない。
- `fold_wavelength`、`fold_periodicity`：符号付き曲率を弧長方向に再サンプルし、FFT の最大非零波数から求める代表波長 `L_contour / mode` と、その振幅の非零スペクトル総和に対する比。短い fixture、非周期形状、ほぼ一定曲率では未定義または低値になり得る。

エネルギーはモデルの `energy_components()` を使い、`energy_stretch`、`energy_bend`、`energy_contact`、`energy_total` を保存する。エネルギーの各項は接触ノード項との互換加算を含むため、接触力だけの寄与を意味しない。

## 仕事と散逸の診断

同じ節点数の accepted state 間では、受理 Euler 軌道から

```text
W_diss,n = dt * sum_i Gamma_i |v_i|^2
v_i = (r_i^(n+1) - r_i^n) / dt
```

を計算する。`Gamma_i` はモデルと同じ参照長重み付き drag である。成長仕事は、離散的なエネルギー収支

```text
W_growth,n = (E^(n+1) - E^n) + W_diss,n
```

で定義する。これは連続体の厳密な成長仕事の導出ではなく、成長による参照長更新、離散積分、接触、丸め誤差を含む診断値である。

中点再メッシュにより節点数が変わる区間では、節点の対応を推測しない。その区間の `dissipation_work_step` は0とし、`remesh_interval=true` を記録する。したがって、累積仕事は再メッシュ区間で完全な連続軌道の仕事積分ではない。

## 制御ケースと相図

設定は `benchmarks/configs/p2_contact_buckling.json` に置く。`cases` の代表・control・剛性比較・折りたたみ比較に加えて、`phase_grid` の `growth_rate`、`contact_stiffness`、`diameter`、`bending_stiffness` の直積を展開する。`bending_stiffness` と `axial_stiffness`、`length` の組から `chi`、`contact_stiffness` と `diameter` から `Pi_c` が再計算されるため、設定値だけを相図座標とみなさない。

`convergence.cases` では `dt` と `k_c` を変えた再計算を登録する。`convergence_summary.csv/json` には、`dt` の細分化を相対貫入差と分類一致で評価する行、`k_c` の増加を貫入非増加で評価する行を出力する。これは判定を自動化するための基準であり、`converged=false` は失敗ではなく未収束または計測不足を意味する。評価では、次を別々に確認する。

1. `dt` を細分化したとき、`max_penetration / D` が低下または安定し、分類と接触時系列が極端に変化しないか。
2. `k_c` を増加したとき、有限径 penalty の貫入が小さくなるか。ただし、硬い接触は明示 Euler の安定条件と step rejection に強く依存する。
3. 同じ設定・同じ Git/Python/NumPy 環境で、ケースの state hash、manifest、summary が再現するか。

## 出力と再現性

CLI は次の compact 成果物だけを出力する。

```text
summary.csv              # ケースごとの相図・最大値・onset・仕事
summary.json             # 同内容のJSON
convergence_summary.csv   # dt / k_c refinement の比較と判定
convergence_summary.json  # 同内容のJSON
metrics.csv              # 主要時系列（節点座標・NPZは含まない）
phase_map.png            # Penetration / contact-pair の概観
compact_manifest.json    # config hash、revision、ケース別state hash
suite.json               # 実行結果と限界の要約
```

全ステップの座標 NPZ、動画、per-run の大容量イベント列は保存しない。個別の完全な `OverdampedGrowingFilament` event log はベンチマーク内部で manifest hash の生成に使うが、compact manifest には棄却イベントの要約だけを残す。

`compact_manifest.json` と `suite.json` の `git_revision` は、成果物自身を含むコミットのSHAではなく、成果物を生成した実行時点で参照したコードHEAD（`source_revision` / `parent_revision`）を記録する。成果物を含むコミットのSHAは、その成果物を含めてコミットを作成するまで確定しないため、manifestへ自己参照として記録しない。この来歴を前提に、同じコードrevision・設定hash・Python/NumPy環境で再実行し、ケース別state hashと集計結果を比較する。

再実行例：

```bash
PYTHONPATH="$PWD:$PWD/continuum_filament_model/src" \
python continuum_filament_model/benchmarks/contact_buckling_benchmark.py \
  --config continuum_filament_model/benchmarks/configs/p2_contact_buckling.json \
  --output continuum_filament_model/results/contact_buckling
```

## penalty 法固有の限界

- `δ` は0にならず、有限 `k_c` と有限 `dt` では貫入が残る。`D` を厳密な非貫入境界として解釈しない。
- `Pi_c` が同じでも、時間刻み、軸伸長、曲げ、drag、空間解像度、初期形状が異なれば軌道は一致しない。
- 交差状態では最近接法線が未定義であり、接触力で交差を解消する仕様ではない。モデルは初期交差と試行中の swept crossing を拒否する。
- 非隣接線分の penalty と既存の非隣接節点接触項は同じ `contact_stiffness` で加算される。`energy_contact` は線分項単独ではない。
- 摩擦、接着、履歴、摩擦散逸、接触の粘性、有限要素の拘束解法、連続時間の CCD は含まれない。
- `contact_length`、`fold_count`、`self_loop_count`、FFT 周期性は観測量の proxy であり、実験画像から直接得る物理量や厳密な幾何分類ではない。
- 標準25ケースは初期U字接触および penalty 感度の診断であり、動的な長時間 coiling、複数巻き、相転移を実証しない。長時間計算を追加する場合は、出力サイズ、交差拒否、再メッシュ区間、`dt` 収束を別途監査する。
