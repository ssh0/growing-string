# 実観察中心線のパラメータ同定・検証

## 目的と位置づけ

`benchmarks/video_parameter_fitting.py` は、PR #16 の動画抽出契約と、PR #12 の `synthetic_data_recovery.py` が定義した回収量・欠損処理の境界を、実観察中心線時系列へ接続する観測側ハーネスである。solverの方程式や接触則は変更しない。

このハーネスが扱う量は次のとおり。

- **成長率 `g`**：輪郭長 `L(t)` の指数モデル `log L = log L0 + g t` と、線形モデル `L = L0 + v t` を、同じ適格フレーム集合へロバストHuber回帰する。残差の正規化RMSEでモデルを選び、選択モデルの `g` と95%区間、フレーム残差を保存する。線形モデルを選んだ場合は `g = v/L0` とする。
- **形状適合度**：明示的な pixel/model-unit 登録とモデル中心線軌道がある場合だけ、最近傍時刻で対応づける。端点向きを比較した上で、弧長再サンプリングした離散Fréchet距離、曲率二乗誤差、端点距離を計算する。複数軌道を与えた場合は、`median_i(Frechet_i / L_i) + median_i(curvature_RMSE_i * L_i)` の決定的な正規化損失で候補を選ぶ。
- **`chi`**：定義は `EI/(EA L^2)`。`L` は単一セグメントの `reference_length` ではなく、`.npz` 軌道の初期状態における `sum(rest_lengths)`、またはmetadataに明示された `initial_length` / `length` を使う。代表長が得られない場合は未同定とする。単一の観察中心線から自由に推定する量ではないため、選択されたパラメータ付きモデル軌道（`.npz` の `metadata_json.parameters`、またはJSON sidecar）に条件付けた値として保存する。
- **実効太さ `D`**：入力中心線に `width` / `diameter` / `thickness` 列がある場合のみ、適格フレームの中央値を `diameter_proxy` として保存する。モデルdiameterとの差分は、`width_unit=pixel|model|physical` と、対応する `pixel_per_model_unit` または `model_unit_to_width_unit` が明示され、同一単位へ変換できる場合だけ計算する。PR #16 の現行 `centerline.csv` は幅列を契約していないため、幅測定系を追加しない限り未同定である。

## 入力と欠損・打ち切り

```bash
PYTHONPATH=continuum_filament_model/src \
python continuum_filament_model/benchmarks/video_parameter_fitting.py \
  --manifest continuum_filament_model/results/video_comparison/gray5_manifest.json \
  --manifest continuum_filament_model/results/video_comparison/original_manifest.json \
  --data-root /path/to/uncommitted/video-extraction-output \
  --model /path/to/trajectory.npz \
  --pixel-per-model-unit 12 \
  --output continuum_filament_model/results/video_parameter_fitting
```

`--data-root` は、manifest内の `full_period_run.artifacts.centerline.path` 等を解決するための明示的なローカルディレクトリである。manifestは動画・中心線本体を含まないため、探索的なファイル検索やハッシュ不一致の黙示的な許容は行わない。宣言されたcenterline、summary、lineage、events、metadataの各artifactは、存在する場合にbytes/SHA-256を照合し、ミスマッチまたはcenterline解決後の宣言companion欠落を`integrity_mismatch`として拒否する。入力が`centerline.csv`そのものの場合は`--centerline`を使用できる。

母集団は summary/lineage/centerline と、抽出metadataのprocessed frame rangeの和集合から作る。選択lineageがまだ存在しない先頭フレームは`unknown`欠損行として保持し、`eligible=false`で分母・除外理由へ含める。次を `eligible=false` とし、推定の分母から除外する。

- `censor=1`、欠損中心線、点数不足、品質閾値未満
- `branched_component`、`loop_component`、`out_of_view`、`roi_clipped`
- `large_jump`、`new_lineage`、`reconnected_after_missing`
- `ambiguous_components`、`components_truncated`、`skeleton_loss` 等のPR #16品質フラグ

除外理由、母集団数、適格数、打ち切り数は `compact_summary.json` に保存する。欠損・分岐・ループを観測できなかった通常フレームとして補完しない。

## 出力

`results/video_parameter_fitting/` のGit管理対象は次の小容量ファイルである。

- `compact_summary.json`：入力ごとの成長率、95%区間、D proxy、形状候補、分母、制約、未同定理由
- `fit_summary.csv`：入力ごとの主要推定値・分母・候補選択のコンパクト表
- `<logical_id>_frame_fits.csv`：フレームごとの適格性、観測長、成長残差、Fréchet距離、曲率MSE、除外理由
- `reproducibility_manifest.json`：入力・モデル・設定・結果のSHA-256とバイト数、解析revision

per-frame画像、mask、動画、NPZ等はこのハーネスから生成・保存しない。結果JSONは時刻を含めず、同じ入力・設定・実行コードでは同じ値になるようにしている。

## 実観察データに対する現時点の結果

リポジトリにある `gray5_manifest.json` と `original_manifest.json` は、PR #16 の抽出出力のハッシュ、動画metadata、処理フレーム数を保持するが、ライセンス未確認の動画および抽出 `centerline.csv` 本体をコミットしていない。作業時点の worktreeにも、そのハッシュに一致する抽出中心線ファイルは存在しなかった。そのため、実行済みのcompact summaryは両入力を `centerline_not_found`、`g=未同定`、`D=未同定`、形状適合度=`not_requested` と記録している。これは実データからの推定値ではない。抽出ディレクトリを `--data-root` で与えれば同じコードで再実行できる。

## 解釈上の限界

- 回帰の95%区間は、Huber IRLSの重み付き線形化共分散と`1.96*SE`（線形モデルの`g`はdelta法）による、宣言した回帰残差モデルだけを反映する。動画抽出誤差、pixel scale、独立試料間ばらつき、欠損機構の不確実性を含まない。
- 指数／線形の選択はAICを異なる応答変換間で比較せず、`normalized_rmse`として、指数の`log(L)` RMSEと線形の`L` RMSEを中央値長で正規化した値を比較する。選択基準と両スコアは結果JSONに保存する。
- 時間隣接フレームを独立replicateとして扱っていない。実験的な信頼区間には、run/filament単位の独立性と校正の階層を追加する必要がある。
- `g` の推定は輪郭長の記述的傾きであり、局所成長、投影誤差、視野外による見かけの長さ変化を分離しない。
- `chi` の数値は初期総参照長を使って候補モデルのパラメータから算出した条件付き値であり、単一動画からの真の剛性比の証明ではない。`EA` と `EI` を中心線だけから個別に同定しない。
- `diameter_proxy` は幅フィールドの測定 proxy であり、モデルdiameterとの差分は単位変換を検証できた場合だけ出力する。有限径penalty、摩擦、接着、接触反力を同定したことを意味しない。
- Fréchet距離・曲率MSEが小さいことは、単一生物フィラメントの追跡成功、モデルの物理妥当性、実験の予測妥当性を意味しない。独立calibration/holdoutと、力・軸方向応答等の追加観測が必要である。

## 検証

新規テストは `tests/test_video_parameter_fitting.py` にあり、次を固定fixtureで確認する。

- 指数成長率の決定性、回収値、95%区間、残差
- 分岐フラグ・censorフレームの母集団除外
- 幅proxyと幅列なしの未同定状態
- 明示的登録を使ったFréchet距離・曲率MSE、および未校正時の抑制
- パラメータ付き軌道からの条件付き `chi` / `D` 比較
- manifestのdata-root解決とハッシュ照合
- compact summary / 再現マニフェストの保存

合成回収の既存契約と同様、ここでのfixture成功を実観察の物性確定へ読み替えない。
