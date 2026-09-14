# 長時間 free/free 反復折りたたみ検証（C1 staged unit）

## 目的

`benchmarks/repeated_folding_validation.py` は、PR #33 で導入した C1 baseline の上に、接触後の複数 episode を観測するための長時間 runner である。C1 の摩擦なし有限径 segment penalty だけを有効にし、摩擦・接着・接触履歴は有効化しない。初期接触 fixture は control として残すが、それだけで実験の折りたたみ再現を主張しない。

既定設定は `benchmarks/configs/repeated_folding_validation.json` で、主ケースは free/free・成長あり・`t_end=2.4` の長時間 run である。時間・空間・接触剛性の refinement を、反復が期待される deterministic population と非接触 control の non-repeating population に分ける。shape-only sensitivity は geometry/amplitudeだけを変え、deterministic baselineのrest lengths、EA、EI、drag、initial-energy定義を固定した別 population として出力し、refinement 判定へ混ぜない。

```bash
TMP_DIR=$(mktemp -d /tmp/growing-string-repeated-folding.XXXXXX)
PYTHONPATH="$PWD:$PWD/continuum_filament_model/src" \
python continuum_filament_model/benchmarks/repeated_folding_validation.py \
  --config continuum_filament_model/benchmarks/configs/repeated_folding_validation.json \
  --output "$TMP_DIR"
```

runner は軌跡配列・動画を保存しない。`summary.csv`、`metrics.csv`、`refinement_summary.*`、`compact_manifest.json`、`suite.json` の compact 出力だけを指定先へ書く。大容量の軌跡や動画が必要な探索は、Git 管理外の一時ディレクトリを呼び出し側で指定する。

## Episode 表現

各 accepted state の `metrics.csv` は、元の `time`、`step`、`n_nodes`、`accepted_dt`、`remeshed_since_previous`、segment lineage、接触 pair/feature、gap、penetration、接触力、endpoint motion、曲率 proxy、crossing 診断を保持する。Episode tracker は以下を区別する。

- `contact_onset`
- `active_continuation`
- `contact_detachment`
- `recontact`
- `feature_change`
- `pair_change`
- `remesh_boundary`

`summary.json` は集約値を保存し、`suite.json` の `episodes` には onset/detachment の raw 時刻・step・node 数、滞在時間、censor 状態、pair/feature、再メッシュ境界、penetration、tangential slip を保存する。再メッシュ前後は接触履歴として接続せず、`remesh_boundary` で分割する。lineage は分割祖先を追う診断 ID であり、物質 IDや履歴依存接触則ではない。同じ root-pair に子segment接触が複数ある場合も、current stateに同じpair/featureが残る接触だけを継続し、消えた子接触は離脱として閉じる。`evidence_classification` は単一episodeの `single-proxy/contact` と、2 episode以上かつ実接触離脱後の再接触を含む `repeated-folding` を分離する。`expected_repeated` のcaseで反復条件を満たさない場合は `repeated_folding_not_observed` として数値未解決にする。

同じ episode の accepted state 間で最近接点の相対変位を接線方向へ射影し、`relative_tangential_slip` を積算する。これは診断値であり、摩擦力や摩擦散逸を追加しない。finite penalty の `penetration` は有限で、剛性・時間刻み依存のため hard non-penetration の証明ではない。

## Fold proxy

- `fold_spacing_proxy`: 各 state の曲率局所ピーク間隔の中央値。
- `fold_period_proxy`: 曲率符号変化／peak proxy の増加時刻の間隔の中央値。
- `max_fold_count_proxy`: 曲率符号変化と局所ピークから得る proxy。
- `curvature_concentration`: 最大絶対曲率 / 平均絶対曲率。

これらは morphology proxy であり、実験の fold count、topological fold、実験周期を意味しない。短い・非周期的な系列では spacing/period は未定義のまま保存する。

## Refinement の判定

`refinement_summary.json` は、各 pair について次を個別に返す。

- `sequence_status`: episode 数、detach/recontact、feature/pair change、rest-length arc 上の正規化 contact identity/feature、episode順の censor pattern を構造 signature として比較し、contact identity tolerance と remesh boundary tolerance を適用する。segment ID の完全一致や raw timestamp の完全一致は要求しない。
- `penetration_status`: 最大 penetration ratio の相対差。
- `residence_status`: 最大 episode residence の相対差。
- `fold_status`: fold count、spacing、period proxy の差。

episode 時刻は `episode_time_tolerance`（既定 0.08）で比較し、`accepted_dt` の違いによる正当な時刻ずれを許容する。`remesh_boundary_tolerance` は discretization tolerance として明示保存する。いずれかが不一致、solver failure、初期接触違反、crossing guard などの場合、該当 metric と pair は `numerically-unresolved` と分類する。resolved はこの数値条件の範囲内の compact observable に対する判定であり、物理妥当性・実験一致・普遍性を意味しない。

`gray5` 等の候補線はこの runner の入力に使用しない。将来、入力が存在する場合も input-QC または探索比較の provenance に限定し、接触量の quantitative validation や物性 fitting へ流用しない。
