# C1 有限径接触・折りたたみ検証ユニット

## 目的と範囲

free/free 非接触成長座屈の収束作業に続き、摩擦なし・接着なし・接触履歴なしの有限径 segment penalty を C1 baseline として検証する。実装入口は `benchmarks/contact_folding_validation.py`、設定は `benchmarks/configs/contact_folding_baseline.json` である。

このユニットの主動的ケースは、初期には有限径接触していない free/free フィラメントを、成長・過渡緩和させて接触へ至らせる。初期 U 字接触は geometry/force control であり、実験の折りたたみを再現したものとは扱わない。centerline だけから径を推定しない。`img/gray5.mp4` などの確認できない、censor 済み、または接触 lineage/幅が検証できない観測は、input-QC と探索的な図示に限定する。

## C1 の定義

C1 は非隣接線分の最近接距離 `d` と排除径 `D` から

```text
penetration = max(0, D - d)
E_contact = 1/2 * k_c * penetration^2
```

を計算し、`d > 0` の法線反発力を最近接パラメータで4端点へ scatter する。`ModelParameters.enable_legacy_node_contact=False` を明示し、旧来の非隣接 node penalty と混ぜない。既存モデルの既定値は互換性のため変更せず、新ユニットの manifest と時系列には `contact_law=C1 segment penalty only`、`legacy_node_contact=disabled` を記録する。

摩擦、接着、履歴依存接触はこのユニットに追加しない。これらは C1 の観測失敗と、独立して品質確認された観測入力の両方がそろった後に、一つずつ nested alternative として同一初期条件へ適用する候補である。現段階では候補の provenance を記録するだけで、力学則は未実装とする。

## 記録する観測量

`metrics.csv` の各 accepted state は、次を compact に記録する。

- **接触 onset**：有限径 gap contact pair が初めて現れた accepted time/step。
- **active pair/feature set**：現在配列の segment index、remesh lineage ID、root lineage ID、feature、gap、penetration、normal、pair force を含む JSON レコード。
- **gap / penetration**：最近接 gap と最大 penetration、`penetration/D`。有限 penalty の貫入は時間刻み・接触剛性に依存するため、非貫入の証明にはしない。
- **contact force / work**：C1 pair force の大きさ、action-reaction residual、最近接点の相対変位に対する離散 contact work。これは摩擦散逸ではない。
- **contact residence**：同じ lineage pair が連続して active だった accepted interval の累積時間。再メッシュで exact pair が変わった区間は lineage reset として明示し、接触履歴を推定しない。
- **relative tangential slip**：前状態と現状態の最近接点相対変位から法線成分を除いた診断値。摩擦力は加えない。
- **detachment**：再メッシュをまたがない区間で、前状態にあり現状態にない pair。再メッシュ区間は detached と断定せず、lineage transition と記録する。
- **endpoint motion**：初期端点からの左右端点変位と最大値。free/free の運動を固定端 fixture と混同しない。
- **curvature concentration**：最大絶対符号付き曲率 / 平均絶対曲率。
- **fold spacing / count proxy**：符号付き曲率の符号反転数と局所ピーク間隔の中央値。短い・非周期的な形状では未定義になり得る。これは topology の fold count や実験の折りたたみ間隔そのものではない。

## lineage と再メッシュ

`FilamentState.segment_lineage` は中点分割時に `x -> x.0, x.1` として子へ継承する診断 ID である。現在配列の添字だけで接触履歴を追跡しない。root lineage pair は refinement 間の祖先対応を補助するが、物質同定・接触履歴・ヒステリシスを意味しない。再メッシュの前後では contact residence/slip/detachment を無理に接続せず、`lineage_reset_count` と `remesh_contact_transition` を保存する。

## 数値ガードと refinement

各ケースについて、要求 `dt` と受理 `accepted_dt`、棄却数、棄却理由を別々に保存する。swept centerline crossing guard は既存の試行棄却を使うが、有限径の連続時間 CCD や hard non-penetration solver は実装していない。接触 sequence が refinement で変わる、期待した接触が現れない、solver failure がある、受理 `dt` が要求値から大きく縮退する場合は `numerically-unresolved` とする。

config には次の3 refinement を含める。

1. **temporal**：同じ初期条件で `dt` を変え、contact onset、sequence、penetration、residence、accepted `dt` を比較する。
2. **spatial**：`n_nodes` を変え、同じ形状規則・物理条件で比較する。粗い解像度が接触を見逃した場合は未解決として残す。
3. **contact stiffness**：`k_c` を変え、penetration が減るか、sequence が変わらないかを診断する。高剛性化だけで収束・非貫入を断定しない。

比較の許容差は設定の `convergence_tolerance` に保存し、満たさない refinement は `numerically-unresolved` とする。`resolved` はこの compact observable と数値条件の範囲での判定であり、物理的妥当性・実験一致・普遍性を意味しない。

## 受入条件

- C1 ケースの manifest で legacy node contact、friction、adhesion、history が無効と確認できる。
- segment force の有限差分勾配、作用反作用、剛体変換、交差/CCD guard、初期 crossing 拒否が既存 test として維持される。
- dynamic primary case は non-contact から contact へ到達するか、到達しない場合は期待条件と理由を `numerically-unresolved` として残す。初期接触 fixtureだけで合格にしない。
- onset、active pair/feature、gap、penetration、normal、force/work、residence、slip、detachment、endpoint motion、curvature concentration、fold spacing/count proxy が compact output に現れる。
- 時間・空間・接触剛性 refinement の各結果と、requested/accepted `dt` の差が保存される。
- finite penalty の結果を hard non-penetration proof と報告しない。
- 実験の折りたたみ再現、径の同定、摩擦・接着の必要性は、このユニットだけから主張しない。

実行例：

```bash
PYTHONPATH="$PWD:$PWD/continuum_filament_model/src" \
python continuum_filament_model/benchmarks/contact_folding_validation.py \
  --config continuum_filament_model/benchmarks/configs/contact_folding_baseline.json \
  --output /tmp/growing-string-contact-folding
```

`/tmp` の per-run trajectory、動画、イベント全量は Git に追加しない。リポジトリへ残すのは、必要な場合の compact summary、設定、manifest、研究ノートだけとする。
