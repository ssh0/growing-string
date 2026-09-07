# 有限径 segment contact geometry 契約

このノートは、有限径の線分接触を後続の penalty / barrier / constraint 方式へ渡す前の、方式非依存な幾何診断契約を記録する。ここで扱うのは中心線の幾何だけであり、接触力・接触エネルギー・摩擦・接着・solver は実装しない。

## 公開API

`growing_filament.geometry` の `segment_contact_geometry(a, b, c, d, diameter)` は、1組の非退化な2次元線分について `SegmentContactGeometry` を返す。`nonlocal_segment_contacts(positions, diameter)` は、開曲線の全非隣接線分対に同じ計算を行う。既存の `segment_closest_points()`、`nonlocal_segment_distances()`、`geometry_diagnostics()` は後方互換のため残す。

各レコードは次を持つ。

- `distance = d`: 中心線線分間の最近接距離
- `gap = d - D`: `D` は引数で与える排除径
- `penetration = max(0, D - d)`
- 最近接点 `point_i` / `point_j`（`closest_point_i` / `closest_point_j` でも参照可能）
- 最近接パラメータ `parameter_i` / `parameter_j`（各線分の `[0, 1]`）
- `diameter`
- `feature`: `endpoint_endpoint`、`endpoint_interior`、`interior_endpoint`、`interior_interior`、`collinear_overlap`、`parallel_overlap`
- `normal` と `normal_status`
- `centerline_intersection` と `diagnostic_type`

接触閾値は `gap <= 0` の包含条件である。閾値の許容幅を使う場合は、呼び出し側が明示的に別途判断する。現在の `geometry_diagnostics()` の従来キー `contact_pairs` は既存の許容誤差を含む診断として保持し、新契約の完全なレコードは `segment_contacts`、`finite_radius_contacts`、`centerline_intersections` で参照する。

## 特異な幾何

- `d > 0` では、法線は `point_j` から `point_i` へ向く単位ベクトルで、`normal_status=defined` とする。
- `d = 0` の中心線交差・collinear overlap では、法線方向を選ばず `normal=None`、`normal_status=undefined_zero_distance` とする。最近接点は診断を再現可能にするため決定的な代表値を返すが、力の作用点や法線の選択を意味しない。
- collinear overlap は最近接点が一意でないため `collinear_overlap` として扱い、重なり区間の中点を代表パラメータにする。
- 平行で投影区間が重なる場合も最近接点が一意でないため `parallel_overlap` とし、投影重なりの中点を代表値にする。ほぼ平行でも法線は距離が正の場合だけ返す。
- 中心線交差は `diagnostic_type=centerline_intersection`、交差しない有限径の gap 接触は `finite_radius_gap_contact` とする。中心線交差を有限径 gap 接触だけとして報告しない。

## 非隣接対と識別子の限界

`nonlocal_segment_contacts()` は既存契約と同じく、現在の `positions` 配列に対する0始まりの線分添字 `(i, j)` で `j >= i + 2` の対だけを返す。共有端点を持つ隣接線分は除外する。これは自己接触の物理的な判定や material identity の契約ではない。

添字はその配列スナップショットに限った一時的な識別子である。中点再メッシュでは節点・線分の数と添字が変わるため、再メッシュ前後で同じ対・同じ物質を追跡できない。現行実装には lineage ID、要素IDの継承、安定した追跡キーはない。将来、接触履歴やヒステリシスを追加する場合は、配列添字を流用せず、分割元を記録する明示的な lineage ID とそのシリアライズ範囲を設計する必要がある。大規模な配列や外部形式へ保存する場合は、Pythonの添字を無制限な永続IDとみなさず、形式の整数上限も確認する。

再メッシュ後は、現時点の配列に対して非隣接条件を再評価する。そのため、再メッシュ前に隣接していた線分由来の子線分が、配列上では非隣接対になることがある。この挙動は現行の配列添字契約によるものであり、lineage-aware な近接除外は未実装である。

## 対象外

この契約は静的な最近接幾何と診断イベントを提供するだけで、動的な接触力、penalty / barrier / constraint の選択、接触solver、摩擦、接着、折りたたみsolverを決めない。また、線形試行の `swept crossing` 検出は既存の中心線交差診断であり、有限径の連続時間衝突検出（CCD）や非貫入保証ではない。有限径の動的接触を先に実装しない理由は、法線が未定義な交差・重なり、方式ごとに異なる許容貫入、履歴とlineageの要件を、幾何契約と混ぜずに後続方式の比較対象として残すためである。
