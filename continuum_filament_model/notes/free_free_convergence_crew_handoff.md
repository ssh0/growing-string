# free/free 収束ゲート：crew 引き継ぎ

この文書は、free/free 収束ゲートを再実行・再開する crew 向けの作業契約である。汎用的なソフトウェア開発手順ではなく、このリポジトリの非接触 free/free・一様な参照長成長の収束監査に限る。

## 適用前提

この手順は、PR #29（free/free mechanics convergence gate）が取り込まれた current master を基準にする。handoff 文書だけを先に適用した古い HEAD では実行しない。再開対象の同じ HEAD に、次の gate 固有ファイルが存在することを確認する。いずれかが無ければ、PR #29 を含む current master から disposable worktree を作り直し、stage2 など別の実装・設定・テストで代用しない。

- `continuum_filament_model/notes/free_free_convergence_gate.md`
- `continuum_filament_model/benchmarks/free_free_convergence_gate.py`
- `continuum_filament_model/benchmarks/configs/free_free_convergence_gate.json`
- `continuum_filament_model/tests/test_free_free_convergence_gate.py`

## 最初に読む正本

次の順で確認する。数値結果のファイル名や過去の compact 値だけから条件を復元しない。

1. `continuum_filament_model/notes/model_spec.md` — 状態、エネルギー、成長、再メッシュ、時間積分。
2. `continuum_filament_model/notes/validation_plan.md` — 数値・力学検証の境界と受入条件。
3. `continuum_filament_model/notes/free_end_dynamics.md` — free/free 端点 force、moment、shear の符号と診断契約。
4. `continuum_filament_model/notes/free_free_convergence_gate.md` — このゲートの population、観測量、reason code、収束判定。
5. `continuum_filament_model/benchmarks/free_free_convergence_gate.py`、`continuum_filament_model/benchmarks/configs/free_free_convergence_gate.json`、`continuum_filament_model/tests/test_free_free_convergence_gate.py` — 実行契約の実装・設定・focused test。

## compact 成果物の広い契約

Git に残すのは、全節点軌跡ではなく、suite-level の `compact_summary.json`、`compact_manifest.json`、収束表と population 別の compact CSV/JSON である。compact は「見た目の要約」ではなく、同じ比較を再現し、失敗・未解決を隠さず監査できる最小成果物とする。少なくとも次を同じ schema で保持する。

- `schema_version`、benchmark scope、boundary、contact disabled、実行時の**現在 Git revision**。
- config の SHA-256、compact 各 artifact の bytes/SHA-256、run 数・population 数。各 run では input、initial state、canonical final state、event sequence、perturbation の hash も保持する。
- `growth_work_step` / `growth_work_cumulative` を正規名とする。これは同じ幾何で参照長だけを更新した離散 energy 差であり、連続体の成長仕事の完全な導出ではない。古い別名を混在させない。
- requested `dt` と実際の accepted `dt`（min/max/mean/値集合）、棄却試行数、reason 別の棄却数、event 数、event sequence hash、failure/reason code。
- total/stretch/bend energy の初期値・最終値・span、mechanical energy change、remesh energy jump。接触 energy はこのゲートでは常に無効であることも明記する。
- accepted Euler 区間の `dissipation_estimate`、`mechanical_balance_residual` の cumulative/max、および total/reference length。
- endpoint force residual、bending moment residual、shear-equivalent residual。過渡 run の端点残差を「常にゼロ」と解釈しない。
- onset の定義と時刻、形態 label、peak transverse amplitude、curvature RMS、mode spectrum/fractions、dominant mode。
- `run_kind` / `role` / `representative` / `refinement_axis`、seed、初期条件 perturbation の内容と hash、population label。
- `morphology_status` と `mechanics_status` を分離した refinement 判定、全体 `status`、`numerical_status`、`numerical_reason_codes`。形態が収束しても、work・energy・残差が未収束なら `numerically-unresolved` のままにする。

この provenance と数値監査を compact に含める理由は、後から current HEAD、設定、入力状態、実行イベント、成果物の対応を検証し、時間・空間 refinement の取り違えや「形態だけ収束」を検出するためである。per-run の `metrics.csv`、`events.json`、`manifest.json` は指定した一時出力の `_runs/<run_name>/` に置き、Git へコピーしない。

## population を混ぜない

- **deterministic fixture/refinement**：`straight`、`boundary-near`、`buckled-candidate`。`seed=null` の決定論的初期形状を使い、temporal と spatial refinement の母集団として判定する。
- **control**：成長なし等の control。deterministic refinement の分母へ混ぜず、solver の基準動作を確認する。
- **parameter contrast**：基準 representative から growth rate、EA、EI、drag density、または imperfection の一因子だけを変える。通常は単一解像度の探索的対照なので、時間・空間 refinement がなければ `numerically-unresolved` / `not_refined_across_time_or_space` と明記する。差を収束済みの物理差や臨界値と呼ばない。
- **shape-only initial-condition sensitivity**：seed、振幅、shape noise は初期形状だけを変える population として、deterministic と別集計する。各 member に perturbation descriptor/hash と seed を残す。確率的力学則や実験ノイズの母集団ではない。

shape-only を名乗るには、初期参照長（各 segment と総量）、EA、EI、drag density、成長則・時間条件を固定し、初期 total/stretch/bend energy を契約値として記録・検査する。初期 energy が変わる perturbation は shape-only として比較せず、別 population または `numerically-unresolved` とする。摂動振幅で参照長や材料量を変えない。各 member の perturbation hash と initial-state hash を compact に残す。

## 高コスト工程と出力

既定 config は deterministic 18 run（3 representative × temporal 3 + spatial 3）、control 1、parameter contrast 5、sensitivity 9、計 33 run である。`t_end` と三つの `dt` は config を正本とし、既定値では no-retry の場合でも概算で数万（約 4.4 万）accepted step と per-step の energy/event/observable 計算を行う。実行時間は環境依存なので固定値を主張せず、実測の elapsed time と出力 byte 数を記録する。これは smoke test ではない。

実行前に必ず新しい `/tmp/growing-string-free-free-gate.XXXXXX` を作る。`_runs` の metrics、events、manifests、summary は compact より大きくなり得る。軌跡、raw metrics、event log、動画、途中生成物は Git に入れない。既存 `results/` を上書きせず、compact の採用は provenance/hash を確認した後に行う。

## 比較を壊す二つの落とし穴

1. `t_end` が requested `dt` の全ての整数倍でない場合、solver の最後の試行は `min(requested_dt, t_end - time)` という effective dt になる。requested 値のラベルだけで refinement を比較せず、accepted dt の値集合と実際の終了時刻を確認する。共通の時間 horizon と有効刻みを設計できない run は比較不能または `numerically-unresolved` とする。
2. run 名は population・axis をまたいで一意でなければならない。重複名は `_runs/<run_name>` の上書き、別条件の取り違え、manifest/hash の混在を起こす。生成前に全 run 名を検査し、`<representative>__temporal_###`、`<representative>__spatial_n###`、`control__...`、`contrast__...`、`sensitivity__...` のように衝突しない名前を使う。古い一時ディレクトリを再利用しない。

## 再開時の最短手順

1. **隔離と status**：`pwd -P` と `git rev-parse --show-toplevel` が同じ disposable worktree を指すこと、`git status --short --branch` が想定どおりであることを確認する。上記4つの gate 固有ファイルが同じ HEAD に存在することも確認し、無ければ PR #29 を含む current master から作り直す。`no-mistakes axi status` で自分の branch の実行状態も確認する。
2. **docs**：上記の正本（特に gate note、config、free-end note）を読み、今回の scope が free/free・非接触のままか確認する。
3. **no-mistakes の診断**：`no-mistakes doctor`、続けて `no-mistakes axi status`。他 branch の実行を停止・再起動・横取りしない。
4. **focused tests**：`PYTHONPATH=continuum_filament_model/src python -m unittest continuum_filament_model.tests.test_free_free_convergence_gate continuum_filament_model.tests.test_free_end_dynamics -v`。
5. **full tests**：`PYTHONPATH=continuum_filament_model/src python -m unittest discover -s continuum_filament_model/tests -v`。
6. **bounded benchmark**：新しい一時ディレクトリへ、正本 config を指定して `free_free_convergence_gate.py` を実行する。短縮 config は schema・分類・失敗保持の smoke に限り、収束結論に使わない。
7. **current-HEAD compact regeneration**：clean な実行 HEAD を記録してから同じ suite を再生成する。`compact_summary` と `compact_manifest` の source/current revision、config hash、全 artifact hash、schema version、run 名・件数、population 分離を確認する。生成後にコード/config が変わったら再生成する。
8. **provenance/hash check**：accepted/requested dt、reject/event/reason、energy/work/dissipation/balance、端点 residual、onset/mode/curvature、perturbation hash、`numerically-unresolved` の保持を compact と `_runs` で突合する。spatial は temporal finest dt を使い、morphology-only の合格を mechanics 合格へ昇格しない。
9. **commit**：レビュー済みの compact だけを対象にし、`_runs` や既存成果物を含めない。実行 HEAD と成果物 commit の関係を provenance に残す。
10. **validation**：commit 後に no-mistakes を intent 付きで実行し、各 gate を判断する。`--yes` / `-y` は使わない。失敗・ask-user・unresolved の扱いを勝手に解消せず、根拠を残して再開する。

## 境界

- spatial refinement が未解決のまま残ることは想定内であり、解像度不足を物理的差と解釈しない。
- 接触・折りたたみ・摩擦・接着はこの gate の対象外である。必要なら別の検証段階として設計する。
- `gray5` は観測品質・lineage・校正・time registration の契約が別に必要であり、この gate の物理 validation ではない。gray5 の見た目や shape-only 比較から物性 fit、model adequacy、臨界値を主張しない。
