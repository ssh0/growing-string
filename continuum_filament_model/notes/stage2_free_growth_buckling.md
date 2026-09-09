# Stage 2：free/free 成長誘起座屈の探索

## 目的と範囲

Stage 2 は、マージ済みの free/free 端点診断を使い、非接触のまま

- 一様な参照長成長
- 伸長剛性と曲げ剛性の比
- 初期非対称を含む初期 imperfection
- 時間刻みと空間解像度
- 基板 drag

の因果関係を探索する bounded experiment である。入口は
`benchmarks/free_growth_buckling_stage2.py`、既定設定は
`benchmarks/configs/stage2_free_free.json` である。既存の solver 式や legacy
model directory は変更しない。

この段階では `contact_stiffness=0`、`diameter=0`、free/free とし、接触、摩擦、
接着、折りたたみ、局所成長、パラメータ同定は行わない。`buckling-candidate` は
定義した観測閾値を超えた形態ラベルであり、臨界値・相境界・相転移を意味しない。

## 再現コマンド

リポジトリルートで実行する。全 run の節点軌跡、イベント、動画中間生成物は指定した
出力ディレクトリに置かれるため、通常は `/tmp` を使う。

```bash
TMP_DIR=$(mktemp -d /tmp/growing-string-stage2.XXXXXX)
PYTHONPATH=continuum_filament_model/src \
python continuum_filament_model/benchmarks/free_growth_buckling_stage2.py \
  --config continuum_filament_model/benchmarks/configs/stage2_free_free.json \
  --output "$TMP_DIR"
```

既定デザインは、決定論的 fixture、`n_nodes` と `dt` の refinement、seed付きの
exploratory replicate を別々に実行する。`compact_summary.json` には endpoint
trajectory と sampled time-series が含まれ、`summary.csv` は比較しやすい1行/実験で
ある。`compact_manifest.json` は config hash、source revision、各 run の provenance
を保持する。deterministic fixture/refinement と seeded replicate を同じ統計へ混ぜない。

## 記録する観測量

各 run は、少なくとも以下を記録する。

- endpoint の位置と endpoint distance、輪郭長、参照長
- 線分の `EA*(l-a)/a` による signed axial-force proxy と圧縮 proxy
- free endpoint force residual、曲げ moment residual、shear-equivalent residual
- `max_transverse_amplitude`、曲率 RMS、1--6 mode の分率と dominant mode
- stretch/bend/contact energy、成長 work、Euler trajectory による dissipation 推定、力学収支残差
- accepted/rejected `dt`、棄却理由、event sequence hash、接触イベント数
- Git revision、Python/NumPy、入力・初期・終状態 hash、seed/trial lineage

成長 work は同じ幾何で参照長だけを1ステップ成長させた離散 energy 差であり、完全な
連続体成長仕事の導出ではない。dissipation は accepted Euler trajectory の診断推定値で
ある。端点残差は過渡状態ではゼロである必要はなく、free/free の境界診断として保存する。

座屈 onset は、`max_transverse_amplitude > max(3*initial_amplitude, 0.005*length)`
を初めて満たす accepted observation と定義する。ノイズ付き replicate の結果は、
fixture と独立の exploratory distribution として報告し、この定義から相境界を外挿しない。

## 実動画との比較

実入力が存在する場合は、既存の `growing_filament.video_comparison` を読み取り専用で
呼び出す。たとえばローカルの gray5 は次のように実行する。

```bash
TMP_DIR=$(mktemp -d /tmp/growing-string-stage2-gray5.XXXXXX)
PYTHONPATH=continuum_filament_model/src \
python continuum_filament_model/benchmarks/free_growth_buckling_stage2.py \
  --config continuum_filament_model/benchmarks/configs/stage2_free_free.json \
  --output "$TMP_DIR" \
  --video /Users/fujimotoshotaro/Workspace/growing-string/img/gray5.mp4
```

動画本体、raw centerline、comparison CSV、trajectory は `_video_artifacts/` と `_runs/`
に出力され、Gitへ追加しない。`video_comparison_manifest.json` は入力 hash、ffprobe
metadata、抽出候補数、lineage/censor、calibration/holdout の状態を compact に記録する。
入力が存在しない・読めない・中心線契約を満たさない場合は、`input_missing` または
`unusable` とし、legacy triangular-lattice videoで代用しない。

pixel per model unit と time registration は推測しない。登録値が無い場合は、動画が
支持する pixel の `L(t)`、endpoint distance、curvature、frame coverage を観測 QC として
保持するが、model/pixel の定量 overlay と parameter fitting は抑制する。登録値があっても
品質 censor、new/reconnected lineage、missing、branch/loop は比較母集団から除外し、
calibration と holdout の records を混同しない。入力中心線が censored のとき、見た目の
overlayからモデル不足・入力品質・数値未収束を区別せずに結論を出してはならない。

## 結果の解釈

- `input_quality` / `censor`：動画抽出、lineage、ROI、branch、missing、品質フラグの問題。
- `model_inadequacy`：非接触 free/free の仮定で観測差が残る場合。接触が必要だと推測して
  接触 solver をこの Stage 2 に追加せず、観測差を証拠として follow-up に残す。
- `numerical_nonconvergence`：accepted/rejected dt、refinement、event、run failure の問題。
- fixture と replicate の形態差：seed付き初期 imperfection の分布として保存し、実験ノイズ
  の同定や相境界の証拠とはしない。

## 実行済み探索の要約

`results/stage2_free_free/` は、source revision
`187434c4e18a21f3d3bdfc33fb133b7f25570c37`、`t_end=4.0` の18 run（決定論的 fixture・
refinement 9 run、seed付き replicate 9 run）を compact に保存する。全 run は
`contact_enabled=false`、失敗0、拒否0であった。決定論的な fast-growth/low-bend 条件は
`t≈3.30` に定義した onset を通過し、`n_nodes=9,13` と `dt=0.002,0.001` の4 refinement
でも onset は `3.298--3.316`、peak transverse amplitude は約 `0.0876--0.0894` であった。
slow-growth/low-bend は閾値未満で、high-bend は同じ fast growth でも閾値未満だった。
fast-growth/low-bend replicate は3本中2本が candidate、1本が閾値未満であり、これは
初期 imperfection の探索的な感度としてのみ扱う。したがって、この結果は「free/freeで
成長・drag・伸長・曲げの競合により非接触の座屈候補が現れ得る」ことの bounded evidence
であり、相境界・臨界成長率・実験パラメータ同定ではない。

`gray5.mp4` の full-period・frame stride 15 抽出は24 sampled frames、20 candidate、
selected lineageを含む2 lineage、candidate censor 20で、centerline contract自体はvalid
だった。一方、pixel/model scale・time registrationは未指定で、比較母集団24行の定量metric
eligibleは0行となった。従って動画側の `L(t)`、endpoint distance、curvature候補、lineage/
censorはQC用に保存したが、モデル overlay、onset matching、parameter fittingは抑制した。
これは入力品質/censorと未校正を数値未収束やモデル不足と混同しないための結果である。

この段階の compact result だけから、実動画のパラメータ同定、接触・摩擦・折りたたみの
再現、臨界値、普遍性を主張しない。
