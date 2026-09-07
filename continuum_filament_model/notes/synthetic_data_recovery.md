# 合成時系列による実験データ契約・parameter recovery検証

## 目的と範囲

`benchmarks/synthetic_data_recovery.py` は、実験データ取得前に次のブロッカーを小さく検証するための runner である。

- `experiment_data_contract.md` に沿う raw-like 入力が、中心線・time/frame・width・quality・events・metadata を表現できるか。
- `centerline → 弧長再パラメータ化 → 曲率・速度・長さ・event QC → parameter recovery` の解析経路が動くか。
- 既知の `g, EI, EA, drag, diameter` と seed 付き観測ノイズに対し、回収可能量と回収不能量を区別できるか。

これは実験結果の代替ではなく、連続モデルの妥当性の証明でもない。合成 fixture の生成には既存の P1B の初期 sine fixture・無次元量 API と P1B.2 の `target_overrides` を使う。core physics は変更していない。

## fixture と観測劣化

既定 config は次の限定 fixture を生成する。

- `straight`：静止した直線。
- `sinusoidal_buckling`：既知の第一モード `A(t)=A_0 exp(-t EI π^4/(drag L^4))`。
- `arc`：既知半径の円弧。
- `growth`：`L(t)=L_0 exp(g t)` の直線。

観測側には、seed 付き localization noise、blur、frame drop、明示的 missing point、point-order error、pixel scale mismatch、width variation を個別に入れる。欠損点は `missing=true` と null 座標で保持し、frame drop は期待 frame と観測 frame の差として検出する。point-order error と scale mismatch は、それらを含む run を成功扱いにせず、それぞれ `censored` / `biased` とする。

生の CSV、events、metadata、run manifest は `--output` で指定した一時ディレクトリだけに書く。Git 管理下へ置くのは runner、config、schema、note と compact summary だけであり、per-frame/per-run データは commit しない。

## 回収量と非識別量

この合成観測で回収するのは、fixture が対応する場合に限る。

- `g`：成長 fixture の `log(contour length)` の時間傾き。
- `EI/drag`：第一モード振幅の緩和率と観測長さから得る比。`EI` と `drag` を個別には回収しない。
- `diameter_proxy`：width の中央値。これは幅測定の proxy であり、有限径の接触則を同定したことを意味しない。

中心線・時刻・width・quality だけでは、校正済みの軸方向応答または力がないため `EA` を回収できない。受動的な形状緩和だけでは `EI` と `drag` は分離できない。真の端点反力、線分接触の反発・摩擦・接着などの contact law も、この入力からは識別できない。runner の各 manifest に、これらを `unidentifiable` として理由付きで保存する。

## 再現性と compact summary

manifest には少なくとも次を保存する。

- `seed`
- raw CSV の `raw_synthetic_hash`
- `truth_config` / `noise_config`
- `analysis_revision`
- 解析結果の `analysis_result_hash`
- 検出イベント、status、識別量、非識別量

同一 seed・同一設定は raw hash と解析結果 hash の一致を別 check として検証する。複数 seed は一致を要求せず、誤差・status の分布として集計する。compact summary の acceptance check は noise=0 の真値許容誤差、missing/frame drop/order error/scale mismatch の非黙示的扱いを確認する。結果は `results/synthetic_data_recovery/` の compact ファイルを参照する。

## この結果から言えること／言えないこと

### 言えること

- 現行のデータ契約で、中心線・時刻・幅・品質・欠損・イベント・metadata を一つの入力単位として保存できる。
- 弧長再パラメータ化後の曲率・速度・長さの算出と、品質イベントの検出経路を再実行できる。
- 宣言した calibration と fixture の範囲では、`g` と `EI/drag` の回収可能性を定量的に smoke-test できる。
- noise、blur、frame drop、missing、point-order error、scale mismatch を valid / degraded / censored / biased に分けられる。

### 言えないこと

- 実験動画で同じ誤差分布・欠損機構・セグメンテーション品質になること。
- 合成 fixture での parameter recovery が、モデルが実験を説明すること。
- `EA`、`EI`、`drag`、真の反力、contact law を中心線だけから同定できること。
- 合成時系列の回収誤差が、実験に対する予測誤差やモデル妥当性を表すこと。

特に、合成 generator が真値を知っていることと、実験でその値を識別できることを混同しない。

## 実験取得後の calibration / holdout 計画

1. **calibration 用試料・run を分離する。** まず pixel scale、width の測定系、frame interval、中心線抽出品質を校正する。単純な成長 run から `g`、単純な曲げ緩和から `EI/drag` を推定する場合も、試料単位で分ける。
2. **holdout を先に固定する。** 条件、試料、独立 run を holdout として解析前に固定し、隣接 frame を独立 replicate と数えない。推定する校正パラメータ、除外規則、欠損・視野外の扱いを manifest に記録する。
3. **識別性を追加データで確認する。** `EA` には校正済みの軸方向変位と力、反力には力センサーまたは力学的 proxy、contact law には接触距離・接触時間・離脱挙動など、中心線だけではない測定を要求する。
4. **holdout では未使用観測量を比較する。** 曲率分布、速度、輪郭長、接触長、端点間距離、慣性半径などを、時系列平均・分布・replicate 間ばらつき・時間相関で比較する。「見た目が似ている」だけでは合格にしない。

## 解析 pipeline とモデル妥当性の区別

- **解析 pipeline 検証**：入力契約、単位換算、点順序、欠損・frame drop、弧長再パラメータ化、観測量算出、hash/manifest、再現性を確認する。今回の主対象。
- **力学モデル検証**：既知解、時間・空間収束、散逸、境界条件、接触診断などを確認する。P1A/P1B/P1B.2 の benchmark と `validation_plan.md` に分けて記録する。
- **実験妥当性評価**：calibration を holdout に再利用せず、独立試料・条件・観測量で予測を評価する。合成 recovery の成功を実験妥当性の成功へ昇格させない。

この区別を崩さない限り、合成検証は実験データ取得時の入力・品質管理・識別性に関する早期の受入条件として利用できる。
