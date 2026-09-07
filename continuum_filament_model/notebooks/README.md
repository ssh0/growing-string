# Marimo visual explorer

`filament_explorer.py` は、`continuum_filament_model/src/growing_filament/` の公開APIを呼び出して、成長するフィラメントの挙動を小規模に視覚探索するノートブックです。モデル実装を複製していません。

## 起動

リポジトリルートから起動してください。ノートブック冒頭でも同じimport経路を確認できます。

```bash
marimo edit continuum_filament_model/notebooks/filament_explorer.py
```

ノートブックは `continuum_filament_model/src` を `PYTHONPATH` 相当で先頭に追加し、`growing_filament.model`、`growing_filament.observables`、`growing_filament.geometry`、`growing_filament.reproducibility` を明示的にimportします。別のディレクトリから起動する場合は、リポジトリルートをカレントディレクトリにしてください。

## 使い方とfixture

フォームでプリセットとパラメータを選び、**この設定で実行**を押します。`プリセット推奨値を適用`を有効にすると、選択したfixtureを再現しやすい設定が優先されます。すべての入力値をそのまま試す場合は無効にしてください。節点数・刻み数・再メッシュ後の推定節点数にはノートブック内の上限があります。

- `straight`: 成長なしの直線。初期エネルギーがほぼゼロの基準ケース。
- `perturbed_fixed_growth`: 微小摂動・両端固定・成長あり。
- `u_shape` / `s_shape`: 決定論的な初期曲線。
- `contact`: 3節点のU字状fixture。直径・接触剛性を有効にすると節点接触エネルギーを確認できます。
- `crossing_rejection`: 既存テストと同じ非交差初期形状から、swept crossing の棄却を確認するfixture。`RuntimeError`は失敗ではなく、受理できなかった理由と棄却イベントを表示するための期待される診断です。

同じ設定を成功実行した場合は、ノートブックが決定論的に2回実行し、canonical state hash・イベント列を比較します。乱数は導入していません。

このノートブックは視覚的な探索用です。表示結果は物理的妥当性、数値収束性、実験再現性を証明しません。

## ブラウザE2E確認

marimoの実行画面で中心線アニメーションが描画され、再生後にフレームが変化することを確認するハーネスを用意しています。リポジトリルートから実行してください。

```bash
python continuum_filament_model/notebooks/e2e_marimo_animation.py
```

この確認は依存関係をインストールしません。実行環境に `marimo` と、既存のChromeを操作する `chrome-devtools-axi` が必要です。ハーネスは専用の既定ポート `27991` で `marimo run` を起動し、HTTP応答を待ってからブラウザを起動します。ポート、Chromeセッション名、証跡出力先を変更する場合は、次のオプションを使えます。

```bash
python continuum_filament_model/notebooks/e2e_marimo_animation.py \
  --port 27992 \
  --session growing-string-marimo-animation-local \
  --evidence-dir /tmp/growing-string-marimo-animation-local
```

ハーネスはデフォルトfixtureの実行ボタンを、`marimo-form` の `data-submit-button-label` を使って選択します。その後、「中心線アニメーション」見出しを基準に、再生コントロールの可視性、画像の非ゼロサイズと `naturalWidth` / `naturalHeight`、再生後の画像データURIまたはフレーム値の変化を条件待ちで確認します。成功時は `initial.png` と `after-play.png` を保存します。

証跡は既定で `/tmp/growing-string-marimo-animation-YYYYmmdd-HHMMSS/` に保存され、Git管理対象ではありません。失敗時は `failure.png`、`failure-state.json`、`browser-failure-dom.log`、`browser-failure-console.log`、`browser-failure-console-errors.log`、`marimo.log` を確認してください。ハーネスは成功・失敗のどちらでもmarimoサーバーとブラウザセッションを回収します。
