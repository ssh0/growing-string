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
