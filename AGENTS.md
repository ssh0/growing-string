# AGENTS.md

## このリポジトリについて

このリポジトリは、非平衡系物理学の研究対象として扱った「成長するひも状オブジェクト」の数値シミュレーション、研究メモ、発表資料、解析結果をまとめたものです。

コードだけのプロジェクトではありません。次のものが同居しています。

- 連続平面上の弾性ひもモデル
- 三角格子上の自己回避・成長モデルとその派生モデル
- シミュレーション実行コード、解析コード、Notebook
- LaTeX・Markdown・HTML・PDFによる研究文書
- 大量のシミュレーション結果、図、動画
- 新しい連続フィラメントモデルの設計・検証・論文草稿

現状は2016〜2017年頃の研究資産を保存したリポジトリであり、現代的なパッケージ構成・依存関係固定・テスト構成が整備されているとは限りません。実行可能性、再現性、生成物の最新性を推測で断定しないでください。

今後の主たる発展作業は `continuum_filament_model/` で行います。既存の連続モデル・三角格子モデルは、過去の研究資産または比較対象として保持し、明示的な依頼なしに変更しません。

## 最初に読むファイル

作業対象に応じて、次の順で確認してください。

1. `README.md` — リポジトリの短い概要。ただし、記載された実行例と画像パスには現状との不一致があります。
2. `TODO.txt` — 自然長成長モデルに関する日付付きの作業メモ。
3. `triangular_lattice/TODO.txt` — 三角格子モデルに関する作業メモ。完了項目・未完了項目・将来案が混在しています。
4. 研究内容を確認する場合は `source/main.tex` と `source/main.md`。
5. 個別モデルを確認する場合は、下記の対応するREADME・章ファイル・実装を読む。
6. 実験結果を確認する場合は、実行コードと `triangular_lattice/results/` の両方を確認する。結果ファイル名だけから意味や生成条件を決めないでください。

## ディレクトリ地図

| パス | 内容 | 主な入口・関連資料 |
| --- | --- | --- |
| `source/` | 研究報告の原稿 | `main.tex`、`main.md`、3つの章ファイル |
| `doc/` | 発表資料、文書PDF、旧APIドキュメント、図版ソース | `160708.tex`、`160708.pdf`、`main.pdf` |
| `img/` | 研究文書・発表資料で使う図、格子図、数式画像 | `source/*.tex` の `\includegraphics` から参照 |
| `growing_natural_length_model/` | 自然長が成長する連続平面モデル | `README.md`、`proto.py` |
| `constant_length_model/` | 自然長・ばね定数を一定にした連続平面モデル | `constant_length.py` |
| `triangular_lattice/` | 三角格子上のひも、成長、統計解析、派生モデル | `growing_string.py`、各種 `*_run.py` / `*_analyze.py` |
| `triangular_lattice/doc/` | 三角格子関連のモジュール別APIドキュメントと技術メモ | `*.pdf`、`how_to_detect_inside_of_the_string.md` |
| `triangular_lattice/results/` | 保存済みの数値データ、画像、動画 | `data/`、`img/`、`video/` |
| `continuum_filament_model/` | 新しい連続フィラメントモデル、検証、研究メモ、論文初稿 | `README.md`、`notes/`、`src/`、`tests/`、`paper_draft.md` |

## 研究文書の所在と関係

### `source/`: 研究報告の原稿

`source/main.tex` がLaTeX側のルートです。次の3ファイルを `\input` しています。

- `source/01_growing_natural_length_model.tex`
  - 質点とばねによる弾性ひもモデル
  - ばねの自然長の増大
  - 運動方程式、Euler法・4次Runge–Kutta法
  - 自然長を閾値で分割する成長規則
  - 自己排他的モデルに関する試行と限界
- `source/02_constant_length_model.tex`
  - 自然長・ばね定数を一定に保つモデル
  - ランダムな点追加と斥力ポテンシャルによる自己回避の試行
  - 期待通りに動かなかったという結果・要確認事項
- `source/03_triangular_lattice.tex`
  - 三角格子の構成
  - 6方向ベクトルによるひも状オブジェクトの表現
  - 自己回避ランダムウォーク、deadlock、配置規則

`source/main.md` は同じ研究題目のMarkdown草稿ですが、`main.tex`の完全な変換版であることや、どちらが正本であるかを示す記述はありません。内容も途中で終わっています。文書を変更する際は、LaTeXとMarkdownの両方を確認し、同期関係を推測で決めないでください。

`source/tmptex.pdf` は `source/` 内にあるPDFですが、生成元と用途は確認できていません。

### `doc/`: 発表資料と生成済み文書

- `doc/160708.tex` / `doc/160708.pdf`
  - 2016-07-08の中間発表資料。
  - 研究背景、研究目的、モデル、三角格子モデル、今後の課題、参考文献を含みます。
- `doc/160708.bib`
  - 上記発表資料の参考文献データ。
- `doc/160708.html.html` / `doc/160708.html_bib.html`
  - 発表資料・参考文献のHTML化された資料。生成物であることを示すメタデータがあります。
- `doc/main.pdf`
  - `source/main.tex` と同じ研究題目のPDF成果物に見えますが、現行ソースから生成されたものか、最新であるかは未確認です。
- `doc/points.pdf`、`doc/proto.pdf`、`doc/runner.pdf`
  - 旧実装に対応するAPIドキュメント風PDF。内部に現在のリポジトリと異なる絶対パスが含まれるため、現行コードの正確な仕様として扱わないでください。
- `doc/images.odg`
  - OpenDocument Drawing形式の図版ソース。

### `img/`: 文書用の図版

`img/` には、弾性ひものモデル、座標、折りたたみ、三角格子、単位ベクトル、画面例などのPDF・EPS・PNG・ODG・TeX図版があります。主な参照先は `source/*.tex` と `doc/160708.tex` です。

ルート `README.md` は `img/screen_001.png` を参照していますが、確認できる同名の画像は `img/screen_001.pdf` です。READMEの例を実行可能な手順や正しい画像リンクとみなさず、変更時にはこの不一致を意識してください。

### コード内のドキュメント

古いPythonコードにはdocstringやコメントによるモデル説明・実験メモがあります。特に次を確認してください。

- `growing_natural_length_model/points.py`, `runner.py`
- `constant_length_model/mystring.py`, `runner.py`
- `triangular_lattice/triangular.py`, `strings.py`, `base.py`
- `triangular_lattice/growing_string.py`, `SAW.py`
- `triangular_lattice/save_data.py`, `save_meta.py`
- `triangular_lattice` 配下の解析・可視化モジュール

コメントやdocstringは実装の意図を知る手掛かりですが、現行コードで検証済みの仕様とは限りません。

## モデルとコードの地図

### 連続平面モデル

#### 自然長成長モデル

`growing_natural_length_model/` は実数座標上の質点列を扱います。

- `proto.py` — 初期条件・パラメータ・CLI解析・`String_Simulation` の起動。
- `runner.py` — 時間発展、描画、アニメーション制御。
- `points.py` — 点列、自然長・ばね定数の更新、線分分割。
- `euler.py`、`runge_kutta.py` — 数値積分器。
- `README.md` — モデルの簡単な説明。

#### 定長モデル

`constant_length_model/` は自然長・ばね定数を一定にした別モデルです。

- `constant_length.py` — 初期条件・パラメータ・シミュレーション起動。
- `runner.py` — 時間発展、点追加、描画。
- `mystring.py` — 点・線分操作と点追加。
- `euler.py`、`runge_kutta.py` — 数値積分器。

両モデルとも、実行入口と同じディレクトリにあるモジュールを直接importする構成です。ルートからのimportや、現代のパッケージとしての実行が保証されているわけではありません。

### 三角格子モデル

`triangular_lattice/` の基本構造は次のとおりです。

- `triangular.py` — 三角格子、近傍、座標変換、境界条件。
- `strings.py` — 始点と6方向のベクトル列で表すひも。
- `base.py` — 格子モデルの基底シミュレーション。
- `growing_string.py` — 曲げの重み（`beta`）で候補を選ぶ成長モデル。
- `SAW.py` — 自己回避ランダムウォーク系の派生。
- `growing_string_inside.py`、`fill_bucket.py`、`filled_kagome.py` — 閉曲線内部の判定・塗りつぶし・内部構造の解析。
- `surface.py`、`optimize.py`、`span_fitting.py` — 表面抽出・フィッティング・範囲選択の共通処理。
- `save_data.py`、`save_meta.py` — `.npz` / `.json` の保存ヘルパー。

主な解析テーマは、ファイル名から次のように対応付けられます。

- `radius*.py` — 慣性半径・半径に関する解析。
- `mass*.py`、`mass_in_r*.py` — 半径内質量。
- `roughness.py` — roughness。
- `distances*.py`、`tortuosity.py` — 距離・経路長・tortuosity。
- `correlation*.py` — 相関。
- `box_counting*.py`、`fractal_dim*.py` — box counting・フラクタル次元。
- `cutting_profile*.py`、`diecutting/` — 型抜き・切断プロファイル・サブクラスター。
- `count_bending.py`、`straight_lines.py` — 曲げ角・連続直線長。

派生モデルは次の場所にあります。

- `triangular_lattice/moving_string/` — 移動するひも、弾性・相互作用・deadlock解析。
- `triangular_lattice/random_lattice/` — ランダム化した格子上のモデル。
- `triangular_lattice/eden/` — Edenモデルと半径・質量・roughness解析。
- `triangular_lattice/vicsek/` — Vicsekモデル、自己回避版、動画出力版。
- `triangular_lattice/diecutting/` — 六角形のdie-cutting実験と結果解析。

### 新しい主モデル：`continuum_filament_model/`

今後の研究発展は、既存コードを直接改修せず、`continuum_filament_model/` に独立して追加します。目的は、成長する半柔軟フィラメントを、エネルギー・散逸・参照長成長・自己接触から一貫して記述し、実験データとの整合を検証することです。

- `README.md` — 新しい作業領域の方針、入口、実行方法。
- `notes/research_direction.md` — 研究課題、物理との接合、三角格子モデルの位置づけ。
- `notes/model_spec.md` — 状態変数、エネルギー、散逸、成長、再メッシュの仕様。
- `notes/validation_plan.md` — 数値・力学・実験・粗視化比較の検証計画。
- `notes/experiment_data_contract.md` — 実験中心線データの入力形式と品質管理。
- `src/growing_filament/` — 独立したPythonプロトタイプ。
- `tests/` — 新モデルの不変条件・数値検証。
- `paper_draft.md` — 最終研究論文の初稿。未実施の結果を確定結果として書かない。

初期版の主モデルは、2次元・開曲線・過減衰・伸長＋曲げ＋軟接触・参照長成長・中点再メッシュです。これは検証可能な最小モデルであり、実験を再現済みの完成モデルではありません。新しい物理仮定を追加する場合は、コードだけでなく `notes/model_spec.md` と `notes/validation_plan.md` も更新してください。

三角格子モデルは、新モデルへ直接importして統合しません。共通観測量、無次元パラメータ、seed付きの実験条件を介して、連続モデルの粗視化系・対照系として比較します。既存の `triangular_lattice/results/` を新モデルの出力先として再利用しないでください。

## 実行入口と解析入口

以下はファイル名・コード上で確認できる代表的な入口です。「この一覧がすべての `__main__` を網羅する」「現在の環境で実行できる」とは限りません。

### 基本シミュレーション

- `growing_natural_length_model/proto.py`
- `constant_length_model/constant_length.py`
- `triangular_lattice/growing_string_run.py`
- `triangular_lattice/growing_string_inside_run.py`
- `triangular_lattice/growing_string_sticky.py`
- `triangular_lattice/eden/eden.py`
- `triangular_lattice/random_lattice/growing.py`
- `triangular_lattice/vicsek/vicsek.py`
- `triangular_lattice/moving_string/` 配下の各派生スクリプト

### `*_run.py` と対応する解析

- `triangular_lattice/box_counting_run.py` / `box_counting_analyze.py`
- `triangular_lattice/cutting_profile_run.py` / `cutting_profile_analyze.py`
- `triangular_lattice/radius_run.py` / `radius_analyze.py`
- `triangular_lattice/mass_in_r_run.py` / `mass_in_r_analyze.py`
- `triangular_lattice/growing_string_run.py` / 関連する可視化・解析モジュール
- `triangular_lattice/diecutting/diecutting_hexagonal_run.py` / `diecutting/result_*.py`
- `triangular_lattice/moving_string/moving_string_deadlock.py` / `moving_string_analyze.py`

多くのスクリプトは、パラメータをコード内に直接書き、解析対象の結果ファイルもコード内の相対パスで指定しています。起動前に、引数、既定値、入力結果、出力先を対象ファイルで確認してください。

## Notebook・HTML・APIドキュメント

### Notebook

- `triangular_lattice/growing_string_notebook.ipynb` — 三角格子上の成長シミュレーションとベクトル対の集計。
- `triangular_lattice/book_diecutting_hexagonal.ipynb` — 六角形die-cutting結果の読み込み・可視化・フィッティング。
- `triangular_lattice/book_distances.ipynb` — 頂点間距離、ユークリッド距離、tortuosityの解析。
- `triangular_lattice/diecutting/count_on_edge.ipynb` — 外縁点のカウント。
- `triangular_lattice/vicsek/Vicsek_trilattice.ipynb` — 三角格子上のVicsekモデル。

Notebookは結果ファイルを相対パスで参照します。実行前にKernelのPython環境、作業ディレクトリ、入力データの存在を確認してください。Notebookの再実行可能性や現在のimport名は検証されていません。

### `triangular_lattice/doc/`

このディレクトリには、三角格子関連のPythonモジュールに対応する多数のPDF APIドキュメントがあります。`base.pdf`、`triangular.pdf`、`strings.pdf`、`growing_string*.pdf`、`SAW.pdf`、各種解析・保存・可視化モジュールのPDFが含まれます。PDFの生成方法と更新時点は確認できていないため、現行コードの正本ではなく、過去のAPI参照・設計資料として扱ってください。

`triangular_lattice/doc/how_to_detect_inside_of_the_string.md` / `.html` は、閉曲線の内部検出に関する技術メモです。次の内容を含みます。

- 三角格子の境界を `2Lx × Ly` の配列に写像する方法
- 方向ベクトルから境界線を検出するコード例
- 行走査、`flag`、XORによる内部判定
- `i=0` をまたぐ場合の既知の問題

HTMLはMarkdownから変換された成果物です。技術メモを修正する場合は、まずMarkdownを編集対象とし、HTMLを直接編集して正本にしないでください。変換手順自体は確認できていません。

## 結果・生成物の扱い

`triangular_lattice/results/` は次の3系統に分かれています。

```text
triangular_lattice/results/
├── data/   # npz/json/csv などの数値データ・メタデータ
├── img/    # png/pdf/eps/html などの画像・可視化
└── video/  # mp4 などの動画
```

`data/` の主なテーマは `box_counting/`、`correlation/`、`cutting_profile/`、`diecutting/`、`distances/`、`mass_in_r/`、`moving_string/`、`radius/` です。`img/` にはそれ以外に `count_bending/`、`eden/`、`fill_bucket/`、`fractal_dim/`、`inside/`、`max_radius/`、`roughness/`、`tortuosity/` などもあります。`video/` には `inside/`、`random_lattice/`、`vicsek/` などがあります。

結果は多数がGit管理対象です。確認時点でリポジトリには大量の `.npz`、`.json`、画像、PDF、動画が含まれ、`results` は `.gitignore` で除外されていません。

したがって、次を守ってください。

- 既存の結果を削除・移動・一括再生成しない。
- シミュレーションを実行する前に、既存出力の上書き有無、対象パラメータ、計算量、保存先を確認する。
- 新しい結果を保存する場合は、既存の命名規則とテーマ別ディレクトリを確認する。
- `save_data.py` はタイムスタンプ付き `.npz`、`save_meta.py` はJSONを保存するが、すべてのスクリプトがこのヘルパーを使うわけではない。
- 結果ファイル名から、乱数seed、パラメータの完全な組、再現性、科学的な結論を推測しない。
- 結果の追加・更新を依頼されていない場合、コード変更だけで重いシミュレーションや図の再生成を行わない。

## 実行環境と既知の制約

### 依存関係・互換性

確認できる範囲では、ルートに `pyproject.toml`、`setup.py`、`setup.cfg`、`requirements.txt`、`environment.yml`、`Makefile`、標準的なCI設定はありません。依存関係は対象スクリプトのimportから確認してください。

コードには、Python 2系を想定した構文・APIと、Python 3向けの記述が混在しています。例えば、古い `print` 文、`has_key()`、`iterkeys()` などが残っています。Pythonのバージョンや必要ライブラリを推測して環境手順に追加しないでください。

必要になった場合は、まず対象ファイルのimportと構文を確認し、既存環境を壊さない隔離環境で最小の動作確認を行ってください。

### 相対パスと作業ディレクトリ

多くのモジュールが同じディレクトリ内のファイルを直接importし、結果出力には `results/...` のような相対パスを使います。`triangular_lattice/diecutting/diecutting_hexagonal_run.py` のように、親ディレクトリを参照する例もあります。

そのため、実行前に必ず次を確認してください。

1. どのディレクトリをカレントディレクトリにするか。
2. `sys.path` や直接importが対象環境で解決できるか。
3. 入力結果の相対パスが存在するか。
4. 出力先が意図した場所か。
5. GUI表示・FFmpeg・動画コーデックなどの実行時依存があるか。

READMEの `python proto.py` はルート直下の現状のファイル配置とは一致しません。実行コマンドを新たに案内する際は、対象ディレクトリ、Pythonバージョン、依存関係、出力先を検証してから記載してください。

### 計算量と副作用

既定値には大きな格子、長いフレーム数、多数のサンプルを使う実験があります。例えば、`growing_string_run.py`、`cutting_profile_run.py`、`radius_run.py`、`mass_in_r_run.py` などは、設定によって長時間実行・大量出力になります。

小さなパラメータでの確認なしに本番設定を実行しないでください。GUI表示、動画保存、結果ファイルの生成は、依頼範囲と出力先を確認してから行ってください。

### 内部判定の既知の問題

`triangular_lattice/doc/how_to_detect_inside_of_the_string.md` は、走査線による内部判定について `i=0` をまたぐ場合の不自然な `flag` 設定を既知の課題として記載しています。初期条件を制限する暫定策も書かれていますが、任意の閉曲線について正しいことは保証されていません。

## 文書・コードを変更するときの方針

- まず変更対象の一次資料、参照元、対応する生成物を確認する。
- `source/main.tex` の章を変更する場合は、関連する `source/*.tex`、`source/main.md`、`doc/main.pdf`、`img/` の参照を確認する。
- 発表資料を変更する場合は、`doc/160708.tex`、`doc/160708.bib`、対応するPDF/HTMLを確認する。
- APIドキュメントPDFやHTMLなどの生成物を直接編集しない。生成元と生成コマンドが確認できない場合は、生成物の更新を推測で行わない。
- TODOのチェック状態を、現行コードで検証済みの機能一覧とみなさない。
- 物理モデルの意味、結果の解釈、科学的妥当性は、コードの存在だけから断定しない。
- 既存の日本語・数式・引用・ファイル名・結果データを、読みやすさだけを理由に意味変更しない。
- 依頼範囲外のリファクタリング、依存関係追加、結果再生成、PDF再ビルドを行わない。
- 新しい連続モデルの変更は `continuum_filament_model/` に限定し、既存の `growing_natural_length_model/`、`constant_length_model/`、`triangular_lattice/` を変更しない。
- 新モデルの物理仮定を変更した場合は、`notes/model_spec.md`、`notes/validation_plan.md`、必要に応じて `paper_draft.md` を同時に確認する。
- `paper_draft.md` では、実施済み・未実施・仮説・計画を明確に分ける。

## 検証の方針

このリポジトリには、標準的なテストスイート・lint設定・CI設定が確認できません。変更後は、実施したものだけを明示してください。

最低限、次を行います。

1. `git diff -- AGENTS.md` または対象ファイルの差分を確認する。
2. パス、ファイル名、リンク、出力先の記述が実在するか確認する。
3. 文書変更では、見出し・リンク・数式ファイル参照の整合性を確認する。
4. コード変更では、対象コードに適合した最小の構文確認・単体確認・小規模実行を選ぶ。Pythonバージョン未確認のまま一律のコマンドを成功扱いしない。
5. シミュレーション、Notebook、LaTeX、動画生成を実行していない場合は、検証済みと報告しない。
6. 既存の作業ツリー変更がある場合は、変更者・目的を推測せず、差分を保持したまま作業する。

## 調査時点で確認できていないこと

次の事項は、このファイルの作成時点では確定していません。

- 推奨Pythonバージョンと完全な依存ライブラリ一覧
- LaTeX、HTML、APIドキュメント、Notebookの正式な生成コマンド
- `source/main.tex` と `source/main.md` の正本・同期関係
- 各PDF・画像・結果データの最新性と再生成条件
- 結果ファイルの完全なスキーマ、乱数seed、再現性
- 物理モデルやシミュレーション結果の科学的妥当性
- すべての実行入口と、現在の環境での実行成功

これらを確認した場合は、推測でこのファイルに追記せず、根拠となる設定・コマンド・実行結果を記録してから更新してください。
