# Python formatter / lint policy

## 方針

このリポジトリの新しいPythonコードは、Ruffのformatterとlintを使います。BlackとFlake8は導入しません。formatterとlintを同じRuffの固定バージョンで実行でき、既存の依存関係や研究コードへ追加の実行時依存を持ち込まないためです。

- ツール: `ruff==0.13.1`
- formatter: `ruff format --check`
- lint: `ruff check`
- Pythonの対象: Python 3.10以降の構文（`target-version = "py310"`）
- 行長: formatterの目標値を100文字に設定
- E501: lintルールには含めない。長いURL、文字列、コメント、数式表現など、formatterが安全に分割できない行を機械的に変更しないため

formatterチェック、lintチェック、既存のunittestは、GitHub Actionsで別々のjobとして表示します。設定は`pyproject.toml`、ツールのバージョンは`requirements-lint.txt`が正本です。

## 対象範囲

初期導入の品質対象は、保守対象としてPython 3系で記述されている次の変更済みファイルです。

- `continuum_filament_model/**/*.py`
- `scripts/**/*.py`

`continuum_filament_model/notebooks/`、PDF・動画・結果データなどの生成物、既存の`growing_natural_length_model/`、`constant_length_model/`、`triangular_lattice/`は対象外です。後者にはPython 2系の構文/APIを含む過去の研究資産があり、今回の品質基盤導入で一括整形・移行すると、研究成果物や科学的挙動と無関係な大規模差分になります。

既存ファイルのベースラインを一括整形しません。品質runnerはPRまたはローカルの変更差分に含まれる対象ファイルだけを検査します。このため、既存の未整形コードを段階的に変更する際は、その変更ファイル全体がformatter/lint規約に適合している必要があります。対象外の既存違反を今回のPRで修正する場合は、物理モデル変更や結果ファイルと混在させず、別の明示的な変更として扱います。

導入時点の維持対象ベースラインには、`continuum_filament_model`全体で`ruff check`の未修正違反16件と、`ruff format --check`の未整形ファイル42件があります。これらは今回一括修正せず、変更差分に含まれない限り受理します。対象ファイルを変更するPRでは、runnerがファイル全体を検査するため、変更を提出する側がそのファイルの既存違反も解消します。

## ローカル実行

リポジトリルートで、CIと同じrunnerを実行します。

```bash
python -m pip install -r requirements-lint.txt
python scripts/check_python_quality.py format
python scripts/check_python_quality.py lint
PYTHONPATH=continuum_filament_model/src \
python -m unittest discover -s continuum_filament_model/tests -v
```

runnerは`PYTHON_QUALITY_BASE`またはGitHub ActionsのPR baseを優先し、ローカルでは`origin/master`、次に`master`を差分の基準にします。baseが利用できない場合は、未コミット・未追跡の変更だけを検査します。対象Pythonファイルの変更がない場合は、品質runnerは成功して「検査対象なし」と表示します。

## GitHub Actions

`.github/workflows/continuum-filament-tests.yml`は、次の3つを分離して実行します。

1. `formatter`: `python scripts/check_python_quality.py format`
2. `lint`: `python scripts/check_python_quality.py lint`
3. `unittest`: 既存の`unittest discover`コマンド

PRでは`actions/checkout`の履歴を取得し、`github.event.pull_request.base.sha`を`PYTHON_QUALITY_BASE`として渡します。したがって、ローカルrunnerのコマンド文字列とCIのコマンド文字列は同じで、差分基準だけが環境変数で明示されます。`unittest`のNumPy、Matplotlib、ffmpegの依存関係とPython 3.11.5は既存workflowの設定を維持します。
