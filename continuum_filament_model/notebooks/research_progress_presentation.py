"""連続体フィラメントモデル：研究進捗中間発表。

研究室内の中間発表を想定した、読み物と数値成果物のレビュー用 marimo ノートブック。
生成済みの compact artifact を読み込み、物理的な問い・モデル・検証・現時点の限界を
一つの流れで確認する。新しいシミュレーションを暗黙に実行せず、表示する結果の由来を
各章で明示する。

リポジトリルートから起動する::

    marimo edit continuum_filament_model/notebooks/research_progress_presentation.py
"""

import marimo

__generated_with = "0.23.6"
app = marimo.App(width="full")


@app.cell
def _():
    import csv
    import json
    import math
    from pathlib import Path

    import marimo as mo
    import matplotlib
    import matplotlib.font_manager as fm
    import matplotlib.pyplot as plt
    import numpy as np

    _jp_font_candidates = [
        "Hiragino Sans",
        "Hiragino Maru Gothic Pro",
        "YuGothic",
        "AppleGothic",
        "Noto Sans CJK JP",
        "IPAexGothic",
        "TakaoGothic",
    ]
    _available_fonts = {font.name for font in fm.fontManager.ttflist}
    _matched_jp_fonts = [font for font in _jp_font_candidates if font in _available_fonts]
    if _matched_jp_fonts:
        matplotlib.rcParams["font.family"] = "sans-serif"
        matplotlib.rcParams["font.sans-serif"] = _matched_jp_fonts + [
            font for font in matplotlib.rcParams.get("font.sans-serif", []) if font not in _matched_jp_fonts
        ]
    matplotlib.rcParams["axes.unicode_minus"] = False
    jp_font_name = _matched_jp_fonts[0] if _matched_jp_fonts else "matplotlib default (Japanese fallback unavailable)"

    return Path, csv, json, jp_font_name, mo, np, plt


@app.cell
def _(mo):
    mo.md(r"""
    # 成長する連続体フィラメント
    ## 研究進捗・中間発表

    **中心仮説**：過減衰環境では、フィラメントの形態は「長さが増える速さ」と
    「曲げ変形が緩和する速さ」の競合によって選択される。さらに、有限の太さを
    持つ場合は自己接触が許容される形状を拘束し、座屈から折りたたみへの経路を
    変える。

    このノートブックでは、コード実装の一覧ではなく、次の順に研究の論理を追う。

    1. どの現象を説明したいのか
    2. どの状態変数とエネルギーで記述するのか
    3. 数値結果を物理的な結果と呼ぶために何を検証したのか
    4. 非接触成長、実観察との接続、有限径接触で何が分かったのか
    5. 何がまだ分かっておらず、次に何を測るべきか

    > **読み方**：青字の「確認済み」は保存済み artifact に基づく記述、
    > 「解釈」はそこからの物理的な読み、赤字の「未解決」は現時点で結論に
    > 使わない領域を表す。
    """)
    return


@app.cell
def _(Path, csv, json):
    repo_root = Path.cwd()
    artifact_root = repo_root / "continuum_filament_model"
    if not artifact_root.is_dir():
        raise RuntimeError(
            "リポジトリルートから起動してください: "
            f"{artifact_root} が見つかりません（cwd={repo_root}）"
        )

    results_root = artifact_root / "results"

    def load_json(path):
        with path.open(encoding="utf-8") as handle:
            return json.load(handle)

    def load_csv(path):
        with path.open(encoding="utf-8", newline="") as handle:
            return list(csv.DictReader(handle))

    p0b = load_json(results_root / "p0b" / "compact_summary.json")
    p1b2_map = load_json(results_root / "p1b2" / "regime_map.json")
    p1b2_manifest = load_json(results_root / "p1b2" / "compact_manifest.json")
    p1b2_convergence = load_json(results_root / "p1b2" / "convergence_summary.json")
    contact_rows = load_csv(results_root / "contact_buckling" / "summary.csv")
    contact_suite = load_json(results_root / "contact_buckling" / "suite.json")
    video_summary = load_json(results_root / "video_parameter_fitting" / "compact_summary.json")

    result_paths = {
        "p1b2_map": results_root / "p1b2" / "regime_map.png",
        "p1b2_convergence": results_root / "p1b2" / "convergence_summary.png",
        "contact_map": results_root / "contact_buckling" / "phase_map.png",
    }
    return (
        contact_rows,
        contact_suite,
        p0b,
        p1b2_convergence,
        p1b2_manifest,
        p1b2_map,
        result_paths,
        video_summary,
    )


@app.cell
def _(jp_font_name, mo, p1b2_manifest, p1b2_map, result_paths, video_summary):
    report_lines = [
        f"- P1B.2 非接触 suite: **{p1b2_manifest['run_count']} run**、seed `{p1b2_manifest['seed_set']}`、3×3 grid",
        f"- P1B.2 regime map: `{result_paths['p1b2_map'].relative_to(result_paths['p1b2_map'].parents[2])}`（各cellは5 seed trial）",
        f"- P2 接触 suite: `{result_paths['contact_map'].relative_to(result_paths['contact_map'].parents[2])}`（compact summary）",
        "- 動画同定: 保存済みartifactは `gray5` と `original` ともに中心線本体が未同梱で、現時点の実観察推定値は未同定",
    ]
    mo.md(
        "### このページが参照する成果物\n"
        + "\n".join(report_lines)
        + "\n\n"
        + f"P1B.2 のregime分類は `{len(p1b2_map['rows'])}` cellを対象にし、動画側のcompact reportは `{len(video_summary['reports'])}` 入力を記録しています。"
        + f"\n\nMatplotlib Japanese font: `{jp_font_name}`"
    )
    return


@app.cell
def _(mo, np, plt):
    # 生物・高分子系で現れる「成長して形を選ぶ」問題を、特定データの再現ではなく
    # 問いの導入として示す概念図。数値結果の主張には使わない。
    s = np.linspace(0.0, 1.0, 240)
    straight = np.column_stack((s, np.zeros_like(s)))
    buckled = np.column_stack((s, 0.08 * np.sin(np.pi * s)))
    folded = np.column_stack((s, 0.18 * np.sin(2.5 * np.pi * s) * (0.3 + 0.7 * s)))

    _figure, _axes = plt.subplots(1, 3, figsize=(12, 3.2), constrained_layout=True)
    for _axis, _curve, _title, _color in zip(
        _axes,
        (straight, buckled, folded),
        ("伸長", "座屈", "折りたたみの候補"),
        ("tab:blue", "tab:orange", "tab:red"),
    ):
        _axis.plot(_curve[:, 0], _curve[:, 1], color=_color, linewidth=3)
        _axis.scatter(_curve[[0, -1], 0], _curve[[0, -1], 1], color="black", s=18, zorder=3)
        _axis.set_title(_title)
        _axis.set_aspect("equal", adjustable="box")
        _axis.set_xticks([])
        _axis.set_yticks([])
        _axis.set_xlim(-0.05, 1.05)
        _axis.set_ylim(-0.28, 0.28)
    _figure.suptitle("長さ成長と曲げ緩和の競合が、同じ材料でも形態を分ける", fontsize=14)

    mo.vstack(
        [
            mo.md(
                r"""
                ## 1. 背景と学術的問い

                枯草菌のような細菌細胞や高分子フィラメントでは、成長・伸長に伴って
                座屈、coiling、folding が起こり得る。観察される形は、単に「長くなった
                結果」ではなく、成長で注入される幾何学的な圧縮と、曲げ・伸長・粘性抵抗
                による緩和の時間競合の結果として理解したい。

                **問い**：

                - 成長率と曲げ緩和時間の比から、直線維持から単一モード座屈へ移る傾向を
                  どこまで整理できるか。
                - 有限の径を持つとき、自己接触は座屈後の自由度をどのように制限し、
                  折りたたみの形を選ぶのか。
                - 画像から見える形の一致は、成長率・曲げ剛性・接触パラメータのどこまでを
                  同定しているのか。

                離散格子モデルは形態統計や粗視化比較には有用だが、格子方向・成長 step・
                占有判定が物理量と直結しない。固定長モデルでは、成長が生む参照長の変化を
                力学の中で扱えない。そこで、Euler--Bernoulli 型の連続体を離散化し、
                **非線形成長・過減衰・有限径接触**を同じ状態方程式に入れる。
                """
            ),
            _figure,
        ]
    )
    return


@app.cell
def _(mo, np, plt):
    _formulation_md = mo.md(
        r"""
        ## 2. 連続体フィラメントの定式化

        開曲線を節点位置 $\mathbf{x}_i\in\mathbb{R}^2$ と線分ごとの局所参照長
        $a_i>0$ で表す。現在の線分長は $l_i=\lVert\mathbf{x}_{i+1}-\mathbf{x}_i\rVert$、
        単位接線は $\mathbf{t}_i=(\mathbf{x}_{i+1}-\mathbf{x}_i)/l_i$ である。

        ### エネルギーと運動方程式

        $$
        E_s=\sum_i\frac{EA}{2a_i}(l_i-a_i)^2,
        \qquad
        E_b=\sum_i\frac{EI}{2h_i}\lVert\mathbf{t}_i-\mathbf{t}_{i-1}\rVert^2,
        \quad h_i=\frac{a_{i-1}+a_i}{2}.
        $$

        参照長は一様な指数成長 $\dot a_i=g a_i$（実装上は
        $a_i(t+\Delta t)=a_i(t)e^{g\Delta t}$）に従う。参照長が閾値を超えたら線分を
        中点で二分し、総参照長・端点・幾何学的輪郭長を保存する。これは物理的な分裂
        ではなく、空間解像度を保つ再メッシュである。

        過減衰ダイナミクスは
        $$
        \boldsymbol{\Gamma}\dot{\mathbf{x}}
        =-\frac{\partial(E_s+E_b+E_c)}{\partial\mathbf{x}},
        \qquad
        \boldsymbol{\Gamma}_i=\zeta w_i\mathbf{I},
        $$
        とする。有限径接触は、線分間距離 $d_{ij}$、径 $D$ に対して
        $\delta_{ij}=[D-d_{ij}]_+$、$E_{c,ij}=k_c\delta_{ij}^2/2$ とし、反発力を
        最近接点の双線形形状関数で4端点へ散布する。中心線交差では法線を作らず、
        試行を棄却する。

        ### 形態を整理する無次元量

        $$
        \chi=\frac{EI}{EA L^2},\qquad
        G_b=g\tau_b,
        \quad \tau_b=\frac{\zeta L^4}{EI\pi^4},
        \qquad
        G_s=\frac{g\zeta L^2}{EA},
        \qquad
        \Pi_c=\frac{k_cD^2}{EI}.
        $$

        $G_b$ は成長と曲げ緩和の競合、$G_s$ は成長と伸長緩和の競合、
        $\Pi_c$ は penalty 接触の相対的な硬さを表す。ただし、これらが普遍的な
        相図を保証するわけではなく、境界条件・初期摂動・時間刻み・解像度も残る。
        """
    )

    # 状態変数と双対セルの対応を図示する。
    points = np.asarray([[0.0, 0.0], [1.0, 0.25], [2.0, 0.0], [3.0, -0.12]])
    _figure, _axis = plt.subplots(figsize=(10, 2.8), constrained_layout=True)
    _axis.plot(points[:, 0], points[:, 1], "o-", color="tab:blue", linewidth=2)
    for i, (x, y) in enumerate(points):
        _axis.text(x, y + 0.08, rf"$\mathbf{{x}}_{i}$", ha="center", fontsize=12)
    for i in range(len(points) - 1):
        midpoint = (points[i] + points[i + 1]) / 2.0
        _axis.text(midpoint[0], midpoint[1] - 0.12, rf"$a_{i}$", ha="center", color="tab:orange")
    _axis.axvline(1.5, color="0.7", linestyle="--", linewidth=1)
    _axis.text(1.5, 0.27, "局所双対セルの境界", ha="center", color="0.35")
    _axis.set_title("節点位置・参照長・局所接線差分")
    _axis.set_aspect("equal", adjustable="datalim")
    _axis.set_axis_off()
    mo.vstack([_formulation_md, mo.md("**離散化の直感**：曲げは角度そのものではなく、隣接接線の差を局所参照長で重み付けする。"), _figure])
    return


@app.cell
def _(mo):
    length_control = mo.ui.slider(1.0, 4.0, value=2.0, step=0.1, label="代表長 L")
    growth_control = mo.ui.slider(0.05, 0.35, value=0.19, step=0.01, label="成長数 G_b")
    chi_control = mo.ui.slider(0.0001, 0.0006, value=0.00025, step=0.000025, label="剛性比 chi")
    contact_pi_control = mo.ui.slider(0.0, 50.0, value=12.25, step=0.25, label="接触数 Pi_c")
    controls = mo.vstack(
        [
            mo.md("### 無次元パラメータの探索（換算のみ。新しいシミュレーションは実行しない）"),
            mo.hstack([length_control, growth_control]),
            mo.hstack([chi_control, contact_pi_control]),
        ]
    )
    return (
        chi_control,
        contact_pi_control,
        controls,
        growth_control,
        length_control,
    )


@app.cell
def _(
    chi_control,
    contact_pi_control,
    controls,
    growth_control,
    length_control,
    mo,
    np,
    plt,
):
    L_selected = float(length_control.value)
    Gb_selected = float(growth_control.value)
    chi_selected = float(chi_control.value)
    Pi_selected = float(contact_pi_control.value)
    EA_reference = 100.0
    zeta_reference = 1.0
    EI_selected = chi_selected * EA_reference * L_selected**2
    tau_b_selected = zeta_reference * L_selected**4 / (EI_selected * np.pi**4)
    g_selected = Gb_selected / tau_b_selected
    Gs_selected = g_selected * zeta_reference * L_selected**2 / EA_reference
    D_reference = 0.35
    kc_selected = Pi_selected * EI_selected / D_reference**2

    gb_grid = np.linspace(0.05, 0.35, 100)
    g_grid = gb_grid / tau_b_selected
    _figure, _axes = plt.subplots(1, 2, figsize=(10, 3.2), constrained_layout=True)
    _axes[0].plot(gb_grid, g_grid, color="tab:blue")
    _axes[0].scatter([Gb_selected], [g_selected], color="tab:red", zorder=3)
    _axes[0].set_xlabel(r"$G_b$")
    _axes[0].set_ylabel(r"換算した $g$（基準 $L, EA, EI, \zeta$）")
    _axes[0].set_title("成長数から成長率への換算")
    _axes[0].grid(alpha=0.25)
    _axes[1].bar(["EI", "k_c"], [EI_selected, kc_selected], color=["tab:orange", "tab:green"])
    _axes[1].set_title(r"選択条件の剛性スケール（$D=0.35$）")
    _axes[1].set_yscale("log")
    _axes[1].grid(axis="y", alpha=0.25)

    mo.vstack(
        [
            controls,
            mo.md(
                f"""
                **現在の換算**：`L={L_selected:.2f}`, `G_b={Gb_selected:.3f}`, `chi={chi_selected:.6f}`, `Pi_c={Pi_selected:.2f}`<br>
                `EI={EI_selected:.4g}`, `tau_b={tau_b_selected:.4g}`, `g={g_selected:.4g}`, `G_s={Gs_selected:.4g}`, `k_c={kc_selected:.4g}`

                ここで表示しているのは定義式の換算であり、この操作だけで形態や臨界値を
                予測しているわけではない。実験へ移るときは、単位・代表長・drag の校正を
                別途用意する必要がある。
                """
            ),
            _figure,
        ]
    )
    return


@app.cell
def _(mo, p0b, plt):
    mode_records = [_record for _record in p0b["linear_mode"]["records"] if _record["record_type"] == "linear_eigenmode"]
    decay_records = [_record for _record in p0b["linear_mode"]["records"] if _record["record_type"] == "linear_decay"]
    max_hessian_error = max(_record["eigenvalue_relative_error"] for _record in mode_records)
    decay_pass_count = sum(_record["decay_rate_status"] == "pass" for _record in decay_records)
    _figure, _axis = plt.subplots(figsize=(7.5, 3.2), constrained_layout=True)
    grouped = {}
    for _record in mode_records:
        grouped.setdefault(_record["n_nodes"], []).append(_record["eigenvalue_relative_error"])
    nodes = sorted(grouped)
    _axis.bar([str(node) for node in nodes], [max(grouped[node]) for node in nodes], color="tab:blue")
    _axis.set_yscale("log")
    _axis.set_xlabel("節点数")
    _axis.set_ylabel("Hessian固有値の相対誤差")
    _axis.set_title("P0-B：離散曲げHessianと有限差分gradientの整合")
    _axis.grid(axis="y", alpha=0.25)

    mo.vstack(
        [
            mo.md(
                f"""
                ## 3. 実験計画と数値計算の検証

                数値の軌跡を物理的な観察と呼ぶ前に、次の層を分けて検証する。

                - **幾何**：剛体並進・回転でエネルギーが変わらない、線分長・接線・再メッシュ保存量が整合する。
                - **力学**：各エネルギー項の解析力が有限差分の負のgradientに一致する。成長なしでは受理軌跡のエネルギーが非増加になる。
                - **トポロジー**：非隣接線分の交差を検出し、試行中の swept crossing を受理しない。
                - **時間・空間**：製造解、線形mode、再メッシュ収束を、形態・エネルギー・散逸の主張に分けて確認する。

                ### P0-B 線形mode

                端点**位置は固定、接線は自由**という境界条件で、有限差分Hessianと離散
                tangent-difference Hessianを比較した。保存済み結果では、固有値相対誤差の
                最大は **{max_hessian_error:.3g}**、線形減衰率の診断は **{decay_pass_count}/{len(decay_records)}** recordがthreshold内でpassだった。

                これはsolverの線形化と実装の整合を示す数値ゲートであり、成長中の非線形座屈や
                実験再現を証明するものではない。
                """
            ),
            _figure,
        ]
    )
    return


@app.cell
def _(mo, p1b2_map):
    gb_axis = p1b2_map["axes"]["G_b"]
    chi_axis = p1b2_map["axes"]["chi"]
    gb_options = {f"{value:.6g}": value for value in gb_axis}
    chi_options = {f"{value:.6g}": value for value in chi_axis}
    map_gb_control = mo.ui.dropdown(options=gb_options, value="0.19", label="map上の G_b")
    map_chi_control = mo.ui.dropdown(options=chi_options, value="0.00025", label="map上の chi")
    return map_chi_control, map_gb_control


@app.cell
def _(map_chi_control, map_gb_control, mo, p1b2_map, plt, result_paths):
    map_gb = float(map_gb_control.value)
    map_chi = float(map_chi_control.value)
    map_rows = p1b2_map["rows"]
    selected_row = min(
        map_rows,
        key=lambda row: abs(float(row["actual_G_b"]) - map_gb) + abs(float(row["actual_chi"]) - map_chi) * 100.0,
    )
    _colors = {
        "resolved-straight": "tab:blue",
        "resolved-buckled": "tab:orange",
        "trial-mixed": "tab:purple",
        "numerically-unresolved": "tab:red",
    }

    _figure, _axes = plt.subplots(1, 2, figsize=(12, 4.0), constrained_layout=True)
    _axes[0].imshow(plt.imread(result_paths["p1b2_map"]))
    _axes[0].set_title("保存済み P1B.2 regime map")
    _axes[0].axis("off")
    for _regime, _regime_color in _colors.items():
        _map_subset = [row for row in map_rows if row["regime"] == _regime]
        if _map_subset:
            _axes[1].scatter(
                [float(row["actual_G_b"]) for row in _map_subset],
                [float(row["actual_chi"]) for row in _map_subset],
                s=420,
                marker="s",
                color=_regime_color,
                edgecolor="black",
                linewidth=0.7,
                label=_regime,
            )
    _axes[1].scatter([map_gb], [map_chi], s=180, facecolors="none", edgecolors="black", linewidth=2.5, label="選択点")
    _axes[1].set_xlabel(r"$G_b=g\tau_b$")
    _axes[1].set_ylabel(r"$\chi=EI/(EA L^2)$")
    _axes[1].set_title("compact JSONからの再描画")
    _axes[1].legend(fontsize=8, loc="best")
    _axes[1].grid(alpha=0.2)

    counts = selected_row["aggregate"]["label_counts"]
    mo.vstack(
        [
            mo.md("## 4. Phase 1：非接触 成長–緩和のregime map"),
            mo.hstack([map_gb_control, map_chi_control]),
            mo.md(
                f"""
                **選択 cell**：`G_b={float(selected_row['actual_G_b']):.6g}`, `chi={float(selected_row['actual_chi']):.6g}`<br>
                **分類**：`{selected_row['regime']}`、seed trialの件数（straight / buckled-single / unresolved）=
                **{counts.get('straight', 0)} / {counts.get('buckled-single', 0)} / {counts.get('unresolved', 0)}**

                分類は相転移のラベルではなく、既定の判定規則に基づく compact 集計である。
                `unresolved` を直線または座屈へ読み替えない。
                """
            ),
            _figure,
        ]
    )
    return


@app.cell
def _(mo, p1b2_convergence):
    convergence_rows = [
        {
            "代表点": _record["pilot_role"],
            "判定": _record["status"],
            "run数": _record["n_runs"],
            "ラベル": ", ".join(_record["labels"]),
        }
        for _record in p1b2_convergence["representatives"]
    ]
    convergence_table = mo.ui.table(
        convergence_rows,
        pagination=False,
        selection=None,
        show_data_types=False,
        show_download=False,
        label="P1B.2 convergence summary",
    )
    mo.vstack(
        [
            mo.md(
                """
                ### P1B.2 の読み方：傾向は見えたが、境界域は未解決

                保存済み suite は **87 run**（pilot、解像度・刻み幅の収束、3×3 grid、サイズ比較）
                で構成される。3×3 gridでは、各 cell に5 seedの初期imperfection trialを置いた。
                中央付近には `trial-mixed` が現れ、成長数・剛性比が同じでも初期条件と有限時間の
                影響を無視できないことが分かる。
                """
            ),
            convergence_table,
            mo.md(
                """
                **物理的な解釈**：成長が速く、曲げ緩和が追いつかないほど、横変位が増えやすい
                方向の傾向はある。ただし、boundary-near と buckled-single の代表点は解像度・
                `dt` に対して分類が安定せず、臨界曲線や普遍性はまだ主張できない。ここで重要な
                成果は「座屈境界を決めた」ことではなく、「どこが数値的に解像できていないかを
                明示できた」ことである。
                """
            ),
        ]
    )
    return


@app.cell
def _(mo, video_summary):
    video_options = {report["logical_id"]: report["logical_id"] for report in video_summary["reports"]}
    video_control = mo.ui.dropdown(options=video_options, value=next(iter(video_options)), label="観察入力")
    return (video_control,)


@app.cell
def _(mo, plt, video_control, video_summary):
    selected_report = next(report for report in video_summary["reports"] if report["logical_id"] == video_control.value)
    population = selected_report["population"]
    growth = selected_report["growth"]
    shape = selected_report["shape"]
    _figure, _axes = plt.subplots(1, 2, figsize=(9, 3.4), constrained_layout=True)
    _axes[0].bar(["母集団", "適格", "censor"], [population["population_rows"], population["eligible_rows"], population["censored_rows"]], color=["0.6", "tab:blue", "tab:red"])
    _axes[0].set_title("現時点の動画artifactの分母")
    _axes[0].set_ylabel("frame rows")
    _axes[0].grid(axis="y", alpha=0.25)
    _axes[1].axis("off")
    _axes[1].text(
        0.02,
        0.98,
        "\n".join(
            [
                f"status: {population['status']}",
                f"growth: {growth['status']}",
                f"shape: {shape['status']}",
                "",
                "推定値は表示しない",
                "centerline本体が未同梱",
            ]
        ),
        va="top",
        family="sans-serif",
    )
    mo.vstack(
        [
            mo.md(
                r"""
                ## 5. Phase 2：実観察動画との接続

                観察側では、動画から中心線を抽出した後、分岐・ループ・視野外・追跡飛び・
                skeleton loss を **censor** として残す。適格フレームの輪郭長に対して、
                指数モデル $\log L=\log L_0+gt$ と線形モデルをHuber回帰し、選択モデルの
                $g$ と回帰残差ベースの95%区間を出す。形状の比較は、明示的なpixel/model
                登録がある場合だけ、弧長再サンプリングした離散Fréchet距離と曲率残差を使う。

                しかし、**保存済みの `gray5` / `original` artifactには中心線本体が含まれていない**。
                したがって、ここでの実データに対する `g`、Fréchet距離、曲率残差、`chi`、径の
                推定値は未同定である。これは解析の失敗ではなく、入力データ契約と分母を保った
                結果である。

                - `g` は輪郭長の記述的な傾きであって、真の局所成長則とは限らない。
                - `chi` は単一の受動的中心線からは同定できず、パラメータ付きモデル比較または
                  力–伸長データが必要である。
                - 形が似ることは、物性定数や生物学的 lineage の妥当性を証明しない。
                """
            ),
            video_control,
            _figure,
        ]
    )
    return


@app.cell
def _(contact_rows, mo):
    named_cases = [
        "representative_u_contact",
        "no_contact_control",
        "soft_penalty",
        "stiff_penalty",
        "s_shape_folding",
    ]
    available_cases = [name for name in named_cases if any(row["case"] == name for row in contact_rows)]
    contact_options = {name: name for name in available_cases}
    contact_case_control = mo.ui.dropdown(options=contact_options, value=available_cases[0], label="接触ケース")
    return (contact_case_control,)


@app.cell
def _(contact_case_control, contact_rows, mo, plt, result_paths):
    selected_contact = next(row for row in contact_rows if row["case"] == contact_case_control.value)
    phase_rows = [row for row in contact_rows if row["group"] == "phase_map"]

    _figure, _axes = plt.subplots(1, 2, figsize=(12, 4.0), constrained_layout=True)
    _axes[0].imshow(plt.imread(result_paths["contact_map"]))
    _axes[0].set_title("保存済み P2 contact phase map")
    _axes[0].axis("off")
    diameter_values = sorted({float(row["diameter"]) for row in phase_rows})
    for _diameter in diameter_values:
        _contact_subset = [row for row in phase_rows if float(row["diameter"]) == _diameter]
        _axes[1].scatter(
            [float(row["Pi_c"]) for row in _contact_subset],
            [float(row["max_penetration_ratio"]) for row in _contact_subset],
            s=55,
            label=f"D={_diameter:g}",
            alpha=0.85,
        )
    _axes[1].scatter(
        [float(selected_contact["Pi_c"])],
        [float(selected_contact["max_penetration_ratio"])],
        s=180,
        facecolors="none",
        edgecolors="black",
        linewidth=2.5,
        label="選択ケース",
    )
    _axes[1].set_xlabel(r"$\Pi_c=k_cD^2/EI$")
    _axes[1].set_ylabel("最大貫入率")
    _axes[1].set_title("compact summaryの再描画")
    _axes[1].grid(alpha=0.2)
    _axes[1].legend(fontsize=8)

    mo.vstack(
        [
            mo.md(
                r"""
                ## 6. Phase 2：有限径接触と折りたたみ

                節点間距離だけの接触から、線分間最近接距離に基づく有限径 penalty へ進めた。
                線分対 $(i,j)$ の貫入を $\delta=[D-d_{ij}]_+$、接触エネルギーを
                $E_c=k_c\delta^2/2$ とし、法線力を最近接パラメータ $(s,u)$ で4端点へ
                双線形 scatter する。これにより、径を持つ中心線が接触したまま成長する
                ケースを、接触時刻・接触pair・接触長proxy・貫入・仕事の収支で追跡できる。
                """
            ),
            contact_case_control,
            mo.md(
                f"""
                **選択ケース `{selected_contact['case']}`**<br>
                分類=`{selected_contact['classification']}`、`t_contact={selected_contact['contact_time'] or '—'}`、
                最大貫入率=`{float(selected_contact['max_penetration_ratio']):.3f}`、
                接触長proxy=`{float(selected_contact['max_contact_length']):.3f}`、
                成長仕事=`{float(selected_contact['final_growth_work']):.3g}`、
                散逸仕事=`{float(selected_contact['final_dissipation_work']):.3g}`

                `representative_u_contact` では初期から自己接触があり、有限径拘束によって
                接触pairが維持される。一方、penalty 法は厳密非貫入ではない。保存済み収束
                診断では、`dt` を `0.002` から `0.0005` に変えると最大貫入率が
                `0.378` から `0.059` に変わり、時間刻み依存が明確だった。`k_c` の比較は
                `Pi_c` を増やすほど貫入が減る方向だが、有限 `k_c` の限界は残る。
                """
            ),
            _figure,
        ]
    )
    return


@app.cell
def _(contact_suite, mo):
    contact_convergence = contact_suite["convergence_summary"]
    mo.md(
        f"""
        ### 接触結果をどう読むか

        - **得られた洞察**：有限径接触は、自由に交差できる中心線モデルとは異なり、
          折りたたみ可能な配置の集合を狭める。成長仕事と粘性散逸の競合を測るための
          diagnostics を導入できた。
        - **確認できた数値的傾向**：接触剛性 refinement（`k_c=5→20`）では、比較した
          条件で最大貫入率が `0.0602→0.0585` と大きくは増えなかった。
        - **未解決**：時間刻み refinement は `converged=false`。接触の長時間 coiling、
          複数巻き、摩擦・接着・履歴、厳密な非貫入はまだ扱っていない。

        したがって、phase mapは「折りたたみ相図の確定版」ではなく、有限径・penalty
        ・初期形状・時間刻みをそろえて比較するための診断地図である。
        """
    )
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 7. 結論と研究の展望

    ### 現時点で言えること

    1. **問いを連続体の言葉に置けた**：局所参照長の成長、伸長・曲げエネルギー、
       過減衰、有限径接触を、同じ開曲線モデルとして記述した。
    2. **数値結果の前提を分離できた**：剛体不変性、energy-gradient整合性、
       線形mode、交差棄却、再現性を、成長・座屈・接触の物理解釈より前のゲートに置いた。
    3. **非接触では競合の軸が見えた**：`G_b` と `chi` の小規模gridで、straight、
       resolved buckled、trial-mixed、numerically-unresolved を区別できた。ただし、
       未解決域を座屈境界へ読み替えない。
    4. **観察への接続条件が明確になった**：輪郭長からの成長率、登録済み中心線からの
       Fréchet・曲率残差は計算できるが、現時点の保存artifactには中心線本体がなく、
       実観察の物性同定結果はまだない。
    5. **接触は新しい拘束を持ち込む**：有限径線分 penalty は折りたたみを拘束するが、
       penalty 固有の有限貫入と時間刻み依存を伴う。

    ### 次に目指すこと

    - Barrier法 / C-IPC など、有限時間刻みでも非貫入を保証する接触定式化を比較する。
    - 流路や壁面などの拘束境界で、成長率・曲げ剛性・径が選ぶ成長パターンを調べる。
    - 観察動画の中心線・品質フラグ・時間／pixel校正を完全同期し、calibration用と
      holdout評価用を分けたフィッティングを実施する。
    - 実験では形状だけでなく、輪郭長、曲率分布、接触時間、端点距離、可能なら力・
      伸長応答を同時に測り、「幾何的一致」と「真の物性定数」を分けて検証する。

    **研究としての次の判断**は、より複雑な物理を足すことではなく、まず
    `G_b`–`chi` の境界域、有限径接触の時間刻み依存、実観察の入力欠損を順に解消し、
    どの観測量がモデルを識別しているのかを確かめることである。
    """)
    return


if __name__ == "__main__":
    app.run()
