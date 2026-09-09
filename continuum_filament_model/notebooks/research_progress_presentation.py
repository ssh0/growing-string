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
    presentation_root = results_root / "presentation_data"

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

    dense_heatmap_rows = load_csv(presentation_root / "dense_buckling" / "heatmap.csv")
    dense_heatmap = load_json(presentation_root / "dense_buckling" / "heatmap.json")
    dense_snapshots = load_json(presentation_root / "dense_buckling" / "snapshots.json")
    contact_snapshots = load_json(presentation_root / "contact_snapshots" / "contact_snapshots.json")
    video_presentation = load_json(presentation_root / "video_gray5" / "video_presentation.json")
    video_centerline_rows = load_csv(presentation_root / "video_gray5" / "centerline.csv")
    exploratory_summary = load_json(presentation_root / "exploratory_fitting" / "summary.json")

    result_paths = {
        "p1b2_map": results_root / "p1b2" / "regime_map.png",
        "p1b2_convergence": results_root / "p1b2" / "convergence_summary.png",
        "contact_map": results_root / "contact_buckling" / "phase_map.png",
        "exploratory_fit": presentation_root / "exploratory_fitting" / "best_fit_comparison.png",
    }
    return (
        contact_rows,
        contact_snapshots,
        contact_suite,
        dense_heatmap,
        dense_heatmap_rows,
        dense_snapshots,
        p0b,
        p1b2_convergence,
        p1b2_manifest,
        p1b2_map,
        result_paths,
        video_centerline_rows,
        video_presentation,
        exploratory_summary,
    )


@app.cell
def _(dense_heatmap_rows, exploratory_summary, jp_font_name, mo, p1b2_manifest, p1b2_map, result_paths, video_presentation):
    report_lines = [
        f"- P1B.2 非接触 suite: **{p1b2_manifest['run_count']} run**、seed `{p1b2_manifest['seed_set']}`、3×3 grid",
        f"- P1B.2 regime map: `{result_paths['p1b2_map'].relative_to(result_paths['p1b2_map'].parents[2])}`（各cellは5 seed trial）",
        f"- 高密度座屈相図: `presentation_data/dense_buckling/heatmap.csv`（**{len(dense_heatmap_rows)}条件**、7×8 grid）",
        f"- 実観察 `gray5`: `{video_presentation['status']}`、中心線行を **{video_presentation['validation']['n_rows']}** 行収録",
        f"- 探索的形状fit: `{exploratory_summary['status']}`、候補 **{exploratory_summary['search']['candidate_count']}** 件",
    ]
    mo.md(
        "### このページが参照する成果物\n"
        + "\n".join(report_lines)
        + "\n\n"
        + f"P1B.2 のregime分類は `{len(p1b2_map['rows'])}` cellを対象にしています。動画の中心線は保存済みartifactから読み込み、品質・censorフラグを保持したまま表示します。探索的fitの結果も、同じ保存済みartifactから読み込みます。"
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
            mo.md("## 4A. Phase 1：低密度 pilot 成長–緩和のregime map"),
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
def _(dense_heatmap_rows, mo):
    dense_cell_options = {row["cell"]: row["cell"] for row in dense_heatmap_rows}
    dense_cell_control = mo.ui.dropdown(
        options=dense_cell_options,
        value=next(iter(dense_cell_options)),
        label="高密度相図のセル（A_max/L・mode spectrum・形状を連動表示）",
    )
    return (dense_cell_control,)


@app.cell
def _(dense_cell_control, dense_heatmap, dense_heatmap_rows, dense_snapshots, mo, np, plt):
    dense_rows_by_cell = {row["cell"]: row for row in dense_heatmap_rows}
    selected_dense_row = dense_rows_by_cell[dense_cell_control.value]
    dense_records_by_cell = {str(record["cell"]): record for record in dense_heatmap["records"]}
    selected_dense_record = dense_records_by_cell[selected_dense_row["cell"]]
    _dense_gb_axis = np.asarray(dense_heatmap["axes"]["G_b"], dtype=float)
    _dense_chi_axis = np.asarray(dense_heatmap["axes"]["chi"], dtype=float)

    def _metric_grid(key):
        grid = np.full((len(_dense_chi_axis), len(_dense_gb_axis)), np.nan)
        for row in dense_heatmap_rows:
            i = int(np.where(np.isclose(_dense_chi_axis, float(row["chi"])))[0][0])
            j = int(np.where(np.isclose(_dense_gb_axis, float(row["G_b"])))[0][0])
            grid[i, j] = float(row[key])
        return grid

    _figure, _axes = plt.subplots(1, 2, figsize=(12, 4.4), constrained_layout=True)
    for _axis, _key, _title, _cmap, _vmin, _vmax in (
        (_axes[0], "A_max_over_L", r"最大横変位比 $A_{max}/L$", "viridis", None, None),
        (_axes[1], "first_mode_fraction", "第一モード分率（連続量）", "magma", 0.0, 1.0),
    ):
        _image = _axis.imshow(
            _metric_grid(_key),
            origin="lower",
            aspect="auto",
            interpolation="nearest",
            extent=(_dense_gb_axis[0], _dense_gb_axis[-1], _dense_chi_axis[0], _dense_chi_axis[-1]),
            cmap=_cmap,
            vmin=_vmin,
            vmax=_vmax,
        )
        _axis.scatter(
            [float(row["G_b"]) for row in dense_heatmap_rows if row["label"] == "unresolved"],
            [float(row["chi"]) for row in dense_heatmap_rows if row["label"] == "unresolved"],
            marker="x",
            color="cyan",
            s=55,
            linewidth=2,
            label="unresolved",
        )
        _axis.scatter(
            [float(selected_dense_row["G_b"])],
            [float(selected_dense_row["chi"])],
            facecolors="none",
            edgecolors="white",
            s=170,
            linewidth=2.2,
            label="選択セル",
        )
        _axis.set_xlabel(r"$G_b=g\tau_b$")
        _axis.set_ylabel(r"$\chi=EI/(EA L^2)$")
        _axis.set_xticks(_dense_gb_axis)
        _axis.set_yticks(_dense_chi_axis)
        _axis.tick_params(axis="x", rotation=35)
        _axis.set_title(_title)
        _axis.legend(fontsize=8, loc="best")
        _figure.colorbar(_image, ax=_axis, shrink=0.86)

    _mode_fractions = np.asarray(selected_dense_record["mode_spectrum"]["mode_fractions"], dtype=float)
    _mode_figure, _mode_axis = plt.subplots(figsize=(5.0, 2.8), constrained_layout=True)
    _mode_axis.bar(np.arange(1, len(_mode_fractions) + 1), _mode_fractions, color=["tab:blue"] + ["tab:orange"] * (len(_mode_fractions) - 1))
    _mode_axis.axhline(0.70, color="tab:red", linestyle="--", linewidth=1.2, label="第一モード判定閾値")
    _mode_axis.set_xlabel("mode n")
    _mode_axis.set_ylabel("|A_n| / sqrt(sum |A_n|²)")
    _mode_axis.set_title(f"選択セルの mode spectrum: dominant n={selected_dense_row['dominant_mode']}")
    _mode_axis.set_xticks(np.arange(1, len(_mode_fractions) + 1))
    _mode_axis.legend(fontsize=8)
    _mode_axis.grid(axis="y", alpha=0.2)
    _snapshot_roles_by_cell = {
        case["cell"]: role for role, case in dense_snapshots["cases"].items()
    }
    _snapshot_role = _snapshot_roles_by_cell.get(
        selected_dense_row["cell"],
        "higher_mode" if selected_dense_row["label"] == "unresolved" else "single_buckling",
    )
    _snapshot_case = dense_snapshots["cases"][_snapshot_role]
    _shape_figure, _shape_axes = plt.subplots(1, 3, figsize=(12, 3.6), constrained_layout=True)
    _last_scatter = None
    for _axis, _snapshot in zip(_shape_axes, _snapshot_case["snapshots"]):
        _x = np.asarray(_snapshot["x"], dtype=float)
        _y = np.asarray(_snapshot["y"], dtype=float)
        _curvature = np.zeros(len(_x), dtype=float)
        for _index in range(1, len(_x) - 1):
            _left = np.array([_x[_index] - _x[_index - 1], _y[_index] - _y[_index - 1]])
            _right = np.array([_x[_index + 1] - _x[_index], _y[_index + 1] - _y[_index]])
            _left_norm, _right_norm = np.linalg.norm(_left), np.linalg.norm(_right)
            if _left_norm > 1.0e-12 and _right_norm > 1.0e-12:
                _cross = _left[0] * _right[1] - _left[1] * _right[0]
                _curvature[_index] = abs(np.arctan2(_cross, np.dot(_left, _right))) / (0.5 * (_left_norm + _right_norm))
        _axis.plot(_x, _y, "-", color="0.25", linewidth=1.5, zorder=1)
        _last_scatter = _axis.scatter(_x, _y, c=_curvature, cmap="plasma", s=42, zorder=2)
        _axis.scatter(_x[[0, -1]], _y[[0, -1]], color="black", s=22, zorder=3)
        _axis.set_title(f"{_snapshot['label']}  t={float(_snapshot['time']):.4g}")
        _axis.set_aspect("equal", adjustable="datalim")
        _axis.grid(alpha=0.2)
    if _last_scatter is not None:
        _shape_figure.colorbar(_last_scatter, ax=_shape_axes.tolist(), label="局所曲率の proxy")

    _unresolved = [row for row in dense_heatmap_rows if row["label"] == "unresolved"]
    _higher_mode_count = sum(row["unresolved_reason"] == "higher_mode_wave" for row in _unresolved)
    _failure_count = sum(bool(row["failure_reason"]) for row in _unresolved)
    _mode_detail = dense_heatmap["unresolved_decomposition"]
    _selected_peak_curvature = selected_dense_record.get("peak_max_curvature")
    _selected_peak_text = "—" if _selected_peak_curvature is None else f"{float(_selected_peak_curvature):.3g}"
    mo.vstack(
        [
            mo.md(
                f"""
                ## 4B. Phase 1：高密度座屈相図と `unresolved` の物理解体

                `heatmap.csv` の **{len(dense_heatmap_rows)}条件（$G_b$ 7点 × $\\chi$ 8点）**を、判定ラベルだけでなく
                $A_{{max}}/L$ と第一モード分率の連続値として再描画した。`unresolved` は計算失敗を意味せず、
                第一モード分率 **< 0.70** のため「単一の第一モード座屈」として分類しなかったセルである。

                現在の5セルの内訳は higher-mode wave **{_higher_mode_count}**、mixed-mode **{len(_unresolved) - _higher_mode_count}**、
                failure reason 非空 **{_failure_count}**。higher-mode wave は $n\\ge2$ の成分（代表セルでは dominant mode $n=5$）
                が励起されており、下段の形状スナップショットでは波数の増加と局所曲率の集中を直接確認できる。
                したがって、この `unresolved` は計算制限へ隠すのではなく、**高次モード／混合モードという物理的な応答領域**として開示する。

                選択セル `{selected_dense_row['cell']}`：`{selected_dense_row['unresolved_reason']}`、
                $A_{{max}}/L={float(selected_dense_row['A_max_over_L']):.4f}$、第一モード分率=`{float(selected_dense_row['first_mode_fraction']):.3f}`、
                dominant mode=`{selected_dense_row['dominant_mode']}`、peak curvature=`{_selected_peak_text}`。
                局所曲率 proxy の色は、単一モードの振幅だけでは見えない形状集中を示す。

                - higher-mode wave: {_mode_detail['higher_mode_wave']}
                - mixed-mode: {_mode_detail['mixed_mode']}
                - sub-threshold transient: {_mode_detail['sub_threshold_transient']}
                """
            ),
            dense_cell_control,
            _figure,
            mo.md(f"選択セルの代表形状（snapshot role=`{_snapshot_role}`）と mode spectrum"),
            mo.hstack([_shape_figure, _mode_figure]),
            mo.ui.table(
                [
                    {
                        "cell": row["cell"],
                        "G_b": float(row["G_b"]),
                        "chi": float(row["chi"]),
                        "reason": row["unresolved_reason"],
                        "dominant_mode": int(row["dominant_mode"]),
                        "first_mode_fraction": float(row["first_mode_fraction"]),
                        "A_max_over_L": float(row["A_max_over_L"]),
                        "curvature_rms": float(row["curvature_rms"]),
                    }
                    for row in _unresolved
                ],
                pagination=False,
                selection=None,
                show_data_types=False,
                show_download=False,
                label="unresolved cells（全5条件）",
            ),
        ]
    )
    return


@app.cell
def _(video_presentation, mo):
    _video_representatives = video_presentation.get("representative_frames", [])
    _video_frame_options = {
        f"frame={record['frame']} / t={float(record['time']):g} / {record['filament_id']}": index
        for index, record in enumerate(_video_representatives)
    }
    video_frame_control = mo.ui.dropdown(
        options=_video_frame_options,
        value=next(iter(_video_frame_options)),
        label="gray5 中心線の代表フレーム",
    )
    return (video_frame_control,)


@app.cell
def _(dense_snapshots, mo, np, plt, video_centerline_rows, video_frame_control, video_presentation):
    _video_representatives = video_presentation.get("representative_frames", [])
    _selected_video_record = _video_representatives[int(video_frame_control.value)]
    _centerline_groups = {}
    for _row in video_centerline_rows:
        _key = (int(_row["frame"]), str(_row["filament_id"]))
        _centerline_groups.setdefault(_key, []).append(_row)
    _selected_key = (int(_selected_video_record["frame"]), str(_selected_video_record["filament_id"]))
    _selected_rows = sorted(_centerline_groups.get(_selected_key, []), key=lambda row: int(row["point_id"]))
    if _selected_rows:
        _observed_points = np.asarray([[float(row["x"]), float(row["y"])] for row in _selected_rows], dtype=float)
    else:
        _observed_points = np.column_stack((_selected_video_record["x"], _selected_video_record["y"]))

    _length_rows = video_presentation.get("length_timeseries", [])
    _length_times = np.asarray([float(row["time"]) for row in _length_rows], dtype=float)
    _length_values = np.asarray([float(row["length_px"]) for row in _length_rows], dtype=float)
    _length_censor = np.asarray([int(row.get("censor", 0)) for row in _length_rows], dtype=int)
    _selected_lineage = str(video_presentation.get("selected_filament_id") or "")
    _same_lineage = np.asarray([str(row.get("filament_id", "")) == _selected_lineage for row in _length_rows], dtype=bool)
    _finite_length = np.isfinite(_length_times) & np.isfinite(_length_values) & (_length_values > 0.0)
    _eligible_length = _finite_length & _same_lineage & (_length_censor == 0)
    _fit_count = int(np.count_nonzero(_eligible_length))
    if _fit_count >= 2:
        _growth_slope, _growth_intercept = np.polyfit(_length_times[_eligible_length], np.log(_length_values[_eligible_length]), 1)
        _fit_times = np.linspace(float(np.min(_length_times[_eligible_length])), float(np.max(_length_times[_eligible_length])), 120)
        _fit_lengths = np.exp(_growth_intercept + _growth_slope * _fit_times)
        _growth_text = f"同一lineage（{_selected_lineage}）の適格行だけでlog-linear記述fit: g={float(_growth_slope):.4g} 1/time"
    else:
        _fit_times = np.array([])
        _fit_lengths = np.array([])
        _finite_count = int(np.count_nonzero(_finite_length))
        _censored_count = int(np.count_nonzero(_finite_length & (_length_censor == 1)))
        _unique_lineages = sorted({str(row.get("filament_id", "")) for row in _length_rows})
        _other_lineage_count = sum(
            str(row.get("filament_id", "")) != _selected_lineage for row in _length_rows
        )
        if _fit_count == 0 and _finite_count == _censored_count:
            _growth_text = (
                f"適格観測行なし（全{_finite_count}行がcensor（分岐・ループ等の品質フラグ）、"
                f"{len(_unique_lineages)}種類の候補lineageが混在、選択lineage: {_selected_lineage}、"
                f"選択lineage以外の候補行={_other_lineage_count}行）。"
                "単一フィラメントの成長曲線としてはfit不可"
            )
        else:
            _growth_text = f"適格観測行={_fit_count}<2（同一lineageかつcensor=0の行のみを対象；fit非表示）"

    _profile_groups = {}
    for _row in video_presentation.get("curvature_profile", []):
        _key = (int(_row["frame"]), str(_row["filament_id"]))
        _profile_groups.setdefault(_key, []).append(_row)

    def _align_model_to_observation(model_points, observation_points):
        _model = np.asarray(model_points, dtype=float) - np.asarray(model_points[0], dtype=float)
        _observation = np.asarray(observation_points, dtype=float) - np.asarray(observation_points[0], dtype=float)
        _model_chord = np.linalg.norm(_model[-1])
        _observation_chord = np.linalg.norm(_observation[-1])
        if _model_chord <= 1.0e-12 or _observation_chord <= 1.0e-12:
            return _model, _observation
        _angle = np.arctan2(_observation[-1, 1], _observation[-1, 0]) - np.arctan2(_model[-1, 1], _model[-1, 0])
        _rotation = np.array([[np.cos(_angle), -np.sin(_angle)], [np.sin(_angle), np.cos(_angle)]])
        return (_model @ _rotation.T) * (_observation_chord / _model_chord), _observation

    _model_end = dense_snapshots["cases"]["single_buckling"]["snapshots"][-1]
    _model_points = np.column_stack((_model_end["x"], _model_end["y"]))
    _aligned_model, _aligned_observed = _align_model_to_observation(_model_points, _observed_points)

    _figure, _axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    _axes[0, 0].plot(_observed_points[:, 0], _observed_points[:, 1], "o-", color="tab:blue", ms=3, label="実中心線（CSV）")
    _axes[0, 0].set_title(f"gray5 centerline: frame={_selected_video_record['frame']}, t={float(_selected_video_record['time']):g}")
    _axes[0, 0].set_xlabel("x [pixel]")
    _axes[0, 0].set_ylabel("y [pixel]")
    _axes[0, 0].set_aspect("equal", adjustable="datalim")
    _axes[0, 0].legend(fontsize=8)
    _axes[0, 0].grid(alpha=0.2)

    for _record in _video_representatives:
        _key = (int(_record["frame"]), str(_record["filament_id"]))
        _rows = sorted(_centerline_groups.get(_key, []), key=lambda row: int(row["point_id"]))
        if _rows:
            _x = [float(row["x"]) for row in _rows]
            _y = [float(row["y"]) for row in _rows]
        else:
            _x, _y = _record["x"], _record["y"]
        _axes[0, 1].plot(_x, _y, "-", linewidth=1.5, alpha=0.75, label=f"t={float(_record['time']):g}")
    _axes[0, 1].set_title("代表フレームの中心線時系列")
    _axes[0, 1].set_xlabel("x [pixel]")
    _axes[0, 1].set_ylabel("y [pixel]")
    _axes[0, 1].set_aspect("equal", adjustable="datalim")
    _axes[0, 1].legend(fontsize=7, ncol=2)
    _axes[0, 1].grid(alpha=0.2)

    _axes[1, 0].scatter(
        _length_times[_finite_length & (_length_censor == 0)],
        _length_values[_finite_length & (_length_censor == 0)],
        color="tab:blue",
        s=28,
        label="候補行（censor=0）",
    )
    _axes[1, 0].scatter(
        _length_times[_finite_length & (_length_censor == 1)],
        _length_values[_finite_length & (_length_censor == 1)],
        color="tab:red",
        s=28,
        label="候補行（censor=1）",
    )
    if len(_fit_times):
        _axes[1, 0].plot(_fit_times, _fit_lengths, "k--", linewidth=2, label="同一lineage適格行の記述fit")
    _axes[1, 0].set_title("候補フィラメントの輪郭長 $L(t)$（censor付き；適格fitのみ表示）")
    _axes[1, 0].set_xlabel("time")
    _axes[1, 0].set_ylabel("L [pixel]")
    _axes[1, 0].text(0.02, 0.97, _growth_text + "\n赤点=censorフラグあり", transform=_axes[1, 0].transAxes, va="top", fontsize=8)
    _axes[1, 0].grid(alpha=0.2)

    for _record in _video_representatives:
        _key = (int(_record["frame"]), str(_record["filament_id"]))
        _profile = sorted(_profile_groups.get(_key, []), key=lambda row: int(row["point_id"]))
        if _profile:
            _axes[1, 1].plot(
                [float(row["arc_length_px"]) for row in _profile],
                [float(row["curvature_px_inv"]) for row in _profile],
                linewidth=1.4,
                alpha=0.75,
                label=f"t={float(_record['time']):g}",
            )
    _axes[1, 1].set_title("曲率プロファイル $\\kappa(s)$")
    _axes[1, 1].set_xlabel("arc length s [pixel]")
    _axes[1, 1].set_ylabel(r"$\kappa$ [pixel$^{-1}$]")
    _axes[1, 1].legend(fontsize=7, ncol=2)
    _axes[1, 1].grid(alpha=0.2)

    _comparison_figure, _comparison_axis = plt.subplots(figsize=(6, 4), constrained_layout=True)
    _comparison_axis.plot(_aligned_observed[:, 0], _aligned_observed[:, 1], "-", color="tab:blue", linewidth=2.5, label="実中心線")
    _comparison_axis.plot(_aligned_model[:, 0], _aligned_model[:, 1], "--", color="tab:orange", linewidth=2.2, label="モデル single-buckling t_end")
    _comparison_axis.scatter([0.0, _aligned_observed[-1, 0]], [0.0, _aligned_observed[-1, 1]], color="black", s=18)
    _comparison_axis.set_title("幾何形状の対比（端点整列・長さスケールのみ）")
    _comparison_axis.set_xlabel("aligned x")
    _comparison_axis.set_ylabel("aligned y")
    _comparison_axis.set_aspect("equal", adjustable="datalim")
    _comparison_axis.legend(fontsize=8)
    _comparison_axis.grid(alpha=0.2)

    mo.vstack(
        [
            mo.md(
                f"""
                ## 5. Phase 2：実観察動画 `gray5` の中心線データ

                `video_presentation.json` は status=`{video_presentation['status']}`、
                `raw_centerline_available={video_presentation['raw_centerline_available']}` であり、
                `centerline.csv` の **{len(video_centerline_rows)}行**を実際に読み込んでいる。
                抽出中心線の座標、時系列の輪郭長、曲率を直接表示する。赤点は品質・追跡上の `censor` フラグを持つ行であり、
                データを捨てずにレビュー対象として明示している。

                {_growth_text}。プロットの点は候補フィラメントの輪郭長時系列であり、censor行を除外せず表示する。
                破線は同一lineageかつ `censor=0` の適格行が2件以上ある場合だけ描画し、適格行が不足する場合は物理的な成長率fitを表示しない。
                下のモデル比較は pixel/model の校正・登録なしに端点方向と弦長だけを合わせた**幾何学的対比**である。
                形状の類似は $\\chi$、曲げ剛性、径、摩擦、lineage の同定を意味しない。力学パラメータの同定には、単位校正、品質フラグを考慮した
                holdout、または力–伸長応答など追加観測が必要である。
                """
            ),
            video_frame_control,
            _figure,
            _comparison_figure,
        ]
    )
    return


@app.cell
def _(exploratory_summary, mo, plt, result_paths):
    _best = exploratory_summary["best_fit"]
    _target = _best["target_frame"]
    _baseline = exploratory_summary["baseline"]
    _improvement = exploratory_summary["improvement"]
    _temporal = _best["temporal_features"]
    _figure, _axis = plt.subplots(figsize=(10, 6), constrained_layout=True)
    _axis.imshow(plt.imread(result_paths["exploratory_fit"]))
    _axis.axis("off")
    _axis.set_title("保存済み exploratory fitting comparison")
    mo.vstack(
        [
            mo.md(
                f"""
                ## 5A. 実データへの探索的フィッティングと視覚的整合性

                `exploratory_shape_fitting.py` は、`gray5` の代表フレームを弧長再サンプルし、
                節点数 **{exploratory_summary['search']['n_nodes']}** の連続体モデルを
                `G_b`、`chi`、モデル成長時間の grid で探索した。以下は、選択された
                target frame（`t={_target['time']:g}s`, `frame={_target['frame']}`）について、
                一様スケールと端点方向だけを合わせた重ね合わせである。最新のtarget形状を
                滑らかな初期seedにも使うため、これは予測的な動画再現ではなく、観察形状の
                周囲で力学パラメータを探索する比較である。

                **最良候補**: `G_b={_best['G_b']:.5g}`, `chi={_best['chi']:.5g}`,
                `growth_time={_best['growth_time']:.5g}`、
                **Fréchet距離={_target['frechet_distance_px']:.4g} px**、
                **曲率RMSE={_target['curvature_rmse_px_inv']:.4g} px⁻¹**、
                **特徴量損失={_target['feature_loss']:.4g}**。

                図の下段には、`Amax/chord`、`L/chord`、無次元曲率統計、低次モード比の
                観察／モデル比較と、時刻に対する特徴量の変化を並べた。成長率は
                `d(log L)/dt` として観察={_temporal['observed_growth_rate']:.4g}、
                モデル={_temporal['model_growth_rate']:.4g} と評価している。
                直線中心線 baseline と比較した Fréchet 距離の減少は
                **{_improvement['frechet_distance_px_reduction']:.4g} px
                ({_improvement['frechet_distance_percent']:.2f}%)**、
                特徴量損失の減少は **{_improvement['feature_loss_reduction']:.4g}** である。
                これは視覚的な形状整合性を示す探索指標であり、pixel/model校正の推定、
                holdout評価、物性値の同定ではない。

                選択フレームの `censor` は **{_target['censor']}**、品質フラグは
                `{_target['quality_flags']}` である。したがって、図が重なって見えることと、
                動画追跡・物理モデルが検証済みであることを分けて読む。
                """
            ),
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
def _(contact_snapshots, dense_snapshots, mo):
    _shape_options = {
        "直線維持（dense）": "dense:straight",
        "単一座屈（dense）": "dense:single_buckling",
        "高次モード波（dense）": "dense:higher_mode",
        "接触・U形（contact）": "contact:u_self_contact",
        "接触・折りたたみ（contact）": "contact:s_contact_folding",
    }
    shape_case_control = mo.ui.dropdown(
        options=_shape_options,
        value=next(iter(_shape_options)),
        label="実空間挙動のレジーム／ケース",
    )
    return (shape_case_control,)


@app.cell
def _(contact_snapshots, dense_snapshots, mo, np, plt, shape_case_control):
    _kind, _case_key = str(shape_case_control.value).split(":", 1)
    if _kind == "dense":
        _case = dense_snapshots["cases"][_case_key]
        _case_title = f"dense / {_case_key} / {_case['cell']}"
    else:
        _case = next(case for case in contact_snapshots["cases"] if case["name"] == _case_key)
        _case_title = f"contact / {_case['role']}"

    _figure, _axes = plt.subplots(1, 3, figsize=(12, 3.8), constrained_layout=True)
    _force_figure = None
    for _axis, _snapshot in zip(_axes, _case["snapshots"]):
        _x = np.asarray(_snapshot["x"], dtype=float)
        _y = np.asarray(_snapshot["y"], dtype=float)
        _axis.plot(_x, _y, "o-", color="tab:blue", linewidth=2, ms=3, label="filament")
        _axis.scatter(_x[[0, -1]], _y[[0, -1]], color="black", s=22, zorder=3)
        if _kind == "contact":
            _contacts = _snapshot.get("contacts", [])
            if _contacts:
                _contact_points = np.asarray([contact["contact_point"] for contact in _contacts], dtype=float)
                _forces = np.asarray([contact["normal_force"] for contact in _contacts], dtype=float)
                _magnitudes = np.asarray([contact["normal_force_magnitude"] for contact in _contacts], dtype=float)
                _axis.scatter(
                    _contact_points[:, 0],
                    _contact_points[:, 1],
                    c=_magnitudes,
                    cmap="Reds",
                    s=55,
                    edgecolor="black",
                    linewidth=0.5,
                    label="接触点（色=|F_n|）",
                    zorder=4,
                )
                _axis.quiver(
                    _contact_points[:, 0],
                    _contact_points[:, 1],
                    _forces[:, 0],
                    _forces[:, 1],
                    color="tab:red",
                    angles="xy",
                    scale_units="xy",
                    scale=1.0,
                    width=0.006,
                    zorder=5,
                )
        _axis.set_title(f"{_snapshot['label']}  t={float(_snapshot['time']):.4g}")
        _axis.set_aspect("equal", adjustable="datalim")
        _axis.grid(alpha=0.2)
        if _kind == "contact":
            _axis.legend(fontsize=7, loc="best")

    if _kind == "contact":
        _force_figure, _force_axes = plt.subplots(1, 3, figsize=(12, 2.8), constrained_layout=True)
        for _axis, _snapshot in zip(_force_axes, _case["snapshots"]):
            _contacts = _snapshot.get("contacts", [])
            _values = [float(contact["normal_force_magnitude"]) for contact in _contacts]
            _axis.bar(np.arange(len(_values)), _values, color="tab:red", alpha=0.8)
            _axis.set_title(f"{_snapshot['label']}: {len(_values)} contact pairs")
            _axis.set_xlabel("contact pair")
            _axis.set_ylabel(r"$|F_n|$")
            _axis.grid(axis="y", alpha=0.25)

    _contact_note = ""
    if _kind == "contact":
        _contact_note = "接触点は赤、矢印は法線反発力 $F_n$、棒グラフは各接触pairの $|F_n|$ 分布。"
    mo.vstack(
        [
            mo.md(
                f"""
                ## 6. 実空間挙動：伸び・座屈・高次波・接触折りたたみ

                ケース **{_case_title}** の `t0`、`t_mid`、`t_end` を、軌跡の数値ではなく $x,y$ 形状として直接表示する。
                dense snapshot は非接触の形態変化、contact snapshot は有限径拘束下の形態変化を表す。
                {_contact_note}
                """
            ),
            shape_case_control,
            _figure,
            *([] if _force_figure is None else [_force_figure]),
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
    4. **観察への接続条件が明確になった**：抽出済み中心線から輪郭長の記述的成長率、
       形状・曲率プロファイルを直接可視化できる。一方、pixel/model校正なしの幾何比較だけでは
       $\\chi$、曲げ剛性、径などの力学パラメータ同定には到達しない。
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
    `G_b`–`chi` の境界域、有限径接触の時間刻み依存、実観察の校正・品質管理を順に確認し、
    どの観測量がモデルを識別しているのかを確かめることである。
    """)
    return


if __name__ == "__main__":
    app.run()
