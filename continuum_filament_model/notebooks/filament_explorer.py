import marimo

__generated_with = "0.23.6"
app = marimo.App(width="full")


@app.cell
def _():
    import marimo as mo

    mo.md(
        r"""
        # 成長フィラメント視覚エクスプローラ

        **このノートブックは視覚的な探索用です。** ここで表示する結果は、物理的妥当性・数値収束性・実験再現性を証明しません。乱数は導入していないため、同じ設定は決定論的に再実行できます。

        このノートブックはコアモデルを複製せず、`continuum_filament_model/src` の
        `growing_filament.model`、`growing_filament.observables`、
        `growing_filament.geometry`、`growing_filament.reproducibility` を呼び出します。
        リポジトリルートから次のコマンドで起動してください。

        ```bash
        marimo edit continuum_filament_model/notebooks/filament_explorer.py
        ```

        プリセットを選び、必要なパラメータを変更してから実行ボタンを押してください。
        送信のたびに軌跡と中心線アニメーションを再生成します。アニメーションの再生ボタン・
        スライダー、または下の「表示フレーム」スライダーで、初期・中間・最終状態を確認できます。
        計算フレーム数・描画フレーム数・埋め込みHTMLサイズには上限があります。
        実行セルには計算負荷の上限があり、不正な入力・モデルの `RuntimeError`・可視化依存不足は原因とともに表示します。
        `crossing_rejection` の `RuntimeError` は、交差棄却を確認するための期待される診断です。
        """
    )
    return (mo,)


@app.cell
def _(mo):
    import json
    import sys
    from pathlib import Path

    import numpy as np

    repo_root = Path.cwd()
    source_dir = repo_root / "continuum_filament_model" / "src"
    if not source_dir.is_dir():
        raise RuntimeError(
            "リポジトリルートから起動してください: "
            f"{source_dir} が見つかりません。現在のcwd={repo_root}"
        )
    if str(source_dir) not in sys.path:
        sys.path.insert(0, str(source_dir))

    from growing_filament.geometry import geometry_diagnostics
    from growing_filament.model import (
        FilamentState,
        ModelError,
        ModelParameters,
        OverdampedGrowingFilament,
        straight_state,
    )
    from growing_filament.observables import (
        arc_length_weighted_radius_of_gyration,
        contour_length,
        discrete_curvature,
        summary as observable_summary,
    )
    from growing_filament.reproducibility import compare_reproducibility

    try:
        import matplotlib.pyplot as plt
    except Exception as exc:  # 表示セルで原因を明示するため保持する
        plt = None
        matplotlib_error = exc
    else:
        matplotlib_error = None

    mo.md(
        f"ライブラリimport経路: `{source_dir}`。"
        + (
            " matplotlib は利用可能です。"
            if matplotlib_error is None
            else f" matplotlib は利用できません: `{type(matplotlib_error).__name__}: {matplotlib_error}`"
        )
    )
    return (
        FilamentState,
        ModelError,
        ModelParameters,
        OverdampedGrowingFilament,
        arc_length_weighted_radius_of_gyration,
        compare_reproducibility,
        contour_length,
        discrete_curvature,
        geometry_diagnostics,
        json,
        matplotlib_error,
        np,
        plt,
        straight_state,
    )


@app.cell
def _(
    FilamentState,
    ModelError,
    ModelParameters,
    OverdampedGrowingFilament,
    json,
    np,
    straight_state,
):
    import traceback

    MAX_STEPS = 1200
    MAX_ESTIMATED_NODES = 180
    MAX_DRAW_FRAMES = 90
    MAX_ANIMATION_HTML_BYTES = 1_500_000

    DEFAULT_CONFIG = {
        "preset": "straight",
        "use_fixture_defaults": True,
        "n_nodes": 16,
        "spacing": 0.75,
        "axial_stiffness": 100.0,
        "bending_stiffness": 1.0,
        "drag_density": 1.0,
        "growth_rate": 0.0,
        "dt": 0.005,
        "t_end": 0.2,
        "a_max": 1.5,
        "contact_stiffness": 0.0,
        "diameter": 0.0,
        "fixed_left": False,
        "fixed_right": False,
        "reject_crossing": True,
    }

    PRESET_LABELS = {
        "straight": "直線（成長なし）",
        "perturbed_fixed_growth": "微小摂動・固定端・成長あり",
        "u_shape": "U字",
        "s_shape": "S字",
        "contact": "接触fixture",
        "crossing_rejection": "交差棄却fixture",
    }

    # 推奨値はフォームの値を黙って変更するのではなく、実効設定として表示する。
    # 「プリセット推奨値を適用」を無効にすればフォーム入力をすべて試せる。
    FIXTURE_DEFAULTS = {
        "straight": {
            "n_nodes": 16,
            "spacing": 0.75,
            "growth_rate": 0.0,
            "dt": 0.005,
            "t_end": 0.2,
            "a_max": 1.5,
            "contact_stiffness": 0.0,
            "diameter": 0.0,
            "fixed_left": False,
            "fixed_right": False,
        },
        "perturbed_fixed_growth": {
            "n_nodes": 18,
            "spacing": 0.5,
            "bending_stiffness": 2.0,
            "growth_rate": 0.15,
            "dt": 0.001,
            "t_end": 0.2,
            "a_max": 0.8,
            "fixed_left": True,
            "fixed_right": True,
            "contact_stiffness": 0.0,
            "diameter": 0.0,
        },
        "u_shape": {
            "n_nodes": 21,
            "spacing": 0.5,
            "bending_stiffness": 1.0,
            "growth_rate": 0.0,
            "dt": 0.002,
            "t_end": 0.1,
            "a_max": 1.0,
            "fixed_left": False,
            "fixed_right": False,
            "contact_stiffness": 0.0,
            "diameter": 0.0,
        },
        "s_shape": {
            "n_nodes": 25,
            "spacing": 0.4,
            "bending_stiffness": 1.0,
            "growth_rate": 0.0,
            "dt": 0.001,
            "t_end": 0.1,
            "a_max": 0.8,
            "fixed_left": False,
            "fixed_right": False,
            "contact_stiffness": 0.0,
            "diameter": 0.0,
        },
        "contact": {
            "n_nodes": 3,
            "spacing": 1.0,
            "axial_stiffness": 4.0,
            "bending_stiffness": 0.8,
            "contact_stiffness": 3.0,
            "diameter": 1.0,
            "growth_rate": 0.0,
            "dt": 0.001,
            "t_end": 0.05,
            "a_max": 1.5,
            "fixed_left": False,
            "fixed_right": False,
        },
        "crossing_rejection": {
            "n_nodes": 4,
            "spacing": 1.0,
            "axial_stiffness": 1.0,
            "bending_stiffness": 10.0,
            "drag_density": 1.0,
            "growth_rate": 0.0,
            "dt": 0.2,
            "t_end": 0.2,
            "a_max": 10.0,
            "contact_stiffness": 0.0,
            "diameter": 0.0,
            "fixed_left": False,
            "fixed_right": False,
            "reject_crossing": True,
        },
    }

    def effective_config(raw_config):
        config = dict(DEFAULT_CONFIG)
        config.update(dict(raw_config))
        overrides = {}
        preset = str(config["preset"])
        label_to_preset = {label: key for key, label in PRESET_LABELS.items()}
        preset = label_to_preset.get(preset, preset)
        if preset not in FIXTURE_DEFAULTS:
            raise ModelError(f"unknown preset: {preset!r}")
        if bool(config.get("use_fixture_defaults", True)):
            for key, value in FIXTURE_DEFAULTS[preset].items():
                if config.get(key) != value:
                    overrides[key] = {"input": config.get(key), "effective": value}
                config[key] = value
        config["preset"] = preset
        return config, overrides

    def _geometric_rest_lengths(positions):
        lengths = np.linalg.norm(np.diff(positions, axis=0), axis=1)
        return FilamentState(positions, lengths)

    def make_initial_state(config):
        preset = config["preset"]
        n_nodes = int(config["n_nodes"])
        spacing = float(config["spacing"])
        if preset == "straight":
            return straight_state(n_nodes, spacing=spacing)
        if preset == "perturbed_fixed_growth":
            x = np.arange(n_nodes, dtype=float) * spacing
            phase = np.linspace(0.0, 2.0 * np.pi, n_nodes)
            positions = np.column_stack((x, 0.08 * spacing * np.sin(phase)))
            return _geometric_rest_lengths(positions)
        if preset == "u_shape":
            parameter = np.linspace(-1.0, 1.0, n_nodes)
            x = (parameter + 1.0) * spacing * (n_nodes - 1) / 2.0
            y = 1.4 * spacing * parameter**2
            return _geometric_rest_lengths(np.column_stack((x, y)))
        if preset == "s_shape":
            x = np.arange(n_nodes, dtype=float) * spacing
            parameter = np.linspace(0.0, 1.0, n_nodes)
            y = 0.8 * spacing * np.sin(2.0 * np.pi * parameter)
            return _geometric_rest_lengths(np.column_stack((x, y)))
        if preset == "contact":
            positions = np.asarray(
                [[0.0, 0.0], [0.4, 0.7], [0.8, 0.0]],
                dtype=float,
            )
            return _geometric_rest_lengths(positions)
        if preset == "crossing_rejection":
            positions = np.asarray(
                [[0.0, 0.0], [1.0, 1.0], [0.0, 2.0], [-1.0, 1.1]],
                dtype=float,
            )
            rest_lengths = np.asarray(
                [np.sqrt(2.0), np.sqrt(2.0), 1.465],
                dtype=float,
            )
            return FilamentState(positions, rest_lengths)
        raise ModelError(f"unknown preset: {preset!r}")

    def make_parameters(config):
        preset = config["preset"]
        fixture_crossing = (
            preset == "crossing_rejection"
            and bool(config.get("use_fixture_defaults", True))
        )
        return ModelParameters(
            axial_stiffness=float(config["axial_stiffness"]),
            bending_stiffness=float(config["bending_stiffness"]),
            drag_density=float(config["drag_density"]),
            growth_rate=float(config["growth_rate"]),
            reference_length=float(config["spacing"]),
            dt=float(config["dt"]),
            t_end=float(config["t_end"]),
            a_max=float(config["a_max"]),
            contact_stiffness=float(config["contact_stiffness"]),
            diameter=float(config["diameter"]),
            fixed_left=bool(config["fixed_left"]),
            fixed_right=bool(config["fixed_right"]),
            reject_crossing=bool(config["reject_crossing"]),
            # The crossing fixture deliberately makes one trial and reports its
            # rejection. Other exploratory runs get a small retry budget.
            max_retries=0 if fixture_crossing else 4,
            max_displacement_fraction=1.0 if fixture_crossing else 0.25,
            dt_min=1.0e-10,
        )

    def validate_budget(state, parameters):
        parameters.validate()
        estimated_steps = int(np.ceil(parameters.t_end / parameters.dt))
        if estimated_steps > MAX_STEPS:
            raise ModelError(
                f"計算負荷上限: t_end/dt={estimated_steps} steps > {MAX_STEPS}"
            )
        growth_end = float(
            np.max(state.rest_lengths) * np.exp(parameters.growth_rate * parameters.t_end)
        )
        ratio = max(growth_end / parameters.a_max, 1.0)
        split_level = int(np.ceil(np.log2(ratio))) if ratio > 1.0 else 0
        estimated_split_factor = 2**split_level
        estimated_nodes = 1 + (state.n_nodes - 1) * estimated_split_factor
        if estimated_nodes > MAX_ESTIMATED_NODES:
            raise ModelError(
                "計算負荷上限: 推定再メッシュ後節点数 "
                f"{estimated_nodes} > {MAX_ESTIMATED_NODES}"
            )
        return {
            "estimated_steps": estimated_steps,
            "estimated_nodes": estimated_nodes,
        }

    def error_text(exc):
        detail = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
        if isinstance(exc, ModelError) and exc.event is not None:
            detail += "\nModelError.event:\n" + json.dumps(
                exc.event,
                ensure_ascii=False,
                indent=2,
                default=str,
            )
        return detail

    def execute_once(raw_config):
        config, overrides = effective_config(raw_config)
        state = None
        parameters = None
        model = None
        trajectory = []
        error = None
        error_trace = None
        budget = None
        try:
            state = make_initial_state(config)
            parameters = make_parameters(config)
            budget = validate_budget(state, parameters)
            model = OverdampedGrowingFilament(state, parameters)
            trajectory = model.run()
        except Exception as exc:  # ノートブック上で型名・tracebackを表示する
            error = exc
            error_trace = error_text(exc)
            if model is not None:
                trajectory = [model.initial_state.copy()]
        manifest = None
        if model is not None:
            manifest = model.run_manifest(
                metadata={
                    "notebook": "filament_explorer",
                    "preset": config["preset"],
                    "effective_config": config,
                }
            )
        return {
            "raw_config": dict(raw_config),
            "effective_config": config,
            "overrides": overrides,
            "state": state,
            "parameters": parameters,
            "model": model,
            "trajectory": trajectory,
            "error": error,
            "error_trace": error_trace,
            "budget": budget,
            "manifest": manifest,
        }

    def markdown_table(headers, rows):
        lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join("---" for _ in headers) + " |"]
        lines.extend("| " + " | ".join(str(value) for value in row) + " |" for row in rows)
        return "\n".join(lines)

    return (
        DEFAULT_CONFIG,
        MAX_ANIMATION_HTML_BYTES,
        MAX_DRAW_FRAMES,
        PRESET_LABELS,
        execute_once,
        markdown_table,
    )


@app.cell
def _(DEFAULT_CONFIG, PRESET_LABELS, mo):
    controls = mo.ui.dictionary(
        {
            "preset": mo.ui.dropdown(
                options=PRESET_LABELS,
                value=DEFAULT_CONFIG["preset"],
                label="初期形状 / fixture",
            ),
            "use_fixture_defaults": mo.ui.checkbox(
                value=True,
                label="プリセット推奨値を適用（全入力値を試す場合は無効）",
            ),
            "n_nodes": mo.ui.slider(
                3,
                50,
                value=DEFAULT_CONFIG["n_nodes"],
                step=1,
                show_value=True,
                label="n_nodes",
            ),
            "spacing": mo.ui.number(
                0.05,
                3.0,
                value=DEFAULT_CONFIG["spacing"],
                step=0.05,
                label="spacing",
            ),
            "axial_stiffness": mo.ui.number(
                0.01,
                300.0,
                value=DEFAULT_CONFIG["axial_stiffness"],
                step=1.0,
                label="axial_stiffness",
            ),
            "bending_stiffness": mo.ui.number(
                0.01,
                100.0,
                value=DEFAULT_CONFIG["bending_stiffness"],
                step=0.1,
                label="bending_stiffness",
            ),
            "drag_density": mo.ui.number(
                0.01,
                100.0,
                value=DEFAULT_CONFIG["drag_density"],
                step=0.1,
                label="drag_density",
            ),
            "growth_rate": mo.ui.number(
                0.0,
                5.0,
                value=DEFAULT_CONFIG["growth_rate"],
                step=0.01,
                label="growth_rate",
            ),
            "dt": mo.ui.number(
                1.0e-5,
                0.05,
                value=DEFAULT_CONFIG["dt"],
                step=1.0e-4,
                label="dt",
            ),
            "t_end": mo.ui.number(
                0.001,
                2.0,
                value=DEFAULT_CONFIG["t_end"],
                step=0.01,
                label="t_end",
            ),
            "a_max": mo.ui.number(
                0.05,
                3.0,
                value=DEFAULT_CONFIG["a_max"],
                step=0.05,
                label="a_max",
            ),
            "contact_stiffness": mo.ui.number(
                0.0,
                100.0,
                value=DEFAULT_CONFIG["contact_stiffness"],
                step=0.1,
                label="contact_stiffness",
            ),
            "diameter": mo.ui.number(
                0.0,
                3.0,
                value=DEFAULT_CONFIG["diameter"],
                step=0.05,
                label="diameter",
            ),
            "fixed_left": mo.ui.checkbox(
                value=DEFAULT_CONFIG["fixed_left"],
                label="左端を固定",
            ),
            "fixed_right": mo.ui.checkbox(
                value=DEFAULT_CONFIG["fixed_right"],
                label="右端を固定",
            ),
            "reject_crossing": mo.ui.checkbox(
                value=DEFAULT_CONFIG["reject_crossing"],
                label="交差を棄却",
            ),
        },
        label="シミュレーション設定",
    )
    config_form = mo.ui.form(
        controls,
        submit_button_label="この設定で実行",
        bordered=True,
    )
    mo.vstack(
        [
            mo.md(
                "### パラメータ\n"
                "変更後に送信すると、シミュレーションと決定論的な再実行比較を行います。"
            ),
            config_form,
        ]
    )
    return (config_form,)


@app.cell
def _(
    DEFAULT_CONFIG,
    PRESET_LABELS,
    compare_reproducibility,
    config_form,
    execute_once,
    json,
    mo,
):
    raw_config = config_form.value or dict(DEFAULT_CONFIG)
    first_result = execute_once(raw_config)
    replay_result = None
    replay_comparison = None
    if first_result["error"] is None:
        replay_result = execute_once(raw_config)
        if replay_result["error"] is None:
            replay_comparison = compare_reproducibility(
                first_result["manifest"], replay_result["manifest"]
            )
    run_result = first_result
    run_result["replay_result"] = replay_result
    run_result["replay_comparison"] = replay_comparison

    preset_name = run_result["effective_config"]["preset"]
    status_lines = [
        f"**実効fixture:** `{PRESET_LABELS[preset_name]}` (`{preset_name}`)",
    ]
    if run_result["error"] is None:
        status_lines.append(
            f"✅ 実行完了: accepted={run_result['model'].accepted_steps}, "
            f"rejected={run_result['model'].rejected_steps}, "
            f"states={len(run_result['trajectory'])}"
        )
    else:
        exc = run_result["error"]
        status_lines.append(f"⚠️ `{type(exc).__name__}: {exc}`")
        status_lines.append(
            "エラーは握りつぶしていません。下の診断欄にtracebackと、利用可能なら構造化イベントを表示します。"
        )
    if run_result["budget"] is not None:
        status_lines.append(
            "負荷見積り: "
            + json.dumps(run_result["budget"], ensure_ascii=False, sort_keys=True)
        )
    if run_result["overrides"]:
        status_lines.append(
            "プリセットによる実効値の上書き: "
            + json.dumps(run_result["overrides"], ensure_ascii=False, sort_keys=True)
        )
    if replay_comparison is not None:
        match_text = "一致" if replay_comparison["match"] else "不一致"
        status_lines.append(f"再現性比較: **{match_text}**")
        status_lines.append(
            "canonical state hash: "
            f"`{first_result['manifest']['canonical_state_hash']}`"
        )
        status_lines.append(
            "event count: "
            f"`{first_result['manifest'].get('event_count', 0)}` "
            "（canonical state hashとイベント列を比較済み。詳細は下のmanifest欄）"
        )

    if run_result["trajectory"]:
        frame_selector = mo.ui.slider(
            0,
            len(run_result["trajectory"]) - 1,
            value=len(run_result["trajectory"]) - 1,
            step=1,
            include_input=True,
            show_value=True,
            label="表示フレーム（accepted state index）",
        )
        selector_view = mo.vstack(
            [mo.md("### 表示フレーム"), frame_selector]
        )
    else:
        frame_selector = mo.ui.slider(
            0,
            0,
            value=0,
            step=1,
            show_value=True,
            label="表示フレーム（実行状態なし）",
        )
        selector_view = mo.md("表示可能なaccepted stateがありません。")

    status_view = mo.vstack([mo.md("\n".join(status_lines)), selector_view])
    status_view
    return frame_selector, run_result


@app.cell
def _(
    MAX_ANIMATION_HTML_BYTES,
    MAX_DRAW_FRAMES,
    arc_length_weighted_radius_of_gyration,
    contour_length,
    discrete_curvature,
    frame_selector,
    geometry_diagnostics,
    markdown_table,
    matplotlib_error,
    mo,
    np,
    plt,
    run_result,
):
    from collections import Counter

    if not run_result["trajectory"]:
        if run_result["error_trace"]:
            error_view = mo.md(
                "### 実行診断\n"
                "```text\n"
                + run_result["error_trace"]
                + "\n```"
            )
        else:
            error_view = mo.md("### 実行診断\n結果状態がありません。")
        display_view = error_view
        report_view = error_view
    elif plt is None:
        display_view = mo.vstack(
            [
                mo.md("### 可視化できません"),
                mo.md(
                    "matplotlibのimportに失敗しました。シミュレーション自体は実行できますが、"
                    f"可視化依存を確認してください: `{type(matplotlib_error).__name__}: {matplotlib_error}`"
                ),
            ]
        )
        report_view = display_view
    else:
        trajectory = run_result["trajectory"]
        model = run_result["model"]
        config = run_result["effective_config"]
        frame_index = int(np.clip(int(frame_selector.value), 0, len(trajectory) - 1))
        selected_state = trajectory[frame_index]
        selected_energy = model.energy_components(
            selected_state.positions,
            selected_state.rest_lengths,
        )
        selected_geometry = geometry_diagnostics(
            selected_state.positions,
            contact_distance=float(config["diameter"]),
        )
        selected_curvature = discrete_curvature(selected_state)

        times = np.asarray([state.time for state in trajectory], dtype=float)
        energies = {
            name: np.asarray(
                [
                    model.energy_components(state.positions, state.rest_lengths)[name]
                    for state in trajectory
                ],
                dtype=float,
            )
            for name in ("stretch", "bend", "contact")
        }
        energies["total"] = sum(energies.values())
        references = np.asarray(
            [float(np.sum(state.rest_lengths)) for state in trajectory], dtype=float
        )
        contours = np.asarray([contour_length(state) for state in trajectory], dtype=float)
        radii = np.asarray(
            [arc_length_weighted_radius_of_gyration(state) for state in trajectory],
            dtype=float,
        )
        curvatures = np.asarray(
            [
                float(np.max(discrete_curvature(state)))
                if len(discrete_curvature(state))
                else 0.0
                for state in trajectory
            ],
            dtype=float,
        )
        fig_visual, axes_visual = plt.subplots(2, 2, figsize=(12, 8))
        ax_centerline = axes_visual[0, 0]
        positions = selected_state.positions
        ax_centerline.plot(
            positions[:, 0],
            positions[:, 1],
            "-o",
            lw=2,
            ms=4,
            label=f"t={selected_state.time:.6g}, nodes={selected_state.n_nodes}",
        )
        ax_centerline.scatter(positions[0, 0], positions[0, 1], s=70, label="left anchor")
        ax_centerline.scatter(positions[-1, 0], positions[-1, 1], s=70, label="right anchor")
        ax_centerline.set_title("中心線（選択フレーム）")
        ax_centerline.set_xlabel("x")
        ax_centerline.set_ylabel("y")
        x_span = max(float(np.ptp(positions[:, 0])), 1.0)
        y_span = float(np.ptp(positions[:, 1]))
        if y_span <= 1.0e-12:
            # A perfectly straight fixture has zero y-range; explicit limits
            # keep the 2-D centerline panel visible in Matplotlib/tight_layout.
            x_min = float(np.min(positions[:, 0]))
            x_max = float(np.max(positions[:, 0]))
            ax_centerline.set_xlim(x_min - 0.05 * x_span, x_max + 0.05 * x_span)
            ax_centerline.set_ylim(-0.15 * x_span, 0.15 * x_span)
        else:
            ax_centerline.set_aspect("equal", adjustable="box")
        ax_centerline.grid(alpha=0.25)
        ax_centerline.legend(loc="best")

        ax_energy = axes_visual[0, 1]
        for name, values in energies.items():
            ax_energy.plot(times, values, label=name)
        ax_energy.set_title("エネルギー成分")
        ax_energy.set_xlabel("time")
        ax_energy.set_ylabel("energy")
        ax_energy.grid(alpha=0.25)
        ax_energy.legend(loc="best")

        ax_lengths = axes_visual[1, 0]
        ax_lengths.plot(times, references, label="reference length")
        ax_lengths.plot(times, contours, label="contour length")
        ax_lengths.set_title("長さ・節点数")
        ax_lengths.set_xlabel("time")
        ax_lengths.set_ylabel("length")
        ax_lengths.grid(alpha=0.25)
        ax_lengths.legend(loc="best")

        ax_shape = axes_visual[1, 1]
        ax_shape.plot(times, radii, label="arc-length radius")
        ax_shape.plot(times, curvatures, label="max curvature")
        ax_shape.set_title("形状観測量")
        ax_shape.set_xlabel("time")
        ax_shape.set_ylabel("value")
        ax_shape.grid(alpha=0.25)
        ax_shape.legend(loc="best")
        fig_visual.tight_layout()

        animation_html = None
        animation_size_bytes = 0
        animation_error = None
        draw_indices = np.asarray([], dtype=int)
        try:
            from matplotlib.animation import FuncAnimation

            def sampled_indices(n_frames, max_frames):
                count = min(int(n_frames), int(max_frames))
                if count <= 0:
                    return np.asarray([], dtype=int)
                if count == 1:
                    return np.asarray([0], dtype=int)
                return np.unique(
                    np.linspace(0, n_frames - 1, count, dtype=int)
                )

            def render_centerline_animation(indices):
                animation_figure, animation_axis = plt.subplots(figsize=(7, 5))
                animation_positions = np.concatenate(
                    [trajectory[index].positions for index in indices], axis=0
                )
                x_min, y_min = np.min(animation_positions, axis=0)
                x_max, y_max = np.max(animation_positions, axis=0)
                span = max(float(x_max - x_min), float(y_max - y_min), 1.0)
                margin = 0.05 * span
                animation_axis.set_xlim(x_min - margin, x_max + margin)
                animation_axis.set_ylim(y_min - margin, y_max + margin)
                animation_axis.set_aspect("equal", adjustable="box")
                animation_axis.set_xlabel("x")
                animation_axis.set_ylabel("y")
                animation_axis.set_title("中心線の時間発展")
                animation_axis.grid(alpha=0.25)
                line, = animation_axis.plot(
                    [], [], "-o", lw=2, ms=3, label="centerline"
                )
                left_anchor, = animation_axis.plot(
                    [], [], "o", ms=7, label="left endpoint"
                )
                right_anchor, = animation_axis.plot(
                    [], [], "o", ms=7, label="right endpoint"
                )
                time_label = animation_axis.text(
                    0.02,
                    0.98,
                    "",
                    transform=animation_axis.transAxes,
                    va="top",
                )
                animation_axis.legend(loc="best")

                def update(frame_number):
                    state = trajectory[int(indices[frame_number])]
                    state_positions = state.positions
                    line.set_data(state_positions[:, 0], state_positions[:, 1])
                    left_anchor.set_data(
                        [state_positions[0, 0]], [state_positions[0, 1]]
                    )
                    right_anchor.set_data(
                        [state_positions[-1, 0]], [state_positions[-1, 1]]
                    )
                    time_label.set_text(
                        f"t={state.time:.6g}  state={int(indices[frame_number])}"
                    )
                    return line, left_anchor, right_anchor, time_label

                animation = FuncAnimation(
                    animation_figure,
                    update,
                    frames=len(indices),
                    init_func=lambda: update(0),
                    interval=100,
                    repeat=True,
                    cache_frame_data=False,
                )
                try:
                    return animation.to_jshtml(fps=10, embed_frames=True)
                finally:
                    plt.close(animation_figure)

            draw_indices = sampled_indices(len(trajectory), MAX_DRAW_FRAMES)
            while len(draw_indices):
                candidate_html = render_centerline_animation(draw_indices)
                candidate_size = len(candidate_html.encode("utf-8"))
                if candidate_size <= MAX_ANIMATION_HTML_BYTES:
                    animation_html = candidate_html
                    animation_size_bytes = candidate_size
                    break
                if len(draw_indices) == 1:
                    animation_size_bytes = candidate_size
                    break
                draw_indices = sampled_indices(
                    len(trajectory), max(1, len(draw_indices) // 2)
                )
        except Exception as exc:
            animation_error = exc

        if animation_html is not None:
            first_index = int(draw_indices[0])
            middle_index = int(draw_indices[len(draw_indices) // 2])
            last_index = int(draw_indices[-1])
            animation_view = mo.vstack(
                [
                    mo.md(
                        "### 中心線アニメーション\n"
                        f"accepted state `{first_index}` → `{last_index}` を時間順に表示。 "
                        f"描画フレーム数: `{len(draw_indices)}` / 上限 `{MAX_DRAW_FRAMES}`、"
                        f"HTMLサイズ: `{animation_size_bytes:,}` bytes / 上限 `{MAX_ANIMATION_HTML_BYTES:,}`。\n\n"
                        f"初期・中間・最終 state index: `{first_index}`, `{middle_index}`, `{last_index}`。"
                    ),
                    # mo.Html keeps the controls but does not execute the
                    # script-bearing FuncAnimation HTML; an iframe provides
                    # the intended document boundary for that script.
                    mo.iframe(animation_html, height="550px"),
                ]
            )
        elif animation_error is not None:
            animation_view = mo.md(
                "### 中心線アニメーション\n"
                "アニメーション生成に失敗しました。下のフレームスライダーは利用できます。\n\n"
                f"```text\n{type(animation_error).__name__}: {animation_error}\n```"
            )
        else:
            animation_view = mo.md(
                "### 中心線アニメーション\n"
                "埋め込みHTMLサイズ上限を超えたため、アニメーションは省略しました。"
                f"（生成サイズ: `{animation_size_bytes:,}` bytes / 上限 `{MAX_ANIMATION_HTML_BYTES:,}`）\n\n"
                "下のフレームスライダーで初期・中間・最終フレームを確認できます。"
            )

        rows = [
            ("time", f"{selected_state.time:.8g}"),
            ("step", selected_state.step),
            ("n_nodes", selected_state.n_nodes),
            ("reference_length", f"{np.sum(selected_state.rest_lengths):.8g}"),
            ("contour_length", f"{contour_length(selected_state):.8g}"),
            (
                "arc_length_radius_of_gyration",
                f"{arc_length_weighted_radius_of_gyration(selected_state):.8g}",
            ),
            (
                "max_curvature",
                f"{np.max(selected_curvature) if len(selected_curvature) else 0.0:.8g}",
            ),
            ("min_nonlocal_distance", f"{selected_geometry['min_nonlocal_distance']:.8g}"),
            ("intersection_pairs", selected_geometry["intersection_pairs"]),
            ("contact_pairs", selected_geometry["contact_pairs"]),
            ("energy_stretch", f"{selected_energy['stretch']:.8g}"),
            ("energy_bend", f"{selected_energy['bend']:.8g}"),
            ("energy_contact", f"{selected_energy['contact']:.8g}"),
        ]
        observable_view = mo.md(
            "### 選択フレームの観測量\n"
            + markdown_table(("quantity", "value"), rows)
        )

        event_log = model.event_log
        reason_counts = Counter(str(event.get("reason", "unknown")) for event in event_log)
        event_rows = [(reason, count) for reason, count in sorted(reason_counts.items())]
        rejection_events = [
            event
            for event in event_log
            if event.get("event_type") == "step_attempt"
            and event.get("accepted") is False
        ]
        rejection_detail = "なし"
        if rejection_events:
            rejection_detail = str(rejection_events[-1].get("detail"))
        acceptance_view = mo.md(
            "### 受理 / 棄却の概要\n"
            + markdown_table(
                ("項目", "値"),
                (
                    ("accepted_steps", model.accepted_steps),
                    ("rejected_steps", model.rejected_steps),
                    ("event_count", len(event_log)),
                    ("accepted_dt_count", len(model.accepted_dts)),
                    ("rejected_dt_count", len(model.rejected_dts)),
                    ("reason counts", event_rows),
                    ("last rejection detail", rejection_detail),
                ),
            )
        )

        manifest = run_result["manifest"]
        comparison = run_result["replay_comparison"]
        if comparison is None:
            replay_view = mo.md(
                "### 再現性比較\n"
                "シミュレーションが完了していないため、2回の比較は実施していません。"
            )
        else:
            replay_view = mo.md(
                "### 再現性比較\n"
                f"結果: **{'一致' if comparison['match'] else '不一致'}**\n\n"
                f"- canonical state hash: `{manifest['canonical_state_hash']}`\n"
                f"- initial state hash: `{manifest['initial_state_hash']}`\n"
                f"- input hash: `{manifest['input_hash']}`\n"
                f"- event count: `{manifest['event_count']}`\n"
                f"- 比較差分: `{list(comparison['differences'])}`"
            )

        if run_result["error_trace"]:
            error_view = mo.md(
                "### 実行診断（モデルが返したエラー）\n"
                "```text\n"
                + run_result["error_trace"]
                + "\n```"
            )
        else:
            error_view = mo.md("### 実行診断\nエラーなし")

        report_view = mo.vstack(
            [
                mo.md(f"## 結果（frame {frame_index}/{len(trajectory) - 1}）"),
                observable_view,
                acceptance_view,
                replay_view,
                error_view,
            ]
        )
        display_view = mo.vstack([fig_visual, animation_view])

    display_view
    return (report_view,)


@app.cell
def _(report_view):
    report_view


if __name__ == "__main__":
    app.run()
