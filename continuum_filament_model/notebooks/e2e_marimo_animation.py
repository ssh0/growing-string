#!/usr/bin/env python3
"""Run a browser-level smoke test for the marimo centerline animation.

The harness intentionally drives the published marimo app through its HTTP
entrypoint. It uses the locally available chrome-devtools-axi CLI rather than
installing a browser automation dependency.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any


DEFAULT_PORT = 27991
DEFAULT_TIMEOUT_SECONDS = 90.0


class HarnessError(RuntimeError):
    """An expected E2E assertion or infrastructure failure."""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--timeout", type=float, default=DEFAULT_TIMEOUT_SECONDS)
    parser.add_argument("--evidence-dir", type=Path)
    parser.add_argument(
        "--session",
        default=f"growing-string-marimo-animation-{os.getpid()}",
        help="chrome-devtools-axi isolated session name",
    )
    return parser.parse_args()


def wait_for_http(url: str, timeout: float) -> None:
    deadline = time.monotonic() + timeout
    last_error: Exception | None = None
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=2) as response:
                if 200 <= response.status < 500:
                    return
        except (OSError, urllib.error.URLError) as exc:
            last_error = exc
        time.sleep(0.2)
    raise HarnessError(f"marimo HTTP endpoint did not become ready: {url}: {last_error}")


def run_browser(
    executable: str,
    session: str,
    evidence_dir: Path,
    name: str,
    *args: str,
    timeout: float = 60.0,
) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    env["CHROME_DEVTOOLS_AXI_SESSION"] = session
    result = subprocess.run(
        [executable, *args],
        cwd=evidence_dir,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=timeout,
        check=False,
    )
    (evidence_dir / f"browser-{name}.log").write_text(
        result.stdout, encoding="utf-8"
    )
    return result


def eval_json(
    executable: str,
    session: str,
    evidence_dir: Path,
    name: str,
    script: str,
    timeout: float = 60.0,
) -> Any:
    result = run_browser(
        executable, session, evidence_dir, name, "eval", script, timeout=timeout
    )
    if result.returncode != 0:
        raise HarnessError(f"browser eval failed ({name}); see browser-{name}.log")
    result_line = next(
        (line for line in result.stdout.splitlines() if line.startswith("result: ")),
        None,
    )
    if result_line is None:
        raise HarnessError(f"browser eval returned no result ({name})")
    try:
        value: Any = json.loads(result_line.removeprefix("result: "))
        if isinstance(value, str):
            value = json.loads(value)
        return value
    except json.JSONDecodeError as exc:
        raise HarnessError(f"could not parse browser eval result ({name})") from exc


def write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def poll_browser_eval(
    executable: str,
    session: str,
    evidence_dir: Path,
    name: str,
    script: str,
    predicate: Any,
    timeout: float,
) -> Any:
    deadline = time.monotonic() + timeout
    last: Any = None
    last_error: Exception | None = None
    while time.monotonic() < deadline:
        try:
            last = eval_json(executable, session, evidence_dir, name, script)
            last_error = None
            if predicate(last):
                return last
        except Exception as exc:  # The page may still be attaching to marimo.
            last_error = exc
        time.sleep(0.5)
    if last is not None:
        return last
    raise HarnessError(f"browser condition timed out ({name}): {last_error}")


def has_visible_controls(state: dict[str, Any]) -> bool:
    controls = state.get("controls") or {}
    play = controls.get("play") or {}
    slider = controls.get("slider") or {}
    return bool(play.get("visible") and slider.get("visible"))


def has_rendered_frame(state: dict[str, Any]) -> bool:
    frame = state.get("image") or state.get("canvas")
    if not frame:
        return False
    if not frame.get("visible"):
        return False
    if "natural_width" in frame:
        return bool(
            frame.get("complete")
            and frame.get("natural_width", 0) > 0
            and frame.get("natural_height", 0) > 0
        )
    return bool(
        frame.get("width", 0) > 0
        and frame.get("height", 0) > 0
        and frame.get("nonempty_pixels")
        and frame.get("screenshot")
    )


def visual_ready(state: dict[str, Any]) -> bool:
    iframe = state.get("iframe")
    iframe_ready = iframe is None or bool(iframe.get("visible") and iframe.get("loaded"))
    return bool(
        state.get("heading")
        and iframe_ready
        and has_visible_controls(state)
        and has_rendered_frame(state)
    )


def main() -> int:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[2]
    marimo = shutil.which("marimo")
    chrome_axi = shutil.which("chrome-devtools-axi")
    if not marimo:
        raise HarnessError("marimo was not found on PATH")
    if not chrome_axi:
        raise HarnessError("chrome-devtools-axi was not found on PATH")

    if args.evidence_dir is None:
        evidence_dir = Path("/tmp") / (
            "growing-string-marimo-animation-" + time.strftime("%Y%m%d-%H%M%S")
        )
    else:
        evidence_dir = args.evidence_dir
    evidence_dir.mkdir(parents=True, exist_ok=True)
    (evidence_dir / "invocation.json").write_text(
        json.dumps(
            {
                "command": sys.argv,
                "repo_root": str(repo_root),
                "port": args.port,
                "session": args.session,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    server_log_path = evidence_dir / "marimo.log"
    server_log = server_log_path.open("w")
    server = subprocess.Popen(
        [
            marimo,
            "run",
            "continuum_filament_model/notebooks/filament_explorer.py",
            "--headless",
            "--no-token",
            "--host",
            "127.0.0.1",
            "--port",
            str(args.port),
        ],
        cwd=repo_root,
        stdout=server_log,
        stderr=subprocess.STDOUT,
        env=os.environ.copy(),
        text=True,
    )
    browser_started = False
    failure: Exception | None = None
    result: dict[str, Any] = {"passed": False, "evidence_dir": str(evidence_dir)}

    section_state_script = r"""
() => {
  const heading = [...document.querySelectorAll("h3")]
    .find((node) => node.textContent.includes("中心線アニメーション"));
  const section = heading?.parentElement?.parentElement || null;
  const iframe = section?.querySelector("iframe") || null;
  const nativeAnimation = section?.querySelector(".animation") || null;
  const doc = iframe?.contentDocument || null;
  const root = doc || nativeAnimation?.ownerDocument || null;
  const animation = doc?.querySelector(".animation") || nativeAnimation;
  const image = animation?.querySelector("img") || null;
  const canvas = animation?.querySelector("canvas") || null;
  const play = animation?.querySelector('button[title="Play"]') || null;
  const slider = animation?.querySelector('input[type="range"]') || null;
  const visible = (node) => {
    if (!node) return false;
    const rect = node.getBoundingClientRect();
    const style = node.ownerDocument.defaultView.getComputedStyle(node);
    return rect.width > 0 && rect.height > 0 && style.display !== "none"
      && style.visibility !== "hidden";
  };
  const rect = (node) => node ? node.getBoundingClientRect().toJSON() : null;
  const imageSrc = image?.currentSrc || image?.src || "";
  let canvasHasPixels = false;
  if (canvas && canvas.width > 0 && canvas.height > 0) {
    const context = canvas.getContext("2d");
    if (context) {
      const pixels = context.getImageData(0, 0, canvas.width, canvas.height).data;
      for (let index = 0; index < pixels.length; index += 4) {
        if (pixels[index] || pixels[index + 1] || pixels[index + 2] || pixels[index + 3]) {
          canvasHasPixels = true;
          break;
        }
      }
    }
  }
  return {
    heading: Boolean(heading),
    section: Boolean(section),
    iframe: iframe ? {visible: visible(iframe), rect: rect(iframe), loaded: Boolean(doc)} : null,
    native_animation: nativeAnimation ? {visible: visible(nativeAnimation)} : null,
    controls: {
      play: play ? {visible: visible(play), title: play.title} : null,
      slider: slider ? {visible: visible(slider), value: slider.value, max: slider.max} : null,
    },
    image: image ? {
      visible: visible(image),
      rect: rect(image),
      complete: image.complete,
      natural_width: image.naturalWidth,
      natural_height: image.naturalHeight,
      src_fingerprint: imageSrc ? `${imageSrc.length}:${imageSrc.slice(0, 24)}:${imageSrc.slice(-24)}` : "",
    } : null,
    canvas: canvas ? {
      visible: visible(canvas),
      rect: rect(canvas),
      width: canvas.width,
      height: canvas.height,
      nonempty_pixels: canvasHasPixels,
      screenshot: canvas.toDataURL().slice(0, 96),
    } : null,
    body_text_has_heading: document.body.innerText.includes("中心線アニメーション"),
    script_count: root?.scripts?.length || 0,
  };
}
"""

    find_submit_script = r"""
() => {
  const form = [...document.querySelectorAll("marimo-form")]
    .find((node) => node.dataset.submitButtonLabel?.includes("この設定で実行"));
  const button = form?.shadowRoot?.querySelector("button");
  return {found: Boolean(button), form: Boolean(form), shadow: Boolean(form?.shadowRoot)};
}
"""

    click_submit_script = r"""
() => {
  const form = [...document.querySelectorAll("marimo-form")]
    .find((node) => node.dataset.submitButtonLabel?.includes("この設定で実行"));
  const submit = form?.shadowRoot?.querySelector("button");
  if (!submit) return {clicked: false, form: Boolean(form)};
  submit.click();
  return {clicked: true};
}
"""

    output_state_script = r"""
() => ({
  ready: document.body.innerText.includes("中心線アニメーション")
    && Boolean([...document.querySelectorAll("h3")]
      .find((node) => node.textContent.includes("中心線アニメーション"))),
  url: location.href,
})
"""

    click_play_script = r"""
() => {
  const heading = [...document.querySelectorAll("h3")]
    .find((node) => node.textContent.includes("中心線アニメーション"));
  const section = heading?.parentElement?.parentElement || null;
  const doc = section?.querySelector("iframe")?.contentDocument || null;
  const play = doc?.querySelector('.animation button[title="Play"]') || null;
  if (!play) return {clicked: false};
  play.click();
  return {clicked: true};
}
"""

    scroll_script = r"""
() => {
  const heading = [...document.querySelectorAll("h3")]
    .find((node) => node.textContent.includes("中心線アニメーション"));
  heading?.scrollIntoView({block: "center", behavior: "instant"});
  return Boolean(heading);
}
"""

    try:
        wait_for_http(f"http://127.0.0.1:{args.port}/", args.timeout)
        opened = run_browser(
            chrome_axi,
            args.session,
            evidence_dir,
            "open",
            "open",
            f"http://127.0.0.1:{args.port}/",
            timeout=args.timeout,
        )
        if opened.returncode != 0:
            raise HarnessError("browser could not open the marimo page")
        browser_started = True

        submit_ready = poll_browser_eval(
            chrome_axi,
            args.session,
            evidence_dir,
            "find-submit",
            find_submit_script,
            lambda value: bool(value.get("found")),
            timeout=args.timeout,
        )
        write_json(evidence_dir / "submit-ready.json", submit_ready)
        submit = eval_json(
            chrome_axi,
            args.session,
            evidence_dir,
            "submit-default-fixture",
            click_submit_script,
        )
        write_json(evidence_dir / "submit.json", submit)
        if not submit.get("clicked"):
            raise HarnessError("default fixture submit button was not found")

        output = poll_browser_eval(
            chrome_axi,
            args.session,
            evidence_dir,
            "wait-for-output",
            output_state_script,
            lambda value: bool(value.get("ready")),
            timeout=args.timeout,
        )
        write_json(evidence_dir / "output-ready.json", output)
        if not output.get("ready"):
            raise HarnessError("centerline animation section did not appear")

        initial_state = poll_browser_eval(
            chrome_axi,
            args.session,
            evidence_dir,
            "wait-for-initial-frame",
            section_state_script,
            visual_ready,
            timeout=min(args.timeout, 15.0),
        )
        initial = {"ready": visual_ready(initial_state), "state": initial_state}
        write_json(evidence_dir / "initial-state.json", initial)
        if not initial.get("ready"):
            raise HarnessError(
                "initial centerline frame is not visibly rendered; see initial-state.json"
            )

        eval_json(
            chrome_axi,
            args.session,
            evidence_dir,
            "scroll-initial",
            scroll_script,
        )
        screenshot = run_browser(
            chrome_axi,
            args.session,
            evidence_dir,
            "initial-screenshot",
            "screenshot",
            str(evidence_dir / "initial.png"),
        )
        if screenshot.returncode != 0:
            raise HarnessError("could not save initial screenshot")

        before = {
            "src_fingerprint": initial_state.get("image", {}).get("src_fingerprint"),
            "slider": initial_state.get("controls", {}).get("slider", {}).get("value"),
        }
        play = eval_json(
            chrome_axi,
            args.session,
            evidence_dir,
            "click-play",
            click_play_script,
        )
        if not play.get("clicked"):
            playback = {"ready": False, "changed": False, "reason": "play control not found"}
        else:
            after_state = poll_browser_eval(
                chrome_axi,
                args.session,
                evidence_dir,
                "wait-for-frame-change",
                section_state_script,
                lambda value: (
                    value.get("image", {}).get("src_fingerprint") != before["src_fingerprint"]
                    or value.get("controls", {}).get("slider", {}).get("value") != before["slider"]
                ),
                timeout=8.0,
            )
            after = {
                "src_fingerprint": after_state.get("image", {}).get("src_fingerprint"),
                "slider": after_state.get("controls", {}).get("slider", {}).get("value"),
            }
            playback = {
                "ready": True,
                "changed": after != before,
                "before": before,
                "after": after,
            }
        write_json(evidence_dir / "playback.json", playback)
        if not playback.get("ready") or not playback.get("changed"):
            raise HarnessError(
                "animation did not change frame after Play; see playback.json"
            )

        screenshot = run_browser(
            chrome_axi,
            args.session,
            evidence_dir,
            "after-play-screenshot",
            "screenshot",
            str(evidence_dir / "after-play.png"),
        )
        if screenshot.returncode != 0:
            raise HarnessError("could not save after-play screenshot")

        final_state = eval_json(
            chrome_axi,
            args.session,
            evidence_dir,
            "final-state",
            section_state_script,
        )
        write_json(evidence_dir / "final-state.json", final_state)
        result.update(
            {
                "passed": True,
                "checks": {
                    "controls_visible": True,
                    "initial_frame_rendered": True,
                    "frame_changed_after_play": True,
                },
                "initial": initial,
                "playback": playback,
                "final": final_state,
            }
        )
    except Exception as exc:  # preserve failure artifacts before cleanup
        failure = exc
        (evidence_dir / "failure.txt").write_text(
            f"{type(exc).__name__}: {exc}\n", encoding="utf-8"
        )
        if browser_started:
            try:
                eval_json(
                    chrome_axi,
                    args.session,
                    evidence_dir,
                    "failure-scroll",
                    scroll_script,
                )
            except Exception:
                pass
            run_browser(
                chrome_axi,
                args.session,
                evidence_dir,
                "failure-screenshot",
                "screenshot",
                str(evidence_dir / "failure.png"),
            )
            try:
                state = eval_json(
                    chrome_axi,
                    args.session,
                    evidence_dir,
                    "failure-state",
                    section_state_script,
                )
                write_json(evidence_dir / "failure-state.json", state)
            except Exception:
                pass
            run_browser(
                chrome_axi,
                args.session,
                evidence_dir,
                "failure-dom",
                "snapshot",
                "--full",
            )
            run_browser(
                chrome_axi,
                args.session,
                evidence_dir,
                "failure-console",
                "console",
            )
            run_browser(
                chrome_axi,
                args.session,
                evidence_dir,
                "failure-console-errors",
                "console",
                "--type",
                "error",
            )
    finally:
        if browser_started:
            run_browser(chrome_axi, args.session, evidence_dir, "stop", "stop")
        server.terminate()
        try:
            server.wait(timeout=5)
        except subprocess.TimeoutExpired:
            server.kill()
            server.wait(timeout=5)
        server_log.close()
        server_log_path.touch(exist_ok=True)
        result["server_returncode"] = server.returncode
        write_json(evidence_dir / "result.json", result)

    if failure is not None:
        print(f"FAIL: {failure}", file=sys.stderr)
        print(f"evidence: {evidence_dir}", file=sys.stderr)
        return 1
    print("PASS: marimo animation E2E")
    print(f"evidence: {evidence_dir}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except HarnessError as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        raise SystemExit(1)
