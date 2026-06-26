"""
Sarashina TTS Launcher
=====================
Pystray-based launcher for the Gradio web UI.
"""

import os
import socket
import subprocess
import sys
import time
import webbrowser
from pathlib import Path

import pystray
from PIL import Image, ImageDraw

APP_NAME = "Sarashina TTS Launcher"
HOST = "127.0.0.1"
PORT = 7860
URL = f"http://localhost:{PORT}"
STARTUP_TIMEOUT_SECONDS = 60
GRADIO_CHILD_ARG = "--gradio-child"
MAX_LOG_SIZE_BYTES = 1_000_000

gradio_process = None
tray_icon = None


def get_base_path() -> Path:
    """Return base path for both source and PyInstaller bundle execution."""
    if getattr(sys, "frozen", False):
        return Path(sys._MEIPASS)
    return Path(__file__).resolve().parent


def get_gradio_app_path() -> Path:
    """Return the Gradio application path."""
    return get_base_path() / "server" / "gradio_app.py"


def get_tray_icon_path() -> Path:
    """Return the tray icon image path."""
    return get_base_path() / "static" / "tts_icon.png"


def get_log_path() -> Path:
    """Return launcher log file path."""
    log_dir = Path.home() / ".sarashina_tts" / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    return log_dir / "tts_launcher.log"


def rotate_log_if_needed(log_path: Path) -> None:
    """Rotate the launcher log when it exceeds the maximum size."""
    if not log_path.exists() or log_path.stat().st_size <= MAX_LOG_SIZE_BYTES:
        return

    rotated_log_path = log_path.with_name(f"{log_path.name}.1")
    if rotated_log_path.exists():
        rotated_log_path.unlink()
    log_path.rename(rotated_log_path)


def write_log(message: str) -> None:
    """Write a message to the launcher log file."""
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    log_path = get_log_path()
    rotate_log_if_needed(log_path)
    with open(log_path, "a", encoding="utf-8") as log_file:
        log_file.write(f"[{timestamp}] {message}\n")


def is_port_open(host: str, port: int) -> bool:
    """Check whether the Gradio server port is accepting connections."""
    try:
        with socket.create_connection((host, port), timeout=1):
            return True
    except OSError:
        return False


def wait_for_gradio() -> bool:
    """Wait until Gradio is ready or timeout is reached."""
    deadline = time.time() + STARTUP_TIMEOUT_SECONDS
    while time.time() < deadline:
        if is_port_open(HOST, PORT):
            return True
        time.sleep(0.5)
    return False


def create_icon_image() -> Image.Image:
    """Create a simple tray icon image."""
    tray_icon_path = get_tray_icon_path()
    if tray_icon_path.exists():
        return Image.open(tray_icon_path).convert("RGBA")

    image = Image.new("RGBA", (64, 64), (0, 0, 0, 0))
    draw = ImageDraw.Draw(image)
    draw.rounded_rectangle((8, 8, 56, 56), radius=12, fill=(168, 85, 247, 255))
    draw.text((18, 20), "TTS", fill=(255, 255, 255, 255))
    return image


def start_gradio() -> None:
    """Start Gradio server if it is not already running."""
    global gradio_process

    if is_port_open(HOST, PORT):
        write_log(f"Gradio is already running at {URL}")
        return

    gradio_app_path = get_gradio_app_path()
    write_log(f"Starting Gradio with app path: {gradio_app_path}")

    # Activate virtual environment if it exists
    venv_path = get_base_path() / "venv"
    if venv_path.exists():
        python_path = venv_path / "bin" / "python"
        if not python_path.exists():
            python_path = venv_path / "Scripts" / "python.exe"  # Windows
    else:
        python_path = sys.executable

    if getattr(sys, "frozen", False):
        command = [str(python_path), GRADIO_CHILD_ARG, str(gradio_app_path)]
    else:
        command = [
            str(python_path),
            str(Path(__file__).resolve()),
            GRADIO_CHILD_ARG,
            str(gradio_app_path),
        ]

    write_log(f"Gradio command: {' '.join(command)}")
    log_file = open(get_log_path(), "a", encoding="utf-8")
    gradio_process = subprocess.Popen(
        command,
        cwd=str(get_base_path()),
        env={
            **os.environ,
            "SARASHINA_TTS_CHILD": "1",
        },
        stdout=log_file,
        stderr=log_file,
    )
    write_log(f"Gradio process started with PID: {gradio_process.pid}")


def stop_gradio() -> None:
    """Stop Gradio server if it is running."""
    global gradio_process

    if gradio_process is not None and gradio_process.poll() is None:
        write_log(f"Stopping Gradio process (PID: {gradio_process.pid})")
        gradio_process.terminate()
        try:
            gradio_process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            write_log("Gradio process did not terminate gracefully, killing...")
            gradio_process.kill()
            gradio_process.wait()
        gradio_process = None
        write_log("Gradio process stopped")
    else:
        # Try to kill process on port 7860
        try:
            result = subprocess.run(
                ["lsof", "-ti", str(PORT)],
                capture_output=True,
                text=True,
                check=False,
            )
            if result.stdout.strip():
                pids = result.stdout.strip().split("\n")
                for pid in pids:
                    write_log(f"Killing process {pid} on port {PORT}")
                    subprocess.run(["kill", "-9", pid], check=False)
        except Exception as e:
            write_log(f"Error killing process on port {PORT}: {e}")


def open_tts_ui() -> None:
    """Open Sarashina TTS Web UI in the default browser."""
    webbrowser.open(URL)


def show_logs() -> None:
    """Open the launcher log file in the default text editor."""
    log_path = get_log_path()
    if log_path.exists():
        subprocess.run(["open", str(log_path)], check=False)
    else:
        write_log(f"Log file not found: {log_path}")


def quit_app(icon: pystray.Icon) -> None:
    """Stop Gradio and quit the tray application."""
    global gradio_process

    stop_gradio()
    icon.stop()


def setup_tray() -> pystray.Icon:
    """Create and return tray icon."""
    menu = pystray.Menu(
        pystray.MenuItem("Open Sarashina TTS", lambda: open_tts_ui()),
        pystray.Menu.SEPARATOR,
        pystray.MenuItem("Start Server", lambda: start_gradio()),
        pystray.MenuItem("Stop Server", lambda: stop_gradio()),
        pystray.MenuItem("Show Logs", lambda: show_logs()),
        pystray.Menu.SEPARATOR,
        pystray.MenuItem("Quit", quit_app),
    )
    return pystray.Icon(APP_NAME, create_icon_image(), APP_NAME, menu)


def main() -> None:
    """Start Gradio, open browser, and run tray icon."""
    global tray_icon

    write_log("Launcher started")
    start_gradio()
    if wait_for_gradio():
        write_log(f"Gradio is ready at {URL}")
        open_tts_ui()
    else:
        if gradio_process is not None:
            write_log(
                f"Gradio did not become ready. Return code: {gradio_process.poll()}"
            )
        else:
            write_log("Gradio did not become ready. No process was started.")
        webbrowser.open(URL)

    tray_icon = setup_tray()
    tray_icon.run()


def run_gradio_child() -> None:
    """Run Gradio inside the child process without starting the launcher."""
    try:
        child_arg_index = sys.argv.index(GRADIO_CHILD_ARG)
        app_path = sys.argv[child_arg_index + 1]
    except (ValueError, IndexError):
        raise SystemExit("Missing Gradio app path.")

    # Run the Gradio app directly
    import subprocess

    result = subprocess.run(
        [sys.executable, app_path],
        cwd=str(get_base_path()),
        check=False,
    )
    sys.exit(result.returncode)


if __name__ == "__main__":
    if GRADIO_CHILD_ARG in sys.argv:
        run_gradio_child()
    else:
        main()
