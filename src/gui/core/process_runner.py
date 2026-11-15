# -*- coding: utf-8 -*-
"""
src.gui.core.process_runner - Subprocess executor using QProcess.
CLI araçlarını QProcess ile arka planda izole bir şekilde çalıştırır.
"""

import os
import sys
from PySide6.QtCore import QObject, Signal, QProcess, QProcessEnvironment


_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


def resolve_python_interpreter(project_root: str = None, environ=None) -> str:
    """Alt süreç yorumlayıcısını çözer.

    Sıra: AI_CORE_PYTHON (dosya varsa) -> proje .venv python -> sys.executable.
    """
    env = os.environ if environ is None else environ
    root = project_root or _PROJECT_ROOT
    override = env.get("AI_CORE_PYTHON")
    if override and os.path.isfile(override):
        return override
    if os.name == "nt":
        venv_python = os.path.join(root, ".venv", "Scripts", "python.exe")
    else:
        venv_python = os.path.join(root, ".venv", "bin", "python")
    if os.path.isfile(venv_python):
        return venv_python
    return sys.executable


class ProcessRunner(QObject):
    """
    QProcess tabanlı izole alt süreç yöneticisi.
    CLI komutlarını tetikler ve stdout/stderr loglarını GUI'ye sinyallerle taşır.
    """
    output_received = Signal(str)
    started = Signal()
    finished = Signal(int, str)  # exit_code, status_message

    def __init__(self, parent=None):
        super().__init__(parent)
        self.process = QProcess(self)
        self.process.readyReadStandardOutput.connect(self._handle_stdout)
        self.process.readyReadStandardError.connect(self._handle_stderr)
        self.process.finished.connect(self._handle_finished)
        self.process.setProcessChannelMode(QProcess.SeparateChannels)

        # AI_CORE_PYTHON -> proje .venv -> sys.executable
        # (attribute adı geriye dönük uyumluluk için korunur)
        self.conda_python = resolve_python_interpreter()

    def start_process(self, module_name: str, args: list[str]):
        """
        Belirtilen modülü arka planda çalıştırır.
        Örnek: python -m src.cli.batch --stocks TUPRS
        """
        if self.is_running():
            self.output_received.emit("[Süreç Yöneticisi] Çalışan bir işlem zaten var. Önce onun bitmesini bekleyin veya iptal edin.\n")
            return

        cmd_args = ["-m", module_name] + args

        # Çalışma dizinini projenin kök dizini yapalım (imports vb. düzgün çalışsın)
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
        self.process.setWorkingDirectory(project_root)

        self.output_received.emit(f"[Süreç Yöneticisi] Çalıştırılıyor: {self.conda_python} {' '.join(cmd_args)}\n")
        self.output_received.emit(f"[Süreç Yöneticisi] Çalışma Dizini: {project_root}\n")
        self.output_received.emit("-" * 60 + "\n")

        env = QProcessEnvironment.systemEnvironment()
        env.insert("PYTHONIOENCODING", "utf-8")
        env.insert("PYTHONUTF8", "1")
        env.insert("PYTHONUNBUFFERED", "1")
        self.process.setProcessEnvironment(env)

        self.process.start(self.conda_python, cmd_args)
        self.started.emit()

    def kill_process(self):
        """
        Çalışmakta olan alt süreci zorla durdurur.
        """
        if self.is_running():
            self.output_received.emit("\n" + "=" * 60 + "\n")
            self.output_received.emit("[Süreç Yöneticisi] Kullanıcı tarafından durdurma isteği gönderildi. İşlem sonlandırılıyor...\n")
            self.process.kill()
            self.process.waitForFinished(3000)

    def is_running(self) -> bool:
        """
        Sürecin çalışıp çalışmadığını sorgular.
        """
        return self.process.state() == QProcess.Running

    def _handle_stdout(self):
        """
        Standart çıktıyı (stdout) okur ve sinyal fırlatır.
        """
        data = self.process.readAllStandardOutput()
        stdout_str = self._decode_output(data.data())
        self.output_received.emit(stdout_str)

    def _handle_stderr(self):
        """
        Standart hatayı (stderr) okur ve sinyal fırlatır.
        """
        data = self.process.readAllStandardError()
        stderr_str = self._decode_output(data.data())
        self.output_received.emit(stderr_str)

    def _handle_finished(self, exit_code: int, exit_status: QProcess.ExitStatus):
        """
        Süreç tamamlandığında tetiklenir.
        """
        self._drain_remaining_output()
        self.output_received.emit("\n" + "-" * 60 + "\n")
        if exit_status == QProcess.NormalExit and exit_code == 0:
            msg = f"[Süreç Yöneticisi] İşlem başarıyla tamamlandı. (Çıkış Kodu: {exit_code})"
        else:
            msg = f"[Süreç Yöneticisi] İşlem durduruldu veya hata ile sonlandı. (Çıkış Kodu: {exit_code}, Durum: {exit_status})"

        self.output_received.emit(msg + "\n")
        self.finished.emit(exit_code, msg)

    @staticmethod
    def _decode_output(raw: bytes) -> str:
        """
        CLI çıktıları normalde UTF-8 üretilir; eski Windows/CP1254 çıktıları için
        fallback uygular. Böylece Türkçe metinler replacement karakterine düşmez.
        """
        if not raw:
            return ""
        try:
            return raw.decode("utf-8")
        except UnicodeDecodeError:
            return raw.decode("cp1254", errors="replace")

    def _drain_remaining_output(self):
        stdout = self.process.readAllStandardOutput().data()
        stderr = self.process.readAllStandardError().data()
        if stdout:
            self.output_received.emit(self._decode_output(stdout))
        if stderr:
            self.output_received.emit(self._decode_output(stderr))
