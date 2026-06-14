# -*- coding: utf-8 -*-
"""
src.gui.core.process_runner - Subprocess executor using QProcess.
CLI araÃ§larÄ±nÄ± QProcess ile arka planda izole bir ÅŸekilde Ã§alÄ±ÅŸtÄ±rÄ±r.
"""

import os
import sys
from PySide6.QtCore import QObject, Signal, QProcess


class ProcessRunner(QObject):
    """
    QProcess tabanlÄ± izole alt sÃ¼reÃ§ yÃ¶neticisi.
    CLI komutlarÄ±nÄ± tetikler ve stdout/stderr loglarÄ±nÄ± GUI'ye sinyallerle taÅŸÄ±r.
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

        # KullanÄ±cÄ±nÄ±n belirttiÄŸi conda ortamÄ± python yorumlayÄ±cÄ±sÄ±
        self.conda_python = r"C:\Users\ysfygc\anaconda3\envs\dl_env\python.exe"
        if not os.path.exists(self.conda_python):
            # Fallback to current sys.executable if conda path doesn't exist
            self.conda_python = sys.executable

    def start_process(self, module_name: str, args: list[str]):
        """
        Belirtilen modÃ¼lÃ¼ arka planda Ã§alÄ±ÅŸtÄ±rÄ±r.
        Ã–rnek: python -m src.cli.batch --stocks TUPRS
        """
        if self.is_running():
            self.output_received.emit("[SÃ¼reÃ§ YÃ¶neticisi] Ã‡alÄ±ÅŸan bir iÅŸlem zaten var. Ã–nce onun bitmesini bekleyin veya iptal edin.\n")
            return

        cmd_args = ["-m", module_name] + args

        # Ã‡alÄ±ÅŸma dizinini projenin kÃ¶k dizini yapalÄ±m (imports vb. dÃ¼zgÃ¼n Ã§alÄ±ÅŸsÄ±n)
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
        self.process.setWorkingDirectory(project_root)

        self.output_received.emit(f"[SÃ¼reÃ§ YÃ¶neticisi] Ã‡alÄ±ÅŸtÄ±rÄ±lÄ±yor: {self.conda_python} {' '.join(cmd_args)}\n")
        self.output_received.emit(f"[SÃ¼reÃ§ YÃ¶neticisi] Ã‡alÄ±ÅŸma Dizini: {project_root}\n")
        self.output_received.emit("â”€" * 60 + "\n")

        self.process.start(self.conda_python, cmd_args)
        self.started.emit()

    def kill_process(self):
        """
        Ã‡alÄ±ÅŸmakta olan alt sÃ¼reci zorla durdurur.
        """
        if self.is_running():
            self.output_received.emit("\n" + "â•" * 60 + "\n")
            self.output_received.emit("[SÃ¼reÃ§ YÃ¶neticisi] KullanÄ±cÄ± tarafÄ±ndan durdurma isteÄŸi gÃ¶nderildi. Ä°ÅŸlem sonlandÄ±rÄ±lÄ±yor...\n")
            self.process.kill()
            self.process.waitForFinished(3000)

    def is_running(self) -> bool:
        """
        SÃ¼recin Ã§alÄ±ÅŸÄ±p Ã§alÄ±ÅŸmadÄ±ÄŸÄ±nÄ± sorgular.
        """
        return self.process.state() == QProcess.Running

    def _handle_stdout(self):
        """
        Standart Ã§Ä±ktÄ±yÄ± (stdout) okur ve sinyal fÄ±rlatÄ±r.
        """
        data = self.process.readAllStandardOutput()
        stdout_str = data.data().decode("utf-8", errors="replace")
        self.output_received.emit(stdout_str)

    def _handle_stderr(self):
        """
        Standart hatayÄ± (stderr) okur ve sinyal fÄ±rlatÄ±r.
        """
        data = self.process.readAllStandardError()
        stderr_str = data.data().decode("utf-8", errors="replace")
        self.output_received.emit(stderr_str)

    def _handle_finished(self, exit_code: int, exit_status: QProcess.ExitStatus):
        """
        SÃ¼reÃ§ tamamlandÄ±ÄŸÄ±nda tetiklenir.
        """
        self.output_received.emit("\n" + "â”€" * 60 + "\n")
        if exit_status == QProcess.NormalExit and exit_code == 0:
            msg = f"[SÃ¼reÃ§ YÃ¶neticisi] Ä°ÅŸlem baÅŸarÄ±yla tamamlandÄ±. (Ã‡Ä±kÄ±ÅŸ Kodu: {exit_code})"
        else:
            msg = f"[SÃ¼reÃ§ YÃ¶neticisi] Ä°ÅŸlem durduruldu veya hata ile sonlandÄ±. (Ã‡Ä±kÄ±ÅŸ Kodu: {exit_code}, Durum: {exit_status})"

        self.output_received.emit(msg + "\n")
        self.finished.emit(exit_code, msg)
