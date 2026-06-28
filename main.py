import sys
import time
import json
import subprocess
from pathlib import Path
import cv2
import numpy as np
from ultralytics import YOLO
from PyQt5 import QtCore, QtGui, QtWidgets
from picamera2 import Picamera2

# ==========================
# НАСТРОЙКИ
# ==========================
MODEL_PATH = "/home/restartdmg/best.pt"  # <-- путь к твоим YOLO-seg весам
CAMERA_SIZE = (640, 480)  # ширина, высота live preview
YOLO_IMGSZ = 640
CONF_THRES = 0.25
SAVE_RESULTS = True
OUTPUT_DIR = Path("captures")

# Настройки "Призрака" (Ghosting)
GHOST_ALPHA = 0.35  # Прозрачность маски (0.0 - полностью прозрачная, 1.0 - непрозрачная)
GHOST_COLOR = (0, 255, 0)  # Цвет подсветки уже посчитанных труб (BGR формат для OpenCV)


# ==========================
# ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ
# ==========================
def rgb_to_qpixmap(rgb_image: np.ndarray) -> QtGui.QPixmap:
    h, w, ch = rgb_image.shape
    bytes_per_line = ch * w
    qimg = QtGui.QImage(rgb_image.data, w, h, bytes_per_line, QtGui.QImage.Format_RGB888).copy()
    return QtGui.QPixmap.fromImage(qimg)


def gray_to_qpixmap(gray_image: np.ndarray) -> QtGui.QPixmap:
    h, w = gray_image.shape
    bytes_per_line = w
    qimg = QtGui.QImage(gray_image.data, w, h, bytes_per_line, QtGui.QImage.Format_Grayscale8).copy()
    return QtGui.QPixmap.fromImage(qimg)


def make_binary_mask_from_result(result, image_shape) -> np.ndarray:
    h, w = image_shape[:2]
    if result.masks is None:
        return np.zeros((h, w), dtype=np.uint8)
    masks = result.masks.data.cpu().numpy()
    if masks.size == 0:
        return np.zeros((h, w), dtype=np.uint8)
    combined = np.zeros((h, w), dtype=np.uint8)
    for mask in masks:
        mask_bin = (mask > 0.5).astype(np.uint8)
        if mask_bin.shape != (h, w):
            mask_bin = cv2.resize(mask_bin, (w, h), interpolation=cv2.INTER_NEAREST)
        combined = np.maximum(combined, mask_bin)
    return combined * 255


def apply_ghost_overlay(frame_rgb: np.ndarray, mask: np.ndarray, alpha: float, color_bgr: tuple) -> np.ndarray:
    """
    Накладывает полупрозрачный цветной оверлей на кадр в местах, где маска != 0.
    frame_rgb: исходный кадр в формате RGB
    mask: бинарная маска (255 - объект, 0 - фон)
    alpha: прозрачность оверлея
    color_bgr: цвет оверлея в формате BGR
    """
    if mask is None or mask.max() == 0:
        return frame_rgb

    # Создаем пустой цветной оверлей того же размера, что и кадр
    overlay = np.zeros_like(frame_rgb)

    # Маска в формате (H, W), а кадр (H, W, 3). Создаем булеву маску для индексации.
    mask_bool = (mask == 255)

    # Красим оверлей в нужный цвет (переводим BGR в RGB для нашего кадра)
    color_rgb = (color_bgr[2], color_bgr[1], color_bgr[0])
    overlay[mask_bool] = color_rgb

    # Смешиваем оригинал и оверлей с учетом прозрачности
    blended = cv2.addWeighted(overlay, alpha, frame_rgb, 1 - alpha, 0)

    return blended


# ==========================
# WORKER ДЛЯ ИНФЕРЕНСА
# ==========================
class InferenceWorker(QtCore.QObject):
    finished = QtCore.pyqtSignal(object, object, str, int)
    failed = QtCore.pyqtSignal(str)

    def __init__(self, model, frame_rgb):
        super().__init__()
        self.model = model
        self.frame_rgb = frame_rgb.copy()

    @QtCore.pyqtSlot()
    def run(self):
        try:
            frame_bgr = cv2.cvtColor(self.frame_rgb, cv2.COLOR_RGB2BGR)
            result = self.model.predict(
                source=frame_bgr, imgsz=YOLO_IMGSZ, conf=CONF_THRES,
                retina_masks=True, verbose=False
            )[0]

            objects_count = 0 if result.masks is None else len(result.masks.data)
            mask = make_binary_mask_from_result(result, self.frame_rgb.shape)

            saved_text = ""
            if SAVE_RESULTS:
                OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
                ts = time.strftime("%Y%m%d_%H%M%S")

                photo_path = OUTPUT_DIR / f"photo_{ts}.png"
                mask_path = OUTPUT_DIR / f"mask_{ts}.png"

                cv2.imwrite(str(photo_path), frame_bgr)
                cv2.imwrite(str(mask_path), mask)

                # Сохраняем JSON для веб-интерфейса
                metadata = {
                    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "count": objects_count,
                    "photo": photo_path.name,
                    "mask": mask_path.name
                }
                json_path = OUTPUT_DIR / f"scan_{ts}.json"
                with open(json_path, "w", encoding="utf-8") as f:
                    json.dump(metadata, f, ensure_ascii=False, indent=2)

                saved_text = f"Сохранено: {photo_path.name}"

            self.finished.emit(self.frame_rgb, mask, saved_text, objects_count)

        except Exception as e:
            self.failed.emit(str(e))


# ==========================
# ОСНОВНОЕ ОКНО
# ==========================
class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("YOLO-seg Raspberry Pi Camera GUI")

        screen = QtWidgets.QApplication.primaryScreen()
        screen_geometry = screen.availableGeometry()
        self.setGeometry(screen_geometry)

        self.model = None
        self.picam2 = None
        self.current_frame_rgb = None
        self.worker_thread = None
        self.worker = None
        self.showing_mask = False
        self.last_frame_rgb = None
        self.last_mask = None

        # Переменная для фонового процесса Streamlit
        self.streamlit_process = None

        self._build_ui()
        self._load_model()
        self._start_camera()
        self._start_streamlit()
        self._start_timer()

    def _build_ui(self):
        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        main_layout = QtWidgets.QVBoxLayout(central)
        main_layout.setContentsMargins(0, 0, 0, 0)

        self.stacked_widget = QtWidgets.QStackedWidget()
        main_layout.addWidget(self.stacked_widget)

        self.camera_widget = QtWidgets.QLabel("Live camera")
        self.camera_widget.setAlignment(QtCore.Qt.AlignCenter)
        self.camera_widget.setStyleSheet("background-color: #222; color: white;")

        self.mask_widget = QtWidgets.QLabel("Mask")
        self.mask_widget.setAlignment(QtCore.Qt.AlignCenter)
        self.mask_widget.setStyleSheet("background-color: #222; color: white;")

        self.stacked_widget.addWidget(self.camera_widget)
        self.stacked_widget.addWidget(self.mask_widget)

        controls_layout = QtWidgets.QHBoxLayout()
        main_layout.addLayout(controls_layout)

        self.capture_button = QtWidgets.QPushButton("Сделать фото и маску")
        self.capture_button.setMinimumHeight(48)
        self.capture_button.clicked.connect(self.on_capture_clicked)

        self.back_button = QtWidgets.QPushButton("Вернуться к камере")
        self.back_button.setMinimumHeight(48)
        self.back_button.clicked.connect(self.back_to_camera)
        self.back_button.setEnabled(False)

        # Новая кнопка для сброса "призрака"
        self.clear_ghost_button = QtWidgets.QPushButton("Очистить призрак")
        self.clear_ghost_button.setMinimumHeight(48)
        self.clear_ghost_button.clicked.connect(self.clear_ghost)
        self.clear_ghost_button.setEnabled(False)

        self.status_label = QtWidgets.QLabel("Инициализация...")
        self.status_label.setWordWrap(True)

        controls_layout.addWidget(self.capture_button)
        controls_layout.addWidget(self.back_button)
        controls_layout.addWidget(self.clear_ghost_button)
        controls_layout.addWidget(self.status_label, stretch=1)

    def _load_model(self):
        model_path = Path(MODEL_PATH)
        if not model_path.exists():
            raise FileNotFoundError(f"Файл весов не найден: {MODEL_PATH}")
        self.status_label.setText("Загружаю YOLO-seg модель...")
        QtWidgets.QApplication.processEvents()
        self.model = YOLO(str(model_path))
        self.status_label.setText("Модель загружена.")

    def _start_camera(self):
        self.picam2 = Picamera2()
        config = self.picam2.create_preview_configuration(main={"size": CAMERA_SIZE, "format": "RGB888"})
        self.picam2.configure(config)
        self.picam2.start()
        self.status_label.setText("Камера запущена.")

    def _start_streamlit(self):
        self.status_label.setText("Запускаю веб-интерфейс...")
        QtWidgets.QApplication.processEvents()
        try:
            # Узнаем IP адрес для красивого вывода
            import socket
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            s.connect(("8.8.8.8", 80))
            local_ip = s.getsockname()[0]
            s.close()

            self.streamlit_process = subprocess.Popen(
                [sys.executable, "-m", "streamlit", "run", "web_app.py",
                 "--server.headless", "true", "--server.port", "8501", "--server.address", "0.0.0.0"],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
            )
            self.status_label.setText(f"✅ Готово. Веб-отчет: http://{local_ip}:8501")
        except Exception as e:
            self.status_label.setText(f"Веб-интерфейс не запущен: {e}")

    def _start_timer(self):
        self.timer = QtCore.QTimer(self)
        self.timer.timeout.connect(self.update_camera_frame)
        self.timer.start(30)

    def update_camera_frame(self):
        try:
            if not self.showing_mask:
                frame_rgb = self.picam2.capture_array()
                if frame_rgb is None: return
                self.current_frame_rgb = frame_rgb

                # >>> ФИЧА GHOSTING: Накладываем "призрак" предыдущей маски <<<
                if self.last_mask is not None:
                    frame_rgb = apply_ghost_overlay(
                        frame_rgb,
                        self.last_mask,
                        alpha=GHOST_ALPHA,
                        color_bgr=GHOST_COLOR
                    )

                pixmap = rgb_to_qpixmap(frame_rgb)
                pixmap = pixmap.scaled(self.camera_widget.size(), QtCore.Qt.KeepAspectRatio,
                                       QtCore.Qt.SmoothTransformation)
                self.camera_widget.setPixmap(pixmap)
        except Exception as e:
            self.status_label.setText(f"Ошибка камеры: {e}")

    def on_capture_clicked(self):
        if self.current_frame_rgb is None:
            self.status_label.setText("Кадр с камеры ещё не получен.")
            return
        if self.worker_thread is not None:
            self.status_label.setText("Обработка уже идет...")
            return

        self.last_frame_rgb = self.current_frame_rgb.copy()
        frame = self.current_frame_rgb.copy()

        self.capture_button.setEnabled(False)
        self.back_button.setEnabled(False)
        self.status_label.setText("Обрабатываю снимок через YOLO-seg...")

        self.worker_thread = QtCore.QThread()
        self.worker = InferenceWorker(self.model, frame)
        self.worker.moveToThread(self.worker_thread)

        self.worker_thread.started.connect(self.worker.run)
        self.worker.finished.connect(self.on_inference_finished)
        self.worker.failed.connect(self.on_inference_failed)
        self.worker.finished.connect(self.worker_thread.quit)
        self.worker.failed.connect(self.worker_thread.quit)
        self.worker_thread.finished.connect(self.worker.deleteLater)
        self.worker_thread.finished.connect(self.worker_thread.deleteLater)
        self.worker_thread.finished.connect(self._clear_worker)

        self.worker_thread.start()

    def on_inference_finished(self, photo_rgb, mask, saved_text, objects_count):
        self.last_mask = mask
        mask_pixmap = gray_to_qpixmap(mask)
        mask_pixmap = mask_pixmap.scaled(self.mask_widget.size(), QtCore.Qt.KeepAspectRatio,
                                         QtCore.Qt.SmoothTransformation)
        self.mask_widget.setPixmap(mask_pixmap)

        self.showing_mask = True
        self.stacked_widget.setCurrentIndex(1)
        self.back_button.setEnabled(True)
        self.clear_ghost_button.setEnabled(True)  # Активируем кнопку очистки

        self.status_label.setText(f"Готово. Найдено: {objects_count} шт. {saved_text}")
        self.capture_button.setEnabled(True)

    def back_to_camera(self):
        self.showing_mask = False
        self.stacked_widget.setCurrentIndex(0)
        self.back_button.setEnabled(False)
        self.status_label.setText("Возврат к камере. Зеленые зоны - уже посчитаны!")

    def clear_ghost(self):
        """Сбрасывает маску-призрак"""
        self.last_mask = None
        self.clear_ghost_button.setEnabled(False)
        self.status_label.setText("Призрак очищен. Можно сканировать новую зону.")

    def on_inference_failed(self, error_text):
        self.status_label.setText(f"Ошибка обработки: {error_text}")
        self.capture_button.setEnabled(True)
        if self.showing_mask: self.back_button.setEnabled(True)

    def _clear_worker(self):
        self.worker_thread = None
        self.worker = None

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if hasattr(self, 'current_frame_rgb') and self.current_frame_rgb is not None:
            if not self.showing_mask:
                frame = self.current_frame_rgb
                if self.last_mask is not None:
                    frame = apply_ghost_overlay(frame, self.last_mask, GHOST_ALPHA, GHOST_COLOR)
                pixmap = rgb_to_qpixmap(frame)
                pixmap = pixmap.scaled(self.camera_widget.size(), QtCore.Qt.KeepAspectRatio,
                                       QtCore.Qt.SmoothTransformation)
                self.camera_widget.setPixmap(pixmap)
        if hasattr(self, 'last_mask') and self.last_mask is not None:
            if self.showing_mask:
                mask_pixmap = gray_to_qpixmap(self.last_mask)
                mask_pixmap = mask_pixmap.scaled(self.mask_widget.size(), QtCore.Qt.KeepAspectRatio,
                                                 QtCore.Qt.SmoothTransformation)
                self.mask_widget.setPixmap(mask_pixmap)

    def closeEvent(self, event):
        try:
            if hasattr(self, "timer"): self.timer.stop()
            if self.picam2 is not None:
                self.picam2.stop()
                self.picam2.close()

            if self.streamlit_process:
                self.streamlit_process.terminate()
                try:
                    self.streamlit_process.wait(timeout=3)
                except subprocess.TimeoutExpired:
                    self.streamlit_process.kill()
        except Exception:
            pass
        event.accept()


# ==========================
# ЗАПУСК
# ==========================
def main():
    app = QtWidgets.QApplication(sys.argv)
    app.setAttribute(QtCore.Qt.AA_EnableHighDpiScaling, True)
    app.setAttribute(QtCore.Qt.AA_UseHighDpiPixmaps, True)

    try:
        window = MainWindow()
        window.showMaximized()
        sys.exit(app.exec_())
    except Exception as e:
        QtWidgets.QMessageBox.critical(None, "Ошибка запуска", str(e))
        raise


if __name__ == "__main__":
    main()