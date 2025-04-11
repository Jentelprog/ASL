import cv2
import sys
import numpy as np
import tensorflow as tf
from cvzone.HandTrackingModule import HandDetector
from PyQt5.QtWidgets import QApplication, QWidget, QLabel, QVBoxLayout, QPushButton, QStackedWidget
from PyQt5.QtGui import QFont, QImage, QPixmap
from PyQt5.QtCore import QTimer, Qt


class Accueil(QWidget):
    def __init__(self, stacked_widget):
        super().__init__()
        self.stacked_widget = stacked_widget
        self.setWindowTitle("Accueil")
        self.setGeometry(100, 100, 800, 600)
        self.setStyleSheet("""
            QWidget { background: qlineargradient(spread:pad, x1:0, y1:0, x2:1, y2:1, 
                                                stop:0 rgba(173,163,172,1), 
                                                stop:0.5 rgba(97,67,89,0.9), 
                                                stop:1 rgba(50,38,57,0.8)); }
        """)

        self.image_label = QLabel(self)
        pixmap = QPixmap("ASL-main/image/image1.png")
        self.image_label.setPixmap(pixmap)
        self.image_label.setAlignment(Qt.AlignCenter)

        self.button = QPushButton("Start", self)
        self.button.setFont(QFont('Arial', 16, QFont.Bold, italic=True))
        self.button.setStyleSheet("""
            QPushButton { background-color: rgba(176,118,151,0.9); color: white; padding: 12px; border-radius: 10px; }
            QPushButton:hover { background-color: rgba(97,67,89,1); }
        """)
        self.button.clicked.connect(self.ouvrir_detection)

        layout = QVBoxLayout()
        layout.addWidget(self.image_label)
        layout.addWidget(self.button)
        self.setLayout(layout)

    def ouvrir_detection(self):
        self.stacked_widget.widget(1).start_camera()
        self.stacked_widget.setCurrentIndex(1)


class InterfaceASL(QWidget):
    def __init__(self, stacked_widget, model):
        super().__init__()
        self.stacked_widget = stacked_widget
        self.model = model
        self.setWindowTitle("Détection ASL en temps réel")
        self.setGeometry(100, 100, 800, 600)
        self.setStyleSheet("""
            QWidget { background: qlineargradient(spread:pad, x1:0, y1:0, x2:1, y2:1, 
                                                stop:0 rgba(34,29,39,1), 
                                                stop:0.5 rgba(97,67,89,0.9), 
                                                stop:1 rgba(176,118,151,0.8)); }
        """)

        self.label = QLabel('Detected Word :', self)
        self.label.setFont(QFont('Arial', 20, QFont.Bold, italic=True))
        self.label.setWordWrap(True)

        self.image_label = QLabel(self)
        self.prediction_label = QLabel(self)
        self.prediction_label.setFont(QFont('Arial', 18, QFont.Bold))
        self.prediction_label.setAlignment(Qt.AlignCenter)

        self.retour_button = QPushButton("Go back", self)
        self.retour_button.setFont(QFont('Arial', 16, QFont.Bold, italic=True))
        self.retour_button.setStyleSheet("""
            QPushButton { background-color: rgba(176,118,151,0.9); color: white; padding: 12px; border-radius: 10px; }
            QPushButton:hover { background-color: rgba(97,67,89,1); }
        """)
        self.retour_button.clicked.connect(self.retour_accueil)

        layout = QVBoxLayout()
        layout.addWidget(self.image_label)
        layout.addWidget(self.prediction_label)
        layout.addWidget(self.label)
        layout.addWidget(self.retour_button)
        self.setLayout(layout)

        self.capture = None
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_frame)

        self.detector = HandDetector(maxHands=1)
        self.labels = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ") + ["SPACE"]
        self.recognized_text = ""

    def start_camera(self):
        if self.capture:
            self.capture.release()
        self.capture = cv2.VideoCapture(0)
        self.timer.start(30)

    def stop_camera(self):
        if self.capture:
            self.timer.stop()
            self.capture.release()
            self.capture = None

    def update_frame(self):
        if self.capture is not None:
            ret, frame = self.capture.read()
            if ret:
                self.show_prediction(frame)
                rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                h, w, c = rgb_image.shape
                bytes_per_line = c * w
                q_image = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
                self.image_label.setPixmap(QPixmap.fromImage(q_image))

    def show_prediction(self, frame):
        letter = self.detect_letter(frame)
        if letter:
            self.prediction_label.setText(f"Detected Letter: {letter}")
        else:
            self.prediction_label.setText("")

    def detect_letter(self, frame):
        hands, _ = self.detector.findHands(frame, draw=True)
        if hands:
            x, y, w, h = hands[0]['bbox']
            imgCrop = frame[max(0, y - 20):min(frame.shape[0], y + h + 20),
                      max(0, x - 20):min(frame.shape[1], x + w + 20)]
            if imgCrop.size == 0:
                return None
            imgGray = cv2.cvtColor(imgCrop, cv2.COLOR_BGR2GRAY)
            imgResized = cv2.resize(imgGray, (128, 128))
            imgInput = np.expand_dims(imgResized.astype('float32') / 255.0, axis=(0, -1))
            predictions = self.model.predict(imgInput)
            return self.labels[np.argmax(predictions)] if np.max(predictions) > 0.8 else None
        return None

    def retour_accueil(self):
        self.stop_camera()
        self.recognized_text = ""
        self.label.setText("Detected Word :")
        self.prediction_label.setText("")
        self.stacked_widget.setCurrentIndex(0)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    stacked_widget = QStackedWidget()
    model = tf.keras.models.load_model("model/hand_sign_model.h5")
    stacked_widget.addWidget(Accueil(stacked_widget))
    stacked_widget.addWidget(InterfaceASL(stacked_widget, model))
    stacked_widget.setCurrentIndex(0)
    stacked_widget.show()
    sys.exit(app.exec_())
