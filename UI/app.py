from PyQt6.QtWidgets import QApplication, QWidget, QVBoxLayout, QStackedWidget, QPushButton, QLabel, QHBoxLayout
from PyQt6.QtCore import Qt
from keywords_ui import Keyword_UI
from eval_ui import Eval_UI
from ex_lec import Ex_lec_UI

from ocr import OCR_UI

class MainWidget(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('Slide Recommendation System')
        self.resize(500, 500)
        # Create a stacked widget to manage multiple screens
        self.stacked_widget = QStackedWidget()

        # Create each of the screens and add them to the stacked widget
        self.screen1 = Eval_UI()
        self.screen2 = Keyword_UI()
        self.screen3 = Ex_lec_UI()
        self.screen4 = OCR_UI()

        self.stacked_widget.addWidget(self.screen1)
        self.stacked_widget.addWidget(self.screen2)
        self.stacked_widget.addWidget(self.screen3)
        self.stacked_widget.addWidget(self.screen4)

        # Create a back button and connect it to the back method


        # Create buttons for each screen and connect them to the corresponding screen
        self.button1 = QPushButton('Manual Mapping')
        #self.button1.clicked.connect(lambda: self.stacked_widget.setCurrentWidget(self.screen1))
        self.button1.clicked.connect(self.manual_mapping)

        self.button2 = QPushButton("Keyword Input")
        self.button2.clicked.connect(self.keyword_input)


        self.button3 = QPushButton("Exercise-Lecture Mapping")
        self.button3.clicked.connect(self.recommender_window)


        self.button4 = QPushButton("OCR Window")
        self.button4.clicked.connect(self.ocr_ui)

        self.button1.setFixedSize(200, 50)
        self.button2.setFixedSize(200, 50)
        self.button3.setFixedSize(200, 50)
        self.button4.setFixedSize(200, 50)

        self.button1.setStyleSheet("background-color: #2C3E4C; color: white;")
        self.button2.setStyleSheet("background-color: #2C3E4C; color: white;")
        self.button3.setStyleSheet("background-color: #2C3E4C; color: white;")
        self.button4.setStyleSheet("background-color: #2C3E4C; color: white;")

        # Create a layout to arrange the widgets
        layout = QVBoxLayout()
        layout.addWidget(self.stacked_widget)

        hbox = QHBoxLayout()
        hbox.addWidget(self.button1)
        hbox.addWidget(self.button2)
        hbox.addWidget(self.button3)
        hbox.addWidget(self.button4)

        layout.addLayout(hbox)
        self.setLayout(layout)
    def manual_mapping(self):
        self.stacked_widget.setCurrentWidget(self.screen1)
        self.setWindowTitle('Manual Mapping Window')

    def keyword_input(self):
        self.stacked_widget.setCurrentWidget(self.screen2)
        self.setWindowTitle('Input Keyword Window')

    def recommender_window(self):
        self.stacked_widget.setCurrentWidget(self.screen3)
        self.setWindowTitle('Exercise-lecture Mapping Window')

    def ocr_ui(self):
        self.stacked_widget.setCurrentWidget(self.screen4)
        self.setWindowTitle('OCR Window')



if __name__ == '__main__':
    app = QApplication([])
    widget = MainWidget()
    widget.show()
    app.exec()
