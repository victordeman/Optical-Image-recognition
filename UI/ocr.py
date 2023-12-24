import json
import os
from PyQt6.QtWidgets import QApplication, QMainWindow, QComboBox, QLineEdit
from qtpy import QtWidgets

from Backend.py_files.try3 import read_data
from Backend.py_files.try4 import read_data_english


class OCR_UI(QMainWindow):
    def __init__(self):
        super().__init__()

        # Set the window title
        self.setWindowTitle('Keyword Utility')

        # Create a drop-down menu
        self.language_combo =QtWidgets.QComboBox(self)
        self.language_combo.addItem("Select Any")
        self.language_combo.addItems(['English', 'German'])
        self.selection = QtWidgets.QComboBox(self)
        self.selection.addItem("Select Any")
        self.selection.addItems(['Lecture', 'Exercise'])
        self.slide_path = QtWidgets.QLineEdit()
        slide_path_label = QtWidgets.QLabel('Enter Path')

        self.submit_button = QtWidgets.QPushButton('Submit', self)

        self.submit_button.setStyleSheet("background-color: #466D1D; color: white;")
        selection_label = QtWidgets.QLabel('Select Lecture/Exercise')

        language_label = QtWidgets.QLabel('Select Language:')

        self.display_label = QtWidgets.QLabel()

        layout = QtWidgets.QFormLayout()
        layout.addRow(language_label, self.language_combo)
        layout.addRow(selection_label, self.selection)
        layout.addRow(slide_path_label, self.slide_path)

        layout.addRow(self.submit_button)

        layout.addRow(self.display_label)

        central_widget = QtWidgets.QWidget()
        central_widget.setLayout(layout)
        self.setCentralWidget(central_widget)
        self.submit_button.clicked.connect(self.submit_form)

        self.submit_button.setFixedSize(200, 50)



    def submit_form(self):
        selected_language=  self.language_combo.currentText()
        selected_pdf=self.selection.currentText()
        path=self.slide_path.text()
        self.display_label.clear()

        if (selected_language=='Select Any'):
            self.display_label.setText("Please select the language")
        else:
            if(selected_pdf=='Select Any'):
                self.display_label.setText("Please exercise or lecture dropdown ")
            elif(path==''):
                self.display_label.setText("Enter path")
            else:

                if(selected_language=='German'):
                    print(selected_pdf)
                    if (selected_pdf=='Lecture'):
                        self.display_label.setText("Started OCR for lecture")

                        if(read_data(path)=='success'):

                            self.display_label.setText("OCR Process completed")

                    else:
                        self.display_label.setText("Started OCR for exercise")
                        if(read_data(path,1)=='success'):
                            self.display_label.setText("OCR Process completed")

                else:
                        print(selected_pdf)
                        if (selected_pdf == 'Lecture'):
                            self.display_label.setText("Started OCR for lecture")

                            if (read_data_english(path) == 'success'):
                                self.display_label.setText("OCR Process completed")

                        else:
                            self.display_label.setText("Started OCR for exercise")
                            if (read_data_english(path, 1) == 'success'):
                                self.display_label.setText("OCR Process completed")

#if __name__ == '__main__':
 #   app = QtWidgets.QApplication([])
  #  window = MainWindow()
   # window.show()
    #app.exec()
