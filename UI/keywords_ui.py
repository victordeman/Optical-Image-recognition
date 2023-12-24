import json
import os
import sys
from PyQt6.QtWidgets import QApplication, QMainWindow, QComboBox, QLineEdit
from qtpy import QtWidgets


class Keyword_UI(QMainWindow):
    def __init__(self):
        super().__init__()

        # Set the window title
        self.setWindowTitle('Keyword Utility')

        # Create a drop-down menu
        self.language_combo =QtWidgets.QComboBox(self)
        self.language_combo.addItem("Select Any")
        self.language_combo.addItems(['English', 'German'])

        # Create two text boxes
        self.alternate_word = QtWidgets.QLineEdit(self)
        self.keyword = QtWidgets.QLineEdit(self)
        self.submit_button = QtWidgets.QPushButton('Submit', self)
        self.delete_button = QtWidgets.QPushButton('Delete', self)

        self.submit_button.setStyleSheet("background-color: #466D1D; color: white;")
        self.delete_button.setStyleSheet("background-color: lightcoral; color: white;")

        language_label = QtWidgets.QLabel('Select Language:')
        alternate_word_label = QtWidgets.QLabel('Input alternate word')
        sql_keyword_label = QtWidgets.QLabel('Input sql keyword')
        self.display_label = QtWidgets.QLabel()

        layout = QtWidgets.QFormLayout()
        layout.addRow(language_label, self.language_combo)
        layout.addRow(alternate_word_label, self.alternate_word)
        layout.addRow(sql_keyword_label, self.keyword)
        layout.addRow(self.submit_button)
        layout.addRow(self.delete_button)

        layout.addRow(self.display_label)

        central_widget = QtWidgets.QWidget()
        central_widget.setLayout(layout)
        self.setCentralWidget(central_widget)
        self.submit_button.clicked.connect(self.submit_form)
        self.delete_button.clicked.connect(self.delete_keyword)

        self.submit_button.setFixedSize(200, 50)
        self.delete_button.setFixedSize(200, 50)



    def submit_form(self):
        selected_language=  self.language_combo.currentText()
        alternate_word=self.alternate_word.text().lower()
        sql_keyword=self.keyword.text().lower()

        if (selected_language=='Select Any') or (alternate_word=='') or (sql_keyword==''):
            self.display_label.setText("Please select all values")
        else:
            if(selected_language=='German'):
                keyword_file= 'dataset/keywords/german_keywords.json'
            else:
                keyword_file= 'dataset/keywords/english_keywords.json'
            if os.path.exists(keyword_file):
                with open(keyword_file, "r") as f:
                    data = json.load(f)
            else:
                with open(keyword_file, "w") as f:
                    json.dump({}, f)
            print(data)
            if alternate_word in data.keys():
                data[alternate_word]=sql_keyword
                self.display_label.setText("Key Already exists and modified")

            else:
                data[alternate_word]=sql_keyword

                self.display_label.setText("New key added")
            print(data)
            with open(keyword_file, 'w', encoding='utf-8') as f:
                json.dump(data ,f,ensure_ascii=False)

    def delete_keyword(self):
        selected_language = self.language_combo.currentText()
        alternate_word = self.alternate_word.text().lower()

        if (selected_language == 'Select Any') or (alternate_word == '') :
            self.display_label.setText("Please provide language and Word to be deleted")
        else:
            if (selected_language == 'German'):
                keyword_file = 'dataset/keywords/german_keywords.json'
            else:
                keyword_file = 'dataset/keywords/english_keywords.json'
            if os.path.exists(keyword_file):
                with open(keyword_file, "r") as f:
                    data = json.load(f)
            else:
                with open(keyword_file, "w") as f:
                    json.dump({}, f)
            print(data)
            if alternate_word in data.keys():
                del data[alternate_word]
                self.display_label.setText("Key Deleted")

            else:
                self.display_label.setText("Word not found")
            print(data)
            with open(keyword_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False)


#if __name__ == '__main__':
 #   app = QtWidgets.QApplication([])
  #  window = MainWindow()
   # window.show()
    #app.exec()
