import json
import os
import sys
from PyQt6.QtWidgets import QApplication, QMainWindow, QComboBox, QLineEdit
from qtpy import QtWidgets

from Backend.py_files.compute_similarity import compute_similarity_english,compute_similarity_german


class Recommender_UI(QMainWindow):
    def __init__(self):
        super().__init__()

        # Set the window title
        self.setWindowTitle('Reccomender System')
        #self.setFixedSize(400, 300)

        self.exercise_combo = QtWidgets.QComboBox(self)
        self.question_combo = QtWidgets.QComboBox(self)
        self.language_combo = QtWidgets.QComboBox(self)

        self.language_combo.addItem("Select Any")
        self.language_combo.addItems(['English', 'German'])

        exercise_label = QtWidgets.QLabel('Select Exercise No:')
        question_label = QtWidgets.QLabel('Select Question No')
        language_label = QtWidgets.QLabel('Select Language')

        self.submit_button = QtWidgets.QPushButton('Get Reccomendation', self)
        self.reset_recommender = QtWidgets.QPushButton('Reset System', self)
        self.display_label = QtWidgets.QLabel()


        self.submit_button.clicked.connect(self.submit_form)
        self.reset_recommender.clicked.connect(self.reset_recommender_func)

        layout = QtWidgets.QFormLayout()
        layout.addRow(language_label, self.language_combo)
        layout.addRow(exercise_label, self.exercise_combo)
        layout.addRow(question_label, self.question_combo)
        layout.addRow(self.submit_button)
        layout.addRow(self.reset_recommender)

        layout.addRow(self.display_label)

        self.submit_button.setFixedSize(200, 50)
        self.reset_recommender.setFixedSize(200, 50)

        self.submit_button.setStyleSheet("background-color: #466D1D; color: white;")
        self.reset_recommender.setStyleSheet("background-color: lightcoral; color: white;")

        central_widget = QtWidgets.QWidget()
        central_widget.setLayout(layout)
        self.setCentralWidget(central_widget)

        self.language_combo.currentTextChanged.connect(self.update_exercise_combo)
        self.exercise_combo.currentTextChanged.connect(self.update_question_combo)

        # Set the list of elements for the second combo box

        self.question_combo_list_german = {
            'Select Any': ['Select Any'],
            '3': ['1', '2', '3','4', '5(a)','5(b)','5(c)', '6(a)','6(b)','6(c)'],
            '9': ['1(a)', '1(b)', '1(c)','1(e)','1(f)','1(g)','2(b)','2(e)','3(a)',
                  '3(b)','3(c)','3(d)','3(e)','3(f)','4(c)','5(a)','5(b)','5(c)','5(d)'],
        }

        self.question_combo_list_english = {
            'Select Any': ['Select Any'],
            '3': ['1', '2', '3','4', '5', '6(a)','6(b)','6(c)'],
            '9': ['1(a)', '1(b)', '1(c)','1(d)','1(e)','2(a)','2(b)','2(c)','3(a)','3(b)'],
        }
        self.update_exercise_combo(self.language_combo.currentText())
        self.update_question_combo(self.exercise_combo.currentText())

    def update_question_combo(self, index):
        selected_language = self.language_combo.currentText()
        selected_exercise = self.exercise_combo.currentText()
        self.question_combo.clear()

        if (selected_language == 'Select Any'):
            self.display_label.setText('Pick Langauge')
        else:
            self.display_label.setText('')

            if not (selected_exercise == ''):
                if (selected_exercise == 'Select Any'):
                    self.display_label.setText('Pick Exercise')
                else:
                    if (selected_language == 'German'):
                        self.question_combo.addItems(self.question_combo_list_german[selected_exercise])
                    if (selected_language == 'English'):
                        self.question_combo.addItems(self.question_combo_list_english[selected_exercise])

    def update_exercise_combo(self, index):

        selected_language = self.language_combo.currentText()
        self.exercise_combo.clear()

        if (selected_language == 'German'):
            self.exercise_combo.addItem('Select Any')
            self.exercise_combo.addItems(['3', '9'])

        if (selected_language == 'English'):
            self.exercise_combo.addItem('Select Any')
            self.exercise_combo.addItems(['3', '8'])
    def submit_form(self):
        selected_language=  self.language_combo.currentText()
        selected_exercise = 'exercise' + self.exercise_combo.currentText()
        selected_task = self.question_combo.currentText()
        if ('Select Any' in [self.exercise_combo.currentText(),
                              selected_language, selected_task]):
            self.display_label.setText("Please select all values")
        else:
            if (selected_language == 'German'):
                filename = 'evaluation/evaluation_german_predicted.json'
            else:
                filename = 'evaluation/evaluation_predicted_english.json'

            if not os.path.exists(filename):
                self.display_label.setText("Please Reset Recommender")
            else:
                with open(filename, "r") as f:
                    data = json.load(f)
                exercise_flag=0
                task_flag=0
                for i in data:
                    if(selected_exercise==i['exercise']):
                        exercise_flag = 1

                        for j in i['tasks']:
                            if (str(selected_task) == j['task_number']):
                                task_flag=1
                                listToStr = ','.join([str(elem) for elem in j['pages'][:8]])

                                self.display_label.setText('Recommended Pages: '+listToStr)

                if(exercise_flag==0 or task_flag==0 ):
                    self.display_label.setText("Please Reset Recommender")

    def reset_recommender_func(self):
        compute_similarity_english()
        compute_similarity_german()
        self.display_label.setText("Recommender Reset successfully")


if __name__ == '__main__':
    app = QtWidgets.QApplication([])
    window = Recommender_UI()
    window.show()
    app.exec()
