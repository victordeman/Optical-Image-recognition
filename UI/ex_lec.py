import json
import os
import sys

from PyQt6 import QtWidgets
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QAction
from PyQt6.QtWidgets import QApplication, QWidget, QComboBox, QCheckBox, QVBoxLayout, QLabel, QMenu


class Ex_lec_UI(QWidget):
    def __init__(self):
        super().__init__()

        # Create the first dropdown
        self.display_label = QLabel()

        language_label = QLabel('Select Language')
        self.language_combo = QComboBox()
        self.language_combo.addItem("Select Any")
        self.language_combo.addItems(['English', 'German'])

        # Create the second dropdown
        exercise_label = QLabel('Select exercise No')
        self.exercise_combo = QComboBox()
        self.exercise_combo.addItem("Select Any")
        self.exercise_combo.addItems(['3', '7','8','9'])

        lecture_label = 'Enter Lecture numbers'
        self.lineedit = QtWidgets.QLineEdit()

        # Set the layout
        self.submit_button = QtWidgets.QPushButton('Submit', self)
        self.submit_button.clicked.connect(self.submit_form)
        self.submit_button.setFixedSize(200, 50)

        self.submit_button.setStyleSheet("background-color: #466D1D; color: white;")

        # Create the layout
        layout = QtWidgets.QFormLayout()
        layout.addRow(language_label, self.language_combo)
        layout.addRow(exercise_label, self.exercise_combo)
        layout.addRow(lecture_label, self.lineedit)
        layout.addRow(self.submit_button)
        layout.addRow(self.display_label)

        self.setLayout(layout)

    def submit_form(self):
        selected_language = self.language_combo.currentText()

        selected_exercise = self.exercise_combo.currentText()

        selected_lectures = self.lineedit.text()
        if('Select Any' in [self.exercise_combo.currentText(),self.language_combo.currentText()] or
                selected_lectures==''):
            self.display_label.setText("Please select all values")
        else:
            selected_exercise='exercise'+selected_exercise
            lectures= [f'dbs-{value.strip()}' for value in selected_lectures.split(',')]
            filename = 'dataset/keywords/ex_lec_mapping.json'
            if os.path.exists(filename):
                with open(filename, "r") as f:
                    data = json.load(f)
                # print(f"File {filename} exists and contains: {data}")
            else:
                with open(filename, "w") as f:
                    json.dump({}, f)
                    data = {}
            if not (data):
                final_list=[]
                new_json={}
                temp={}
                new_json['language']=selected_language
                temp['exercise']=selected_exercise
                temp['lecture']=lectures

                new_json['mapping'] = []
                new_json['mapping'].append(temp)

                final_list.append(new_json)
                print(final_list)
                with open(filename, "w") as f:
                    json.dump(final_list, f)
                self.display_label.setText('Added record')
            else:
                flag_exercise = 0

                for i in data:
                    if(i['language']==selected_language):
                        for j in i['mapping']:
                            if (j['exercise'] == selected_exercise):
                                flag_exercise = 1
                                exisiting_lecs = j['lecture']
                                j['lecture']=lectures
                        if(flag_exercise==0):
                            new_json = {}
                            temp = {}
                            temp['exercise'] = selected_exercise
                            temp['lecture'] = lectures
                            i['mapping'].append(temp)
                            flag_exercise = 1

                if (flag_exercise == 0):
                    temp = {}
                    new_json={}
                    new_json['language'] = selected_language
                    temp['exercise'] = selected_exercise
                    temp['lecture'] = lectures

                    new_json['mapping'] = []
                    new_json['mapping'].append(temp)

                    data.append(new_json)

                with open(filename, "w") as f:
                    json.dump(data, f)
                    self.display_label.setText('Added record')
            self.display_label.setText(" Lecture and Exercise remapping success")






if __name__ == '__main__':
    app = QApplication(sys.argv)
    widget = Ex_lec_UI()
    widget.show()
    sys.exit(app.exec())
