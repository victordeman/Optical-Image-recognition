import json
import os

from PyQt6 import QtCore, QtGui, QtWidgets
from PyQt6.QtWidgets import QLabel


class Eval_UI(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()

        # Set the window title
        self.setWindowTitle('Slide Recommendation Mapping Tool')

        # Create the two QComboBoxes
        self.exercise_combo = QtWidgets.QComboBox(self)
        self.question_combo = QtWidgets.QComboBox(self)
        self.language_combo=   QtWidgets.QComboBox(self)
        self.lecture_combo=   QtWidgets.QComboBox(self)

        self.language_combo.addItem("Select Any")
        self.language_combo.addItems(['English', 'German'])
        self.lecture_combo.addItem("Select Any")
        self.lecture_combo.addItems(['dbs-2','dbs-6', 'dbs-7'])

        # Set labels for the QComboBoxes

        exercise_label = QtWidgets.QLabel('Select Exercise No:')
        question_label = QtWidgets.QLabel('Select Question No')
        language_label = QtWidgets.QLabel('Select Language')
        lecture_label = QtWidgets.QLabel('Enter Lecture Number')
        slides_label = QtWidgets.QLabel('Enter Page Numbers(PDF page number)')

        self.display_label = QLabel()

        # Create the QLineEdit for entering numbers
        self.lineedit = QtWidgets.QLineEdit()

        self.submit_button = QtWidgets.QPushButton('Submit', self)
        self.submit_button.clicked.connect(self.submit_form)
        self.reset_mapping = QtWidgets.QPushButton('Reset Mapping', self)
        self.reset_mapping.clicked.connect(self.reset_mapping_func)
        # Layout the QComboBoxes using a QFormLayout
        layout = QtWidgets.QFormLayout()
        layout.addRow(language_label, self.language_combo)
        layout.addRow(exercise_label, self.exercise_combo)
        layout.addRow(question_label, self.question_combo)
        layout.addRow(lecture_label, self.lecture_combo)
        layout.addRow(slides_label, self.lineedit)

        layout.addRow(self.submit_button)
        layout.addRow(self.reset_mapping)

        layout.addRow(self.display_label)

        self.language_combo.currentTextChanged.connect(self.update_exercise_combo)
        self.exercise_combo.currentTextChanged.connect(self.update_question_combo)

        self.submit_button.setFixedSize(200, 50)
        self.reset_mapping.setFixedSize(200, 50)

        self.submit_button.setStyleSheet("background-color: #466D1D; color: white;")
        self.reset_mapping.setStyleSheet("background-color: lightcoral; color: white;")
        # Set the list of elements for the second combo box

        self.question_combo_list_german = {
            'Select Any':['Select Any'],
            '3': ['1', '2', '3','4', '5(a)','5(b)', '6(a)','6(b)','6(c)'],
            '9': ['1(a)', '1(b)', '1(c)', '1(d)', '1(e)', '1(f)', '1(g)',
                  '2(a)', '2(b)', '2(c)', '2(d)', '2(e)',
                  '3(a)', '3(b)', '3(c)', '3(d)', '3(e)', '3(f)',
                  '4(c)',
                  '5(a)', '5(b)', '5(c)', '5(d)'
                  ],
        }


        self.question_combo_list_english = {
            'Select Any': ['Select Any'],
            '3': ['1', '2', '3','4'],
            '8': ['1(a)', '1(b)', '1(c)', '1(d)', '1(e)', '1(f)', '1(g)',
                  '2(a)', '2(b)', '2(c)', '2(d)', '2(e)',
                  '3(a)', '3(b)', '3(c)', '3(d)', '3(e)', '3(f)',
                  '4(a)','4(b)'
                  ],
        }

        self.update_exercise_combo(self.language_combo.currentText())
        self.update_question_combo(self.exercise_combo.currentText())

        central_widget = QtWidgets.QWidget()
        central_widget.setLayout(layout)
        self.setCentralWidget(central_widget)
    def update_question_combo(self,index):
        selected_language = self.language_combo.currentText()
        selected_exercise=self.exercise_combo.currentText()
        self.question_combo.clear()

        if(selected_language=='Select Any'):
            self.display_label.setText('Pick Langauge')
        else:
            self.display_label.setText('')

            if not (selected_exercise==''):
                if(selected_exercise=='Select Any'):
                    self.display_label.setText('Pick Exercise')
                else:
                    if(selected_language=='German'):
                        self.question_combo.addItems(self.question_combo_list_german[selected_exercise])
                    if (selected_language == 'English'):
                        self.question_combo.addItems(self.question_combo_list_english[selected_exercise])

    def update_exercise_combo(self, index):

        selected_language = self.language_combo.currentText()
        self.exercise_combo.clear()

        if(selected_language=='German'):
            self.exercise_combo.addItem('Select Any')
            self.exercise_combo.addItems(['3','9'])

        if(selected_language=='English'):
            self.exercise_combo.addItem('Select Any')
            self.exercise_combo.addItems(['3','8'])

        # Add the new items to the second combo box
        #self.question_combo.addItems(self.question_combo_list_german[selected_value])

    def update_list(self, text):
        # Split the text into a list of numbers using commas as the delimiter
        numbers = [x.strip() for x in text.split(',')]

        return(numbers)

    def submit_form(self):
        selected_exercise= 'exercise'+self.exercise_combo.currentText()
        selected_task= self.question_combo.currentText()
        selected_language=self.language_combo.currentText()

        selected_lecture=self.lecture_combo.currentText()

        given_pages= self.lineedit.text()
        tasks= self.update_list(given_pages)

        if('Select Any' in [self.exercise_combo.currentText(),
                            selected_task,selected_language,selected_lecture]
                or tasks[0]==''):
            self.display_label.setText("Please select all values")

        else:
            tasks = [selected_lecture +'-page_'+item for item in tasks]
            if(selected_language=='German'):
                filename = 'evaluation/evaluation_ui_german.json'
            else:
                filename = 'evaluation/evaluation_ui_english.json'

            if os.path.exists(filename):
                with open(filename, "r") as f:
                    data = json.load(f)
                #print(f"File {filename} exists and contains: {data}")
            else:
                with open(filename, "w") as f:
                    json.dump({}, f)
                    data = {}
                #print(f"File {filename} created with empty JSON object")

            if not (data):
                final_list=[]
                new_json={}
                temp={}
                new_json['exercise']=selected_exercise
                new_json['tasks'] = []
                temp['task_number']=selected_task
                temp['pages']=tasks
                new_json['tasks'].append(temp)
                final_list.append(new_json)
                print(final_list)
                with open(filename, "w") as f:
                    json.dump(final_list, f)
                self.display_label.setText('Added record')


            else:
                flag_exercise=0
                flag_task=0
                for i in data:
                    if(i['exercise']==selected_exercise):

                        flag_exercise=1
                        exisiting_tasks=i['tasks']
                        for j in exisiting_tasks:
                            if(j['task_number']==selected_task):

                                flag_task=1
                                j['pages'] =tasks

                        if(flag_task==0):

                            temp={}
                            temp['task_number'] = selected_task
                            temp['pages'] = tasks
                            exisiting_tasks.append(temp)

                if(flag_exercise==0):

                    new_json = {}
                    temp = {}
                    new_json['exercise'] = selected_exercise
                    new_json['tasks'] = []
                    temp['task_number'] = selected_task
                    temp['pages'] = tasks
                    new_json['tasks'].append(temp)
                    data.append(new_json)
                with open(filename, "w") as f:
                    json.dump(data, f)
                    self.display_label.setText('Added record')

    def reset_mapping_func(self):
        selected_language=self.language_combo.currentText()
        if(selected_language=='Select Any'):
            self.display_label.setText('Please select Language before Resetting')
        else:
            if (selected_language == 'German'):
                    filename = 'evaluation/evaluation_ui_german.json'
            if (selected_language == 'English'):
                    filename = 'evaluation/evaluation_ui_english.json'

            if os.path.exists(filename):
                os.remove(filename)
            self.display_label.setText('Reset Complete')




#if __name__ == '__main__':
 #   app = QtWidgets.QApplication([])
  #  window = MainWindow()
   # window.show()
    #app.exec()
