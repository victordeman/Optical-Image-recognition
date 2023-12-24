import json
import os

import numpy as np
from statistics import mean

def start_eval(language='german'):
    if(language=='german'):
        predicted_json_list = json.load(open('evaluation/bert_evaluation_german_predicted.json'))
        actual_json_list = json.load(open('evaluation/evaluation_ui_german.json'))
    else:
        predicted_json_list = json.load(open('evaluation/bert_evaluation_english_predicted.json'))
        actual_json_list = json.load(open('evaluation/evaluation_ui_english.json'))

    print(os.getcwd())

    overall_precision=[]
    overall_recall=[]
    for predicted in predicted_json_list:

        print('Overall Predicted')
        predicted_tasks=predicted['tasks']
        print(predicted_tasks)

        for actual in actual_json_list:

            if(actual['exercise']==predicted['exercise']):
                actual_tasks = actual['tasks']
                print('Overall Actual')

                print(actual_tasks)
                print('\n')
                for i in predicted_tasks:
                    task_no=i['task_number']
                    print(task_no)

                    for j in actual_tasks:
                        if(j['task_number']==task_no):

                            actual_pages = j['pages']

                          #  print('predicted trimmed')

                            predicted_pages = i['pages']

                            print('Actual ',actual_pages)
                            print('Predicted ',predicted_pages)
                            tp = len(set(actual_pages) & set(predicted_pages))  # True positives
                            fp = len(set(predicted_pages) - set(actual_pages))  # False positives
                            fn = len(set(actual_pages) - set(predicted_pages))  # False negatives

                            precision = tp / (tp + fp)
                            recall = tp / (tp + fn)
                            print(tp,fp,fn)
                            print("Precision: {:.2f}".format(precision))
                            print("Recall: {:.2f}".format(recall))

                            print('\n\n')
                            overall_precision.append(precision)
                            overall_recall.append(recall)


    print('Overall Precision ',mean(overall_precision))
    print('Overall Recall ',mean(overall_recall))
    print('F1-score',(2*mean(overall_precision)*mean(overall_recall)) /
          (mean(overall_precision)+mean(overall_recall)))



if __name__ == "__main__":
        start_eval()
