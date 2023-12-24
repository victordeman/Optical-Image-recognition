import os
from collections import Counter

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import json
import numpy as np
ex_lect='dataset/keywords/ex_lec_mapping.json'
if os.path.exists(ex_lect):
        with open(ex_lect, "r") as f:
                ex_lec_mapping = json.load(f)
print(ex_lec_mapping)
for i in ex_lec_mapping:
        if i['language']=='German':
                german_ex_lec=i['mapping']
        else:
                english_ex_lec=i['mapping']
def compute_similarity_german():
        lecture_matrix = pd.read_csv('Backend/py_files/german_lecture_csv.csv',index_col=0)
        lecture_matrix = lecture_matrix[lecture_matrix['FrequentWords'].notna()]
        exercise_matrix = pd.read_csv('Backend/py_files/german_exercise.csv',index_col=0)
        exercise_matrix = exercise_matrix[exercise_matrix['FrequentWords'].notna()]

        tfidf_vectorizer = TfidfVectorizer()
        combined_texts = pd.concat([lecture_matrix['FrequentWords'],
                                    exercise_matrix['FrequentWords']])
        tfidf_vectorizer.fit(combined_texts)
        tfidf_matrix1 = tfidf_vectorizer.transform(lecture_matrix['FrequentWords'])
        flag=0
        final_json=[]
        prev_exercise=''
        predicted_json = {}

        for i in exercise_matrix.index:
                print('\n')
                temp = {}
                exercise=i.split('_')[0]
                question_number = (i.split('_')[-1])
                if(exercise!=prev_exercise):
                    flag=0
                    if(predicted_json):
                            final_json.append(predicted_json)
                if(flag==0):
                        predicted_json = {}

                        predicted_json['exercise']=exercise
                        predicted_json['tasks']=[]
                if(flag==1):
                        print(lecture_matrix.columns)
                        lecture_matrix=lecture_matrix.drop('cosine_scores_for'+exercise+'_'+prev,axis=1)

                print('Question:',i)
                print('Question content: ',exercise_matrix.loc[i]['FrequentWords'])
                print('Useful Slides sorted with cosine similarity\n')
                tfidf_matrix2 = tfidf_vectorizer.transform([exercise_matrix.loc[i]['FrequentWords']])

                cosine_similarities = cosine_similarity(tfidf_matrix1, tfidf_matrix2)
                # print(cosine_similarities)
                lecture_matrix['cosine_scores_for'+str(i)]=cosine_similarities
                useful_df=lecture_matrix[lecture_matrix['cosine_scores_for'+str(i)]>0.1]
                useful_df=useful_df.sort_values('cosine_scores_for'+str(i),ascending=False)

                top=useful_df.index
                filtered_top=[]
                for j in german_ex_lec:
                        if (j['exercise'] == exercise):
                                map = j['lecture']
                                for k in map:
                                        for l in top:
                                                if k in l:
                                                        filtered_top.append(l)
                print('\n')
                flag=1
                prev_exercise=exercise
                temp['task_number']=question_number
                temp['pages']=list(filtered_top[:5])
                predicted_json['tasks'].append(temp)
                prev=question_number
        final_json.append(predicted_json)
        print(final_json)
        with open("evaluation/evaluation_german_predicted.json", "w") as write_file:
                json.dump(final_json, write_file)

def compute_similarity_english():
        lecture_matrix = pd.read_csv('Backend/py_files/english_lecture_csv.csv',index_col=0)
        lecture_matrix = lecture_matrix[lecture_matrix['FrequentWords'].notna()]
        exercise_matrix = pd.read_csv('Backend/py_files/english_exercise.csv',index_col=0)
        exercise_matrix = exercise_matrix[exercise_matrix['FrequentWords'].notna()]

        tfidf_vectorizer = TfidfVectorizer()
        combined_texts = pd.concat([lecture_matrix['FrequentWords'],
                                    exercise_matrix['FrequentWords']])
        tfidf_vectorizer.fit(combined_texts)
        tfidf_matrix1 = tfidf_vectorizer.transform(lecture_matrix['FrequentWords'])
        flag=0
        final_json=[]
        prev_exercise=''
        predicted_json = {}

        for i in exercise_matrix.index:
                print('\n')
                temp = {}
                exercise=i.split('_')[0]

                question_number = (i.split('_')[-1])
                if(exercise!=prev_exercise):
                    flag=0
                    if(predicted_json):
                            final_json.append(predicted_json)
                if(flag==0):
                        predicted_json = {}

                        predicted_json['exercise']=exercise
                        predicted_json['tasks']=[]
                if(flag==1):
                        print(lecture_matrix.columns)
                        lecture_matrix=lecture_matrix.drop('cosine_scores_for'+exercise+'_'+prev,axis=1)

                print('Question:',i)
                print('Question content: ',exercise_matrix.loc[i]['FrequentWords'])
                print('Useful Slides sorted with cosine similarity\n')
                tfidf_matrix2 = tfidf_vectorizer.transform([exercise_matrix.loc[i]['FrequentWords']])

                cosine_similarities = cosine_similarity(tfidf_matrix1, tfidf_matrix2)
                # print(cosine_similarities)
                lecture_matrix['cosine_scores_for'+str(i)]=cosine_similarities
                useful_df=lecture_matrix[lecture_matrix['cosine_scores_for'+str(i)]>0.2]
                useful_df=useful_df.sort_values('cosine_scores_for'+str(i),ascending=False)
                top = useful_df.index
                filtered_top = []
                for j in english_ex_lec:
                        if ('exc'+j['exercise'][-1] == exercise):
                                map = j['lecture']
                                for k in map:
                                        for l in top:
                                                if k in l:
                                                        filtered_top.append(l)
                print('\n')
                flag = 1
                prev_exercise = exercise
                temp['task_number'] = question_number
                temp['pages'] = list(filtered_top[:5])
                predicted_json['tasks'].append(temp)
                prev = question_number
        final_json.append(predicted_json)
        print(final_json)
        with open("evaluation/evaluation_english_predicted.json", "w") as write_file:
                json.dump(final_json, write_file)


if __name__ == "__main__":
        compute_similarity_german()
        compute_similarity_english()