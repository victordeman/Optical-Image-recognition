import os
import pytesseract
from PIL import Image
from pdf2image import convert_from_path
import pathlib
import json
from nltk.corpus import stopwords
import numpy as np
import re
stopwords=stopwords.words('german')
import pandas as pd

global keywords, slidekeyword, questionkeyword,mappings
# keywords=['select','join','where','having','table','update','insert','truncate','delete','get','count','distinct','unique','union','values','outer','inner','order','natural','limit','like','having']
keywords = ['datenbanken', 'legen', 'sie', 'mittels', 'legen sie mittels', 'ausgeben', 'beschreiben', 'listen',
            'wo', 'where-klausel', 'where', 'löschen', 'delete', 'geordnet nach', 'anzahl', 'durchschnittlich',
            'durchschnittliche', 'durchschnittliches', 'durchschnittlichen', 'aggregatfunktionen', 'sum', 'zwischen',
            'natural-join', 'natural join', 'left-outer-join', 'right-outer-join', 'full-outer-join', 'gleichverbund',
            'erzeugen', 'erstellen', 'entwerfen', 'anlegen', 'tabelle', 'relation', 'relationen', 'spalte',
            'spalten', 'eindeutig', 'eindeutigkeit', 'wertebereich', 'distinct', 'verschieden', 'verschiedene',
            'eindeutig', 'eindeutige', 'niedrigsten', 'niedrigste', 'fremdschlüssel', 'primärschlussel',
            'cardinalitäten', 'sortieren', 'geordnet nach', 'aktualisieren','select','join','where','having','table',
            'update','insert','truncate','delete','get','count','distinct','unique','union','values','outer','inner',
            'order','natural','limit','like','having','create table']


with open('dataset/keywords/german_keywords.json') as json_file:
    mappings = json.load(json_file)

def read_data(folderPath,exercise=0):
    path = pathlib.Path(folderPath)

    slidekeyword = dict()

    if(exercise==1):
        temp_path = 'Backend/Out/exercise_images/'

        for f in os.listdir(temp_path):
            os.remove(os.path.join(temp_path, f))
            print("removed all the temp files")
        questionkeyword_df = dict()
        for item in path.rglob('*.pdf'):
            doc = convert_from_path(item)
            all_sentences = list()
            for count, img in enumerate(doc):
                img_name = f"{str(item).split('/')[-1].split('.')[0]}"
                img.save(os.path.join(temp_path, img_name + ".png"), "png")
                extracted_text = pytesseract.image_to_string(Image.open(os.path.join(temp_path, img_name + ".png")),lang="deu",config='--psm 6')
                lowercased_extracted_text = extracted_text.lower()
                
                lowercased_extracted_text = lowercased_extracted_text.replace('\n(','\n\n(').replace('\n1.','\n\n1.').replace('5. gegeben',' \n\n5. gegeben')
               # lowercased_extracted_text = lowercased_extracted_text.replace('\n', '\n\n')
                sentences = lowercased_extracted_text.split("\n\n")
                modified_sentences = list()
                for i in sentences:
                    if re.match(r'^\d.|^\([abcdefghijklmnopqrstuvwxyz]\)|\n\([a-zA-Z]+\)|\n\d.', i):
                        modified_sentences.append(i)
                    else:
                        try:
                            modified_sentences[len(modified_sentences)-1] = modified_sentences[len(modified_sentences)-1] + i 
                        except:
                            print('First Lines')
                
                sentences = modified_sentences


                print('original sentences:')
                print(sentences)
                print('After Stopword Removal and preprocessing:')
                preprocessed_text = [' '.join(w for w in p.split() if w not in stopwords) for p in sentences]
                print(preprocessed_text)
                final_result = list(filter(lambda x: re.match(r'^\d.|^\([abcdefghijklmnopqrstuvwxyz]\)', x), preprocessed_text))
                all_sentences.extend(final_result)
                print("final_Result in list")
                print(all_sentences)
                all_questions= dict()
                print("Final Object")
            for i in range(len(all_sentences)) :
                questions_dict= dict()
                #questions_dict["page_no"]=count+1
                if(re.match(r'^\d.', all_sentences[i])):
                    questions_dict["question_no"] = all_sentences[i][0]
                    questions_dict["question"]=all_sentences[i]
                    #questions_dict["sub_parts"] =""
                    question_line = i
                    if(i < len(all_sentences) - 1 and re.match(r'^\([abcdefghijklmnopqrstuvwxyz]\)', all_sentences[i+1])):                            
                        #sub_parts=[]
                        for j in range(i+1,len(all_sentences)):
                            if(re.match(r'^\([abcdefghijklmnopqrstuvwxyz]\)', all_sentences[j])):
                                questions_dict= dict()
                                questions_dict["question"] = all_sentences[question_line] + all_sentences[j][3:]
                                questions_dict["question_no"] = all_sentences[question_line][0] + all_sentences[j][0:3]
                                all_questions[img_name + '_' + questions_dict["question_no"]] = questions_dict
                                i= i+1
                            else:
                                break
                        #questions_dict["sub_parts"] =sub_parts
                else:
                    continue
                try:
                    all_questions[img_name + '_' + questions_dict["question_no"]] = questions_dict
                except:
                    print("duplicate key")
            print(all_questions)
            questionkeyword = dict()
            
            for question in all_questions.keys():
                questionkeyword[question] = dict()
                questionkeyword_df[question] = ""
                for key in mappings.keys():
                    keyword = mappings[key]
                    mainpulatedKey = " " + key.lower()
                    questionText = (all_questions[question])["question"]
                    questionText = questionText.replace(".", " ").replace("(", " ").replace(")", " ").replace("!", " ").replace("?", " ").replace(",", " ")
                    questionText = questionText.replace("[", " ").replace("]", " ").replace("{", " ").replace("}", " ").replace(">", " ").replace("<", " ")
                    questionText = questionText.replace(":", " ").replace(";", " ")
                    (questionkeyword[question])[keyword] = questionText.count(mainpulatedKey)
                    # for subpart in (all_questions[question])["sub_parts"]:
                    #     subpart = subpart.replace(".", " ").replace("(", " ").replace(")", " ").replace("!", " ").replace("?", " ").replace(",", " ")
                    #     subpart = subpart.replace("[", " ").replace("]", " ").replace("{", " ").replace("}", " ").replace(">", " ").replace("<", " ")
                    #     (questionkeyword[question])[keyword] = (questionkeyword[question])[keyword]  + subpart.count(mainpulatedKey)
                    for i in range((questionkeyword[question])[keyword]):
                        questionkeyword_df[question] += keyword
                        questionkeyword_df[question] += " "
                    
        #df = pd.DataFrame(questionkeyword)
        df = pd.DataFrame(questionkeyword_df,index=["FrequentWords"])
        df = df.transpose()
        print(df)
        df.to_csv("Backend/py_files/german_exercise.csv")
        with open("Backend/py_files/german_exercises_dataset.json", "w") as outfile:
            json.dump(questionkeyword, outfile)

    else:
        temp_path ='Backend/Out/lecture_images/'

        for f in os.listdir(temp_path):
            os.remove(os.path.join(temp_path, f))
            print("removed all the temp files")
        slidekeyword_df = dict()
        for item in path.rglob('*.pdf'):
            doc = convert_from_path(item)
            
            for count, img in enumerate(doc):
                keywordfrequency = dict()

                img_name = f"{str(item).split('/')[-1].split('.')[0]}-page_{count+1}"
                print(f'page_{count}')
                slidekeyword[img_name] = keywordfrequency
                img_arr = np.array(img)
                img_arr[60: 250, 1650: 2400] = (255, 255, 255)
                img = Image.fromarray(img_arr)

                img.save(os.path.join(temp_path, img_name + ".png"), "png")
                extracted_text = pytesseract.image_to_string(Image.open(os.path.join(temp_path, img_name + ".png")),lang = "deu",config='--psm 6')
                lowercased_extracted_text = extracted_text.lower()
                sentences = lowercased_extracted_text.split("\n")
                print('original sentences:')
                print(sentences)
                print('After Stopword Removal and preprocessing:')
                preprocessed_text = [' '.join(w for w in p.split() if w not in stopwords) for p in sentences]
                print(preprocessed_text)
                dict1={}
                if(len(preprocessed_text[-2].split('-'))>1):
                    dict1['page_no']=preprocessed_text[-2].split('-')[1].lstrip()[:2].rstrip()
                else:
                    dict1['page_no']=''

                dict1['heading']=preprocessed_text[0]
                dict1['content']=preprocessed_text[1:-1]
                print(dict1)
                print('\n')
                
                slidekeyword_df[img_name] = ""
                for key in mappings.keys():
                    KeywordCount = 0
                    keyword = mappings[key]
                    mainpulatedKey = " " + key.lower()
                    for preprocessed in preprocessed_text:                       
                        preprocessed = preprocessed.replace(".", " ").replace("(", " ").replace(")", " ").replace("!", " ").replace("?", " ").replace(",", " ")
                        preprocessed = preprocessed.replace("[", " ").replace("]", " ").replace("{", " ").replace("}", " ").replace(">", " ").replace("<", " ")
                        preprocessed= preprocessed.replace(":", " ").replace(";", " ")
                        preprocessed = " " + preprocessed + " "
                        KeywordCount = KeywordCount + preprocessed.count(mainpulatedKey)
                        (slidekeyword[img_name])[keyword] = KeywordCount
                    for i in range(KeywordCount):
                        slidekeyword_df[img_name] += keyword 
                        slidekeyword_df[img_name] += " "

        df = pd.DataFrame(slidekeyword_df,index=["FrequentWords"])
        #df = pd.DataFrame(slidekeyword)
        df = df.transpose()
        print(df)
        df.to_csv("Backend/py_files/german_lecture_csv.csv")
        with open("Backend/py_files/german_lecture_dataset.json", "w") as outfile:
            json.dump(slidekeyword, outfile)


class RecommendationDetails:
    recommentationScore = 0
    scoreDetails = dict()


if __name__ == "__main__":
    dataFolderPath = 'dataset/Test/lecture/'
    exerciseFolderPath = 'dataset/Test/exercise/'
    read_data(dataFolderPath)
    #read_data(exerciseFolderPath,exercise=1)
