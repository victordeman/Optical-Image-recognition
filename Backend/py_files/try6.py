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
from transformers import BertTokenizer, BertModel
import torch
from sklearn.metrics.pairwise import cosine_similarity
import PyPDF2
import torch
from transformers import BertModel, BertTokenizer
import string
import nltk
from nltk.corpus import stopwords

global keywords, slidekeyword, questionkeyword,mappings

with open('dataset/keywords/german_keywords.json') as json_file:
    mappings = json.load(json_file)

def get_similar_slides(query_embeddings, slides_embeddings, threshold=0.8):
    similarities = cosine_similarity(query_embeddings, slides_embeddings)
    similar_pairs = np.argwhere(similarities > threshold)
    return similar_pairs[:, 1]

def embed_sql_text(text):
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertModel.from_pretrained('bert-base-uncased')
    input_ids = tokenizer.encode(text, add_special_tokens=True, return_tensors='pt')
    with torch.no_grad():
        embeddings = model(input_ids)[0]
    return embeddings.squeeze(0).numpy()


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
                preprocessed_text = [' '.join([word for word in sentence.split() if word.lower() not in stop_words]) for sentence in sentences]
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
        return df
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
                preprocessed_text = [' '.join([word for word in sentence.split() if word.lower() not in stop_words]) for sentence in sentences]
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
        df.to_csv("Backend/py_files/bert_german_lecture_csv.csv")
        with open("Backend/py_files/bert_german_lecture_dataset.json", "w") as outfile:
            json.dump(slidekeyword, outfile)
        return df


def GetLectureEmbeddings(dataFolderPath):

    lectures_slides_text = read_data(dataFolderPath)
    lecture_embeddings = []
    for index, row in lectures_slides_text.iterrows():
        text = row['FrequentWords']
        # Preprocess and clean the text
        text = text.lower()  # convert to lowercase
        text = re.sub(r'\d+', '', text)  # remove digits
        text = re.sub(r'[^\w\s]', '', text)  # remove punctuation
        text = re.sub(r'\s+', ' ', text)  # remove extra whitespace

        # Tokenize and encode the text
        tokens = tokenizer.tokenize(text)
        # Truncate or pad the tokens to a fixed length
        max_length = 512
        if len(tokens) > max_length:
            tokens = tokens[:max_length]
        else:
            tokens += ['[PAD]'] * (max_length - len(tokens))
        # Convert the tokens to input IDs and attention mask
        input_ids = tokenizer.convert_tokens_to_ids(tokens)
        attention_mask = [1 if token != '[PAD]' else 0 for token in tokens]
        
        # Generate the BERT embedding
        input_ids = torch.tensor(input_ids).unsqueeze(0)  # add batch dimension
        attention_mask = torch.tensor(attention_mask).unsqueeze(0)  # add batch dimension
        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            embedding = outputs[1][0] 
        lecture_embeddings.append(embedding.numpy())
    return lecture_embeddings
    
def GetExerciseEmbeddings(exerciseText):
    exercise_slides_text = read_data(exerciseFolderPath,exercise=1)
    # Load pre-trained model and tokenizer
    exercise_embeddings = []
    for index, row in exercise_slides_text.iterrows():
        text = row['FrequentWords']
        # Preprocess and clean the text
        text = text.lower()  # convert to lowercase
        text = re.sub(r'\d+', '', text)  # remove digits
        text = re.sub(r'[^\w\s]', '', text)  # remove punctuation
        text = re.sub(r'\s+', ' ', text)  # remove extra whitespace

        # Tokenize and encode the text
        tokens = tokenizer.tokenize(text)
        # Truncate or pad the tokens to a fixed length
        max_length = 512
        if len(tokens) > max_length:
            tokens = tokens[:max_length]
        else:
            tokens += ['[PAD]'] * (max_length - len(tokens))
        # Convert the tokens to input IDs and attention mask
        input_ids = tokenizer.convert_tokens_to_ids(tokens)
        attention_mask = [1 if token != '[PAD]' else 0 for token in tokens]
        
        # Generate the BERT embedding
        input_ids = torch.tensor(input_ids).unsqueeze(0)  # add batch dimension
        attention_mask = torch.tensor(attention_mask).unsqueeze(0)  # add batch dimension
        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            embedding = outputs[1][0] 
        exercise_embeddings.append(embedding.numpy())
    return exercise_embeddings
    
# Print the shape of the embedding
    print(cls_embedding.shape)
    return cls_embedding

def load_slide_paths(file_path):
    """
    Reads slide file paths from a text file and returns them as a list of strings.

    Args:
    - file_path (str): Path to the text file containing slide file paths, one per line.

    Returns:
    - slide_paths (list): List of slide file paths, as strings.
    """
    with open(file_path, 'r', encoding='iso-8859-1') as f:
        slide_paths = f.read().splitlines() 
    return slide_paths

def compute_cosine_similarity(lecture_embeddings, exercise_embedding):
    slide_paths = load_slide_paths('dataset/Test/english/lecture/dbs-2.pdf')
# flatten the page_embeddings array
    lecture_embeddings_flat = np.reshape(lecture_embeddings, (len(lecture_embeddings), -1))
    exercise_embedding = np.array(exercise_embedding).reshape(-1, 768)
# calculate cosine similarities
    similarity_scores = cosine_similarity(exercise_embedding, lecture_embeddings)

# get the index of the highest similarity score
    #recommended_page_index = np.argmax(similarity_scores)
    top_indices = np.argsort(similarity_scores)[0][::-1][:7]
    return top_indices
    #recommended_page_embedding = page_embeddings[recommended_page_index]
    # Get the paths of recommended lecture slides
    print(top_indices)

if __name__ == '__main__':
    dataFolderPath = 'dataset/Test/german/lecture/'
    exerciseFolderPath = 'dataset/Test/german/exercise/'
    nltk.download('stopwords')
        # Get stop words in German
    stop_words = set(stopwords.words('german'))
    # Load pre-trained BERT model and tokenizer
    model = BertModel.from_pretrained('bert-base-uncased')
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    lectureEmbeddings = GetLectureEmbeddings(dataFolderPath)
    exerciseEmbeddings = GetExerciseEmbeddings(exerciseFolderPath)
    #questions = read_data(exerciseFolderPath,exercise=1)
    #print(questions)
    final_list = []
    obj = {}
    obj['exercise'] = "exercise3"
    tasks = []
    for i in range(len(exerciseEmbeddings)):
        #for j in range(len(lectureEmbeddings)):
        # Get the indexes of recommended lecture slides
        recommended_slide_indexes = compute_cosine_similarity(lectureEmbeddings,exerciseEmbeddings[i])
        task_obj = {}
        if ("(" in question):
            task_obj['task_number'] = question[-4:]
        else:
            task_obj['task_number'] = question[-1:]
        modified_slides = ["dbs-2-page_" + str(s) for s in recommended_slide_indexes]
        task_obj['pages'] = list(modified_slides)
        tasks.append((task_obj))
    obj['tasks'] = tasks
    final_list.append(obj)
    with open("evaluation/bert_keyword_evaluation_german_predicted.json", "w") as write_file:
        json.dump(final_list, write_file)

