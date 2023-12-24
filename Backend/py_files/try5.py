import os
import pytesseract
from PIL import Image
from pdf2image import convert_from_path
import pathlib
import json
from nltk.corpus import stopwords
import numpy as np
import re

stopwords = stopwords.words('german')
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

global keywords, slidekeyword, questionkeyword, mappings

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

def read_lecture_data(folderPath):
        path = pathlib.Path(folderPath)
        temp_path ='Backend/Out/lecture_images/'
        slidekeyword = dict()
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
                preprocessed_text = [' '.join([word for word in sentence.split() if word.lower() not in stop_words]) for sentence in sentences ]
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
                        #KeywordCount = KeywordCount + preprocessed.count(mainpulatedKey)
                        #(slidekeyword[img_name])[keyword] = KeywordCount
                        slidekeyword_df[img_name] += preprocessed
                    # for i in range(KeywordCount):
                    #     slidekeyword_df[img_name] += keyword 
                    #     slidekeyword_df[img_name] += " "

        df = pd.DataFrame(slidekeyword_df,index=["FrequentWords"])
        #df = pd.DataFrame(slidekeyword)
        df = df.transpose()
        print(df)
        df.to_csv("Backend/py_files/german_lecture_csv.csv")
        with open("Backend/py_files/german_lecture_dataset.json", "w") as outfile:
            json.dump(slidekeyword, outfile)
        return df


def read_data(folderPath):
    path = pathlib.Path(folderPath)

    slidekeyword = dict()
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
            extracted_text = pytesseract.image_to_string(Image.open(os.path.join(temp_path, img_name + ".png")),
                                                         lang="eng", config='--psm 6')
            lowercased_extracted_text = extracted_text.lower()

            lowercased_extracted_text = lowercased_extracted_text.replace('\n(', '\n\n(').replace('\n1.', '\n\n1.')
            # lowercased_extracted_text = lowercased_extracted_text.replace('\n', '\n\n')
            sentences = lowercased_extracted_text.split("\n\n")
            modified_sentences = list()
            for i in sentences:
                if re.match(r'^\d.|^\([abcdefghijklmnopqrstuvwxyz]\)|\n\([a-zA-Z]+\)|\n\d.', i):
                    modified_sentences.append(i)
                else:
                    try:
                        modified_sentences[len(modified_sentences) - 1] = modified_sentences[
                                                                              len(modified_sentences) - 1] + i
                    except:
                        print('First Lines')

            sentences = modified_sentences

            print('original sentences:')
            print(sentences)
            print('After Stopword Removal and preprocessing:')
            # preprocessed_text = [' '.join(w for w in p.split() if w not in stopwords) for p in sentences]
            preprocessed_text = [' '.join([word for word in sentence.split() if word.lower() not in stop_words]) for
                                 sentence in sentences]
            print(preprocessed_text)
            final_result = list(
                filter(lambda x: re.match(r'^\d.|^\([abcdefghijklmnopqrstuvwxyz]\)', x), preprocessed_text))
            all_sentences.extend(final_result)
            print("final_Result in list")
            print(all_sentences)
            all_questions = dict()
            print("Final Object")
        for i in range(len(all_sentences)):
            questions_dict = dict()
            # questions_dict["page_no"]=count+1
            if (re.match(r'^\d.', all_sentences[i])):
                questions_dict["question_no"] = all_sentences[i][0]
                questions_dict["question"] = all_sentences[i]
                # questions_dict["sub_parts"] =""
                question_line = i
                if (i < len(all_sentences) - 1 and re.match(r'^\([abcdefghijklmnopqrstuvwxyz]\)',
                                                            all_sentences[i + 1])):
                    # sub_parts=[]
                    for j in range(i + 1, len(all_sentences)):
                        if (re.match(r'^\([abcdefghijklmnopqrstuvwxyz]\)', all_sentences[j])):
                            questions_dict = dict()
                            questions_dict["question"] = all_sentences[question_line] + all_sentences[j][3:]
                            questions_dict["question_no"] = all_sentences[question_line][0] + all_sentences[j][0:3]
                            all_questions[img_name + '_' + questions_dict["question_no"]] = questions_dict
                            i = i + 1
                        else:
                            break
                    # questions_dict["sub_parts"] =sub_parts
            else:
                continue
            try:
                all_questions[img_name + '_' + questions_dict["question_no"]] = questions_dict
            except:
                print("duplicate key")
        print(all_questions)
    return all_questions

def GetLecturesEmbeddings(lectureEmbeddings):
    lecture_embeddings=[]
    for index, row in lectureEmbeddings.iterrows():
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

def GetLectureEmbeddings():
    # Open PDF file and extract text from each page
    pdf_file = open('dataset/Test/german/lecture/dbs-2.pdf', 'rb')
    pdf_reader = PyPDF2.PdfReader(pdf_file)
    # page_texts = [pdf_reader.pages[i].extract_text() for i in range(len(pdf_reader.pages))]
    page_texts = []
    for i in range(len(pdf_reader.pages)):
        page = pdf_reader.pages[i]
        page_text = page.extract_text()

        # Preprocess and clean the text
        page_text = page_text.lower()  # convert to lowercase
        page_text = re.sub(r'\d+', '', page_text)  # remove digits
        page_text = re.sub(r'[^\w\s]', '', page_text)  # remove punctuation
        page_text = re.sub(r'\s+', ' ', page_text)  # remove extra whitespace

        # Tokenize the text into words
        # words = nltk.word_tokenize(page_text)

        # Remove stop words
        # words = [word for word in words if word not in stop_words]

        # Join the remaining words back into a string
        # page_text = ' '.join(words)

        # Add the cleaned text to the list
        page_texts.append(page_text)

    # Tokenize and encode each page of text
    page_inputs = []
    for page_text in page_texts:
        # Tokenize the text
        tokens = tokenizer.tokenize(page_text)
        # Truncate or pad the tokens to a fixed length
        max_length = 512
        if len(tokens) > max_length:
            tokens = tokens[:max_length]
        else:
            tokens += ['[PAD]'] * (max_length - len(tokens))
        # Convert the tokens to input IDs and attention mask
        input_ids = tokenizer.convert_tokens_to_ids(tokens)
        attention_mask = [1 if token != '[PAD]' else 0 for token in tokens]
        # Add the input IDs and attention mask to the page inputs list
        page_inputs.append({'input_ids': input_ids, 'attention_mask': attention_mask})
    page_embeddings = []
    for page_input in page_inputs:
        input_ids = torch.tensor(page_input['input_ids']).unsqueeze(0)  # add batch dimension
        attention_mask = torch.tensor(page_input['attention_mask']).unsqueeze(0)  # add batch dimension

        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            page_embedding = outputs[1][0]

        page_embeddings.append(page_embedding.numpy())

    pdf_file.close()
    return page_embeddings


def GetExerciseEmbeddings(exerciseText):
    # Load pre-trained model and tokenizer
    model = BertModel.from_pretrained('bert-base-uncased')
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    # Tokenize text and add special tokens [CLS] and [SEP]
    tokens = tokenizer.tokenize(exerciseText)
    tokens = ['[CLS]'] + tokens + ['[SEP]']

    # Convert tokens to input ids and attention mask
    input_ids = torch.tensor(tokenizer.convert_tokens_to_ids(tokens)).unsqueeze(0)
    attention_mask = torch.ones_like(input_ids)

    # Get BERT embeddings for input text
    with torch.no_grad():
        outputs = model(input_ids, attention_mask=attention_mask)
        embeddings = outputs[0]

    # Get the embedding for the [CLS] token
    cls_embedding = embeddings[:, 0, :]

    # Convert tensor to numpy array
    cls_embedding = cls_embedding.numpy()

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


def compute_cosine_similarity(question, page_embeddings, exercise_embedding):
    slide_paths = load_slide_paths('dataset/Test/german/lecture/dbs-2.pdf')
    # flatten the page_embeddings array
    page_embeddings_flat = np.reshape(page_embeddings, (len(page_embeddings), -1))

    # calculate cosine similarities
    similarity_scores = cosine_similarity(exercise_embedding, page_embeddings_flat)

    # get the index of the highest similarity score
    # recommended_page_index = np.argmax(similarity_scores)
    top_indices = np.argsort(similarity_scores)[0][::-1][:5]
    return top_indices
    # recommended_page_embedding = page_embeddings[recommended_page_index]
    # Get the paths of recommended lecture slides
    recommended_slide_paths = [slide_paths[i] for i in top_indices]
    print(recommended_slide_paths)


if __name__ == '__main__':

    nltk.download('stopwords')
    # Get stop words in German
    stop_words = set(stopwords.words('german'))
    # Load pre-trained BERT model and tokenizer
    model = BertModel.from_pretrained('bert-base-uncased')
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    #

    exerciseFolderPath = 'dataset/Test/german/exercise/'
    lectures = read_lecture_data("dataset/Test/german/lecture/")
    lectureEmbeddings = GetLecturesEmbeddings(lectures)
    questions = read_data(exerciseFolderPath)
    print(questions)
    final_list=[]
    obj={}
    obj['exercise'] = "exercise3"
    tasks=[]
    for question in questions.keys():
        questionText = (questions[question])["question"]
        questionText = questionText.replace(".", " ").replace("(", " ").replace(")", " ").replace("!", " ").replace("?",
                                                                                                                    " ").replace(
            ",", " ")
        questionText = questionText.replace("[", " ").replace("]", " ").replace("{", " ").replace("}", " ").replace(">",
                                                                                                                    " ").replace(
            "<", " ")
        questionText = questionText.replace(":", " ").replace(";", " ")
        exerciseEmbeddings = GetExerciseEmbeddings(questionText)
        print(exerciseEmbeddings)
        # Get the indexes of recommended lecture slides
        recommended_slide_indexes = compute_cosine_similarity(question, lectureEmbeddings, exerciseEmbeddings)
        print(recommended_slide_indexes)
        task_obj={}
        if("(" in question):
            task_obj['task_number']=question[-4:]
        else:
            task_obj['task_number'] = question[-1:]
        modified_slides=["dbs-2-page_" + str(s) for s in recommended_slide_indexes]
        task_obj['pages']=list(modified_slides)
        tasks.append((task_obj))
    obj['tasks']=tasks
    final_list.append(obj)
    with open("evaluation/bert_evaluation_german_predicted.json", "w") as write_file:
        json.dump(final_list, write_file)
