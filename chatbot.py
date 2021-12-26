import tensorflow as tf
import numpy as np
import re
import time
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import Embedding,Input,Dense,LSTM
from tensorflow.keras.models import Model


#importing dataset
lines=open('movie_lines.txt',encoding='utf-8',errors='ignore').read().split('\n')
conversations=open('movie_conversations.txt',encoding='utf-8',errors='ignore').read().split('\n')

#creating dictionary that maps each line with id
id2line={}
for line in lines:
    _line=line.split(' +++$+++ ')
    if len(_line)==5:
        id2line[_line[0]]=_line[4]

#create a list of converstaion
conversations_ids=[]
for conversation in conversations:
    _conversation=conversation.split(' +++$+++ ')[-1][1:-1].replace("'","").replace(" ","")
    conversations_ids.append(_conversation.split(','))

#creating questions and answer list
questions=[]
answers=[]
for conversation in conversations_ids:
    for i in range(len(conversation)-1):
        questions.append(id2line[conversation[i]])
        answers.append(id2line[conversation[i+1]])


#doing first cleaning of the texts
#re.sub is used for replacement
def clean_text(text):
    text=text.lower()
    text=re.sub(r"i'm","i am",text)
    text=re.sub(r"he's","he is",text)
    text=re.sub(r"she's","she is",text)
    text=re.sub(r"that's","that is",text)
    text=re.sub(r"what's","what is",text)
    text=re.sub(r"where's","where is",text)
    text=re.sub(r"\'ll"," will",text)
    text=re.sub(r"\'ve"," have",text)
    text=re.sub(r"\'re"," are",text)
    text=re.sub(r"\'d"," would",text)
    text=re.sub(r"won't","will not",text)
    text=re.sub(r"can't","cannot",text)
    text=re.sub(r"[-()\"#/@;:<>{}+=|.?,]","",text)
    return text

#cleaning questions
clean_questions=[]
for question in questions:
    clean_questions.append(clean_text(question))

#cleaning answers
clean_answers=[]
for answer in answers:
    clean_answers.append(clean_text(answer))

del(questions,answers,conversations_ids,id2line)

clean_answers=clean_answers[:30000]
clean_questions=clean_questions[:30000]
#creating dictionnary that maps each word to its number of occurance
#if you want to map each unique word with unique integer the use Tokenizer.
# (from tensorflow.keras.preprocessing.text import Tokenizer)
word2count={}
for question in clean_questions:
    for word in question.split():
        if word not in word2count:
            word2count[word]=1
        else:
            word2count[word]+=1
for answer in clean_answers:
    for word in answer.split():
        if word not in word2count:
            word2count[word]=1
        else:
            word2count[word]+=1

# creating two dictionaries that maps the question words and the answer words to a unique integer
threshold=15
questionswords2int={}
word_number=1
for word,count in word2count.items():
    if(count>=threshold):
        questionswords2int[word]=word_number
        word_number+=1
answerswords2int={}
word_number=1
for word,count in word2count.items():
    if(count>=threshold):
        answerswords2int[word]=word_number
        word_number+=1

#adding tokens to these two dictionaries
tokens=['<PAD>','<EOS>','<OUT>','<SOS>']
for token in tokens:
    questionswords2int[token] = len(questionswords2int) + 1
for token in tokens:
    answerswords2int[token] = len(answerswords2int) + 1

questionswords2int['Cameron'] = questionswords2int['<PAD>']
questionswords2int['<PAD>'] = 0
answerswords2int['Cameron'] = answerswords2int['<PAD>']
answerswords2int['<PAD>'] = 0
#creating reverse dictionary
#word_index:word
answersints2words={w_i:w for w,w_i in answerswords2int.items()}

#adding <EOS> at every end of the answer string
for i in range(len(clean_answers)):
    clean_answers[i]='<SOS> '+clean_answers[i]+'<EOS>'

#translating all the questions and answers into integers
#and filtered out words with <OUT>
questions_into_int=[]
for question in clean_questions:
    ints=[]
    for word in question.split():
        if(word not in questionswords2int):
            ints.append(questionswords2int['<OUT>'])
        else:
            ints.append(questionswords2int[word])
    questions_into_int.append(ints)
answers_into_int=[]
for answer in clean_answers:
    ints=[]
    for word in answer.split():
        if(word not in answerswords2int):
            ints.append(answerswords2int['<OUT>'])
        else:
            ints.append(answerswords2int[word])
    answers_into_int.append(ints)


# add padding
encoder_inp = pad_sequences(questions_into_int,13,padding='post',truncating='post')
decoder_inp = pad_sequences(answers_into_int,13,padding='post',truncating='post')
del(questions_into_int,answers_into_int)
decoder_final_output=[]
for i in decoder_inp:
    decoder_final_output.append(i[1:])

del(i)
decoder_final_output = pad_sequences(decoder_final_output,13,padding='post',truncating='post')
decoder_final_output = to_categorical(decoder_final_output,num_classes = len(answerswords2int))

V=len(answerswords2int)
D=50
enc_inp = Input(shape=(13,))
dec_inp = Input(shape=(13,))

# encoder(enc_inp)
embed = Embedding(V+1,output_dim=D,input_length=13,trainable=True)
enc_embed = embed(enc_inp)
enc_lstm = LSTM(400,return_sequences=True,return_state=True)
enc_op,h,c = enc_lstm(enc_embed)
enc_states = [h,c]

# decoder(dec_inp)
embed = Embedding(V+1,output_dim=D,input_length=13,trainable=True)
dec_embed = embed(dec_inp)
dec_lstm = LSTM(400,return_sequences=True,return_state=True)
dec_op,_,_ = dec_lstm(dec_embed,initial_state=enc_states)

#dense
dense = Dense(V,activation='softmax')
dense_op = dense(dec_op)


model = Model([enc_inp,dec_inp],dense_op)
r = model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])

model.fit([encoder_inp,decoder_inp],decoder_final_output,epochs=40)

##############################################################################################
#building interface model
enc_model = Model([enc_inp],enc_states)

#decoder model
decoder_state_input_h = Input(shape=(400,))
decoder_state_input_c = Input(shape=(400,))

decoder_states_inputs = [decoder_state_input_h,decoder_state_input_c]

decoder_outputs,state_h,state_c = dec_lstm(dec_embed,initial_state=decoder_states_inputs)

decoder_states = [state_h, state_c]

dec_model = Model([dec_inp]+ decoder_states_inputs,[decoder_outputs]+ decoder_states)

###############################################################################################
prepro1=""
while prepro1 != 'q':
    prepro1 = input("you : ")
    prepro1 = clean_text(prepro1)
    prepro = [prepro1]

    txt = []
    for x in prepro:
        lst = []
        for y in x.split():
            try:
                lst.append(questionswords2int[y])
            except:
                lst.append(questionswords2int['<OUT>'])
        txt.append(lst)

    txt = pad_sequences(txt,13, padding='post')

    stat = enc_model.predict(txt)

    
    empty_target_sequence = np.zeros((1,1))
    empty_target_sequence[0,0] = questionswords2int['<SOS>']

    stop_condition = False
    decoded_translation = ''
    while not stop_condition:
        dec_outputs,h,c=dec_model.predict([empty_target_sequence]+stat)
        decoder_concat_input = dense(dec_outputs)
        sample_word_index = np.argmax(decoder_concat_input[0,-1,:])
        sample_word = answersints2words[sample_word_index]

        if sample_word != '<EOS> ':
            decoded_translation += sample_word 

        if sample_word == '<EOS> ' or len(decoded_translation.split()) > 13:
            stop_condition = True 

        empty_target_sequence = np.zeros( ( 1 , 1 ) )  
        empty_target_sequence[ 0 , 0 ] = sample_word_index
        ## <SOS> - > hi
        ## hi --> <EOS>
        stat = [h, c]  

    print("chatbot attention : ", decoded_translation )
    print("==============================================")  

