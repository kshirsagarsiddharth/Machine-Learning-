# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %% [markdown]
# ## Importing all the dependencies

# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
from bs4 import BeautifulSoup
from nltk.tokenize import WordPunctTokenizer
import nltk
from bs4 import BeautifulSoup
import urllib
from nltk.tokenize import sent_tokenize
from nltk.corpus import cmudict
from nltk.corpus import stopwords
from nltk.tokenize import wordpunct_tokenize
import syllables

# %% [markdown]
# ## Importing the ciklist into Pandas dataframe and doing some preliminary analysis

# %%
df = pd.read_excel(r'D:\nlp_internship\Data Science\cik_list.xlsx')


# %%
df.shape


# %%
df.head()


# %%
df.describe()

# %% [markdown]
# ## We need to extract the files from the web, hence hence lets add the hyperlink and extract all the text data.

# %%
# we need to extract the files from the web hence hence lets add the hyperlink and ectract all the text data
hyperlink = 'https://www.sec.gov/Archives/'
df['SECFNAME'] = hyperlink + df['SECFNAME']

# %% [markdown]
# ### Importing stopwords, positive words and negative words from the list

# %%
stop_words = pd.read_csv(r'D:\nlp_internship\Data Science\stop_words\StopWords_GenericLong.txt',header=None) # importing stop words
stopword_list = stop_words[0].to_list()
master_dict = pd.read_csv(r'D:\nlp_internship\Data Science\stop_words\LoughranMcDonald_MasterDictionary_2018.csv') # importing the master dict
negative_words = master_dict[master_dict['Negative'] == 2009]['Word']
negative_dict =[word.lower() for word in negative_words if word not in stop_words] # creating negative words dictonary
positive_words = master_dict[master_dict['Positive'] == 2009]['Word']
positive_dict = [word.lower() for word in positive_words if word not in stop_words] # creating positive words dictonary

# %% [markdown]
# ### Now all the dependencies are defined and all files are imported to proceed with the analysis. I saw that there are nearly 152 documents in the structure. Instead of downloading each file storing it in my SSD and loading it into my ram, and perform  analysis the text I decided to use, urllib library which allows me to load data from any webpage and store it in RAM. which is fast and efficient.

# %%
def preliminary_cleaner(text):
    '''
    This function is used to remove all the 
    numerical data in the files. And next line in the file
    '''
    if text:
        array = []
        soup =text
        soup = re.sub('[\d%/$]','',str(soup))
        soup = re.sub("\\\\n",'',soup)
        soup = ' '.join(soup.split())
        return soup
        
    else:
        return None

# %% [markdown]
# ### Now what I did is created a regular expression for all the three given titles. And then scraped the website for this data and stored in three python arrays. Later I merged the three arrays into a pandas data frame. Which is a highly efficient data-structure for processing the text files.

# %%
array_one,array_two,array_three = [],[],[] # creating arrays
# regular expression for the three components.
mgt_disc_ana = r"item[^a-zA-Z\n]*\d\s*\.\s*management\'s discussion and analysis.*?^\s*item[^a-zA-Z\n]*\d\s*\.*"
qlty_qnt_disc = r"item[^a-zA-Z\n]*\d[a-z]?\.?\s*Quantitative and Qualitative Disclosures about Market Risk.*?^\s*item\s*\d\s*"
rsk_fct = r"item[^a-zA-Z\n]*\d[a-z]?\.?\s*Risk Factors.*?^\s*item\s*\d\s*"
for hyperlink in df['SECFNAME']:
    text = urllib.request.urlopen(hyperlink).read().decode('utf-8') # eastiblishint the client server connection and downloading the data
    # into ram.
    matches_mda = re.findall(mgt_disc_ana, text, re.IGNORECASE | re.DOTALL | re.MULTILINE)
    matches_qlty_qnt_disc= re.findall(qlty_qnt_disc,text,re.IGNORECASE | re.DOTALL | re.MULTILINE)
    matches_rsk_fct = re.findall(rsk_fct,text,re.IGNORECASE | re.DOTALL | re.MULTILINE) 
    array_one.append(preliminary_cleaner(matches_mda))
    array_two.append(preliminary_cleaner(matches_qlty_qnt_disc))
    array_three.append(preliminary_cleaner(matches_rsk_fct))

# %% [markdown]
# # Merging all three arrays and storing it as csv for later analysis.

# %%
df1,df2,df3 = pd.DataFrame(array_one),pd.DataFrame(array_two),pd.DataFrame(array_three) #
merged_dataframe = pd.concat([df1,df2,df3],axis = 1)
merged_dataframe.columns = ['matches_mda','matches_qlty_qnt_disc','matches_rsk_fct']
merged_dataframe.to_csv('merged_dataframe_all.csv')


# %%
df2 = pd.read_csv(r'D:\nlp_internship\submission\merged_dataframe_all.csv') # loading the merged dataframe

# %% [markdown]
# # concatenating the main data frame and newly merged data frame.

# %%
final_dataframe = pd.concat([df,df2],axis = 1)
final_dataframe.drop(columns='Unnamed: 0',inplace = True)
final_dataframe.fillna(' ',inplace = True)


# %%
final_dataframe.columns


# %%
def cleaning_stopwords(columns):
    """This function is used to tokenize and removing everything except alphabets in the text files."""
    columns = columns.apply(lambda x : re.sub("[^a-zA-Z]"," ",str(x))) # removing everything except letters
    columns = columns.apply(lambda x : wordpunct_tokenize(x)) # tokenizing the frame
    # cleaning stopwords
    columns = columns.apply(lambda x : [word.lower() for word in x if word not in stopword_list])
    return columns

# %% [markdown]
# # Creating a new data frame called the final data frame which consists of all the tokenized documents.

# %%
final_dataframe['mda_bulk'] = cleaning_stopwords(final_dataframe['matches_mda'])
final_dataframe['qltqnt_bulk'] = cleaning_stopwords(final_dataframe['matches_qlty_qnt_disc'])
final_dataframe['rskfct_bulk'] = cleaning_stopwords(final_dataframe['matches_rsk_fct'])

# %% [markdown]
# ### I have tried to compute the required variables using three methods this method is highly efficient and calculates all the variables in the number of minutes because I am creating a class called text analysis which will initialize the four variables words,complex_words,positive_words and negative words. hence for each of the three sections, I will create a new object for that section and compute all the 14 concerned variables.

# %%
class text_analysis:
    def __init__(self,column_with_words,column_with_sentences):
        self.column_with_words = column_with_words # initializing the column which contaions words.

        self.column_with_sentences = column_with_sentences # initializing the words which contains sentence tokens.

        self.words = self.column_with_words.apply(lambda x : len([word for word in x])) # initialing the variable wors count

        self.complex_words = self.column_with_words.apply(lambda x : len([word for word in x if syllables.estimate(word) > 2])) # initializing the variable complex words

        self.positive_words_column = self.column_with_words.apply(lambda x : len([word for word in x if word in np.array(positive_dict)])) # initializing the column number of positive words

        self.negative_words_column = self.column_with_words.apply(lambda x : len([word for word in x if word in np.array(negative_dict)])) # initializing the column number of positive words

    def positive_score(self):
        '''This score is calculated by assigning the value of +1 for each word if 
        found in the Positive Dictionary and then adding up all the values. '''
        return self.positive_words_column
        #return self.column_with_words.apply(lambda x : len([word for word in x if word in positive_dict]))

    def negative_score(self):
        '''This score is calculated by assigning the value of +1 for each word if 
        found in the Negative Dictionary and then adding up all the values.'''
        return self.negative_words_column
        #return self.column_with_words.apply(lambda x : len([word for word in x if word in negative_dict]))

    def polarity_score(self):
        """This is the score that determines if a given text is positive or negative in nature. It is calculated by using the formula: 
            Polarity Score = (Positive Score – Negative Score)/ ((Positive Score + Negative Score) + 0.000001)
        """
        val1 = (self.positive_words_column - self.negative_words_column)
        val2 = (self.positive_words_column + self.negative_words_column + 0.00001)
        return val1 / val2

    def average_sentence_length(self):
        """
        Average Sentence Length = the number of words / the number of sentences
        we use sent_tokenizer liabrary and we replace null values with 0

        """
        #words = self.column_with_words.apply(lambda x : len([word for word in x]))
        sentences = self.column_with_sentences.apply(lambda x : len(sent_tokenize(str(x))))
        ans =  self.words / sentences
        ans = ans.fillna(0)
        return ans

    def syllables_count(self,word):
        """This function is used to count the syllables. In this case, we use the cmu dictionary which consists 
        of all the syllables of given words but I found that there are some words which are not defined in the 
        cmu dictionary hence I used the algorithm to calculate the number of syllables."""
        phoneme_dictonary = dict(cmudict.entries())
        count = 0
        vowels = 'aeiouy'
        word = word.lower()
        if word[0] in vowels:
            count += 1
        for i in range(1,len(word)):
            if word[i] in vowels and word[i - 1] not in vowels:
                count += 1
        if word[-1] == 'e':
            count -= 1
        if word[-2:] == 'le':
            count += 1
        if count == 0:
            count += 1
        # comparing the values with cmudict
        try:
            second_count = 0
            for val in phoneme_dictonary[word.lower()]:
                for val1 in val:
                    if val1[-1].isdigit():
                        second_count += len(val1)
            return second_count
        except KeyError: # this is in case the values are not found in the cmu dict
            return count

    def complex_words_proportion(self):
        """Percentage of Complex words = the number of complex words / the number of words """
        #complex_words = self.column_with_words.apply(lambda x : len([word for word in x if self.syllables_count(word) > 2]))
        #words = self.column_with_words.apply(lambda x : len([word for word in x]))
        ans = self.complex_words / self.words
        ans = ans.fillna(0)
        return ans

    def fog_index(self):
        """Fog Index = 0.4 * (Average Sentence Length + Percentage of Complex words)
        The Gunning Fog Index gives the number of years of education that your reader 
        hypothetically needs to understand the paragraph or text. The Gunning Fog Index 
        formula implies that short sentences written in plain English achieve a better 
        score than long sentences written in complicated language."""
        return 0.4 * (self.average_sentence_length() + self.complex_words)

    def complex_words_count(self):
        """ Complex words are words in the text that contain more than two syllables.
        hence we compare it with our words."""
        #complex_words = self.column_with_words.apply(lambda x : len([word for word in x if self.syllables_count(word) > 2]))
        ans = self.complex_words.fillna(0) 
        
        return ans
    def word_counts(self):
        """This function is used to calculate number of words in each document axcept stopwords found in nltk library"""
        stopwords_nltk = np.array(list(set(stopwords.words('english'))))
        words = self.column_with_words.apply(lambda x : len([word for word in x if word not in stopwords_nltk]))
        ans =  words
        ans = ans.fillna(0)
        return ans

    def uncertainty_words_count(self):
        uncertainty_dict = pd.read_excel(r'D:\nlp_internship\Data Science\uncertainty_dictionary.xlsx')
        uncertainty_dict = uncertainty_dict['Word'].str.lower().to_list()
        words = self.column_with_words.apply(lambda x : len([word for word in x if word  in uncertainty_dict]))
        return words

    def constraning_words_count(self):
        constrain_dict = pd.read_excel(r'D:\nlp_internship\Data Science\constraining_dictionary.xlsx')
        constrain_dict = constrain_dict['Word'].str.lower().to_list()
        words = self.column_with_words.apply(lambda x : len([word for word in x if word  in constrain_dict]))
        return words

    def positive_words_proportion(self):
        #words = self.column_with_words.apply(lambda x : len([word for word in x]))
        ans = self.positive_words_column / self.words
        ans = ans.fillna(0)
        return ans

    def negative_words_proportion(self):
        #words = self.column_with_words.apply(lambda x : len([word for word in x]))
        ans = self.negative_words_column / self.words
        ans = ans.fillna(0)
        return ans

    def uncertaninty_words_proportion(self):
        #words = self.column_with_words.apply(lambda x : len([word for word in x]))
        ans = self.uncertainty_words_count() / self.words
        ans = ans.fillna(0)
        return ans
        
    def constraining_words_proportion(self):
        #words = self.column_with_words.apply(lambda x : len([word for word in x]))
        ans = self.constraning_words_count() / self.words
        ans = ans.fillna(0)
        return ans


# %%
final_dataframe.columns


# %%
# defining the mda object and giving it the concerned values.


# %%
mda = text_analysis(final_dataframe['mda_bulk'],final_dataframe['matches_mda']) 


# %%
MDA = pd.DataFrame() #creating mda dataframe
MDA['mda_positive_score'] = mda.positive_score()
MDA['mda_negative_score'] = mda.negative_score()
MDA['mda_polarity_score'] = mda.polarity_score()
MDA['mda_average_sentence_length'] = mda.average_sentence_length()
MDA['mda_percenmdage_of_complex_words'] = mda.complex_words_proportion()
MDA['mda_fog_index'] = mda.fog_index()
MDA['mda_complex_word_count'] = mda.complex_words_count()
MDA['mda_word_count'] = mda.word_counts()
MDA['mda_uncertainty_score'] = mda.uncertainty_words_count()
MDA['mda_constraining_score'] = mda.constraning_words_count()
MDA['mda_positive_word_proportion'] = mda.positive_words_proportion()
MDA['mda_negative_word_proportion'] = mda.negative_words_proportion()
MDA['mda_uncertainty_word_proportion'] = mda.uncertaninty_words_proportion()
MDA['mda_constraining_word_proportion'] = mda.constraining_words_proportion()
 


# %%
MDA.to_csv('mda.csv')


# %%
qqdmr = text_analysis(final_dataframe['qltqnt_bulk'],final_dataframe['matches_qlty_qnt_disc'])
QQDMR = pd.DataFrame()
QQDMR['qqdmr_positive_score'] = qqdmr.positive_score()
QQDMR['qqdmr_negative_score'] = qqdmr.negative_score()
QQDMR['qqdmr_polarity_score'] = qqdmr.polarity_score()
QQDMR['qqdmr_average_sentence_length'] = qqdmr.average_sentence_length()
QQDMR['qqdmr_percenqqdmrge_of_complex_words'] = qqdmr.complex_words_proportion()
QQDMR['qqdmr_fog_index'] = qqdmr.fog_index()
QQDMR['qqdmr_complex_word_count'] = qqdmr.complex_words_count()
QQDMR['qqdmr_word_count'] = qqdmr.word_counts()
QQDMR['qqdmr_uncertainty_score'] = qqdmr.uncertainty_words_count()
QQDMR['qqdmr_constraining_score'] = qqdmr.constraning_words_count()
QQDMR['qqdmr_positive_word_proportion'] = qqdmr.positive_words_proportion()
QQDMR['qqdmr_negative_word_proportion'] = qqdmr.negative_words_proportion()
QQDMR['qqdmr_uncertainty_word_proportion'] = qqdmr.uncertaninty_words_proportion()
QQDMR['qqdmr_constraining_word_proportion'] = qqdmr.constraining_words_proportion()

QQDMR.to_csv('qqdmr.csv')


# %%
rf = text_analysis(final_dataframe['rskfct_bulk'],final_dataframe['matches_rsk_fct'])
RF = pd.DataFrame()
RF['rf_positive_score'] = rf.positive_score()
RF['rf_negative_score'] = rf.negative_score()
RF['rf_polarity_score'] = rf.polarity_score()
RF['rf_average_sentence_length'] = rf.average_sentence_length()
RF['rf_percenrfge_of_complex_words'] = rf.complex_words_proportion()
RF['rf_fog_index'] = rf.fog_index()
RF['rf_complex_word_count'] = rf.complex_words_count()
RF['rf_word_count'] = rf.word_counts()
RF['rf_uncertainty_score'] = rf.uncertainty_words_count()
RF['rf_constraining_score'] = rf.constraning_words_count()
RF['rf_positive_word_proportion'] = rf.positive_words_proportion()
RF['rf_negative_word_proportion'] = rf.negative_words_proportion()
RF['rf_uncertainty_word_proportion'] = rf.uncertaninty_words_proportion()
RF['rf_constraining_word_proportion'] = rf.constraining_words_proportion()
RF.to_csv('rf.csv')


# %%
pd.concat([MDA,QQDMR,RF],axis = 1).to_csv('highly_efficient.csv')


# %%


# %% [markdown]
# # Finding constraining words for the whole report: 
# ## In this I have used numpy vectorization function, to analyse the whole document which is 10 times faster than normal python

# %%
complete_array = []
count = 0
for hyperlink in df['SECFNAME']:
    text = urllib.request.urlopen(hyperlink).read().decode('utf-8')
    count += 1
    complete_array.append(preliminary_cleaner(text))
    print(count)


# %%
whole_document_dataframe = pd.DataFrame()
stopwords_nltk = np.array(list(set(stopwords.words('english'))))
whole_document_dataframe['constraining'] = complete_array
whole_document_dataframe = cleaning_stopwords(whole_document_dataframe['constraining'])
arr = pd.DataFrame(whole_document_dataframe)
union_array = arr['constraining'].apply(lambda x : np.union1d(np.array(x),stopwords_nltk))
uncertainty_dict = pd.read_excel(r'D:\nlp_internship\Data Science\uncertainty_dictionary.xlsx')
uncertainty_dict = np.array(uncertainty_dict['Word'].str.lower().to_list())
ans = union_array.apply(lambda x : len(np.intersect1d(x,uncertainty_dict)))


# %%



# %%
highly_eff = pd.read_csv('highly_efficient.csv')


# %%
highly_eff['constraining_words_whole_report'] = ans


# %%
highly_eff.to_csv('last_and_final_output.csv')


# %%
highly_eff.columns


# %%
df2.to_csv('input_file.csv')


# %%
ola = pd.read_csv('output.csv')


# %%
pd.concat([df,ola],axis = 1).drop(columns = ['Unnamed: 0','Unnamed: 0.1']).to_csv('output.csv')


# %%


