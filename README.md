# Analise_De_Sentimento

Modelo de Análise de Sentimento
Discentes:
 Emerson Moreira Baliza 
Glayson José Oliveira da Silva 

Linguagem
Será usado o Python, esta linguagem e popular para projetos de classificação de sentimento devido à sua vasta gama de bibliotecas especializadas e ferramentas de processamento de texto. Sua sintaxe clara e legível facilita o desenvolvimento e a compreensão do código, bem como, a comunidade ativa e colaborativa do Python, que oferece suporte e recursos valiosos. Em resumo, o Python oferece recursos avançados de aprendizado de máquina, tornando uma opção acessível e eficiente para a criação de modelos de classificação de sentimento.

Dataset

Os dados serão retirados do Kaggle, plataforma conhecida por hospedar uma ampla variedade de bases de dados de alta qualidade para projetos de Data Science, com milhares de conjuntos de dados públicos, com uma variedade de domínios e tópicos, incluindo finanças, saúde, redes sociais e muito mais. Essas bases de dados são frequentemente disponibilizadas por empresas, organizações governamentais e indivíduos, proporcionando uma fonte rica de informações para análise e modelagem. 

Base utilizada : Stop_Words_Personalizadas.csv

Análise Exploratória

O projeto seguirá as etapas abaixo, de modo que inicialmente será compreendido o problema, os dados e os objetivos, posteriormente será feito a manipulação da base, criação de modelo, treinamento, teste e validação com uso de métricas, conforme abaixo.

1.	Compreensão do problema e dos dados: 

A base conta com dados de usuários, com Nome, Avaliações,  Comentários, entre outros, temos como objetivo, avaliar o sentimento dos comentários dos usuários.

2.	Limpeza e pré-processamento de dados: 

Há linhas com campos vazios, registros duplicados, acentos, vírgulas, pontos e números, estes serão removidos, a fim de termos uma base limpa e confiável para manipulação.


3.	Criação de modelo: 

O modelo fará uso de Bibliotecas do Python, funções e base de referência, será feito teste e treinamento do modelo.
Pandas – Manipulação e análise de dados
Numpy – Álgebra linear
re – Expressões regulares
Matplotlib – Visualização gráfica
Seaborn - Visualização gráfica
Scikit-learn - Aprendizado de máquina
Processamento de Linguagem Natural:
Gensim
Spacy
NLTK

4.	Visualização de dados: 

Será feito alguns gráficos a fim de resumir os dados e permitir uma visão estratégica dos dados.


 
Modelo de Análise de Sentimento

Pré-processamento

Importando as bibliotecas necessárias

import pandas as pd
import re
import nltk
import spacy
import matplotlib.pyplot as plt
import seaborn as sns

%matplotlib inline
import spacy.cli
spacy.cli.download("pt_core_news_sm")

import pt_core_news_sm

spc_pt = pt_core_news_sm.load()


Limpando o dataset

Lendo o dataset, primeiramente

data = pd.read_csv("../input/brazilian-ecommerce/olist_order_reviews_dataset.csv")
data.head(15)
 
Filtrar/Ordenar base por colunas mais relevantes 

data.drop(['order_id', 'review_creation_date', 'review_answer_timestamp'],
          1, inplace = True)


data.info()

<class 'pandas.core.frame.DataFrame'>
RangeIndex: 100000 entries, 0 to 99999
Data columns (total 4 columns):
 #   Column                  Non-Null Count   Dtype 
---  ------                  --------------   ----- 
 0   review_id               100000 non-null  object
 1   review_score            100000 non-null  int64 
 2   review_comment_title    11715 non-null   object
 3   review_comment_message  41753 non-null   object
dtypes: int64(1), object(3)


Checagem de duplicados

duplicados = round(sum(data.duplicated("review_id"))/len(data)*100, 2)
print(f"Reviews com id duplicado: {duplicados}%.")

Reviews com id duplicado: 0.83%

data[data.duplicated("review_id", keep =  False)].sort_values(by = "review_id") 

Removendo duplicados 

data.drop_duplicates("review_id", inplace = True)

Concatenar dados para retirar só linhas com texto

data.fillna('', inplace = True) 
data['review'] = data['review_comment_title'] + ' ' + data['review_comment_message']
data = data[data['review'] != ' ']
data.info()

<class 'pandas.core.frame.DataFrame'>
Int64Index: 43152 entries, 3 to 99999
Data columns (total 5 columns):
 #   Column                  Non-Null Count  Dtype 
---  ------                  --------------  ----- 
 0   review_id               43152 non-null  object
 1   review_score            43152 non-null  int64 
 2   review_comment_title    43152 non-null  object
 3   review_comment_message  43152 non-null  object
 4   review                  43152 non-null  object
dtypes: int64(1), object(4)


data.head() 


Review scores
data['review_score'].value_counts()
5    21762
1     9153
4     6296
3     3721
2     2220
Name: review_score, dtype: int64

Ajuste de Score para Positivo ou Negativo, onde se o score for menor ou igual a 3, consideraremos negativa (0) e caso contrário, positiva (1).

labels = []

for score in data['review_score']:
  if score > 3:
    labels.append(1)
  else:
    labels.append(0)

data['label'] = labels
data.head(10)
 

plt.figure(figsize=(8,6))
sns.countplot(data['label'])
plt.show()


 


Pré-processamento do texto

import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
[nltk_data] Downloading package stopwords to /usr/share/nltk_data...
[nltk_data]   Package stopwords is already up-to-date!
stopwords_pt = stopwords.words("portuguese")
stopwords_pt

['de',
 'a',
 'o',
 'que',
..., 
'teriam']

Palavras como 'não' e 'nem' podem ser importantes na análise de sentimentos, por isso vamos tirá-las da lista de stopwords.

stopwords_pt.remove('não')
stopwords_pt.remove('nem')
def limpa_texto(texto):
  '''(str) -> str
    '''
  texto = texto.lower()

  texto = re.sub(r"[\W\d_]+", " ", texto)

  texto = [pal for pal in texto.split() if pal not in stopwords_pt]

  spc_texto = spc_pt(" ".join(texto))
  tokens = [word.lemma_ if word.lemma_ != "-PRON-" else word.lower_ for word in spc_texto]
  
  return " ".join(tokens)
data['review'] = data['review'].apply(limpa_texto)
data.info()
<class 'pandas.core.frame.DataFrame'>
Int64Index: 43152 entries, 3 to 99999
Data columns (total 6 columns):
 #   Column                  Non-Null Count  Dtype 
---  ------                  --------------  ----- 
 0   review_id               43152 non-null  object
 1   review_score            43152 non-null  int64 
 2   review_comment_title    43152 non-null  object
 3   review_comment_message  43152 non-null  object
 4   review                  43152 non-null  object
 5   label                   43152 non-null  int64 
dtypes: int64(2), object(4)

Feature extraction

Será testado dois métodos: Bag of Words com um vetor de componentes binários ou TF-IDF.

Com Bag of Words

from sklearn.feature_extraction.text import CountVectorizer

texto = data['review']

X_bow = vectorizer.fit_transform(texto)
X_bow.toarray()
array([[0, 0, 0, ..., 0, 0, 0],
       [0, 0, 0, ..., 0, 0, 0],
       [0, 0, 0, ..., 0, 0, 0],
       ...,
       [0, 0, 0, ..., 0, 0, 0],
       [0, 0, 0, ..., 0, 0, 0],
       [0, 0, 0, ..., 0, 0, 0]])
print(X_bow.shape, type(X_bow))
(42891, 5000) <class 'scipy.sparse.csr.csr_matrix'>

Com TF-IDF
from sklearn.feature_extraction.text import TfidfVectorizer
tfidf_vect = TfidfVectorizer(max_features=5000)

X_tfidf = tfidf_vect.fit_transform(texto)
print(X_tfidf)

(0, 1883)	0.6933181141682138
  (0, 3677)	0.3055742580640303
  (0, 269)	0.3527978440552724
  (0, 578)	0.4278176238568358

…



Modelos

Os dados serão divididos, base de treino (70%) e teste (30%).

from sklearn.model_selection import train_test_split
X1_train, X1_test, y1_train, y1_test = train_test_split(X_bow, data['label'],
                                                        test_size=0.3, random_state = 10)

X2_train, X2_test, y2_train, y2_test = train_test_split(X_tfidf, data['label'],
                                                        test_size=0.3, random_state = 10)

Importando as métricas que serão usadas para avaliação de cada modelo:

from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, roc_auc_score
def mostra_metricas(y_true, y_pred):
  ''' Função que recebe o y real, o y predito e mostra as
  principais metricas.
  '''
  print("Acurácia: ", accuracy_score(y_true, y_pred))
  print("\nAUROC:", roc_auc_score(y_true, y_pred))
  print("\nF1-Score:", f1_score(y_true, y_pred, average='weighted'))
  print("\nMatriz de confusão:")
  sns.heatmap(confusion_matrix(y_true, y_pred), annot=True)
  plt.show()


Regressão Logística

Texto vetorizado com Bag of Words

from sklearn.linear_model import LogisticRegression
# Instanciando a reg. logistica
reglog = LogisticRegression()

Aplicando o modelo

reglog.fit(X1_train, y1_train)
/opt/conda/lib/python3.7/site-packages/sklearn/linear_model/_logistic.py:765: ConvergenceWarning: lbfgs failed to converge (status=1):
STOP: TOTAL NO. of ITERATIONS REACHED LIMIT.

Increase the number of iterations (max_iter) or scale the data as shown in:
    https://scikit-learn.org/stable/modules/preprocessing.html
Please also refer to the documentation for alternative solver options:
    https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression
  extra_warning_msg=_LOGISTIC_SOLVER_CONVERGENCE_MSG)
LogisticRegression()

Predição
y1_reglog_pred = reglog.predict(X1_test)

Métricas

mostra_metricas(y1_test, y1_reglog_pred)
Acurácia:  0.8937674852346907

AUROC: 0.8803704892147814

F1-Score: 0.8934137960673484

Matriz de confusão

 

Texto vetorizado com tf-idf

reglog2 = LogisticRegression()

reglog2.fit(X2_train, y2_train)

y2_reglog_pred = reglog2.predict(X2_test)
/opt/conda/lib/python3.7/site-packages/sklearn/linear_model/_logistic.py:765: ConvergenceWarning: lbfgs failed to converge (status=1):
STOP: TOTAL NO. of ITERATIONS REACHED LIMIT.

Increase the number of iterations (max_iter) or scale the data as shown in:
    https://scikit-learn.org/stable/modules/preprocessing.html
Please also refer to the documentation for alternative solver options:
    https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression
  extra_warning_msg=_LOGISTIC_SOLVER_CONVERGENCE_MSG)
mostra_metricas(y2_test, y2_reglog_pred)
Acurácia:  0.8987410631022692

AUROC: 0.8888459548745419

F1-Score: 0.8987231587227253

Matriz de confusão:

 
A diferença do desempenho do modelo com os 2 métodos de feature extraction é pouca, mas todas as métricas apontam ele foi melhor com tf-idf.

Naive Bayes Multinomial
BoW
from sklearn.naive_bayes import MultinomialNB
nb1 = MultinomialNB()

nb1.fit(X1_train.toarray(), y1_train)

y1_gnb_pred = nb1.predict(X1_test.toarray())

mostra_metricas(y1_test, y1_gnb_pred)
Acurácia:  0.8877059372085794

AUROC: 0.8793225462615109

F1-Score: 0.8879668745789652

Matriz de confusão:,

 

Tf-idf
nb2 = MultinomialNB()

nb2.fit(X2_train.toarray(), y2_train)

y2_gnb_pred = nb2.predict(X2_test.toarray())

mostra_metricas(y2_test, y2_gnb_pred)
Acurácia:  0.8864625427416848

AUROC: 0.8773549064179689

F1-Score: 0.8866612269671004

Matriz de confusão:

 
Random Forest
from sklearn.ensemble import RandomForestClassifier
BoW
rf1 = RandomForestClassifier()

rf1.fit(X1_train, y1_train)

y1_dt_pred = rf1.predict(X1_test)

mostra_metricas(y1_test, y1_dt_pred)
Acurácia:  0.8952440161641281

AUROC: 0.8884662734564008

F1-Score: 0.8955734175225284

Matriz de confusão:
 

Tf-idf
rf2 = RandomForestClassifier()

rf2.fit(X2_train, y2_train)

y2_dt_pred = rf2.predict(X2_test)

mostra_metricas(y2_test, y2_dt_pred)
Acurácia:  0.8915915449176252

AUROC: 0.884891718915553

F1-Score: 0.8919749125971954

Matriz de confusão:

 


Resultados

Para os modelos, a diferença entre usar Bag of Words ou TF-IDF foi relativamente  pequena, os modelos apresentaram melhores métricas com TF-IDF, com exceção do Naive Bayes.

O melhor resultado, sendo tido como melhor modelo foi a regressão logística (com TF-IDF), com acurácia e F1 de 90% e AUROC de 89%.

Vamos testá-lo com um novo texto:

def nova_predicao(texto):
  texto_vetorizado = tfidf_vect.transform([texto])
  pred = reglog2.predict(texto_vetorizado)

  if pred == 0:
    print("Essa é uma review negativa.")
  else:
    print("Essa é uma review positiva.")

nova_predicao("Demorou muito não gostei")
Essa é uma review negativa.

nova_predicao("Achei cheirosinho")
Essa é uma review positiva.

nova_predicao("Nossa que produto ruim é esse parece que encontrei no lixo")
Essa é uma review negativa.

nova_predicao("Gostei")
Essa é uma review positiva.
nova_predicao("Não gostei")
Essa é uma review negativa.
![image](https://github.com/glaysonbk/Analise_De_Sentimento/assets/100309646/df4dcd0f-c536-4386-9c56-2210a00c7aef)
