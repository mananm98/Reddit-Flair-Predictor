# Reddit-Flair-Predictor
A web application to detect flair of a given Reddit r/india post. The application is live [here](https://predict-this-flair.herokuapp.com/)

## Contents
1. [Data Collection](https://github.com/mananm98/Reddit-Flair-Predictor/blob/master/Part-1%20Reddit%20Data%20Collection.ipynb) : This notebook contains the code to scrape data from reddit and make a dataset.

2. [Data Analysis](https://github.com/mananm98/Reddit-Flair-Predictor/blob/master/Part%20-%202%20Exploratory%20Data%20Analysis.ipynb) : This notebbok contains the code to explore, clean, analyse dataset.

3. [Flair Prediction](https://github.com/mananm98/Reddit-Flair-Predictor/blob/master/Part-3%20Building%20Flair%20Detector.ipynb) : This notebook contains code to test various Machine Learning and Deep Learning models on our dataset.

4. [Web app](https://github.com/mananm98/Reddit-Flair-Predictor/tree/master/Web-app) : Contains code used to develop a flask application and deploy on heroku.

## Getting started
Open Terminal
- `git clone https://github.com/mananm98/Reddit-Flair-Predictor.git`
- `cd web-app`
- `virtualenv venv`
- `source venv/bin/activate`
- `pip install -r requirements.txt`
- `python app1.py`  

Open the displayed URL on your browser to run application

## Approach :bulb:

After Extraction and cleaning of dataset I applied various Machine Learning and Deep Learning model to get good accuracy.

### Choosing the best features and applying Baseline models
- To find the best combination of features, I used Bag of Words model. First text was converted to Tf-idf vectors and then these vectors were fed to the following models :-
1. **Linear SVC**
2. **Naive-Bayes**
3. **Logistic Regression**
4. **Random Forests**

### Results 

#### Title
| Model      | Test Accuracy |                                                     
| :---:        |    :----:   |
| LinearSVC      | 0.7194029850746269       | 
| Naive - Bayes   | 0.7074626865671642        | 
| Logistic Regression      | **0.746268656716418**      | 
| Random Forest Classifier   | 0.7134328358208956        | 


#### URL
| Model      | Test Accuracy |                                                     
| :---:        |    :----:   |
| LinearSVC      | **0.5269461077844312**       | 
| Naive - Bayes   | 0.4431137724550898       | 
| Logistic Regression      | 0.5239520958083832      | 
| Random Forest Classifier   |  0.5        |

#### Comments
| Model      | Test Accuracy |                                                     
| :---:        |    :----:   |
| LinearSVC      | **0.5132450331125827**       | 
| Naive - Bayes   | 0.4105960264900662       | 
| Logistic Regression      | 0.49337748344370863      | 
| Random Forest Classifier   |  0.423841059602649        |

#### Title + Comments + URL
| Model      | Test Accuracy |                                                     
| :---:        |    :----:   |
| LinearSVC      | **0.7761194029850746**       | 
| Naive - Bayes   | 0.5940298507462687       | 
| Logistic Regression      | 0.7343283582089553      | 
| Random Forest Classifier   |  0.7134328358208956        |


#### Title + Body + URL
| Model      | Test Accuracy |                                                     
| :---:        |    :----:   |
| LinearSVC      | **0.8238805970149253**      | 
| Naive - Bayes   | 0.6716417910447762       | 
| Logistic Regression      | 0.7970149253731343      | 
| Random Forest Classifier   |  0.7880597014925373        |

- Using a combination of (Title + Body + URL) performed the best, So I decided to these features for deep learning models

## Deep Learning :boom:

- Deep Learning models can learn more complex functions than traditional Machine Learning models.   
- I used the folowing models :- 
### 1. Word Embeddings + CNN 
 - CNNs are well known for there location invariant feature extraction capability, So I decided to use CNN model for text classification. In hope that CNN model will learn some useful filters, which can be used to distinguish between different text categories 
 
 ### 2. LSTMs
  - LSTMS perform really well on many NLP tasks, because of their ability to remember past information and learn context. While the CNN model tries to classify by learning distinguishing tokens or sequence of tokens. LSTMs on the other hand try to understand the text, they can learn relationships between tokens.
  - But LSTMs didn't perform as accepted, I think the reason for low accuracy is less data

 
  ### 3. Bidirectional LSTMs
  - Bidirectional LSTMs, are more effective than unidirectional ones. This is because training two LSTMs on the same input in forward and reverse direction allows the network to learn better context.
  - Bidirectional LSTM performed better than a unidirectional LSTM.
  
  ### 4. Hybrid-model CNN-LSTM
 - The idea was to combine the advantages of both models. The CNN would act as an effective feature extractor which would pass only the essential information to LSTM. And then LSTM model would learn interesting dependencies in data.

  
| Model      | Test Accuracy |                                                     
| :---:        |    :----:   |
| Word Embeddings + CNN      | **0.8477611940298507**      | 
| LSTM   | 0.7014925373134329       | 
| Bidirectional LSTM      | 0.764179104477612      | 
| Hybrid-model CNN-LSTM   |  0.7522388059701492        |

- LSTM based models lagged behind CNN and even traditional ML models like Logistic Regression and LinearSVC. I think the reason is less data LSTMs require large data to perform well.
- The CNN model performed the best and was deployed in the [web-app](https://predict-this-flair.herokuapp.com/) :tada:

