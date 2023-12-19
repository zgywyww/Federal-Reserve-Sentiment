## Project Description
Our project aims to predict Federal funds rate decision, which will hugely impact monetary policy and macroeconomic indexes in the US, from the Federal Open Market Committee's (FOMC) official documents.


## Dataset
Our datasets include both metadata and textual data.
- Metadata: macroeconomic variables from 1996 to 2023 organized by month, downloaded using FRED API, - [full_df.csv](https://raw.githubusercontent.com/zgywyww/Federal-Reserve-Sentiment/main/full_df.csv)
- Textual data: Fed official documents recording texts to extract sentiments. Organized by meeting dates from 1999-2023 and downloaded using pypl [pypl](https://github.com/zgywyww/Federal-Reserve-Sentiment/blob/main/Fed_Minute_Download.ipynb) and BeautifulSoup.
  
  Federal Minutes: [federal_minute](https://raw.githubusercontent.com/zgywyww/Federal-Reserve-Sentiment/main/Fed_Minutes_1996_2023.csv)
  
  Federal Statement: [federal_statement](https://raw.githubusercontent.com/zgywyww/Federal-Reserve-Sentiment/main/federal_reserve_statement_1999_2023.csv)
  
  Federal Speech: [federal_speech](https://raw.githubusercontent.com/zgywyww/Federal-Reserve-Sentiment/main/federal_reserve_speeches_1996_2023.csv)

## Preprocessing

## Models
- Baseline ML: In the classification task, metadata is used as independent variables to predict the Federal Reserve's rate decisions, which are the dependent variable. Baseline models including SVC, RFC, GBC, Perceptron, and AdaBoost are optimized using grid search and k-fold cross-validation to find the best hyperparameters. Due to an imbalanced dataset skewed towards 'hold' decisions, class weights are used to improve predictive performance by giving more importance to less frequent outcomes.
- LSTM: we refined the LSTM model by incorporating Tfidf vectorization for input data. This enhancement, inspired by Takahashi (Takahashi, 2020), enables the model to retain more pertinent information, potentially improving its predictive accuracy. Additionally, we utilized Ray Tune Hyperband (Liaw et al., 2018), a technique explored in our coursework, for efficient hyperparameter optimization.
- Baseline Fin-BERT: We directly employed the Fin-BERT model (Huang et al., 2022), a variant of BERT specifically pre-trained on financial texts, as the BERT baseline. We applied this model to our aggregated dataset, which consists of FOMC minutes and statements analyzed at the sentence level. The model’s output was then used to calculate a weighted average sentiment score for each text paragraph. This score served as the basis for assigning sentiment labels to each paragraph, facilitating a more nuanced understanding of the underlying sentiment in financial communications. We later applied these sentiment scores to different classification models including Random Forest, SVC and KNN, to observe how these scores would contribute to rates decision classification. Hyper-parameter tuning is applied to these model to explore the best performance.
- Fine-tuned Fin-BERT


## Conclusion
We ended up making improvements on previous work with an accuracy of around 94% by using a fine-tuned Fin-BERT model to input the textual paragraphs even if they are large.
| Model                  | Performance (Accuracy) |
|------------------------|------------------------|
| Baseline ML            | 0.877                  |
| LSTM                   | 0.729                  |
| Baseline Fin-BERT      | 0.864                  |
| Fine-tuned Fin-BERT    | 0.942                  |


## References
