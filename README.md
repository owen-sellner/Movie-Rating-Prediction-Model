# Movie Rating Prediction Model

* Gabrielle Tang 
* Vyomesh Iyenger
* Owen Sellner 
* Edward Ho 

## Introduction

The goal of our project is to predict the performance of a movie in a specific genre based
on the movie’s genre, country of release, year of release, and the principal actor’s previous
experience. Our model outputs a movie rating range based on the actor’s experience and the
genre of the movie they are being considered for. The insights from the model will be looking to
solve two key problems within the film industry. The first is the problem (Hofmann, 2019) that
casting agencies face when determining which actor is the best fit for a potential movie in a
specific genre based on their past work and acting experience. The second is typecasting within
the acting industry and how this can be eliminated in order to improve actor credibility (Scarlett,
2017) and limit discrimination (Scarlett, 2017) for specific role types.

Our model can also help identify trends in actor performances that outline how translatable acting skills are across genres, giving filmmakers insights on casting actors without experience in a specific genre. In addition to actor experience and expertise, we are using genre,
country of production, and release year as parameters. The reason we chose to use the genre as a
feature in our model is because actors with previous experience in the genre they are looking to
act in typically have better performances as they are familiar with the intricacies associated with
the genre. We chose to include the country of production and release years as features because
we found through examining the data that movies produced throughout different years or in
different countries had different ratings, even with the same principal actor. In order to account
for cultural changes, and trends across time we opted to include these features when predicting a
theoretical movie’s rating.

## Description of Data

The dataset we are using for this project is a collection of movie data from movies
released in the years 1986 to 2016. The dataset has 15 columns with features of three different
data types: float, integer and string. In total the dataset contains 7668 rows. Within the dataset,
there are many null values to account for, notably, the “budget” feature which has 2171 null
entries. In addition, using the unique method in pandas it was found that there are 2815 distinct
actors/actresses within the dataset which is much less than the 7668 rows of movies meaning
there are many actors that have worked on a multitude of movies which will help identify actors
that have been in a variety of genres.

![image](https://user-images.githubusercontent.com/43042601/232344094-13591754-dd0c-4c86-aa73-6a181def2b49.png)

Figure 1 - Dataset information breakdown

### Source of Data

All the data used in this project is from the same dataset that was sourced from Kaggle.
Although there is code posted by Kaggle users on the dataset page these projects focused on the
topic of movie recommendations based on user preferences which do not match the goal of our
project. The creator of the dataset states that they collected the information through web scraping
the Internet Movie Database (IMDb) website.

Link to Dataset: <https://www.kaggle.com/datasets/danielgrijalvas/movies>

### Data Visualization

Since our model predicts the audience rating of a movie we wanted to first get an
understanding of the audience score values to see if our dataset was skewed. As seen in Figure 2
the histogram of the “score” feature shows that a significant number of the movies in our dataset
have an audience rating between six and seven on the ten-point scale and that there are very few
movies with a score less than five. This information is used later to help decide the intervals used
when binning this feature.

![image](https://user-images.githubusercontent.com/43042601/232344219-a2fb54c4-90c0-47f2-9d24-31c98edf88ef.png)

Figure 2 - Histogram of audience score

### Cleaning the Data & Feature Engineering

To prepare the dataset for clustering the dataset had to be cleaned. The steps that were
required included removing rows with null values, removing irrelevant features, refactoring
condensed features, reducing infrequently used options on categorical features, and binning the
target variable. To begin we created a new feature called “new\_score\_bin” that converted the
"score" feature values from the original dataset into six bins. This reduced the number of
possible target variable values and accounted for the fact that there is not a meaningful difference
within specific ranges of ratings. For example, a rating of 67% is not significantly better than a
film with a rating of 63%. Furthermore, it decreases our model complexity in requiring
predictions of fewer class values.

![image](https://user-images.githubusercontent.com/43042601/232344297-5ec82ed3-cad7-4043-8a11-eba6cd9d7637.png)

Figure 3 - Dataset audience score category counts

After binning the audience ratings, the features of the dataset were further adjusted to
remove unneeded columns and add a new column for the country that the movie was released in.
The original dataset included a “country” feature but that value represented the location where
the movie was filmed not where it had been initially released. Since the country where the movie
was released would dictate who the audience is it was decided that was a better feature to
consider. To do this, the “release” feature from the original dataset was split to extract the name
of the country to create the feature “release\_country”.

The next step taken was to reduce the number of categories in the “release\_country” and
“genre” features. For “release\_country” it was decided that our model would focus on movies
released in English-speaking countries, specifically the countries of Australia, the United States,
and the United Kingdom as these countries had the highest number of movies. In addition, it was
decided that reducing the number of categories in the “genre” feature would help to improve the
accuracy of our model by reducing the number of options the model would need to consider. It
was found in Figure 4 that many of the genre categories had very few movies so the dataset was
filtered to include only the top nine most frequently occurring genres.

![image](https://user-images.githubusercontent.com/43042601/232344326-6e725a1e-7062-4bc5-91ec-096181384490.png)

Figure 4 - Histogram of “genre” feature before adjustment

![image](https://user-images.githubusercontent.com/43042601/232344341-c834ad09-ce2d-4735-aa38-898358a9a8b1.png)

Figure 5 - Dataset after feature adjustment

### Transforming the Data

To prepare the data frame for clustering by actor type, the “genre” feature had to be
one-hot encoded to be able to count the number of films in each genre for the actors. To do this
the pandas “get\_dummies” method was used on a copy of the original data frame to convert the
category values of the “genre” to columns. In addition, the columns that would not be needed for
clustering were removed from this newly created data frame including the features "genre",
"year", "score", "new\_score\_bin", and "release\_country". One-hot encoding is used again later in
the project to prepare the movie genre, release country, and actor cluster features for the
supervised learning models.

![image](https://user-images.githubusercontent.com/43042601/232344372-37d162c2-1e1a-4360-bf77-56c77e9ea736.png)

Figure 6 - Dataset after one-hot encoding

## Machine Learning

### Methodology

In order to achieve our goal of predicting a movie's rating based on the input features of
genre, year of release, country of production, and principal actor experience in various genres;
we capitalized on the use of both unsupervised and supervised machine learning.
 The first model is a K-clustering algorithm used to cluster the collection of actors based
on the number of films they have done in each of the ten genres as discussed in the *Data
Description* section of this report. In order to determine the optimal number of clusters, we ran
the unsupervised algorithm with K values ranging from 1-514 and plotted their distance metrics
in order to find the elbow of the curve. Initially, we set the maximum K value to be 2471 as it is
equal to the number of unique actors (Appendix - Figure A.1), but found that the residual sum of
squares was converging after a K value of 514 since the number of distinct clusters remained
equal to 514 even as the K value increased. In doing so, we determined the optimal value of K to
be 12, where the least sum of squares distance is minimized relative to maintaining a moderate
level of complexity.

The output of the clustering allowed us to replace the 'star' column in our training dataset
with a 'cluster' column, where we were able to reduce the possible number of values from 2471
to 12. In this manner, the clustering algorithm helps introduce a level of generalization to the
goal of predicting movie ratings, thus also straying from potential overfitting or algorithm
complexity due to the high number of varying values. This also means that our system of
algorithms is able to consider any arbitrary actor, not dependent on a set list of possible
categorical values that it recognizes.

Once the “cluster” column is added to the data set as a feature, the second phase of the
process involves using supervised learning in order to predict the actual movie rating based on
all the features. We approached this aspect of the process with two different models in order to 
determine which provided the greater accuracy while maintaining relatively low complexity. The
first supervised learning model we tried was a decision tree model, and the second, a Naive
Bayes classifier. Both of the models used are from the *sklearn* library.

For feature engineering, we decided on genre, year, country of release, and the actor
cluster. We experimented with additional features like including movie budget, however, after
implementing the budget feature we discovered that our model’s accuracy decreased. A potential
reason why the budget negatively impacted model accuracy is that the dataset we used had over
2000 null values for the budget category. Additionally, film budget is not always an accurate
indicator of movie quality, with low-budget movies capable of achieving critical acclaim, and
high-budget movies capable of critical failure.

### Results

The decision tree model acted as a basis for which we conducted our feature engineering
and initial testing of our prediction capabilities. In training and testing our models using both
K-fold cross-validation, and regular cross-validation, we utilized three different methods of
accuracy visualization and measurement: RMSE, accuracy score (via the *sklearn* library), and
confusion matrices.

Through our initial implementation of the decision tree model, we determined that
changes to the features: such as the removal of the "budget" feature increased accuracy, while
others such as the removal of the "director" feature decreased complexity (as discussed in the
Data Description section of this report). Further, through the tuning of the “min\_sample\_split”,
and “max\_depth” parameters we were able to determine that a “min\_sample\_split” of 100 and a
“max\_depth” of 7 produced the highest accuracy score values.

The finalized version of the decision tree model produces an RMSE of 0.96, as found
through the K-fold cross-validation where we used 10 folds, a value that shows that it is not very
reliable as it is much closer to 1 than 0. In the same vein, we calculated the accuracy score to be
around 43%.

In order to gain a visual understanding of how our model was functioning, we plotted an
instance of the Decision Tree. It is clear that the model follows the skewness of the data, as the
vast majority of branches lead to class label instances of "Class 3" which is equivalent to a movie
rating between 60% and 69%.

Following the implementation of the Decision Tree, we decided to test the accuracy of a
Naive Bayes classifier. The model produces an RMSE of 0.96 as found through the K-fold
cross-validation, and an accuracy score of 43.19%. Both RMSE and the accuracy score show that
our model has poor accuracy.

### Discussion

We discovered that the two models have nearly identical accuracy scores and RMSE as
determined by K-fold cross-validation. The identical accuracy scores may be caused by the small
size of our dataset, which may lead to skewed data and because the Naive Bayes and Decision
Tree models use an identical training and testing data split in both instances of cross-validation.
 After significant testing with both supervised learning models, we discovered that the
models will predict class 3 as the output in a majority of cases, as seen in the confusion matrix. A
potential explanation for why this occurs is because about 41% of the data points in our dataset
are classified as class 3 (rating 60-69%), making it the most common class by a wide margin
(Figure 3). Since class 3 is the majority class, it could potentially cause the models to predict
generic inputs and outliers as the majority class.

The overall runtime of our code is of negligible length except for the code blocks for
obtaining the optimal K value for clustering. Obtaining the optimal K value had a runtime of 1
minute (Appendix - Figure A.2). One of the reasons why our runtime was so short may be due to
our small dataset being only about 6000 rows.

### Unexpected Results

When clustering the actors in our dataset, we discovered that some of the actor clusters
seemingly have no distinguishable patterns. Actor clusters such as cluster 8 distinctly contain
action veterans (Appendix - Figure A.3), however, clusters like cluster 2 have no distinguishable
patterns (Appendix - Figure A.4).

## Conclusion

We found that in order to create a model with robust accuracy, it is required to have a
large amount of training data which spans a variety of outputs. For example, it is not enough to
have 1 or 2 training data points which have high ratings as this would be considered an outlier,
instead of a potential rating output. Comparing the decision tree and Naive Bayes models
resulted in very similar outputs, accuracy scores, and RMSE values. This was not surprising as
both models relied on the same limited number of data points. Since the majority of our training
data related to one of the outputs (class 3 - rating of 60-69%), this caused the results to be
skewed toward that class. We found through our confusion matrices that while the results were
distributed across the different outputs, they were strongly clustered around the third class.
 The real-world problem we are looking to investigate involves how casting agents can
better predict the outcome of a movie based on the previous experience of the principal actor,
country of release, genre of movie, and year of release. Through our model we have found that
the current output is not very reliable and outliers in the data are often grouped with the largest
data group (class 3 - rating of 60-69%), causing inaccurate predictions. In order for this model to
be applied in the real world, it would have to be modified to be trained on a much larger training
set, as well as consider other factors which may influence movie popularity such as current
events, the stardom of actors involved, and size of the target demographic.

An important finding we realized about machine learning throughout the process was that
the accuracy of the model was very dependent on how many data points were provided for each
training feature, as well as how much it changed the accuracy score. When trying to improve our
model, we thought adding budget as a feature would improve accuracy, however, it made it
worse. In order to create machine learning models which have accurate results, it’s important to
critically consider and test the impact of each individual feature on accuracy.

In order to improve our model, two key changes can be done: adding more variety to our
training data with a larger data set, and adding more features on which the model can use to place
the inputs. Potential additional features include principal actor stardom (measured using the
Ulmer scale), the other actors in the movie, director, time of year of release, and public opinion
on movie topics based on recent events. Since some of these factors are difficult to quantify, the
model would have to be very robust and be trained on a wider range of data points. While our
model is a good start, doing so could potentially lead to more significant and accurate insights
into the movie industry.

## Appendix

![image](https://user-images.githubusercontent.com/43042601/232344569-3ccd6e18-c75e-47cd-b1db-9d9e986dafff.png)

Figure A.1: Data frame of actor movie history

![image](https://user-images.githubusercontent.com/43042601/232344584-dad948ac-7ec3-4e2b-8ed1-97d8264f88ba.png)

Figure A.2: Finding optimal K value

![image](https://user-images.githubusercontent.com/43042601/232344610-79c9c8c7-de7c-492e-99f3-cd510ba0710f.png)

Figure A.3: Actor dataframe cluster 8

![image](https://user-images.githubusercontent.com/43042601/232344638-67d69e5e-64c1-4d3f-9440-cda2d3b903f4.png)

Figure A.4: Actor dataframe cluster 2


## References

Gunjal, S. (2020, July 28). *K Fold Cross Validation*. Quality Tech Tutorials. Retrieved March 20, 2023, from https://satishgunjal.com/kfold/

Hofmann, K. H. (2019, October). *The contribution of actors in film production and
distribution ...* ResearchGate. Retrieved February 14, 2023, from

https://www.researchgate.net/publication/336407507\_The\_contribution\_of\_actors\_in\_film
\_production\_and\_distribution\_exploring\_the\_antecedents\_of\_the\_drawing\_power\_of\_stars

Scarlett, N. (2017, December). *The Truth About Casting: An Analysis of Typecasting in the
Boston Theatre Market*. eScholarship@BC. Retrieved February 14, 2023, from
https://dlib.bc.edu/islandora/object/bc-ir%3A107893/datastream/PDF/view
