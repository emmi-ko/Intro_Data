# %% [markdown]
# # Introduction to Data Science 2026
# 
# # Week 2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# %% [markdown]
# Exercise 1 | Titanic: data preprocessing and imputation
# Note: You can find tutorials for NumPy and Pandas under 'Useful tutorials' in the course material.

# %% [markdown]
# Download the Titanic dataset (https://www.kaggle.com/c/titanic) train.csv from Kaggle or directly from the course material, and complete the following exercises. If you choose to download the dataset from Kaggle, you will need to create a Kaggle account unless you already have one, but it is quite straightforward.
# 
# The dataset consists of personal information of all the passengers on board the RMS Titanic, along with information about whether they survived the iceberg collision or not.
# 
# 1. Your first task is to read the data file and print the shape of the data.
# 
#     Hint 1: You can read them into a Pandas dataframe if you wish.
#     
#     Hint 2: The shape of the data should be (891, 12).

# %%
# Use this cell for your code
df = pd.read_csv("titanic.csv")
print(df.shape)

# %% [markdown]
# 2. Let's look at the data and get started with some preprocessing. Some of the columns, e.g Name, simply identify a person and are not useful for prediction tasks. Try to identify these columns, and remove them.
# 
#     Hint: The shape of the data should now be (891, 9).

# %%
# Use this cell for your code
print(df.info())
df = df[['Survived', 'Pclass', 'Sex', 'Age', 'SibSp',
       'Parch', 'Fare', 'Cabin', 'Embarked']]
print(df.shape)
# %% [markdown]
# 3. The column Cabin contains a letter and a number. A smart catch at this point would be to notice that the letter stands for the deck level on the ship. Keeping just the deck information would be more informative when developing, e.g. a classifier that predicts whether a passenger survived. The next step in our preprocessing will be to add a new column to the dataset, which consists simply of the deck letter. You can then remove the original Cabin-column.
# 
#     Hint: The deck letters should be ['A' 'B' 'C' 'D' 'E' 'F' 'G' 'T'].

# %%
# Use this cell for your code
df["Deck"] = df["Cabin"].str.extract("([A-Z]+)")

print(df["Deck"].value_counts())
df = df.drop(columns=["Cabin"])
# %% [markdown]
# 4. Youâ€™ll notice that some of the columns, such as the previously added deck number, are categorical (https://en.wikipedia.org/wiki/Categorical_variable). To preprocess the categorical variables so that they're ready for further computation, we need to avoid the current string format of the values. This means the next step for each categorical variable is to transform the string values to numeric ones, that correspond to a unique integer ID representative of each distinct category. This process is called label encoding and you can read more about it here (https://pandas.pydata.org/docs/user_guide/categorical.html).
# 
#     Hint: Pandas can do this for you.

# %%
# Use this cell for your code
df["Deck"] = df["Deck"].astype("category").cat.codes
df["Sex"] = df["Sex"].astype("category").cat.codes
df["Embarked"]= df["Embarked"].astype("category").cat.codes
print(df.info())
# %% [markdown]
# 5. Next, let's look into missing value imputation. Some of the rows in the data have missing values, e.g when the cabin number of a person is unknown. Most machine learning algorithms have trouble with missing values, and they need to be handled during preprocessing:
# 
#     a) For continuous variables, replace the missing values with the mean of the non-missing values of that column.
# 
#     b) For categorical variables, replace the missing values with the mode of the column.
# 
#         Remember: Even though in the previous step we transformed categorical variables into their numeric representation, they are still categorical.

# %%
# Use this cell for your code

age_mean = df["Age"].mean()
embarked_mode = df["Embarked"].mode() 
deck_mode = df["Deck"].mode() 

df.fillna({"Age": age_mean}, inplace=True)
df.fillna({"Embarked":embarked_mode}, inplace=True)
df.fillna({"Deck": deck_mode}, inplace=True)


print(df.info())

# %% [markdown]
# 6. At this point, all data is numeric. Write the data, with the modifications we made, to a .csv file. Then, write another file, this time in JSON format, with the following structure:

# %%
#[
#    {
#        "Deck": 0,
#        "Age": 20,
#        "Survived", 0
#        ...
#    },
#    {
#        ...
#    }
#]

# %%
# Use this cell for your code
df.to_csv("titanic_numeric.csv")
df.to_json('titanic_numeric.json', orient='records', indent=1)

# %% [markdown]
# Study the records and try to see if there is any evident pattern in terms of chances of survival.

# %% [markdown]
# Remember to submit your code on the MOOC platform. You can return this Jupyter notebook (.ipynb) or .py, .R, etc depending on your programming preferences.

# %% [markdown]
# ## Exercise 2 | Titanic 2.0: exploratory data analysis
# 
# In this exercise, weâ€™ll continue to study the Titanic dataset from the last exercise. Now that we have done some preprocessing, itâ€™s time to look at the data with some exploratory data analysis.

# %% [markdown]
# 1. First investigate each feature variable in turn. For each categorical variable, find out the mode, i.e., the most frequent value. For numerical variables, calculate the median value.

# %%
# Use this cell for your code
survived_mode = df["Survived"].mode()
pclass_mode = df["Pclass"].mode()
sex_mode = df["Sex"].mode()
age_mean = df["Age"].mean()
sibsp_mode = df["SibSp"].mode()
parch = df["Parch"].mode()
fare_mean = df["Fare"].mean()
embarked_mode = df["Embarked"].mode() 
deck_mode = df["Deck"].mode() 

# %% [markdown]
# 2. Next, combine the modes of the categorical variables, and the medians of the numerical variables, to construct an imaginary â€œaverage survivorâ€. This "average survivor" should represent the typical passenger of the class of passengers who survived. Also following the same principle, construct the â€œaverage non-survivorâ€.
# 
#     Hint 1: What are the average/most frequent variable values for a non-survivor?
#     
#     Hint 2: You can split the dataframe in two: one subset containing all the survivors and one consisting of all the non-survivor instances. Then, you can use the summary statistics of each of these dataframe to create a prototype "average survivor" and "average non-survivor", respectively.

# %%
# Use this cell for your code
survivors_df = df[df["Survived"] == 1]
non_survivor_df = df[df["Survived"] == 0]

survived_mode = survivors_df["Survived"].mode().iloc[0]
pclass_mode = survivors_df["Pclass"].mode().iloc[0]
sex_mode = survivors_df["Sex"].mode().iloc[0]
age_mean = survivors_df["Age"].mean()
sibsp_mode = survivors_df["SibSp"].mode().iloc[0]
parch = survivors_df["Parch"].mode().iloc[0]
fare_mean = survivors_df["Fare"].mean()
embarked_mode = survivors_df["Embarked"].mode().iloc[0]
deck_mode = survivors_df["Deck"].mode().iloc[0]

average_survivor = pd.Series({"survived":survived_mode,"pclass":pclass_mode,"sex":sex_mode,"age":age_mean,"sibsp":sibsp_mode,"parch":parch,"fare":fare_mean,"embarked":embarked_mode,"deck":deck_mode})
print(average_survivor)

survived_mode = non_survivor_df["Survived"].mode().iloc[0]
pclass_mode = non_survivor_df["Pclass"].mode().iloc[0]
sex_mode = non_survivor_df["Sex"].mode().iloc[0]
age_mean = non_survivor_df["Age"].mean()
sibsp_mode = non_survivor_df["SibSp"].mode().iloc[0]
parch = non_survivor_df["Parch"].mode().iloc[0]
fare_mean = non_survivor_df["Fare"].mean()
embarked_mode = non_survivor_df["Embarked"].mode().iloc[0]
deck_mode = non_survivor_df["Deck"].mode().iloc[0] 

average_non_survivor = pd.Series({"survived":survived_mode,"pclass":pclass_mode,"sex":sex_mode,"age":age_mean,"sibsp":sibsp_mode,"parch":parch,"fare":fare_mean,"embarked":embarked_mode,"deck":deck_mode})
print(average_non_survivor)


# %% [markdown]
# 3. Next, let's study the distributions of the variables in the two groups (survivor/non-survivor). How well do the average cases represent the respective groups? Can you find actual passengers that are very similar to the (average) representative of their own group? Can you find passengers that are very similar to the (average) representative of the other group?
# 
#     Note: Feel free to choose EDA methods according to your preference: non-graphical/graphical, static/interactive - anything goes.

# %%
# Use this cell for your code
import seaborn as sns

columns = ['Survived', 'Pclass', 'Sex', 'Age', 'SibSp',
       'Parch', 'Fare', 'Deck', 'Embarked']

sns.pairplot(survivors_df[columns], hue="Survived")
plt.show()
print(len(survivors_df))
similar_to_average = df[(df["Survived"] == 1) & (df["Pclass"] == 1) &  (df["Sex"] == 0) & (df["Age"].between(28, 29))& (df["SibSp"] == 0)& (df["Parch"] == 0) & (df["Embarked"] == 2) ]
print(similar_to_average) # Found one person that matched all expect the fare and deck average. Changing Pclass to 2, matches 4 people. 
# If you just try to match gender, we get 233/342 matches, around 70% of the survivors

similar_to_average_non = df[(df["Survived"] == 0) & (df["Pclass"] == 1) &  (df["Sex"] == 0) & (df["Age"].between(28, 29))& (df["SibSp"] == 0)& (df["Parch"] == 0) & (df["Embarked"] == 2) ]
print(similar_to_average_non)# No same matches found in the nonsurvivors, only when you look at same pclass and sex do you get 3 matches and changing pclass to 2 you get 6 matches and pclass 3 gives 81 matches.


sns.pairplot(non_survivor_df[columns], hue="Survived")
plt.show()
similar_to_average = df[(df["Survived"] == 0) & (df["Pclass"] == 3) &  (df["Sex"] == 1) & (df["Age"].between(30, 31))& (df["SibSp"] == 0)& (df["Parch"] == 0) & (df["Embarked"] == 2) ]
print(similar_to_average) # Found 6 matches, removing the age limit gives 177 matches
similar_to_average_non = df[(df["Survived"] == 1) & (df["Pclass"] == 3) &  (df["Sex"] == 1) & (df["Age"].between(30, 31))& (df["SibSp"] == 0)& (df["Parch"] == 0) & (df["Embarked"] == 2) ]
print(similar_to_average_non)  # Found 2 matches in the survivor group

# %% [markdown]
# 4. Next, let's continue the analysis by looking into pairwise and multivariate relationships between the variables in the two groups. Try to visualize two variables at a time using, e.g., scatter plots and use a different color to encode the survival status.
# 
#     Hint 1: You can also check out Seaborn's pairplot function, if you wish.
# 
#     Hint 2: To better show many data points with the same value for a given variable, you can use either transparency or â€˜jitterâ€™.

# %%
# Use this cell for your code

sns.pairplot(df[columns], hue="Survived", plot_kws=dict(alpha=0.5))
plt.show()
# %% [markdown]
# 5. Finally, recall the preprocessing we did in the first exercise. What can you say about the effect of the choices that were made to use the mode and mean to impute missing values, instead of, for example, ignoring passengers with missing data?

# %% [markdown]
# Use this (markdown) cell for your written answer

# The cabin column has large number of missing values, around 77%. If we had ignored all rows with missing values, we would have missed a large chunk of data. At the same time, I would have rather ignored the Cabin column while keeping all the row information.
# Imputating such a large number of values will skew the data even with a netter imputation method and in my opinion would be better left out of the analysis (so keep the row data and drop the column data). I also think that the passenger class info substitutes the cabin info, even if it doesn't give spesific deck details.

# %% [markdown]
# Remember to submit your code on the MOOC platform. You can return this Jupyter notebook (.ipynb) or .py, .R, etc depending on your programming preferences.

# %% [markdown]
# ## Exercise 3 | Working with text data 2.0
# 
# This exercise is related to the second exercise from last week. Find the saved pos.txt and neg.txt files, or, alternatively, you can find the week 1 example solutions on the MOOC platform after Tuesday.

# %% [markdown]
# 1. Find the most common words in each file (positive and negative). Examine the results. Do they tend to be general terms relating to the nature of the data? How well do they indicate positive/negative sentiment?

# %%
# Use this cell for your code
pos_dict = {}

with open("pos.txt") as f:
    for line in f:
       words = line.split(" ")
       for word in words:
           if word in pos_dict:
              pos_dict[word] = pos_dict[word] +1
           else:
              pos_dict[word] = 1

print(list(sorted(pos_dict, key=pos_dict.get, reverse=True))[0:10])    

neg_dict = {}

with open("neg.txt") as f:
    for line in f:
        words = line.split(" ")
        for word in words:
            if word in neg_dict:
                neg_dict[word] = neg_dict[word] +1
            else:
                neg_dict[word] = 1

print(list(sorted(neg_dict, key=neg_dict.get, reverse=True))[0:10])

#Most common words in pos.txt are words like "great, good, works, easy" that have positive connotations. Some words are more genreal like "product, just, use".
#Most common words in neg.txt are more general words like "fit, just, work, product, time". The only really negative word is "dont", but there is also a positive word "good". 
#The most common words in negative reviews are difficult to see indicating negative because they are more general terms that are also used a lot in the positive reviews.

# %% [markdown]
# 2. Compute a TF/IDF (https://en.wikipedia.org/wiki/Tfâ€“idf) vector for each of the two text files, and make them into a 2 x m matrix, where m is the number of unique words in the data. The problem with using the most common words in a review to analyze its contents is that words that are common overall will be common in all reviews (both positive and negative). This means that they probably are not good indicators about the sentiment of a specific review. TF/IDF stands for Term Frequency / Inverse Document Frequency (here the reviews are the documents), and is designed to help by taking into consideration not just the number of times a term occurs (term frequency), but also how many times a word exists in other reviews as well (inverse document frequency). You can use any variant of the formula, as well as off-the-shelf implementations. Hint: You can use sklearn (http://scikit-learn.org/).

# %%
# Use this cell for your code


from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer

with open("pos.txt") as f:
    pos = f.read()
with open("neg.txt") as f:
    neg = f.read()

corpus = [pos, neg]
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(corpus)
print(X.toarray())
print(X.shape)
tokens = sorted(vectorizer.vocabulary_.keys())


doc_names = ['Doc{:d}'.format(idx) for idx, _ in enumerate(X)]
df = pd.DataFrame(data=X.toarray(), index=doc_names,
                  columns=tokens)

# %% [markdown]
# 3. List the words with the highest TF/IDF score in each class (positive | negative), and compare them to the most common words. What do you notice? Did TF/IDF work as expected?


# %%
# Use this cell for your code
tfidf_transformer = TfidfTransformer(smooth_idf=True,use_idf=True)
tfidf = tfidf_transformer.fit_transform(X)
df_idf = pd.DataFrame(tfidf.toarray(), index=["pos", "neg"], columns=tokens)
print(df_idf)

pos_top_words = df_idf.loc['pos'].sort_values(ascending=False).head(20)
print(pos_top_words)
neg_top_words = df_idf.loc['neg'].sort_values(ascending=False).head(20)
print(neg_top_words)

#Seems like it worked as expected, words like good and great have the highest values in pos and we see some of the same general words having similar values in the both pos and neg, like fit and product.
# %% [markdown]
# 4. Plot the words in each class with their corresponding TF/IDF scores. Note that there will be a lot of words, so youâ€™ll have to think carefully to make your chart clear! If you canâ€™t plot them all, plot a subset â€“ think about how you should choose this subset.
# 
#     Hint: you can use word clouds. But feel free to challenge yourselves to think of any other meaningful way to visualize this information!

# %%
# Use this cell for your code

df_plot_values = df_idf[list(set(pos_top_words.index).union(set(neg_top_words.index)))]
plt.figure(figsize=(10,10))
sns.heatmap(df_plot_values.T, annot=True, cmap="YlGnBu", xticklabels=True, yticklabels=True)
plt.xticks(rotation=45, ha="center")
plt.tight_layout()
plt.show()




# %% [markdown]
# Remember to submit your code on the MOOC platform. You can return this Jupyter notebook (.ipynb) or .py, .R, etc depending on your programming preferences.

# %% [markdown]
# ## Exercise 4 | Junk charts
# 
# Thereâ€™s a thriving community of chart enthusiasts who keep looking for statistical graphics that they find inappropriate, and which they call â€œjunk chartsâ€, and who often also propose ways to improve them.

# %% [markdown]
# 1. Find at least three statistical visualizations you think are not very good and identify their problems. Copying examples from various junk chart websites is not accepted â€“ you should find your own junk charts, out in the wild. You should be able to find good (or rather, bad) examples quite easily since a significant fraction of charts can have at least *some* issues. The examples you choose should also have different problems, e.g., try to avoid collecting three bar charts, all with problematic axes. Instead, try to find as interesting and diverse examples as you can.

# %% [markdown]
# 2. Try to produce improved versions of the charts you selected. The data is of course often not available, but perhaps you can try to extract it, at least approximately, from the chart. Or perhaps you can simulate data that looks similar enough to make the point.
# 
# 

# %% [markdown]
# Submit a PDF with all the charts (the ones you found and the ones you produced).
