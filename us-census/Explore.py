
# coding: utf-8

##### Setup: Python Imports

# In[1]:

import pandas as pd
import numpy as np
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.feature import StringIndexer, VectorAssembler, OneHotEncoder
from pyspark.sql.functions import lit

get_ipython().magic(u'matplotlib inline')


## Business Understanding

# In the context of an (supposedly quick) interview exercise to demonstrate some critical skills as a Data Scientist, the goal of this exercise is not to create the best or the purest model, but rather to describe the steps we’ll take to accomplish it and explain the places that may have been the most challenging.
# 
# Let's find clear insights on the profiles of the people that make more than $50,000 / year. 
# For example, which variables seem to be the most correlated with this phenomenon?

## Data Understanding

##### Data loading

# * Import the learning and test files

# In[2]:

# the meta data 
columnNames = [ w.replace(' ', '_') for w in [
    "age",
    "class of worker",
    "industry code",
    "occupation code",
    #"adjusted gross income",
    "education",
    "wage per hour",
    "enrolled in edu inst last wk",
    "marital status",
    "major industry code",
    "major occupation code",
    "race",
    "hispanic Origin",
    "sex",
    "member of a labor union",
    "reason for unemployment",
    "full or part time employment stat",
    "capital gains",
    "capital losses",
    "divdends from stocks",
    #"federal income tax liability",
    "tax filer status",
    "region of previous residence",
    "state of previous residence",
    "detailed household and family stat",
    "detailed household summary in household",
    "instance weight",
    "migration code-change in msa",
    "migration code-change in reg",
    "migration code-move within reg",
    "live in this house 1 year ago",
    "migration prev res in sunbelt",
    "num persons worked for employer",
    "family members under 18",
    #"total person earnings",
    "country of birth father",
    "country of birth mother",
    "country of birth self",
    "citizenship",
    "own business or self employed",
    "fill inc questionnaire for veteran's admin",
    "veterans benefits",
    "weeks worked in year",
    #"taxable income amount",
    "year",
    "total person income"
]]
targetColumn = "total_person_income"
lessThan50k = " - 50000."
print "Number of columns defined: " + str(len(columnNames))
train = pd.read_csv('us_census_full/census_income_learn.csv', header=None)
print "Number of columns in data frame: " + str(len(train.columns))
train.columns = columnNames
target = (train[targetColumn] != lessThan50k).astype(int)

train.head(1)


##### Simple Data audit

# * Make a quick statistic based and univariate audit of the different columns’ content and produce the results in visual / graphic format.
# * This audit should describe the variable distribution, the % of missing values, the extreme values, and so on.

# In[3]:

def describe(df):
    # FIXME: add better legends
    for i in df.columns:
        print "***************************************"
        print "Describing Column " + str(i) + ":" 
        numValues = pd.Series(df[i].values.ravel()).unique().size
        if df[i].dtype !=  np.object and numValues > 5:
            auditNumericalColumn(df, str(i))
        else:
            auditCategoricalColumn(df, str(i))

def auditNumericalColumn(df, columnName):
    col = df[columnName]
    print col.describe()
    fig = plt.figure()
    if col.dtype == np.int64:
        num_bins = pd.Series(col.values.ravel()).unique().size
    else:
        num_bins = 100
    data = [
        df[df[targetColumn] != lessThan50k][columnName].values,
        df[df[targetColumn] == lessThan50k][columnName].values
    ]
    f, axarr = plt.subplots(2, sharex=True)
    axarr[0].boxplot(data, vert = False, showmeans = True, whis = [2, 99])
    axarr[1].hist(data, num_bins, color = ['g','r'], stacked=True, alpha = 0.6)
    plt.title("Values Histogram of column:" + columnName)
    plt.xlabel("Values")
    plt.ylabel("count") 
    plt.show()
    
def auditCategoricalColumn(df, columnName):
    col = df[columnName]
    total = col.count()
    counts = pd.DataFrame(col.value_counts(), columns=["count"])
    counts["%"] = counts["count"] * 100.0 / total
    print counts
    col.value_counts().plot(kind="bar")
    plt.show()


# In[4]:

describe(train)


##### Audit the test data too

# In[5]:

test = pd.read_csv('us_census_full/census_income_test.csv', header=None, names=columnNames)


# Let's check the <a href="http://ucanalytics.com/blogs/population-stability-index-psi-banking-case-study/">Population Stability Index</a> (PSI) to see if the test data is in a similar context than the training data or if we need to ignore some features because they happened to be too different in the test dataset.
# 

# In[6]:

epsilon = 0.005

def computePSI(actual, expected):
    a = epsilon
    act = actual.value_counts()
    totalAct = 1.0 * act.values.sum()
    exp = expected.value_counts();
    totalExp = 1.0 * exp.values.sum()
    for actKey in act.keys():
        if actKey not in exp.keys():
            exp = exp.append(pd.Series([a], index=[actKey]))
    for expKey in exp.keys():
        if expKey not in act.keys():
            act = act.append(pd.Series([a], index=[expKey]))
    act = act.values / totalAct
    exp = exp.values / totalExp
    try:
        PSI = (act - exp) * np.log(act / exp)
    except:
        print act, exp
    return PSI.sum()

for var_name in columnNames:
    if train[var_name].dtype != np.float64:
        PSI = computePSI(train[var_name], test[var_name])
        if PSI > epsilon:
            print var_name + " has a PSI of " + str(PSI)


# All the PSI measures for each features seems to be quite low so we don't need to worry about it in this exercise but this could be used as a test on production data to detect any shift in the day-to-day data that are fed to the model once it is deployed.
# 
# This kind of verification would ensure that we would get alerted if there is a sudden change in the behavior of the population (which is not the context expected for this model to work properly)

##### Visualize without missing values

# In[7]:

missingValues = [
    " Not in universe",
    " ?"
]
def describeWithoutNull(df):
    for i in df.columns:
        numValues = pd.Series(df[i].values.ravel()).unique().size
        if df[i].dtype ==  np.object or numValues <= 5:
            print "***************************************"
            print "Describing Column " + str(i) + ":" 
            auditCategoricalColumn(df[~df[i].isin(missingValues)], str(i))


# In[8]:

describeWithoutNull(train)


##### Study missing values

# Rows with missing values shouldn't be ignored completely but a deeper analysis of why it is happening and looking for any obvious pattern could be critical to better understand the data (or the way the data was acquired). 

# In[9]:

# FIXME


## Data Preparation

##### Convert to Spark DataFrame

# In[10]:

trainDF = sqlContext.createDataFrame(train)
testDF = sqlContext.createDataFrame(test)


##### Encode categorical features

# In[11]:

# assemble multiple columns into a single vector column
def prepareData(df):
    vectorAssembler = VectorAssembler(inputCols = [ w.replace(' ', '_') for w in [
        "age",
        "class of worker",
        "industry code",
        "occupation code",
        #"adjusted gross income",
        "education",
        "wage per hour",
        "enrolled in edu inst last wk",
        "marital status",
        "major industry code",
        "major occupation code",
        "race",
        "hispanic Origin",
        "sex",
        "member of a labor union",
        "reason for unemployment",
        "full or part time employment stat",
        "capital gains",
        "capital losses",
        "divdends from stocks",
        #"federal income tax liability",
        "tax filer status",
        "region of previous residence",
        "state of previous residence",
        "detailed household and family stat",
        "detailed household summary in household",
        #"instance weight",
        "migration code-change in msa",
        "migration code-change in reg",
        "migration code-move within reg",
        "live in this house 1 year ago",
        "migration prev res in sunbelt",
        "num persons worked for employer",
        "family members under 18",
        #"total person earnings",
        "country of birth father",
        "country of birth mother",
        "country of birth self",
        "citizenship",
        "own business or self employed",
        "fill inc questionnaire for veteran's admin",
        "veterans benefits",
        "weeks worked in year",
        #"taxable income amount",
        "year"
    ]], outputCol = "features")
    return vectorAssembler.transform(df)
    
# for each string feature, build a string indexer model
def modelEncoders(df):
    models = {}
    selectedFeatures = []
    for i in df.columns:
        if df.select(i).dtypes[0][1] == 'string':
            stringIndexer = StringIndexer(inputCol = i, 
                                          outputCol = "str_" + i)
            models[i] = stringIndexer.fit(df)
    return models

# for each string feature, transform it into double index using the string models
def encodeStringFeatures(df, models):
    selectedFeatures = []
    for i in df.columns:
        if df.select(i).dtypes[0][1] == 'string':
            model = models[i]
            indexed = model.transform(df)
            if i == targetColumn:
                df = indexed.drop(i).withColumnRenamed("str_" + i, i)
            else:
                encoder = OneHotEncoder(dropLast=False, inputCol="str_" + i, outputCol=i)
                encoded = encoder.transform(indexed.drop(i))
                df = encoded
            selectedFeatures.append(i)
        else:
            selectedFeatures.append(i)
    return df.select(selectedFeatures)

stringModels = modelEncoders(trainDF)
trainData = prepareData(encodeStringFeatures(trainDF, stringModels))
testData = prepareData(encodeStringFeatures(testDF, stringModels))
print trainData.select(targetColumn, "features").head()


# I am using <a href="https://spark.apache.org/docs/latest/api/python/pyspark.ml.html#pyspark.ml.feature.OneHotEncoder">OneHotEncoder</a> which creates a new boolean column for each category value but I should investigate in the future the use of VectorIndexer instead as it seems to be more adequate to our use case and might improve performances: <a href="https://spark.apache.org/docs/latest/ml-features.html#vectorindexer">see documentation here</a> for handling categorical features
# 
# However, since it would be a <a href="https://en.wikipedia.org/wiki/Lean_software_development#Eliminate_waste">waste</a> (in terms of lean development) to work on this aspect for now, I am leaving it out (ie result could still be achieved without it)

## Modeling

# * Create a model using these variables (you can use whichever variables you want, or even create you own; for example, you could find the ratio or relationship between different variables, the binarisation of “categorical” variables, etc.) to modelize earning more or less than $50,000 / year. Here, the idea would be for you to test one or two algorithms, type regression logistics, or a decision tree. But, you are free to choose others if you’d rather.

##### Random Forest model

# In[12]:

def trainRandomForest(df):
    rf = RandomForestClassifier(labelCol=targetColumn, 
                                featuresCol="features", 
                                rawPredictionCol="rawPrediction",
                                probabilityCol="probability",
                                numTrees=50)
    model = rf.fit(df)
    return model


# * Choose the model that appears to have the highest performance based on a comparison between reality (the 42nd variable) and the model’s prediction.

# From <a href="https://spark.apache.org/docs/latest/mllib-evaluation-metrics.html">MLLib documentation</a>:
# ## Classification model evaluation
# While there are many different types of classification algorithms, the evaluation of classification models all share similar principles. In a supervised classification problem, there exists a true output and a model-generated predicted output for each data point. For this reason, the results for each data point can be assigned to one of four categories:
# 
# * True Positive (TP) - label is positive and prediction is also positive
# * True Negative (TN) - label is negative and prediction is also negative
# * False Positive (FP) - label is negative but prediction is positive
# * False Negative (FN) - label is positive but prediction is negative
# 
# These four numbers are the building blocks for most classifier evaluation metrics. A fundamental point when considering classifier evaluation is that pure accuracy (i.e. was the prediction correct or incorrect) is not generally a good metric. The reason for this is because a dataset may be highly unbalanced. For example, if a model is designed to predict fraud from a dataset where 95% of the data points are not fraud and 5% of the data points are fraud, then a naive classifier that predicts not fraud, regardless of input, will be 95% accurate. For this reason, metrics like precision and recall are typically used because they take into account the type of error. In most applications there is some desired balance between precision and recall, which can be captured by combining the two into a single metric, called the F-measure.
# 
# Here is a list of some possible measurements (from Dataiku's DSS):
# * Accuracy Score: Proportion of correct predictions (positive and negative) in the sample
# * Average Precision Score: Average precision for all classes
# * F1 Score: Harmonic mean of Precision and Recall
# * Hamming Loss: The Hamming loss is the fraction of labels that are incorrectly predicted. (The lower the better)
# * Log Loss: Error metric that takes into account the predicted probabilities
# * Matthews Correlation Coefficient: The MCC is a correlation coefficient between actual and predicted classifications; +1 is perfect, -1 means no correlation
# * Precision Score: Proportion of correct 'positive' predictions in the sample
# * ROC - AUC Score: From 0.5 (random model) to 1 (perfect model).
# * Recall Score

# In[13]:

def measurePerformance(data, model = None):
    print
    print "Perfect Model"
    data.crosstab(targetColumn, targetColumn).show()

    print "Random Forest"
    if model == None:
        model = trainRandomForest(data)
    predictions = model.transform(data)
    confusionMat = predictions.crosstab(targetColumn, "prediction")
    if "1.0" in confusionMat.columns:
        confusionMat.show()
    else:
        confusionMat.withColumn("1.0", lit(0)).show()
        
    evaluator = BinaryClassificationEvaluator(rawPredictionCol="rawPrediction", labelCol=targetColumn)
    print "Area Under ROC Curve: " + str(evaluator.evaluate(predictions))
    print "Area Under Precision-Recall Curve: " + str(evaluator.evaluate(predictions, {evaluator.metricName: "areaUnderPR"}))
        
    print "Precision:", evaluatePrecision(confusionMat)
    return model

def evaluatePrecision(confusionMat):
    tp, fp, tn, fn = getConfusionMatrixCells(confusionMat)
    if (tp + fp) != 0:
        return tp * 1.0 / (tp + fp)
    else:
        return 0.0
        
def getConfusionMatrixCells(confusionMat):
    if "1.0" in confusionMat.columns:
        #True Positive (TP) - label is positive and prediction is also positive
        tp = confusionMat.where(confusionMat[targetColumn + "_prediction"] == "1.0").collect()[0]['1.0']
        #False Positive (FP) - label is negative but prediction is positive
        fp = confusionMat.where(confusionMat[targetColumn + "_prediction"] == "0.0").collect()[0]['1.0'] 
    else:
        tp = 0.0
        fp = 0.0
    if "0.0" in confusionMat.columns:
        #True Negative (TN) - label is negative and prediction is also negative
        tn = confusionMat.where(confusionMat[targetColumn + "_prediction"] == "0.0").collect()[0]['0.0']  
        #False Negative (FN) - label is positive but prediction is negative
        fn = confusionMat.where(confusionMat[targetColumn + "_prediction"] == "1.0").collect()[0]['0.0'] 
    else:
        tn = 0.0
        fn = 0.0
    return tp, fp, tn, fn


# In[14]:

print "Always returning 0"
trainData.withColumn("always0", lit(0)).crosstab(targetColumn, "always0").withColumn("1.0", lit(0)).show()

#print "Always returning 1"
#trainData.withColumn("always1", lit(1)).crosstab(targetColumn, "always1").withColumn("0.0", lit(0)).show()

firstModel = measurePerformance(trainData)


# Obviously, we see that something terribly wrong is happening as our very complicated piece of machine learning, Random Forest, is making exactly the same predictions as returning the same label (0.0) all the time. This is caused by the very unbalanced dataset as we have seen on previous charts...
# 
# However, I wanted to make sure that i would be able to produce some models as fast as possible, regardless of the accuracy results at first. Now there is a full pipeline in place, we should be able to start the creative work and iterate as fast as possible while trying different methods and compare results with previous runs and slowly improve our predictions in a more efficient manner.
# 
# So, let's now follow the "standard" Data Mining process (CRISP-DM) and go back to improve our predictions:
# 
# <img src="https://upload.wikimedia.org/wikipedia/commons/b/b9/CRISP-DM_Process_Diagram.png", width=512>

## Data Preparation

##### Stratified Sampling: Down sample

# Since the dataset is very unbalanced, the easiest (and fastest solution to implement) to this issue is to throw away data by "down sampling" the rows with the majority class (warning: this might not be the best solution!)

# In[15]:

def downSample(df, factor = 1.0):
    targetCount = (df
       .groupBy(targetColumn).count()
       .sort("count", ascending= True)
       .map(lambda x: (x[targetColumn], x["count"]))
       .collect())
    fractions = { 
        targetCount[0][0] : 1.0,
        targetCount[1][0] : (targetCount[0][1] * 1.0 / targetCount[1][1]) * factor
    }
    return df.sampleBy(targetColumn, fractions)

sampleTrainData = downSample(trainData)
sampleTrainData.groupBy(targetColumn).count().show()


## Modeling

# In[16]:

secondModel = measurePerformance(sampleTrainData)


# The Random Forest model is now properly making more predictions instead of classifying everything in the same category all the time...

##### Cross-Validation

# To avoid overfitting and have a more objective view of our model performances, we can implement a <a href="https://en.wikipedia.org/wiki/Cross-validation_(statistics)#k-fold_cross-validation">cross-validation</a> where we split our training data into multiple training and evaluation partitions.

# In[17]:

# FIXME


### Evaluation

# * Apply your model to the test file and measure its real performance on it (same method as above).

# In[18]:

measurePerformance(testData, firstModel)
measurePerformance(testData, secondModel)


# Fine-tune the evaluation function to reflect some imaginary business cost of misclassifications instead...
# 
# This is only an example to show that we should adapt this evaluation to the business use case. It is also an effective way to communicate the results back to non-technical audience.

# In[19]:

def measureBusinessPerformance(data, model = None):
    if model == None:
        model = trainRandomForest(data)
    predictions = model.transform(data)
    confusionMat = predictions.crosstab(targetColumn, "prediction")
    if "1.0" in confusionMat.columns:
        confusionMat.show()
    else:
        confusionMat.withColumn("1.0", lit(0)).show()

    tp, fp, tn, fn = getConfusionMatrixCells(confusionMat)
    print "We offered a $50 gift to", int(tp), "customers that might be interested in a luxury saving account, avg ROI is $100"
    print "We offered a $50 gift to", int(fp), "customers that were not interested and declined the offer"
    print int(tn + fn), "customers were not contacted and they never purchased our banking services"
    print "Total marketing investment was: $" + str((tp + fp) * 50)
    print "ROI: $" + str( (tp * 100) + 
                         (fp * -50) +
                         (tn *   0) + 
                         (fn *   0) )
    return model

measureBusinessPerformance(testData, firstModel)
measureBusinessPerformance(testData, secondModel)


# If we were to roll out that model, we would lose money!
# 
# Let's try to learn more about people not having more than $50000 so we make less costly mistakes...
# 
# We can do this by rebalancing the dataset but give more importance to the label 0.0 by providing 3 times more data.
# (note that we shouldn't use the testData to figure what the right parameters are but use the cross validation of train data instead)

# In[20]:

sample2TrainData = downSample(trainData, 3.0)
sample2TrainData.groupBy(targetColumn).count().show()
thirdModel = measurePerformance(sample2TrainData)
measurePerformance(testData, thirdModel)


# In[21]:

measureBusinessPerformance(testData, thirdModel)


# Of course, this is far from being the best model (I even picked some parameters randomly) but with more time to work on this exercise, we can iterate on and on through the CRISP-DM process to bring the evaluation function up to a satisfactory point where we can decide to deploy it into some "production" environment.
# 
# My point here is that I wanted to show a base pipeline that is quickly built in few hours of work (in this context of interview exercise) rather than a "perfect" solution handling all the data aspects in order to get the best score out of this. This is best described in this article: http://www.techrepublic.com/article/start-tackling-data-science-inefficiencies-by-properly-defining-waste/
# 
# However here are some ideas of aspects to work on to improve even more the predictions:

##### Resampling: Up sampling (SMOTE?)

# Instead of throwing away data, we can synthetize or duplicate the rows from minority class instead to rebalance the dataset.

# In[22]:

# FIXME


##### Rescaling

# Not necessary for Random Forest but might be useful for other modeling algorithms

# In[23]:

from sklearn.preprocessing import scale

# FIXME


##### Binning

# Combine variables into bins based on <a href="http://www.listendata.com/2015/03/weight-of-evidence-woe-and-information.html">Weight Of Evidence</a> while increasing <a href="http://ucanalytics.com/blogs/information-value-and-weight-of-evidencebanking-case/">Information Value</a>
# 
# #### Grouping Categorical Nominal characteristic:
# 
# 1. For each Categorical Nominal variable, we start with one value per bin. So if we have N possible values, we have N bins.
# 2. We compute the Information Value.
# 3. Then we try every combinations of two bins and simulate a merge to recompute the new Information Value if they were grouped together.
# 4. We pick the pair of bins that provides the biggest change to Information Value between step 2 and 3 and we apply the merge
# 5. We reiterate with step 2 until we can't find any pairs that increase the Information Value anymore.
# 
# #### Grouping Binary characteristic:
# 
# Binary characteristics contains by definition only 2 potential values, therefore, no further grouping is possible.
# 
# #### Grouping Categorical Ordinal and Numerical Continuous characteristics:
# 
# 0. We start by transforming each value into an interval tuple with a lower and upper bound
# 1. For each Categorical Ordinal variable, we start with one value per bin. So if we have N possible values, we have N bins.
# 2. We compute the Information Value.
# 3. Then we try every combinations of two adjacent bins (the bin at index i and index i + 1) and simulate a merge to recompute the new Information Value if they were grouped together so that we produce a range of values. Grouping two bins results into taking the lower bound of the first bin and the upper bound of the second bin to produce the merged bin.
# 4. We pick the pair of bins that provides the biggest change to Information Value between step 2 and 3 and we apply the merge
# 5. We reiterate with step 2 until we can't find any pairs that increase the Information Value anymore.
# 
# Note that with ordinal value, we make sure to merge only adjacent values. we are simply growing the intervals when possible without skipping values.

# In[24]:

# FIXME


##### Parameter Optimization - fine tuning

# In[25]:

# FIXME


##### Feature engineering

# <i>Coming up with features is difficult, time-consuming, requires expert knowledge. "Applied machine learning" is basically feature engineering.</i>
# 
# — Andrew Ng

# In[26]:

# FIXME

