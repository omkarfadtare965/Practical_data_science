# Practical data science Specialization:
- In this course, you will learn how to take your data science project from idea to production. You'll discover how to build, deploy, and scale data science projects, serving thousands of models to millions of end-users. You will learn to analyze and clean datasets, extract relevant features, train models, and construct automated pipelines to orchestrate and scale your data science projects. You will also learn to tackle complex data science challenges with sophisticated tools.
- One of the biggest benefits of running data science projects in the cloud is the agility and elasticity it offers, allowing you to scale and process virtually any amount of data.

![image](https://github.com/omkarfadtare/Practical_data_science/assets/154773580/cc5d2bf9-5e51-4d90-928d-6da501560da6)

__Course [1]:__
- In course1, you will learn how to ingest the data into a central repository and explore the data using various tools, explore and analyze the dataset using interactive queries and learn how to visualize the results.
- You will perform exploratory data analysis and detect statistical data biases.
- Next, you will learn to train machine learning models using automated machine learning techniques and build a multi-class text classification model using state-of-the-art algorithms.
- you will understand how to describe the concept of statistical bias, and use metrics to measure imbalances in data sets.
- You will understand how to detect statistical bias in your data and generate bias reports. You will further explore how to generate feature importance reports.

__Course [2]:__
- In course2, you will dive deeper into building custom NLP models.
- You will construct a machine learning pipeline, perform feature engineering, and share your features with the organization using a scalable feature store.
- You will train, tune, and deploy your model, orchestrate the model workflow using ML pipelines and MLOps strategies.

__Course [3]:__
- In course3, you will optimize machine learning models and learn best practices for tuning hyperparameters and performing distributed model training.
- You will explore advanced model deployment and monitoring options.
- You will discover how to handle large-scale data labeling and build human-in-the-loop pipelines to enhance model accuracy and performance by combining machine intelligence with human intelligence.

__Prerequisites for this Specialization:__
- Proficiency in Python and SQL programming.
- Familiarity with building neural networks using deep learning Python frameworks like TensorFlow or PyTorch.
- Understanding the concept of building, training, and evaluating machine learning models.

__Brief introduction:__
- Artificial intelligence/AI, is generally described as a technique that lets machines mimic human behavior.
- Machine learning/ML, is a subset of AI, that uses statistical methods and algorithms that are able to learn from data, without being explicitly programmed.
- Deep learning is yet another subset of machine learning, that uses artificial neural networks to learn from data.
- Data science is an interdisciplinary field that combines business and domain knowledge with mathematics, statistics, data visualization, and programming skills.
- Practical data science helps you to improve your data science and machine learning skills, work with almost any amount of data, and implement their use cases in the most efficient way.

![image](https://github.com/omkarfadtare/Practical_data_science/assets/154773580/54ea4e85-d127-4fc2-9f1c-4eadd5c6b69b)

- Popular machine learning tasks are classification and regression problems, which are examples of supervised learning.
- In supervised learning, you learn by providing the algorithm with labeled data. - In classification, the goal is to assign the input sample a defined class. For example, is this email I received spam or not spam?
- In contrast, regression applies statistical methods to predict a continuous value, such as a house price, given a set of related and non-related input variables.
- Another popular task is clustering, it is an example of unsupervised learning where the data is not labelled. The clustering algorithm tries to find patterns in the data and starts grouping the data points into distinct clusters. 
- Image processing is a major task of computer vision where you need to classify images into pictures of dogs and cats, distinguish between speed signs and trees.
- The field of Natural Language Processing (NLP), or Natural Language Understanding (NLU) includes machine translations, sentiment analysis, question answering, etc.

![image](https://github.com/omkarfadtare/Practical_data_science/assets/154773580/7ae4c166-5944-4539-aeac-ba763619bf55)

__Benifits of performing data science on cloud:__
- Practical data science is geared towards handling massive datasets, that could originate from social media channels, mobile and web applications, public or company internal data sources, and much more, depending on the use case you're working on. This data is often messy, potentially error-ridden, or even poorly documented.
- Practical data science tackles these issues by providing tools to analyze and clean the data and to extract relevant features and leads to knowledge distillation and gaining insight from those large datasets.
- Developing and running data science projects in the Cloud is the agility and elasticity that the Cloud offers. If you develop data science projects in a local notebook or IDE environment, for example, hosted on your laptop or company-owned server pool, you are limited by the existing hardware resources. You have to carefully watch how much data you process, how much CPU processing power you have available to train and tune your model. And if you need more, you need to buy additional computer resources. This process doesn't allow you to move and develop quickly. Maybe your model training takes too long because it consumes all of the CPU resources of the compute instance you have chosen.
- Using practical data science, you can switch to using a compute instance that has more CPU resources, or even switch up to a GPU-based compute instance.
- Cloud allows training your model on single CPU instance __(scaling up)__, performing distributed model training in parallel across various compute instances __(scaling out)__.

![image](https://github.com/omkarfadtare/Practical_data_science/assets/154773580/16fcd327-e77f-4b16-9de1-4b9356a53b41)

- On cloud when the model training is completed, the instances are terminated as well. This means you only pay for what you actually use.
- CLoud allows you to store and process almost any amount of data.
- Cloud also comes with a large data science and machine learning toolbox you can choose from, to perform your tasks as fast and efficiently as possible.

__Data science and machine learning toolbox:__

![image](https://github.com/omkarfadtare/Practical_data_science/assets/154773580/8a5def2a-5426-4587-a335-0f453dfc15de)

## Use case: Multi-class Classification for Sentiment analysis of Product reviews
- Assume you work at an e-commerce company, selling many different products online. Your customers are leaving product feedback across all the online channels. Whether it is through sending email, writing chat FAQ messages on your website, maybe calling into your support center, or posting messages on your company's mobile app, popular social networks, or partner websites. And as a business, you want to be able to capture this customer feedback as quickly as possible to spot any change in market trends or customer behavior and then be alerted about potential product issues.
- Your task is to build an NLP model that will take those product reviews as input. You will then use the model to classify the sentiment of the reviews into the three classes of positive, neutral, and negative.
- Multi-class classification is a supervised learning task, hence you need to provide your tax classifier model with examples how to correctly learn to classify the products and the product reviews into the right sentiment classes. 
- You can use the review text as the input feature for the model training and the sentiment as a label for model training. The sentiment class is usually expressed as an integer value for model training such as 1 for positive sentiment, 0 for neutral sentiment, and -1 for negative sentiment.

### Data ingestion and Exploration:
- Imagine your e-commerce company is collecting all the customer feedback across all online channels. You need to capture, suddenly, customer feedback streaming from social media channels, feedback captured and transcribed through support center calls, incoming emails, mobile apps, and website data, and much more.
- To do that, you need a flexible and elastic repository that can store, not only the different file formats, such as dealing with structured data, CSV files, as well as unstructured data, such as support center call audio files. It also needs to elastically scale the storage capacity as new data arrives.
- Cloud-based data lakes address this problem. __Data lake__ is a centralized and secure repository that can store, discover, and share virtually any amount and any type of your data.
- You can ingest data in its raw format without any prior data transformation. Whether it's structured relational data in the form of CSV or TSV files, semi-structured data such as JSON or XML files, or unstructured data such as images, audio, and media files.
- You can also ingest streaming data, such as an application delivering a continuous feed of log files, or feeds from social media channels, into your data lake.
- A data lake needs to be governed. With new data arriving at any point in time you need to implement ways to discover and catalog the new data. A data lake needs to be governed. With new data arriving at any point in time you need to implement ways to discover and catalog the new data.
- Data lakes are often built on top of object storage, such as Amazon S3. __File storage__ stores and manages data as individual files organized in hierarchical file folder structures. In contrast __Block storage__ stores and manages data as individual chunks called the blocks. Each block receives a unique identifier, but no additional metadata is stored with that block. __Object storage__ stores data with its metadata  such as when the object was last modified, and a unique identifier. Object storage is particularly helpful for storing and retrieving growing amounts of data of any type.
- __Amazon S3__ gives you access to durable and high-available object storage in the cloud. You can ingest virtually anything, from just a few dataset files, to exabytes of data. AWS also provides additional tools and services to assist you in building a secure, compliant, and auditable data lake on top of S3. with this you can now use this centralized data repository to enable data warehousing analytics and also machine learning.
- __AWS Data wrangler__ is an open-source Python library. The library connects Pandas DataFrame with AWS data-related services. AWS Data Wrangler offers abstracted functions to load or unload data from data lakes, data warehouses, or databases on AWS. You can install the library through the PIP install AWS wrangler command.
> To read csv data from S3 data lake run below commands:
```ruby
!pip install awswrangler

import awswrangler as wr
import pandas as pd

df = wr.s3.read_csv(path = 's3://bucket/prefix/')
```
- __AWS Glue Data catalog__ is used to register or catalog the data stored in S3. It's like inventory to know what data you have stored in S3 date lake or bucket. Using the Data Catalog Service, you create a reference to the data, basically S3 to table mapping. The AWS Glue table, which is created inside an AWS Glue database, only contains the metadata information such as the data schema. Catalog is used to simplify where to find the data and which schema should be used, to query the data.
- Instead of manually registering the data, you can also use __AWS Glue Crawler__. A Crawler can be used and set up to run on a schedule or to automatically find new data, which includes inferring the data schema and also to update the data catalog.
- To register the data you can use AWS Data Wrangler tool. Follow the below steps:
> 1) Create a database in the AWS Glue data catalog database using below command:
```ruby
import awswrangler as wr

wr.catalog.create_database(name = name_for_the_database)
```
> 2) Create CSV table (metadataonly) in the AWS glue data catalog using below command:
```ruby
wr.catalog.create_csv_table(table = name_of_the_database, column_types = ..., )
```
- __Amazon Athena__ is used to query the data stored in S3. Athena is an interactive query service that lets you run standard SQL queries to explore your data. Athena is serverless, which means you don't need to set up any infrastructure to run those queries. No matter how large the data is that you want to query, you can simply type your SQL query, referencing the dataset schema you provided in the AWS Glue Data Catalog.
- To run the query follow below steps from the python environment you are using:
> 1) Create amazon athena S3 bucket using below commands/query:
```ruby
import awswrangler as wr

wr.athena.create_athena_bucket()
```
> 2) Execute SQL query on amazon athena
```ruby
df = wr.athena.read_sql_query(sql = 'sql_query', database = name_of_the_database)
```
- Athena then runs the query on the specified dataset and stores the results in S3, and it also returns the results in a Pandas DataFrame.
- Athena will automatically scale out and split the query into simpler queries to run in parallel against your data when building highly complex analytical queries to run against not just gigabytes, or terabytes, or petabytes of data. Because athena is based on Presto, an open source distributed SQL engine, developed for this exact use case.

### Data Visualization:
- Depending on what kind of data you are exploring and what kind of relationships in the data you're looking for, the type of visualizations you use might be different.
- Pandas an open source library, is used for data analysis and data manipulation. - NumPy an open source library, is used to perform scientific computing in Python.
- Matplotlib helps to create static animated and interactive visualizations.
- Seaborn is based on matplotlib, and adds statistical data visualizations.

### Statistical bias and Feature importance:
- Statistical bias and Feature importance allow you to gain a better understanding of your data and a better understanding of the quality of your data.
- These concepts also allow you to explore how the individual features of your datasets contribute to the final model.
- A data set is considered to be biased if it cannot completely and accurately represent the underlying problem space. Statistical bias is a tendency of a statistic to either overestimate or underestimate a parameter.
- For example, a dataset where fraudulent credit card transactions are rare, can lead to ineffective fraud detection models as they are unlikely to recognize fraudulent transactions due to lack of exposure to them.
- Another example is, let's say, we have products review data set in which one product category A has large number of reviews and fewer number of reviews for other product category B and C.
- When you build a product sentiment prediction model with this biased data set, the resulting model could very well detect sentiment of new products that belong to product category A. But for newer products that belong to other product category B and C, your sentiment model is not going to be really accurate.
- One way to address this problem is to add more examples of fraud transactions to your training dataset.

__Types of Bias:__
> 1) Activity bias:
- Activity bias occurs when certain groups or individuals are overrepresented or underrepresented in the data due to their level of engagement or activity.
- In an online shopping dataset, frequent users may have more data recorded about their preferences and behaviors compared to occasional users, leading to biased predictions.
> 2) Societal bias:
- Societal bias reflects existing societal inequalities and prejudices that are reflected in the data, leading to unfair treatment of certain groups.
- Historical biases against certain demographics (e.g., race, gender) may be perpetuated in datasets, resulting in biased decisions in areas like hiring or lending.
> 3) Selection bias:
- Selection bias occurs when the data collection process systematically favors certain samples over others, leading to an unrepresentative dataset.
- A survey conducted only among tech-savvy individuals may not accurately represent the opinions of the general population, leading to biased conclusions.

__Types of drift in Machine learning operations:__
> 1) Data drift:
- When the data used by a model changes over time, it's called data drift. This can make the model less accurate because it's not used to the new data.
> 2) Concept drift:
- If the relationship between things the model looks at and what it's trying to predict, changes, that's concept drift.
- For example, if you're predicting whether someone will buy a product based on their age, and suddenly younger people start buying more than older people, that's concept drift.
> 3) Covariate drift:
- When the characteristics the model uses to make predictions change, it's covariate drift.
- Let's say you're trying to predict how much ice cream people will buy based on the temperature. If suddenly people start buying more ice cream on colder days instead of hotter ones, that's covariate drift.
> 4) Prior probability drift:
- The shift in how often each outcome happens is called prior probability drift.
- Imagine you're flipping a coin, and at first, it comes up heads 70% of the time and tails 30% of the time. Your model learns from this and gets good at predicting based on that. But then, over time, the coin changes, and now it's heads only 50% of the time and tails 50% of the time.
> 5) Model drift: 
- Model drift is when a model that used to work well starts to become less accurate over time.
- This can happen if the data the model was trained on changes, or if the world changes in a way that the model didn't expect. 
> 6) Population drift:
- Let's say you're building a model to predict what kind of movies people will like, and you train it on data from one country. If you then try to use that model in a different country where people have different tastes, that's population drift.
- It's like the group of people you're trying to predict for has changed, making your model less accurate because it's not used to the new population.
> 7) Label drift:
- Label drift is when the answers or labels you have for your data change over time.
- It's like if you were trying to label pictures of cats and dogs, but then someone changed their mind about what a cat looks like. So, the labels for the pictures change, making it harder for your model to learn from them because the right answers keep changing.

__Measuring statistical bias:__
- Class imbalance, or CI, measures the imbalance in the number of examples that are provided, for different facet values in your dataset.
- A facet is a sensitive feature in your dataset, that you want to analyze for these imbalances.
- When you apply this to the product review dataset, it answers this particular question, does a particular product category, such as product Category A, have disproportionately large number of total reviews than any other category in the dataset?
- Difference in Proportions of Labels(DPL) metric measures the imbalance of positive outcomes between the different facet values. When applied to the product review dataset, what this metric is measuring is if a particular product category, say product Category A, has disproportionately higher ratings than other categories.
- CI, the metric that we just saw as measuring if a particular category has a total number of reviews higher than any other categories, DPL is actually looking for higher ratings than any other product categories.

__AWS tools to detect statistical bias in dataset:__
- The two tools are SageMaker Data Wrangler and SageMaker Clarify.
- Data Wrangler provides you with capabilities to connect to various different sources for your data, visualize the data, and transform the data, by applying any number of transformations in the Data Wrangler environment, and, detect statistical bias in your data sets, and generate reports about the bias detected in those data sets. It also provides capabilities to provide feature importance calculations on your training data set.
- Amazon SageMaker Clarify can perform statistical bias detection and generate bias reports on your training datasets. Additionally, it can also perform bias detection in trained and deployed models. It further provides capabilities for machine learning explainability, as well as detecting drift in data and models.
- To start using Clarify APIs, start by importing the Clarify library from the SageMaker SDK
```ruby
from sagemaker import clarify

clarify_processor = clarify.SageMakerClarifyProcessor(role=role, instance_count=1, instance_type="ml.c5.2xlarge", sagemaker_session=sess)

bias_report_output_path = << Define S3 Path >>
```
- instance_count=1 >>Distributed cluster size,
- instance_type="ml.c5.2xlarge" >> type of each instance
- bias_report_output_path = << Define S3 Path >> >> S3 location to store bias report
