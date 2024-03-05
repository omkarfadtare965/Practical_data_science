# Practical data science on AWS cloud:
- In this course, you will learn how to take your data science project from idea to production. You'll discover how to build, deploy, and scale data science projects, serving thousands of models to millions of end-users. You will learn to analyze and clean datasets, extract relevant features, train models, and construct automated pipelines to orchestrate and scale your data science projects. You will also learn to tackle complex data science challenges with sophisticated tools.
- One of the biggest benefits of running data science projects in the cloud is the agility and elasticity it offers, allowing you to scale and process virtually any amount of data.
> Course [1]:
- In course1, you will learn how to ingest the data into a central repository and explore the data using various tools, explore and analyze the dataset using interactive queries and learn how to visualize the results.
- You will perform exploratory data analysis and detect statistical data biases.
- Next, you will learn to train machine learning models using automated machine learning techniques and build a multi-class text classification model using state-of-the-art algorithms.
- You will understand how to describe the concept of statistical bias, and use metrics to measure imbalances in data sets.
- You will understand how to detect statistical bias in your data and generate bias reports. You will further explore how to generate feature importance reports.
> Course [2]:
- In course2, you will dive deeper into building custom NLP models.
- You will construct a machine learning pipeline, perform feature engineering, and share your features with the organization using a scalable feature store.
- You will train, tune, and deploy your model, orchestrate the model workflow using ML pipelines and MLOps strategies.
> Course [3]:
- In course3, you will optimize machine learning models and learn best practices for tuning hyperparameters and performing distributed model training.
- You will explore advanced model deployment and monitoring options.
- You will discover how to handle large-scale data labeling and build human-in-the-loop pipelines to enhance model accuracy and performance by combining machine intelligence with human intelligence.

__Prerequisites for this course:__
- Proficiency in Python and SQL programming.
- Familiarity with building neural networks using deep learning Python frameworks like TensorFlow or PyTorch.
- Understanding the concept of building, training, and evaluating machine learning models.

__Brief introduction:__
- Artificial intelligence/AI, is generally described as a technique that lets machines mimic human behavior.
- Machine learning/ML, is a subset of AI, that uses statistical methods and algorithms that are able to learn from data, without being explicitly programmed.
- Deep learning is yet another subset of machine learning, that uses artificial neural networks to learn from data.
- Data science is an interdisciplinary field that combines business and domain knowledge with mathematics, statistics, data visualization, and programming skills.
- This course helps you to improve your data science and machine learning skills, work with almost any amount of data, and implement their use cases in the most efficient way.

![image](https://github.com/omkarfadtare/Practical_data_science/assets/154773580/54ea4e85-d127-4fc2-9f1c-4eadd5c6b69b)

- Popular machine learning tasks are classification and regression problems, which are examples of supervised learning.
- In supervised learning, you learn by providing the algorithm with labeled data.
- In classification, the goal is to assign the input sample a defined class. For example, is this email I received spam or not spam?
- In contrast, regression applies statistical methods to predict a continuous value, such as a house price, given a set of related and non-related input variables.
- Another popular task is clustering, it is an example of unsupervised learning where the data is not labelled. The clustering algorithm tries to find patterns in the data and starts grouping the data points into distinct clusters.
- Image processing is a major task of computer vision where you need to classify images into pictures of dogs and cats, distinguish between speed signs and trees.
- The field of Natural Language Processing (NLP), or Natural Language Understanding (NLU) includes machine translations, sentiment analysis, question answering, etc.

![image](https://github.com/omkarfadtare/Practical_data_science/assets/154773580/7ae4c166-5944-4539-aeac-ba763619bf55)

__Benifits of performing data science on cloud:__
- Practical data science is geared towards handling massive datasets, that could originate from social media channels, mobile and web applications, public or company internal data sources, and much more, depending on the use case you're working on. This data is often messy, potentially error-ridden, or even poorly documented.
- Practical data science tackles these issues by providing tools to analyze and clean the data and to extract relevant features and leads to knowledge distillation and gaining insight from those large datasets.
- Developing and running data science projects in the cloud is the agility and elasticity that the cloud offers. If you develop data science projects in a local notebook or IDE environment, for example, hosted on your laptop or company-owned server pool, you are limited by the existing hardware resources. You have to carefully watch how much data you process, how much CPU processing power you have available to train and tune your model. And if you need more, you need to buy additional computer resources. This process doesn't allow you to move and develop quickly. Maybe your model training takes too long because it consumes all of the CPU resources of the compute instance you have chosen.
- Using practical data science, you can switch to a compute instance that has more CPU resources, or even switch up to a GPU-based compute instance.
- Cloud allows training your model on single CPU instance __(scaling up)__, performing distributed model training in parallel across various compute instances __(scaling out)__.
- On cloud when the model training is completed, the instances are terminated as well. This means you only pay for what you actually use.
- CLoud allows you to store and process almost any amount of data. Cloud also comes with a large data science and machine learning toolbox you can choose from, to perform your tasks as fast and efficiently as possible.

![image](https://github.com/omkarfadtare/Practical_data_science/assets/154773580/5d4e0e9a-c2b5-4570-a99f-92ab3355dbec)

## Use case for this course: Multi-class Classification for Sentiment analysis of Product reviews
- __Problem statement:__ Assume you work at an e-commerce company, selling many different products online. Your customers are leaving product feedback across all the online channels. Whether it is through sending email, writing chat FAQ messages on your website, maybe calling into your support center, or posting messages on your company's mobile app, popular social networks, or partner websites. And as a business, you want to be able to capture this customer feedback as quickly as possible to spot any change in market trends or customer behavior and then be alerted about potential product issues.
- Your task is to build an NLP model that will take those product reviews as input. You will then use the model to classify the sentiment of the reviews into the three classes of positive, neutral, and negative.
- Multi-class classification is a supervised learning task, hence, you must furnish your classifier model with examples to correctly learn how to classify products and product reviews into the respective sentiment classes. 
- You can use the review text as the input feature for model training and assign sentiment labels to train the model. Sentiment classes are typically represented as integer values during model training, such as 1 for positive sentiment, 0 for neutral sentiment, and -1 for negative sentiment.

## Data ingestion and Exploration:
- Imagine your e-commerce company collecting customer feedback from various online channels, including social media, support center calls, emails, mobile apps, and website data, among others.
- To achieve this, you require a flexible and scalable repository capable of storing different file formats, including structured data like CSV files and unstructured data like support center call audio files. Additionally, it should dynamically scale storage capacity as new data streams in.
- Cloud-based data lakes address this problem. __Data lake__ is a centralized and secure repository that can store, discover, and share virtually any amount and any type of data.
- You can ingest data in its raw format without any prior data transformation. Whether it's structured relational data in the form of CSV or TSV files, semi-structured data such as JSON or XML files, or unstructured data such as images, audio, and media files.
- you can also ingest streaming data such as continuous log file feeds or social media channel feeds into your data lake.
- Effective governance is crucial for a data lake. With new data arriving continuously, it's essential to implement mechanisms for discovering and cataloging the incoming data.
- Data lakes are often built on top of object storage, such as Amazon S3. __File storage__ stores and manages data as individual files organized in hierarchical file folder structures. In contrast, __Block storage__ stores and manages data as individual chunks called the blocks. Each block receives a unique identifier, but no additional metadata is stored with that block. __Object storage__ stores data with its metadata  such as when the object was last modified, and a unique identifier, making it ideal for storing and retrieving large and diverse data sets.
- __Amazon S3__ provides access to durable and highly-available object storage in the cloud. It allows you to ingest virtually any amount of data, from a few dataset files to exabytes of data. AWS also offers additional tools and services to help you build a secure, compliant, and auditable data lake on top of S3. With this centralized data repository, you can enable data warehousing analytics and machine learning.
- __AWS Data wrangler__ is an open-source Python library, that connects Pandas DataFrame with AWS data-related services. It offers abstracted functions for loading or unloading data from data lakes, data warehouses, or databases on AWS.
> To install AWS Data Wrangler library execute the following command:
```ruby
!pip install awswrangler
```
> To read CSV data from an S3 data lake, execute the following commands:
```ruby
import awswrangler as wr
import pandas as pd

df = wr.s3.read_csv(path='s3://bucket/prefix/')
```
- __AWS Data catalog services__ are used to register or catalog the data stored in S3. It is like a list/inventory that keeps track of all the data stored in S3. It helps you to know what data you have in your S3 storage. 
- When you use the data catalog services, it's like establishing a reference between a file you have stored in Amazon S3 and a table. This table is managed by __AWS Glue__, and it lives in a special database created by AWS Glue. However, this table doesn't contain the actual data, it just holds metadata information such as the data schema.
- Instead of registering data manually, you can use __AWS Glue Crawler__. It's a tool that can automatically scan your data to find out what's there.
- It figures out how the data is structured and keeps your data catalog updated without you having to do it yourself.
- To register data you can use AWS Data Wrangler tool following below steps:
> Start by creating a database in the AWS Glue Data Catalog database using the command below:
```ruby
import awswrangler as wr

wr.catalog.create_database(name=name_for_the_database)
```
> Next, create a CSV table (metadata only) in the AWS Glue Data Catalog using the command below:
```ruby
wr.catalog.create_csv_table(database=name_of_the_database, table=name_of_the_table, column_types=...)
```
- __AWS Athena__ is a tool used to query the data stored in S3. It allows you to use standard SQL queries to explore your data interactively.
- Athena is serverless, meaning you don't have to set up any infrastructure to run your queries. Regardless of the data's size, you can simply write your SQL query, referencing the dataset schema provided in the AWS Glue Data Catalog.
- To execute a query, follow these steps from your Python environment:
> Begin by setting up Amazon Athena to access your data stored in S3.
```ruby
import awswrangler as wr

wr.athena.create_athena_bucket()
```
> Next, execute your SQL query on Amazon Athena:
```ruby
df = wr.athena.read_sql_query(sql='sql_query', database=name_of_the_database)
```
- Athena processes the query on the specified dataset, stores the results in S3, and returns them as a Pandas DataFrame.
- When running highly complex analytical queries against large volumes of data, Athena automatically scales out and divides the query into simpler ones to run in parallel. This capability is possible because Athena is built on Presto, an open-source distributed SQL engine designed for this purpose.

## Data visualization:
- The type of visualizations you use may vary depending on the type of data you're exploring and the relationships you're examining within the data.
- Pandas, an open-source library, is utilized for data analysis and manipulation.
- NumPy, another open-source library, facilitates scientific computing in Python.
- Matplotlib aids in creating static, animated, and interactive visualizations.
- Seaborn, built on top of matplotlib, enhances visualizations with statistical data analysis.

## Statistical bias and Feature importance:
- Statistical bias and feature importance help you gain a better understanding of your data quality and how individual features contribute to your model.
- These concepts also allow you to explore how the individual features of your datasets contribute to the final model.
- A dataset is biased if it fails to accurately represent the underlying problem space. Statistical bias refers to a statistic's tendency to either overestimate or underestimate a parameter.
- For instance, in a dataset where fraudulent credit card transactions are rare, fraud detection models may struggle to identify fraudulent transactions due to lack of exposure.
- Similarly, consider a product review dataset where one product category (A) has a large number of reviews compared to categories B and C. When building a sentiment prediction model using this biased dataset, the model may accurately predict sentiments for category A products but perform poorly for categories B and C.
- One solution to this issue is to augment the training dataset with more examples of fraudulent transactions.

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
- Data drift happens when the data used by a model changes over time, making the model less accurate
> 2) Concept drift:
- Concept drift occurs when the relationship between the model's input and output changes over time.
- For example, if you're predicting whether someone will buy a product based on their age, and suddenly younger people start buying more than older people, that's concept drift.
> 3) Covariate drift:
- Covariate drift happens when the characteristics used by the model to make predictions change over time.
- Let's say you're trying to predict how much ice cream people will buy based on the temperature. If suddenly people start buying more ice cream on colder days instead of hotter ones, that's covariate drift.
> 4) Prior probability drift:
- Prior probability drift is the shift in the frequency of each outcome over time.
- Imagine you're flipping a coin, and at first, it comes up heads 70% of the time and tails 30% of the time. Your model learns from this and gets good at predicting based on that. But then, over time, the coin changes, and now it's heads only 50% of the time and tails 50% of the time.
> 5) Model drift: 
- Model drift is when a model that used to perform well becomes less accurate over time.
- This can happen if the data the model was trained on changes, or if the world changes in a way that the model didn't expect. 
> 6) Population drift:
- Population drift occurs when the population the model is applied to changes over time, making the model less accurate.
- Let's say you're building a model to predict what kind of movies people will like, and you train it on data from one country. If you then try to use that model in a different country where people have different tastes, that's population drift.
> 7) Label drift:
- Label drift is when the answers or labels you have for your data change over time.
- It's like if you were trying to label pictures of cats and dogs, but then someone changed their mind about what a cat looks like. So, the labels for the pictures change, making it harder for your model to learn from them because the right answers keep changing.

__Measuring statistical bias:__
- Class imbalance, or CI, shows if there are more examples of one thing than another in your dataset.
- A facet is a sensitive feature in your dataset, that you want to analyze for these imbalances.
- For example, CI can tell us if one product category has a lot more reviews than others.
- The Difference in Proportions of Labels (DPL) metric checks if one group has more positive outcomes than others. When applied to the product review dataset, what this metric is measuring is if a particular product category, say product category A, has disproportionately higher ratings than other categories.
- So, while CI looks at overall reviews, DPL looks at whether some categories get higher ratings than others.
- For example, consider we have a dataset of customer reviews for an e-commerce platform. Each review is labeled as either "positive" or "negative" sentiment based on the customer's feedback.
- Class imbalance occurs if there are significantly more positive reviews than negative reviews, or vice versa.
- DPL would assess whether certain product categories receive a disproportionately higher proportion of positive reviews compared to others.

__AWS tools to detect statistical bias in dataset:__
- The two AWS tools used to detect statistical bias in datasets are SageMaker Data Wrangler and SageMaker Clarify.
- ata Wrangler allows you to connect to various data sources, visualize, and transform data, detect statistical bias, and generate reports on the detected bias in datasets. It also provides feature importance calculations for your training dataset.
- Amazon SageMaker Clarify can detect statistical bias and generate bias reports on training datasets. Additionally, it can detect bias in trained and deployed models. Moreover, it offers capabilities for machine learning explainability and detects data and model drift.

__Detecting Bias in Datasets Using Amazon SageMaker Clarify:__
- To use Clarify APIs, start by importing the SageMaker Clarify module from the SageMaker SDK and then construct the SageMakerClarifyProcessor object using the SageMakerClarifyProcessor constructor.
> 1) Constructing SageMakerClarifyProcessor object
```ruby
from sagemaker import clarify

clarify_processor = clarify.SageMakerClarifyProcessor(role=role, instance_count=1, instance_type="ml.c5.2xlarge", sagemaker_session=sess)
```

> 2) Define S3 path for bias report output
```ruby
bias_report_output_path = "s3://your-bucket-name/path/to/save/bias/report"
```
- instance_count represents the number of nodes that are included in the cluster 
- instance_type represents the processing capacity of each individual node in the cluster
- The processing capacity is measured by the node's compute capacity, memory and the network
- In the next step you configure the data config object on the clarify library, Data config object represents the details about your data

> 3) Configuring data config object
```ruby
bias_data_config = clarify.DataConfig(s3_data_input_path = "s3://your-bucket-name/path/to/input/data", s3_output_path = bias_report_output_path, label = 'sentiment', headers = df_balanced.columns.to_list(), dataset_type = 'text/csv')
```
label that we are going to predict
- In the next step, you configure the bias config object on clarify library. The bias config object captures the facet or the featured name that you are trying to evaluate for bias or imbalance. In this below case you are trying to find out the imbalances in the product category feature. SO if the sentiment feature is your label what is the desired value for that label. That value goes into the parameter label or threshold.
- The parameter label_values_or_threshold defines the desired values for the labels.

> 4) Configuring bias config object
```ruby
bias_config = clarify.BiasConfig(label_values_or_threshold=[...], 
                                 facet_name='product_category')
```
- In the next step you run the pre training bias method on the clarify processor.
>  Running pre-training bias method
```ruby
clarify_processor.run_pre_training_bias(data_config=bias_data_config, 
                                        data_bias_config=bias_config, 
                                        methods=["CI", "DPL", ...], 
                                        wait=False, 
                                        logs=False)
```
- In addition to specifying the data config and the data bias you already configured, You can also specify the methods that you want to evaluate for bias. These are nothing but the metrics that you already learned about to detetct bias.
- Wait parameter specifies whether this bias detection job should block your rest of the code or should it be executed in the background.
- Similarly logs parameter specify that whether you want to capture the logs or not.
- Once the configuration of the pre-training bias method is done, you launch this job. In the background, SageMaker Clarify is using a construct called SageMaker Processing Job to execute the bias detection at scale. SageMaker Processing Jobs is a construct that allows you to perform any data-related tasks at scale.
- These tasks could be executing pre-processing, or post-processing tasks, or even using data to evaluate your model.

- 
- As you can see in the figure here, the SageMaker Processing Job expects the data to be in an S3 bucket.  The data is collected from the S3 bucket and processed on this processing cluster which contains a variety of containers in the cluster.

![image](https://github.com/omkarfadtare/Practical_data_science/assets/154773580/94a703d1-c887-48ba-914e-24689a8f8525)

 Once the processing cluster has processed the data, the transformed data or the processed data is put back in the S3 bucket.  What do you think happens when you execute this run pre-training bias?
- The result will actually be a very detailed report on the bias on your dataset that has persisted in S3 bucket. You can download the report and review in detail to understand the behavior of your data.  you should see a few familiar metrics, like CI and DPL, and also a few other metrics that you might not know about.



__Which one of these tools should you use, in which situation?__
- Data Wrangler, provides you with more of a UI-based visual experience. 
- So, if you would like to connect to multiple data sources and explore your data in more visual format, and configure what goes into your bias reports by making selections from drop-down boxes and option buttons, and finally, launch the bias detection job using a button click, Data Wrangler is the tool for you.
- Keep in mind that Data Wrangler is only using a subset of your data to detect bias in that data set.
-  SageMaker Clarify provides you with more of an API-based approach. Additionally, Clarify also provides you with the ability to scale out the bias detection process.
-  SageMaker Clarify uses a construct called processing jobs that allow you to configure a distributed cluster to execute your bias detection job at scale. So, if you're thinking of large volumes of data, for example, millions of millions of rows of product reviews, and you want to explore that data set for bias, then SageMaker Clarify is the tool for you, so that you can take advantage of the scale and capacity offered by Cloud.




__Feature importance (SHAP):__
- Feature importance is the idea of explaining the individual features that make up your training data set, using a score called important score.
- Some features from your data set could be more relevant, or more important, to your final model than others. Using feature importance, you can rank the individual features in the order of their importance and contribution to the final model.
- Feature importance allows you to evaluate how useful or valuable a feature is, in relation to the other features that exist in the same data set.
- In case of product review dataset, It consists of multiple different features, and you are trying to build a product sentiment prediction model out of that data set.
- Feature importance is based on a very popular for open source framework called SHAP; SHAP stands for Shapley Additive Explanations.
- The framework itself is based on Shapley values, which in turn is based on game theory. consider a play, or a game, in which multiple players are involved and there is a very specific outcome to the play that could be either a win or a loss. Shapley values allow you to attribute the outcome of the game to the individual players involved in the game. you can use the same concept to explain the predictions made by the machine learning model.
- In this case, the individual players would be the individual features that make up the data set, and the outcome of the play would be the machine learning model prediction.
- Using the SHAP framework, you can provide both local and global explanations. While the local explanation focuses on indicating how an individual feature contributes to the final model, the global explanation takes a much more comprehensive view in trying to understand how the data in its entirety contributes to the final outcome from the machine learning model.
- SHAP framework it considers all possible combinations of feature values along with all possible outcomes for your machine learning model.

__How to use Data Wrangler to calculate feature importance on your data set__
- Amazon sagemaker studio >> New data flow >> S3/Athena >> navigate to the right bucket >> navigate to right data that you want to calculate the feature importance on >>  After selecting right csv you will see the preview of the columns in that csv file >> import dataset (THis action will bring the data from s3 bucket to data wrangler environment) >> Once the data is imported click of + sign >> Add analysis >> select type of analysis (quick model) >> name analysis >> select the label that you want to ndicate in your data set >> preview >> Create analysis

combine positive feedback count + recommender indicator + new feature

Week2 
- Detecting statistical bias in the training dataset can help you gain insight into how imbalanced the dataset could be. I demonstrated using Data Wrangler, to detect statistical bias in your training data and generate bias reports. I also introduced SageMaker Clarify API, that'll help you perform the bias detection at scale. In the lab exercise this week, you will use the Clarify APIs to generate the bias reports and explore the report in a bit more detail. I further introduced feature importance and demonstrated using Data Wrangler how to generate the feature importance report on your training dataset. The generated report give you an insight into how the individual features of the training dataset are contributing to the final model. I hope you enjoyed the Week 2 content, and you're ready and excited to explore the lab assignment.










### Week3:
- Discuss some of the challenges, or repetitive tasks, that you can often run into when building machine learning models. learn about some of the benefits of using AutoML
- After this week's course, you'll be able to describe the concept of AutoML as well as be able to describe how you can train a text classifier using AutoML.
- you will learn about how Amazon Sagemaker uniquely implements AutoML capabilities through Sagemaker autopilot. For this, I'll walk you through the steps on how you can use Sagemaker autopilot to train a text classifier. So let's get started and dive deeper into AutoML.
