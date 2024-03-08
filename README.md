# Practical data science on AWS cloud:
In this project, you'll learn how to turn your data science project ideas into real-world solutions. Then, you'll learn to analyze and clean datasets, extract relevant features, train models, and construct automated pipelines to orchestrate and scale your data science projects, serving thousands of models to millions of end-users using AutoML. Following that, you'll learn to create and deploy ML pipelines that manage the flow of data and models, especially using a powerful tool called BERT. Additionally, you'll learn how to deploy human-in-the-loop pipelines, processes that involve humans in decision-making, to make your models work even better.

__Prerequisites for this project:__
- Proficiency in Python and SQL programming.
- Familiarity with building neural networks using deep learning Python frameworks like TensorFlow or PyTorch.
- Understanding of the concept of building, training, and evaluating machine learning models.

__Brief introduction to Artificial intelligence and Machine learning:__
- ___`Artificial intelligence`___, is generally described as a technique that lets machines mimic human behavior.
- ___`Machine learning`___, is a subset of AI, that uses statistical methods and algorithms that are able to learn from data, without being explicitly programmed.
- ___`Deep learning`___ is yet another subset of ML, that uses artificial neural networks to learn from data.
- ___`Data science`___ is an interdisciplinary field that combines business and domain knowledge with mathematics, statistics, data visualization, and programming skills.

![image](https://github.com/omkarfadtare/Practical_data_science/assets/154773580/54ea4e85-d127-4fc2-9f1c-4eadd5c6b69b)

- Popular ML tasks are classification and regression problems, which are examples of supervised learning.
- In ___`Supervised learning`___, you learn by providing the algorithm with labeled data.
- In ___`Classification`___, the goal is to assign the input sample a defined class. For example, is this email I received spam or not spam?
- In contrast, ___`Regression`___ applies statistical methods to predict a continuous value, such as a house price, given a set of related and non-related input variables.
- Another popular task is clustering, it is an example of ___`Unsupervised learning`___ where the data is not labelled. The ___`Clustering`___ algorithm tries to find patterns in the data and starts grouping the data points into distinct clusters.
- Image processing is a major task of ___`Computer vision`___ where you need to classify images into pictures of dogs and cats, distinguish between speed signs and trees.
- The field of ___`Natural Language Processing`___ (NLP), or ___`Natural Language Understanding`___ (NLU) includes machine translations, sentiment analysis, question answering, etc.

![image](https://github.com/omkarfadtare/Practical_data_science/assets/154773580/7ae4c166-5944-4539-aeac-ba763619bf55)

__Benifits of performing data science project on cloud:__
- Data science project on cloud is geared towards handling massive datasets, that could originate from social media channels, mobile and web applications, public or company internal data sources, and much more, depending on the use case you're working on. This data is often messy, potentially error-ridden, or even poorly documented.
- Data science project on cloud tackles these issues by providing tools to analyze and clean the data and to extract relevant features and leads to knowledge distillation and gaining insight from those large datasets.
- Developing and running data science projects on cloud offers the agility and elasticity. If you develop data science projects in a local notebook or IDE environment, for example, hosted on your laptop or a company-owned server pool, you are limited by the existing hardware resources.You have to carefully monitor how much data you process and how much CPU processing power you have available to train and tune your model. If you need more resources, you must purchase additional computing resources. This process doesn't allow for quick development and movement. Perhaps your model training takes too long because it consumes all of the CPU resources of the compute instance you have chosen.
- Using the cloud, you can switch to a compute instance that has more CPU resources, or even switch to a GPU-based compute instance.
- The cloud allows you to train your model on a single CPU instance i.e. ___`Scaling up`___ or perform distributed model training in parallel across various compute instances i.e. ___`Scaling out`___
- In the cloud, when the model training is completed, the instances are terminated as well. This means you only pay for what you actually use.
- The cloud allows you to store and process almost any amount of data. It also comes with a large data science and machine learning toolbox from which you can choose, to perform your tasks as fast and efficiently as possible.

![image](https://github.com/omkarfadtare/Practical_data_science/assets/154773580/5d4e0e9a-c2b5-4570-a99f-92ab3355dbec)

## Project use case: Multi-class classification for Sentiment analysis of Product reviews
- Assume you work at an e-commerce company, selling many different products online. Your customers are leaving product feedback across all the online channels. Whether it is through sending email, writing chat FAQ messages on your website, maybe calling into your support center, or posting messages on your company's mobile app, popular social networks, or partner websites. And as a business, you want to be able to capture this customer feedback as quickly as possible to spot any change in market trends or customer behavior and then be alerted about potential product issues.
- Your task is to build an NLP model that will take those product reviews as input. You will then use the model to classify the sentiment of the reviews into the three classes of positive, neutral, and negative to spot any changes in market trends or customer behavior, as well as be alerted on any potential product issues.
- Multi-class classification is a supervised learning task, hence, you must furnish your classifier model with examples to correctly learn how to classify products and product reviews into the respective sentiment classes. 
- You can use the review text as the input feature for model training and assign sentiment labels to train the model. Sentiment classes are typically represented as integer values during model training, such as 1 for positive sentiment, 0 for neutral sentiment, and -1 for negative sentiment.

## Data ingestion and Exploration:
-  ___`Data ingestion`___ is the process of bringing data from various sources, such as databases, files, or websites, into a system where it can be used. Think of it like gathering ingredients into your kitchen before cooking a meal.
- Imagine your e-commerce company collecting customer feedback from multiple online channels, including social media, support center calls, emails, mobile apps, and website data.
- To accomplish this, you need a flexible and scalable repository capable of storing different file formats, including structured data like CSV files and unstructured data like support center call audio files. Additionally, it should dynamically scale storage capacity as new data streams in.
- Cloud-based data lakes provide a solution to this problem. A ___`Data lake`___ is a centralized and secure repository that can store, discover, and share virtually any amount and type of data.
- Data can be ingested into a data lake in its raw format without prior transformation. Whether it's structured relational data in CSV or TSV files, semi-structured data like JSON or XML files, or unstructured data such as images, audio, and media files. Additionally, you can ingest streaming data such as continuous log file feeds or social media channel feeds into your data lake.
- Effective governance is crucial for managing a data lake. With new data continuously arriving, it's essential to implement mechanisms for discovering and cataloging incoming data.
- There are three types of storage technologies used in computer systems and data storage solutions:
  > ___`File storage`___ organizes data as individual files stored in a hierarchical directory structure. It is similar to how files are organized on personal computers or file servers. File storage is well-suited for storing structured and semi-structured data, such as documents, spreadsheets, multimedia files, and application data.
  
  > ___`Block storage`___ manages data as individual blocks or chunks, typically at the disk level. Each block is a fixed-size unit of data and is accessed using block-level protocols like SCSI (Small Computer System Interface) or Fibre Channel. Block storage is commonly used in storage area networks and provides high-performance, low-latency access to data.
  
  > ___`Object storage`___ is a storage architecture that manages data as objects. Each object consists of data, metadata (information that describes data, such as when the object was last modified), and a unique identifier.
- Data lakes, such as Amazon S3 are often built on top of object storage. Amazon S3 provides access to durable and highly-available object storage in the cloud, allowing you to ingest virtually any amount of data, from a few dataset files to exabytes of data.
- AWS provides various tools and services that help ensure your data lake, which is built on Amazon S3, is secure, compliant with regulations, and allows for auditing. These tools and services include features like access control, encryption, data governance, and compliance certifications, which help protect your data and ensure it meets legal requirements.
- By using Amazon S3 as a centralized repository for your data lake, you can easily access and analyze your data for data warehousing analytics. Data warehousing analytics involves analyzing large volumes of structured data to gain insights and make informed decisions. Additionally, you can leverage machine learning tools and algorithms to extract valuable insights from your data, identify patterns, and make predictions or recommendations based on your business needs.
- Data lakes and data warehouses are both technologies used for storing and analyzing data, but they have different architectures and purposes
  > ___`Data lake`___ are repositories that store vast amounts of raw, unstructured, and structured data in its native format. They are designed to ingest and store data from various sources, such as databases, sensors, logs, social media, and more, without prior transformation. Data lakes are ideal for exploratory data analysis, data science, and big data processing, as they enable organizations to retain and analyze raw data for future use. ex: ___`AWS S3 Bucket`___
  
  > ___`Data warehouse`___ are structured repositories that store processed and organized data in a structured format. Data warehouses undergo a process of extraction, transformation, and loading (ETL) to clean, transform, and integrate data from different sources before loading it into the warehouse. Data warehouses provide a consolidated view of business data and are optimized for high-performance analytics, reporting, and business intelligence (BI) applications. They are used for historical analysis, generating reports, and supporting decision-making processes. ex: ___`AWS Redshift`___
- ___`AWS Data wrangler`___ is an open-source Python library, that helps you interact with data stored in AWS environments and provides easy-to-use functions to load data from AWS services like S3, Glue, Athena, Redshift, and more into Python environments (such as Jupyter notebooks), and to push processed data back into AWS services.
- For example, if you have a bunch of data stored in an Amazon S3 bucket and you want to use it in your Python code (perhaps for analysis, visualization, or machine learning), AWS Data Wrangler makes it simple to load that data into a Pandas DataFrame. Similarly, if you've processed some data in your Python code and want to store it back into an AWS service like S3 or Redshift, AWS Data Wrangler provides functions to easily accomplish that task too.
- ___`AWS Glue`___ is a fully managed metadata repository catalog service. This metadata includes details about data schemas, column names, data types, file formats, and locations.
- The Data Catalog integrates seamlessly with other AWS services, such as AWS Glue ETL (Extract, Transform, Load), Amazon Athena, Amazon Redshift Spectrum, and Amazon EMR (Elastic MapReduce).
- ___`AWS Glue Crawler`___ is a service provided by AWS as part of the AWS Glue suite. Its primary purpose is to automatically discover and catalog metadata from various data sources, making it easier to analyze and query the data using other AWS services.
- Instead of registering data manually on AWS Glue; AWS Glue Crawler automatically scans and analyzes data stored in different data sources such as Amazon S3 buckets, relational databases (Amazon RDS, Amazon Redshift), data lakes, and other accessible repositories. It detects the structure, format, and schema of the data without requiring manual intervention.
- ___`AWS Athena`___ is an interactive query service that enables you to analyze data stored in Amazon S3 using standard SQL queries. It allows you to query data directly from your S3 buckets without the need to set up or manage any infrastructure.
- It is a serverless service, which means you don't need to provision or manage any infrastructure. You simply define your queries, and Athena automatically scales resources to execute them quickly and efficiently.
- Athena integrates seamlessly with the AWS Glue Data Catalog, allowing you to define tables and schemas for your S3 data
- Athena is compatible with popular business intelligence (BI) tools like Tableau, Power BI, and Amazon QuickSight.
- When running highly complex analytical queries against large volumes of data, Athena automatically scales out and divides the query into simpler ones to run in parallel. This capability is possible because Athena is built on Presto, an open-source distributed SQL engine designed for this purpose.

## Data visualization:
- Pandas, an open-source library, is utilized for data analysis and manipulation.
- NumPy, another open-source library, facilitates scientific computing in Python.
- Matplotlib aids in creating static, animated, and interactive visualizations.
- Seaborn, built on top of matplotlib, enhances visualizations with statistical data analysis.

## Statistical bias and Feature importance:
- Statistical bias and Feature importance help you gain a better understanding of your data quality and allows you to explore how the individual feature of your datasets contribute to the final model.
- A dataset is biased if it fails to accurately represent the underlying problem space. Statistical bias refers to a tendency to either overestimate or underestimate a parameter.
- For instance, in a dataset where fraudulent credit card transactions are rare, fraud detection models may struggle to identify fraudulent transactions due to lack of exposure. One solution to this issue is to augment the training dataset with more examples of fraudulent transactions.
- Similarly, consider a product review dataset where one product category (A) has a large number of reviews compared to categories B and C. When building a sentiment prediction model using this biased dataset, the model may accurately predict sentiments for category A products but perform poorly for categories B and C.
- There are various ways with which bias can be introduced in the dataset:
  > ___`Activity bias`___ Activity bias occurs when certain groups or individuals are overrepresented or underrepresented in the data due to their level of engagement or activity. In an online shopping dataset, frequent users may have more data recorded about their preferences and behaviors compared to occasional users, leading to biased predictions.
  
  > ___`Societal bias`___ reflects existing societal inequalities and prejudices that are reflected in the data, leading to unfair treatment of certain groups. Historical biases against certain demographics (e.g., race, gender) may be perpetuated in datasets, resulting in biased decisions in areas like hiring or lending.
  
  > ___`Selection bias`___ occurs when the data collection process systematically favors certain samples over others, leading to an unrepresentative dataset. A survey conducted only among tech-savvy individuals may not accurately represent the opinions of the general population, leading to biased conclusions.
- ___`Drift`___ refers to a change or deviation in the statistical properties or distribution of data over time. It can occur in various forms:
  > ___`Data drift`___ happens when the data used by a model changes over time, making the model less accurate.
  
  > ___`Concept drift`___ occurs when the relationship between the model's input and output changes over time. For example, if you're predicting whether someone will buy a product based on their age, and suddenly younger people start buying more than older people, that's concept drift.
  
  > ___`Covariate drift`___ happens when the characteristics used by the model to make predictions change over time. Let's say you're trying to predict how much ice cream people will buy based on the temperature. If suddenly people start buying more ice cream on colder days instead of hotter ones, that's covariate drift.
  
  > ___`Prior probability drift`___ is the shift in the frequency of each outcome over time. Imagine you're flipping a coin, and at first, it comes up heads 70% of the time and tails 30% of the time. Your model learns from this and gets good at predicting based on that. But then, over time, the coin changes, and now it's heads only 50% of the time and tails 50% of the time.
  
  > ___`Model drift`___ is when a model that used to perform well becomes less accurate over time. This can happen if the data the model was trained on changes, or if the world changes in a way that the model didn't expect.
  
  > ___`Population drift`___ occurs when the population the model is applied to changes over time, making the model less accurate. Let's say you're building a model to predict what kind of movies people will like, and you train it on data from one country. If you then try to use that model in a different country where people have different tastes, that's population drift.
  
  > ___`Label drift`___ is when the answers or labels you have for your data change over time. It's like if you were trying to label pictures of cats and dogs, but then someone changed their mind about what a cat looks like. So, the labels for the pictures change, making it harder for your model to learn from them because the right answers keep changing.
- ___`Class imbalance`___ refers to the situation where one class (or category) of data is significantly more or less prevalent than another class.  
- The ___`Difference in Proportions of Labels (DPL)`___ calculates the absolute difference in the proportions of a particular outcomes between different groups or categories within a dataset.
- It helps to understand whether there are imbalances in outcomes across different groups or categories. Identifying such differences is essential for understanding biases, making decisions, and designing strategies to address disparities.
- In context of product reviews dataset. understanding the DPL between different product categories is crucial for identifying which categories tend to receive more positive reviews than others. This insight can inform marketing strategies, product development priorities, and resource allocation decisions within a company.
- So, while CI looks at overall reviews, DPL looks at whether some categories get higher ratings than others.
- For example, consider we have a dataset of customer reviews for an e-commerce platform. Each review is labeled as either "positive" or "negative" sentiment based on the customer's feedback.
- ___`Sagemaker Clarify`___ offers functionality for detecting biases in both datasets and machine learning models. It analyzes training and testing datasets to identify biases based on facet/sensitive features (such as gender or race) and generate detailed bias reports. These reports include metrics, visualizations, and insights to help users understand and mitigate biases in their datasets.
- SageMaker Clarify seamlessly integrates with other components of Amazon SageMaker, allowing users to incorporate bias detection and model explainability into their ML workflows. 
- ___`Sagemaker Wrangler`___ focuses on data preparation tasks such as connecting to various data sources, visualizing, transforming data, and generating reports on the data.can help with preparing the data for bias analysis by cleaning and preprocessing it, it does not include built-in features for detecting biases or generating bias reports.
- ___`Feature importance`___ refers to the measure of the impact or significance of each feature in a dataset on the prediction made by a machine learning model. It provides a ranking of features based on their contribution to the model's performance.
- It provides a high-level overview of feature importance across the entire dataset.
- ___`SHapley Additive exPlanations (SHAP)`___ is a method for explaining individual predictions made by machine learning models. It provides a unified framework for understanding the contribution of each feature to the prediction for a specific instance. SHAP values represent the impact of each feature on the difference between the actual prediction and the average prediction across all instances in the dataset.
- SHAP values offer a more detailed and instance-specific explanation of feature contributions, allowing for a deeper understanding of model behavior and individual predictions.
- 
# This is till week2 everything is up to date you just have to walk through the coding part and include importatnt images

### Important links:
[Dataset](https://www.kaggle.com/datasets/nicapotato/womens-ecommerce-clothing-reviews)
[AWS SDK](https://github.com/aws/aws-sdk-pandas)
[AWS Glue](https://aws.amazon.com/glue/)
[AWS Athena](https://aws.amazon.com/athena/)
[matplotlib](https://matplotlib.org/)
[Seaborn](https://seaborn.pydata.org/)
[Pandas](https://pandas.pydata.org/)
[NumPy](https://numpy.org/)
[AWS Wrangler](https://aws-sdk-pandas.readthedocs.io/en/stable/)
[use case](https://aws.amazon.com/sagemaker/canvas/customers/#samsung)


- ___`Automated Machine Learning (AutoMl)`___,
# Week3
- This week, we will discuss the challenges often encountered when building machine learning models and explore the benefits of using AutoML
- We'll also deep dive into the steps comprising the machine learning workflow and identify those that can be automated through AutoML.
- By the end of this course, you will be able to explain the concept of AutoML and demonstrate how to train a text classifier.
- Furthermore, you will learn how to implement AutoML using AWS SageMaker Autopilot.

__What is AutoML and Benifits of using AUtoML:__
__Steps in machine learning workflow and where automl can be used__
__Implementation of AUtoML using AWS sagemaker Autopilot__
__Model Hosting/Deployement__
- To take model and make it available for consumption so applications can use it for predictions
