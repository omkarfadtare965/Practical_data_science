# Turning data science project ideas into real-world solutions using AWS cloud:
__Prerequisites for this project:__
- Proficiency in Python and SQL programming.
- Familiarity with building neural networks using deep learning Python frameworks like TensorFlow or PyTorch.
- Basic understanding of building, training, and evaluating machine learning models.

__Brief introduction to Artificial intelligence and Machine learning:__
- ___`Artificial intelligence`___ is generally described as a technique that lets machines mimic human behaviour.
- ___`Machine learning`___ is a subset of AI, that uses statistical methods and algorithms that can learn from data, without being explicitly programmed.
- ___`Deep learning`___ is yet another subset of ML, that uses artificial neural networks to learn from data.
- ___`Natural language processing (NLP)`___ or ___`Natural language understanding (NLU)`___ is yet another subset of ML focused on the interaction between computers and humans through natural language, which includes machine translations, sentiment analysis, question answering system, etc.
- ___`Computer vision`___ is a subset of ML that enables computers to interpret and make decisions based on visual data from the world, where you need to classify images into pictures of dogs and cats, distinguish between speed signs and trees.
- ___`Data science`___ is an interdisciplinary field that combines business and domain knowledge with mathematics, statistics, data visualization, and programming skills.

![image](https://github.com/omkarfadtare/Practical_data_science/assets/154773580/54ea4e85-d127-4fc2-9f1c-4eadd5c6b69b)
 
![image](https://github.com/omkarfadtare/Practical_data_science/assets/154773580/7ae4c166-5944-4539-aeac-ba763619bf55)

__Benifits of performing data science projects on the cloud:__
- Data science projects are often geared towards handling massive datasets that can originate from social media channels, mobile and web applications, public or company internal data sources, and more, depending on the specific use case. This data is often messy, potentially error-ridden, or poorly documented. Cloud-based data science tackles these issues by providing tools to analyze, and clean the data, extract relevant features, and gain insights from these large datasets.
- Developing and running data science projects on the cloud offers agility and elasticity. In contrast, if you develop data science projects in a local notebook or IDE environment, such as on your laptop or a company-owned server pool, you are limited by the existing hardware resources. You have to carefully monitor how much data you process and the available CPU processing power for training and tuning your model. If you need more resources, you must purchase additional computing resources, which hinders quick development and flexibility. Additionally, your model training might take too long because it consumes all the CPU resources of the compute instance you have chosen. Using the cloud services, you can indeed switch to a compute instance with more CPU resources or even switch to a GPU-based compute instance.
- The cloud allows you to train your model on a single CPU instance, which is referred to as ___`scaling up`___, or perform distributed model training in parallel across various compute instances, which is referred to as ___`scaling out`___.
- In the cloud, when the model training is completed, the instances are terminated, meaning you only pay for what you actually use. The cloud allows you to store and process almost any amount of data and offers a large toolbox of data science and machine learning tools to perform tasks quickly and efficiently.

![image](https://github.com/omkarfadtare/Practical_data_science/assets/154773580/5d4e0e9a-c2b5-4570-a99f-92ab3355dbec)

# Project use case: Multi-Class Classification for Sentiment Analysis of Product Reviews
- Assume you work at an e-commerce company that sells various products online. Customers leave product feedback across multiple online channels, such as sending emails, writing chat FAQ messages on your website, calling your support center, or posting messages on your company's mobile app, popular social networks, or partner websites. As a business, you want to capture this customer feedback quickly to spot any changes in market trends or customer behaviour and be alerted to potential product issues.
- Your task is to build an NLP model that takes these product reviews as input. You will use the model to classify the sentiment of the reviews into three classes: positive, neutral, and negative. This classification will help you spot changes in market trends or customer behaviour and be alerted to potential product issues.
- Multi-class classification is a supervised learning task, so you must provide your classifier model with examples to correctly learn how to classify product reviews into the respective sentiment classes. You can use the review text as the input feature for model training and assign sentiment labels to train the model. Sentiment classes are typically represented as integer values during model training, such as 1 for positive sentiment, 0 for neutral sentiment, and -1 for negative sentiment.

## Data ingestion and Exploration:
-  ___`Data ingestion`___ is the process of bringing data from various sources, such as databases, files, or websites, into a system where it can be used. Think of it like gathering ingredients in your kitchen before cooking a meal.
- Imagine your e-commerce company is collecting customer feedback from multiple online channels, including social media, support centre calls, emails, mobile apps, and website data. 
- To accomplish this, you need a flexible and scalable repository capable of storing different file formats, including structured data like CSV files and unstructured data like support centre call audio files. Additionally, it should dynamically scale storage capacity as new data streams in. Cloud-based data lakes provide a solution to this problem.
- ___`Data lake`___ is a centralized and secure repository that can store, discover, and share virtually any amount and type of data. Data can be ingested into a data lake in its raw format without prior transformation. Whether it's structured relational data in CSV or TSV file formats, semi-structured data like JSON or XML files, or unstructured data such as images, audio, and media files, it can all be ingested. Additionally, you can ingest streaming data such as continuous log file feeds or social media channel feeds into your data lake.
- Effective governance is indeed crucial for managing a data lake, especially given the continuous influx of new data. Implementing mechanisms for discovering and cataloguing incoming data helps ensure that the data lake remains organized, accessible, and compliant with regulatory and organizational standards. This governance framework also facilitates efficient data utilization, enhances data quality, and supports data analytics and decision-making processes within an organization. Cloud services facilitate scalable storage, automated data ingestion, robust security and compliance measures, advanced analytics capabilities, cost efficiency, and seamless integration with machine learning, all crucial for effectively managing a data lake.
- There are three types of storage technologies used in computer systems and data storage solutions:
  > ___`File storage`___ is a storage technology that organizes data as individual files stored in a hierarchical directory structure, similar to how files are organized on personal computers or file servers. File storage is suitable for structured and semi-structured data such as documents, spreadsheets, multimedia files, and application data.

  > ___`Block storage`___ is a storage technology that manages data as individual blocks or chunks at the disk level, accessed using block-level protocols such as SCSI (small computer system interface) or Fibre Channel. It is commonly used in storage area networks (SANs).
  
  > ___`Object storage`___ is a storage architecture that manages data as objects, each consisting of data, metadata (information that describes the object), and a unique identifier. Object storage is suitable for scalable and distributed storage of unstructured data like images, videos, backups, and log files.
- Amazon S3 is commonly used as the underlying object storage for data lakes due to its durability, availability, and scalability. It allows ingestion of vast amounts of data, ranging from small datasets to exabytes. AWS provides tools and services that ensure data lakes built on Amazon S3 are secure, compliant with regulations, and auditable. This includes features like access control, encryption, data governance, and compliance certifications.
- Amazon S3 serves as a centralized repository in data lakes, facilitating easy access and analysis of data for data warehousing analytics. This involves analyzing large volumes of structured data to derive insights and support decision-making. It also supports integration with machine learning tools for extracting deeper insights, identifying patterns, and making predictions based on business requirements.
- Data lakes and data warehouses are indeed different technologies with distinct architectures and purposes. Data lakes are designed for storing vast amounts of raw and unstructured data, enabling flexible data exploration and analysis. In contrast, data warehouses are optimized for querying and analyzing structured data to support business intelligence and reporting.
- ___`Difference between data lake and data warehouse:`___

| ___`Data lake`___                                                                                                                                         | ___`Data warehouse`___                                                                                                                                           |
|-----------------------------------------------------------------------------------------------------------------------------------------------------------|------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| - Data lakes are repositories that store vast amounts of raw, unstructured, and structured data in its native format.                                     | - Data warehouses are structured repositories that store processed and organized data in a structured format.                                                    |
| - They are designed to ingest and store data from various sources without prior transformation.                                                           | - Data undergoes extraction, transformation, and loading (ETL) to clean, transform, and integrate data from different sources before loading into the warehouse. |
| - Ideal for exploratory data analysis, data science, and big data processing, allowing organizations to retain raw data for future analysis and insights. | - Optimized for high-performance analytics, reporting, and business intelligence (BI) applications, supporting historical analysis and decision-making.          |
| - ___`AWS S3 Bucket`___ serves as a common example of a data lake storage solution.                                                                       | - ___`AWS Redshift`___ is an example of a data warehouse service optimized for querying and analyzing structured data.                                           |

- ___`AWS Data Wrangler`___ is an open-source Python library focused on simplifying data preparation and exploration for analytics and machine learning tasks. It provides easy-to-use functions and abstractions for working with data in Pandas DataFrames, facilitating common data engineering tasks. AWS Data Wrangler seamlessly integrates with AWS services such as Amazon S3, Amazon Redshift, Amazon Athena, and Amazon Glue, enabling smooth data integration, processing, and interaction between Python environments (such as Jupyter notebooks) and AWS data services. For instance, it simplifies loading data from S3 into Pandas DataFrames for analysis or machine learning and allows pushing processed data back into AWS services like S3 or Redshift.

> Code to retrieve data directly from s3 bucket to Pandas DataFrame using AWS Data Wrangler:
```python
!pip install awswrangler
!pip install boto3 # Boto3 is the AWS SDK (software development kit) for Python. It allows Python developers to write software that makes use of AWS services like S3, EC2, DynamoDB, and many more. Boto3 provides an easy-to-use, object-oriented API, as well as low-level access to AWS services.

import awswrangler as wr
import boto3
import pandas as pd

# Replace with your actual AWS credentials
aws_access_key_id = 'your_access_key_id'
aws_secret_access_key = 'your_secret_access_key'
region = 'your_aws_region'

# Create a boto3 session with your credentials
boto3_session = boto3.Session(
    aws_access_key_id=aws_access_key_id,
    aws_secret_access_key=aws_secret_access_key,
    region_name=region
)

# Replace with your actual S3 path
s3_path = 's3://your-bucket-name/path/to/your/file.csv'

# Use AWS Data Wrangler to read data from S3
df = wr.s3.read_csv(path=s3_path, 
                    boto3_session=boto3_session,
                    skiprows=0,
                    nrows=None,
                    dtype=None)

# skiprows: number of rows to skip from the beginning
# nrows: number of rows to read ('None' to read all)

print(df.head())
```

- ___`AWS Glue`___ is a fully managed service from AWS designed for extract, transform, and load (ETL) tasks, automating processes such as data discovery, cataloguing, cleaning, and transformation. It simplifies ETL job creation and management through a graphical interface, allowing users to schedule and monitor jobs easily. AWS Glue seamlessly integrates with various AWS services like Amazon S3, Amazon RDS, and Amazon Redshift, facilitating efficient data integration and processing workflows across diverse data sources and formats.
- ___`AWS Glue Crawler`___ is a tool provided within AWS Glue that automates the process of discovering and cataloguing data stored in different sources such as Amazon S3, databases, and data warehouses. It scans these data sources, infers the schema of the data (i.e., its structure and format), and then creates metadata tables in the AWS Glue Data Catalog. This allows AWS users to easily access and query the data using AWS Glue ETL jobs or other AWS analytics services like Amazon Athena, Amazon Redshift Spectrum, and Amazon EMR. The AWS Glue Crawler simplifies the management of data cataloguing and ensures that metadata remains updated as new data is added or existing data changes, thereby supporting efficient data integration and processing workflows within AWS environments.
- ___`AWS Glue Data Catalogue`___ serves as a central metadata repository within AWS, storing comprehensive metadata information about databases and tables across your AWS environment. It operates independently as a persistent metadata store, managing both structural and operational metadata for all data assets. This includes storing definitions for tables, partitions, and other relevant metadata components, providing a consolidated and unified view of your data assets. The AWS Glue Data catalogue integrates seamlessly with AWS Glue itself, leveraging its metadata for tasks such as Extract, Transform, and Load (ETL) jobs. Furthermore, other AWS services like Amazon Athena, Amazon Redshift Spectrum, and Amazon EMR utilize the AWS Glue Data catalogue for efficient querying and processing of data, ensuring consistent and reliable access to metadata across different AWS services and environments.
- While AWS Glue uses the Data Catalog to store metadata, they serve different primary functions: AWS Glue executes ETL tasks, while the Data Catalog manages and stores metadata about those tasks and data assets.
- ___`AWS Athena`___ is an interactive query service provided by Amazon Web Services (AWS) that allows you to analyze and query data stored in Amazon S3 using standard SQL. It enables you to run ad-hoc queries on large amounts of data without needing to set up or manage any infrastructure. Athena is serverless, meaning there is no need for provisioning or scaling of resources, and you pay only for the queries you run. It supports a wide range of data formats, including CSV, JSON, Parquet, and ORC, making it versatile for analyzing different types of data stored in S3.
- An ad-hoc query is a query that you write on the fly whenever you need to quickly get specific information from a database or data source. It's like asking a question directly to the data to get immediate answers, without needing to plan or save the query for future use.

> Code to execute a SQL query on AWS Athena using AWS Data Wrangler, and loading the results into a Pandas DataFrame:
```python
import awswrangler as wr
import pandas as pd

# Set up the boto3 session for AWS Data Wrangler
aws_access_key_id = 'your_access_key_id'
aws_secret_access_key = 'your_secret_access_key'
region = 'your_aws_region'

# Set up the boto3 session for AWS Data Wrangler
wrangler_boto3_session = wr.Session(
    aws_access_key_id=aws_access_key_id,
    aws_secret_access_key=aws_secret_access_key,
    region_name=region
)

# Create an S3 bucket for Athena query results (if it doesn't exist)
wr.athena.create_athena_bucket(boto3_session=wrangler_boto3_session)

# Define your SQL query
sql_query = "SELECT * FROM your_database.your_table LIMIT 10"

# Specify the database
database = "your_database" 

# Execute the query and read results into a DataFrame
df = wr.athena.read_sql_query(
    sql=sql_query,
    database=database,
    boto3_session=wrangler_boto3_session
)

print(df.head())
```

## Data visualization:
- ___`Pandas`___ is indeed an open-source library used for data analysis and manipulation in Python. It provides powerful data structures like DataFrame and Series, along with tools for reading and writing data between in-memory data structures and various file formats.
- ___`NumPy`___ is another open-source library that supports scientific computing in Python. It provides powerful numerical arrays and functions for operations on these arrays, which are essential for numerical computations and data analysis.
- ___`Matplotlib`___ is a widely used library for creating static, animated, and interactive visualizations in Python. It provides a variety of plotting functions to visualize data in different formats, ranging from simple line plots to complex 3D plots. 
- ___`Seaborn`___ is built on top of matplotlib and enhances data visualizations with statistical data analysis capabilities. It provides a high-level interface for drawing attractive and informative statistical graphics, making it easier to create complex visualizations with fewer lines of code.

__Statistical bias and Feature importance:__
- Statistical bias and feature importance are indeed critical tools for evaluating the quality of your data and understanding the role each feature plays in your model. Statistical bias helps you identify any systematic errors in your data, while feature importance helps you understand which features significantly contribute to your model's predictions.
- ___`Statistical bias`___ refers to a systematic error or deviation from the true value or an expected value in the data or a model's predictions. This bias results from the data collection process, model assumptions, or other factors that cause the estimates to be consistently inaccurate in a particular direction.
- ___`Feature importance`___ refers to the measure of the impact or significance of each feature in a dataset on the predictions made by a machine learning model. It evaluates how much each feature contributes to the model's predictive power. Feature importance provides a ranking of features based on their contribution to the model's performance. This ranking helps in understanding which features are most influential in making predictions and which ones have less impact.
- For example, In fraud detection, if fraudulent transactions are rare in the dataset, the model may not perform well in identifying fraud due to insufficient examples of fraudulent transactions. One common solution is to augment the training dataset with more fraudulent examples, which can help the model learn to identify fraud more effectively. 
- Similarly, In our use case, If one product category (e.g., category A) has significantly more reviews than others (e.g., categories B and C), the model may become biased towards predicting sentiments accurately for category A while performing poorly for categories B and C. This happens because the model has more information and training data for category A, making it less effective for underrepresented categories.
- There are various ways in which bias can be introduced in the dataset:
  > ___`Activity bias`___ arises when certain groups or individuals are overrepresented or underrepresented in the data due to their level of engagement or activity. For example, in an online shopping dataset, frequent users might have more recorded data about their preferences and behaviours compared to occasional users. This can lead to biased predictions, as the model may learn more about the frequent users and thus be less accurate for occasional users.
  
  > ___`Societal bias`___ reflects existing societal inequalities and prejudices that are mirrored in the data. This type of bias can lead to unfair treatment of certain groups. Historical biases against certain demographics, such as race or gender, may be embedded in the datasets, resulting in biased outcomes in areas like hiring or lending. For instance, if historical hiring data is biased against a particular gender, a model trained on this data may also discriminate against that gender.
  
  > ___`Selection bias`___ occurs when the data collection process systematically favours certain samples over others, resulting in an unrepresentative dataset. For instance, if a survey is conducted only among tech-savvy individuals, it may not accurately capture the opinions of the general population. This can lead to biased conclusions, as the sample is not representative of the entire population.
  
  > ___`Measurement bias`___ happens when there are inaccuracies in data collection methods, leading to systematically incorrect measurements.
  
  > ___`Sampling bias`___ results from a non-random sample of a population, leading to certain groups being over or under-represented.
  
  > ___`Response bias`___ arises when respondents provide inaccurate or false information, often influenced by the question's wording or the survey environment.

__Drift:__
- Drift refers to any change or deviation in the statistical properties or distribution of data over time that can impact the performance of models. These changes can occur in various forms, affecting different aspects of the data and the model's behaviour:
  > ___`Data drift`___ occurs when the statistical properties of the data used by a model change over time. This can make the model less accurate because it was trained on data with different characteristics. For example, if the features of the data (like user behaviours or product preferences) change, the model's performance may degrade.
  
  > ___`Concept drift`___ refers to a change in the relationship between the input and output variables over time. For instance, if a model predicts product purchases based on age and the purchasing behaviour of different age groups shifts, the model's predictions will become less accurate..
  
  > ___`Covariate drift`___ happens when the distribution of the predictor variables (features) changes over time. For example, if temperature was a key predictor for ice cream sales, and the relationship between temperature and sales changes (people start buying more ice cream on colder days), the model's predictions would be affected.
  
  > ___`Prior probability drift`___ occurs when the frequency of the target outcomes changes over time. For instance, if a coin flip initially results in heads 70% of the time and later changes to a 50-50 distribution, a model trained on the initial data will be less accurate in predicting future outcomes.
  
  > ___`Model drift`___ refers to the phenomenon where a model that used to perform well becomes less accurate over time. This can happen due to changes in the data distribution, changes in the real world that affect the relationships the model learned, or other factors that make the original model less relevant.
  
  > ___`Population drift`___ occurs when the population on which the model is applied changes over time. For example, a model trained on movie preferences in one country may not perform well if applied to a different country with different preferences.
  
  > ___`Label drift`___ happens when the meaning of the target labels changes over time. For example, if the criteria for labelling data as "cat" or "dog" change, the model will struggle to learn and make accurate predictions because the definitions of the labels are inconsistent.

__Matrics to measure imbalance in data:__
- ___`Class imbalance`___ refers to a situation in a classification problem where the number of instances in each class is not evenly distributed.
- ___`Difference in Proportions of Labels (DPL)`___ calculates the absolute difference in the proportions of particular outcomes (e.g., positive vs. negative reviews) between different groups or categories within a dataset. This helps to quantify the degree of imbalance or disparity in outcomes across these groups. DPL helps to understand whether there are imbalances in outcomes across different groups or categories. In the context of a product reviews dataset, understanding the DPL between different product categories is crucial. For instance, if some categories receive disproportionately more positive reviews than others, this insight can inform various strategic decisions such as marketing efforts, product development, and resource allocation within a company. While Cumulative Incidence (CI) or overall reviews look at the total number of reviews and their general trends, DPL specifically focuses on whether some categories get higher or lower ratings than others. This targeted analysis is useful for identifying specific areas where there might be an imbalance. Consider a dataset of customer reviews for an e-commerce platform where each review is labelled as either "positive" or "negative". By calculating the DPL, you can determine if certain product categories receive more positive (or negative) reviews compared to others, highlighting potential biases or areas for improvement.
-  ___`SHapley Additive exPlanations (SHAP)`___ values can also be used to assess feature importance by averaging the absolute SHAP values for each feature across all predictions. It is a method for interpreting individual predictions from machine learning models. They aim to explain the output of a machine learning model by attributing the prediction outcome to different features in the input data. SHAP values are based on the idea of Shapley values in cooperative game theory. Shapley values calculate the contribution of each player (feature) in a coalition (subset of features) to the overall outcome (prediction).

__Tools to detect Statistical bias in the dataset:__
- ___`Amazon SageMaker Data Wrangler`___ is a powerful tool designed to simplify the process of data preparation and feature engineering for machine learning. It is part of Amazon SageMaker, a fully managed service that provides every developer and data scientist with the ability to build, train, and deploy machine learning models quickly and easily. AWS 
SageMaker Data Wrangler offers a visual interface that allows users to interactively prepare their data without writing code. It provides a wide range of built-in data transformers and recipes that simplify common data preparation tasks. These include handling missing values, encoding categorical variables, scaling features, and more. SageMaker Data Wrangler seamlessly integrates with various AWS data sources and services, such as Amazon S3 for data storage, Amazon Athena for querying data in S3 using SQL, Amazon Redshift for data warehousing, and AWS Glue for ETL (Extract, Transform, Load) jobs. The tool includes automated data profiling capabilities that help users understand their datasets better. This includes summary statistics, data distribution visualizations, and insights into potential data quality issues. Users can visualize their data and transformations to gain insights into how features are modified and how data distributions change after processing steps.
- ___`Amazon SageMaker Clarify`___ is indeed a critical tool that helps detect biases in both training data and model predictions. It provides model explainability through tools like SHAP values, which highlight the contribution of each feature to predictions, aiding in understanding model behaviour. SageMaker Clarify generates bias reports that include statistical analyses such as disparate impact analysis and fairness metrics across different demographic groups defined by sensitive attributes (e.g., race, gender). These reports offer insights into biases present in the model and its predictions. SageMaker Clarify not only detects biases but also performs bias detection in both trained and deployed models, ensuring ongoing monitoring and mitigation of biases throughout the model's lifecycle. 
> Code to detect statistical bias using Amazon SageMaker Clarify:
```python
import sagemaker
from sagemaker import clarify

# Specify your SageMaker execution role and session
role = sagemaker.get_execution_role()
session = sagemaker.Session()

# Define the S3 bucket and prefix for input and output data
bucket = 'your-s3-bucket-name'
prefix = 'sagemaker/clarify'

# Specify the path to your training dataset in S3
train_data_uri = f's3://{bucket}/{prefix}/train_data.csv'

# Specify the path to your model artifacts in S3
model_uri = f's3://{bucket}/{prefix}/model.tar.gz'

# Specify the S3 path for bias report output
bias_report_output_path = f's3://{bucket}/{prefix}/bias_reports'

# Create a SageMaker Clarify processor
clarify_processor = clarify.SageMakerClarifyProcessor(role=role,
                                                      instance_count=1,
                                                      instance_type='ml.m5.large',
                                                      sagemaker_session=session)

# Specify configuration for bias detection job
bias_config = clarify.BiasConfig(
    label_name='your_label_column_name',
    facet_name='your_sensitive_attribute_column_name',
    group_name='your_group_id_column_name',
    # Additional configuration options can be set here, such as reference_groups and probability_threshold
)

# Specify data configuration
data_config = clarify.DataConfig(
    s3_data_input_path=train_data_uri,
    s3_output_path=bias_report_output_path,
    label='your_label_column_name',
    headers=train_data.columns.to_list(),  # Optional: Provide headers if your dataset has headers
    dataset_type='text/csv'
)

# Run bias detection job
clarify_processor.run_bias(data_config=data_config,
                            data_bias_config=bias_config,
                            model_uri=model_uri,
                            pre_training_methods='all',
                            post_training_methods='all')

# Wait for the job to finish
clarify_processor.wait()
```
| ___`Amazon SageMaker Data Wrangler`___                                                                                                                                                              | ___`Amazon SageMaker Clarify`___                                                                                                                                                  |
|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| - SageMaker Data Wrangler is designed for data preparation, transformation, and feature engineering.                                                                                                | - SageMaker Clarify is used primarily for bias detection, explainability, and fairness assessment in machine learning models.                                                     |
| -  Use SageMaker Data Wrangler when you need to integrate and preprocess data from multiple sources, benefiting from its visual interface and efficient data handling across distributed locations. | - Use SageMaker Clarify if your priority is to analyze biases and ensure model fairness across a vast dataset, leveraging its scalability and robust bias detection capabilities. |

## AutoML:

![image](https://github.com/omkarfadtare/Practical_data_science/assets/154773580/4e730206-b5e9-4b68-ae6b-e0521ba02aaf)

- ___`AutoML (Automated Machine Learning)`___ refers to the process of automating the end-to-end process of applying machine learning to real-world problems. It includes automating tasks such as data preprocessing, feature selection, model selection, hyperparameter tuning, and model evaluation. 
- AutoML makes machine learning accessible to those without extensive expertise by providing a graphical interface or easy-to-use APIs, automating repetitive tasks, and speeding up the model development process while efficiently utilizing computational resources. It ensures standardized procedures for model training and evaluation, facilitating reproducible results through standardized workflows. Often resulting in better-performing models, AutoML systematically and thoroughly tunes hyperparameters and selects models, employing cutting-edge algorithms and methodologies without requiring users to stay updated with the latest advancements.
- AutoML can be integrated at various stages of the machine-learning workflow:
  > ___`Data Preprocessing:`___ AutoML includes automated cleaning, which handles missing values, outliers, and data normalization, ensuring the dataset is ready for analysis. Additionally, it performs feature engineering by automatically generating and selecting important features, transforming and encoding data, optimizing the dataset for better model performance, and identifying and selecting the most relevant features for modelling.
  
  > ___`Model Selection:`___ AutoML involves algorithm selection, where it chooses the most appropriate algorithms for the dataset, and the use of ensemble methods, which combine multiple models to enhance overall performance. It automates the process of training multiple models and selecting the best one.
  
  > ___`Hyperparameter Tuning::`___ AutoML involves optimization techniques such as grid search, random search, and Bayesian optimization to identify the best hyperparameters, enhancing the model's performance and accuracy.
  
  > ___`Model Evaluation:`___ AutoML involves performing cross-validation and other evaluation techniques to assess model performance. It also includes metrics reporting, which provides comprehensive reports on various performance metrics such as accuracy, precision, recall, and more, ensuring a thorough understanding of the model's effectiveness.
  
  > ___`Model Deployment:`___ AutoML simplifies the process of deploying machine learning models to production environments with minimal manual intervention. It ensures scalability, allowing deployed models to handle varying levels of demand by scaling resources up or down as needed. Additionally, it seamlessly integrates with other tools and platforms, facilitating easy implementation into existing workflows.
  
  > ___`Monitoring and Maintenance:`___ AutoML involves continuous performance tracking to ensure models meet expected standards in the production environment. It includes drift detection to identify data drift and concept drift, signalling when the model's accuracy begins to degrade over time. The system provides alerts and notifications for anomalies or significant performance changes, enabling timely intervention. Additionally, it facilitates automated retraining of models using updated data to maintain their performance and relevance.

- ___`Amazon SageMaker Autopilot`___ is AWS's AutoML solution that automates the end-to-end process of machine learning model development. It starts with data exploration, identifying the machine learning problem, and selecting an appropriate algorithm based on the dataset and problem type. It also transforms the data to the format expected by the selected algorithm and performs training and hyperparameter tuning to find the optimal set of hyperparameters for the best-performing model. SageMaker Autopilot provides transparency by automatically generating and sharing feature engineering code. It also generates Jupyter notebooks that detail how the models were built, including data processing steps, algorithm selection, hyperparameters, and training configurations. This transparency helps users understand and trust the model development process. Users can customize certain aspects of the pipeline, such as selecting specific algorithms or defining custom preprocessing steps. SageMaker Autopilot seamlessly integrates with other AWS services like S3 for data storage, AWS Glue for data cataloguing, and SageMaker Studio for a comprehensive development environment.
- Users can interact with Amazon SageMaker Autopilot in several ways, such as programmatically through the SageMaker API, using the AWS CLI, AWS SDK, or the SageMaker Python SDK. Additionally, users can work with SageMaker Autopilot through SageMaker Studio, which is a workbench for end-to-end machine-learning activities. Regardless of whether you are interacting programmatically or using SageMaker Studio, you are using the same APIs.
![image](https://github.com/omkarfadtare/Practical_data_science/assets/154773580/9da88a47-8348-4b18-bbd8-d26e525dc8c4)
> Code to use SageMaker Autopilot programmatically:
```python
import sagemaker
from sagemaker.automl.automl import AutoML

# Define the S3 path for the input data
input_data_uri = 's3://your-bucket/path/to/data.csv'

# Define the S3 path for the output data
output_data_uri = 's3://your-bucket/path/to/output/'

# Create a SageMaker session
sagemaker_session = sagemaker.Session()

# Create an AutoML job
automl = AutoML(role='your-sagemaker-role', 
                target_attribute_name='target_column', 
                sagemaker_session=sagemaker_session,
                max_candidates=10,  # Number of models to train and evaluate
                output_path=output_data_uri)

# Start the AutoML job
automl.fit(inputs=input_data_uri)

# Deploy the best model
best_candidate = automl.describe_auto_ml_job()['BestCandidate']
model = sagemaker.Model(model_data=best_candidate['InferenceContainerDefinitions'][0]['ModelDataUrl'], 
                        role='your-sagemaker-role', 
                        sagemaker_session=sagemaker_session)
predictor = model.deploy(initial_instance_count=1, instance_type='ml.m5.large')
```

























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


# Week3 

## What is Automl, Concept of Automated machine learning, Benifits of using Automl, How Automl fits into the overall machine learning workflow, Implementation of AutoMl using Sagemaker Autopilot, Model hosting (once having optimised model how do you then take that model and make it available for consumption so applications can then use it for predictions):
- In this case once our text classifier is trained to predict the sentiment for a specific product review how do you then deploy that model so it can actually be used to detect sentiment on new product reviews as they come in.
- Why are we going to use automl for this use case? When you are trying to build a machine learning models to solve everyday problms its common to eun into the model building challenges for a number of different reasons.
  - first step: involved in creating machine learning model typically involve multiple iterations that can often result in increased time to market.
  - Machine learning can also requires specialised skills sets that can be very challenging to find satff from your existing teams.
  - Also machine learnig iterations typically take much longer than traditional developement life cycles. This is due to time it takes to get model performance feedback and the time it takes to run through the numerous experiments using different combinations of data transformations algorithms and hyperparameters untill you find out the model that is meeting at least your minimum objective matrics
  - The nature of machine learning developemnt can also make it difficult to iterate quickly not only from workflow perspective but also from compute resource preferances
- To work out all these challenges this is where automl comes into the picture
- 
- conceptually Automl uses machine learning to automate many tasks in the machine learning workflow
- It reduces time to market by automating resource intensive task like data transformation feature engineering and model tunning.
- Automl can enable your non data scintist to build machine learning models without requiring deep data science skillset
- AutoMl lets you iterate quickly by using machine learning and automation to perform the majority of the tasks in your model building workflows.
- using Automl in combination with cloud computing also addresses potential compute resource challeges
- Automl lets data scientist focus on those really hard to solve machine learning problems.


## Automl workflow:
___TElls you how automl fits into endto-end machine learning workflow___
- __Typical machine learning workflow:__
  - Ingest and analyze your data which includes defining ml problem and exploring and understanding the data, and selecting the right algorirthm or algorithms to use for your experiments
  - Next you prepare and transform your data by performing feature engineering and trasforming data into a format required by the algo 
  - FInaly you typically train multiple models across a number of experiments untill you have a well performing model thats created using a specific combination of algo, data transformation and hyper parameter

- GO through the data, understand the problem, once you identify the potential algo that you want to try as part of your your training experiemets
- After you have done some analysis on your data and detrmine the type of machie learning problem that you are trynig to solve for you can then look at which algo or algo that are best suited for your data and the problem that you are tring to solve.
- When you perform data analysis. You want to understand the data. so getting insight into things like data distribution, attribution correlation or do you have potential quality issues in your data like missing data.
- Then based on your machine learning problem combined with your data analysisyou are able to identify the algorithm that you would like to try for your experiments.
- slecting a right algorithm or algorithms is only the part of the process. for each algorithms there are also a number of hyper parameteres tha t you need to consider as you tuune the model for optimal performance.
- Each algorithms have different expections in terms of the forma of the data that expects on input for traiingsuch as claeaning or text transformation
- in this project we are gonna use TF IDF. Data banacing resampling tech niques
- Once you have done your data transformation then you can use your processed data set to create your training and validation sets.
- For this you reserve the largest portion of your data for traiining yur model. This is the data that your model learns from and you can use it to calculate model matrics such as training accuracy and training loss.
- Validation dataset is the second dataset or holdout dataset created from your fully processed training dataset and you will use it to evaluate your model performance usually after each epoch or full pass though the training set.
- The purpose of this evaluation is to fine tune model hyper parameters and determine how well your model is able to generalise on unseen data.
- Here you can calculate the vvalidation accuracy and validation loss.
- After you have your datasets ready you can move on to model training and tuning.
- MOdel training and validation is highly iterative and typically happens over many experients during this your goal is to determine which combination of the data algo and hyper parameter results in the best performing model.
- FOr each combination that you choose you need to train the model and evaluate it against that holdout data set.
- You repeat these steps until you have a model that is performing well according to your objective metrics whether thats accuracy or something like an f1 score depending on what you are optimizing for.
- All of these iterations can take a lot of compute and you typically want to be able to rapidly iterate without running into bottlenecks so this where training at scale comes in.
- Cloud gives you access to on demand resources that allow you to train and experiement at scale without wait time or schduling training time on on premises resources that are often constrained or limited by GPU CPU or storage.
- WIthout resource limitation you can also further optimize your trainig tie using capabilities like
- Automl allows you to reduce the need for data scientist to build machine learning models
- It uses machine learning to automate machine learning workflow tasks such as (ingest and analyze dataset, prepare and transform and train and tune models)
-  You first provide your labeled dataset which includes the target that you are trying to predict. Then automl is going to automatically do some analysis of that data and determine the type of machine learning problem
-  Then automl will typically explore a number of algorithms and automatically select the algo that besy suits your ml problrm and your data.
-  oce utoml selects algorithm, it will automatically explore various data transformations that are likely to have an impact on the overall performance of your model
-  - Then it will automate the creation of those scripts that will be neccesary to perofm those data transformations across your tuning experiments.
   - Finally automl will select a number of hyper parameter configurations to explore over those trading iterations to determine which combinations of hyper parameters and feature transformation codes results in the best performing model
   - Automl capabilities reduces a lot of repetitive work in terms of building and tuning your models  through numerous iterations
   - Automl is all about the automation even if the automatuon doesnt get all the way there you can still use automl to reduce a lot of the repetitave work but still uses experts to focus on high value task like taking that automl output and applying their domain knowledge or doing additional feature engineering or using data scientist to evaluate and analyze the reults of that automl
   - considerations when selecting an implementation of automl
     - Depending on the implementation of AutoML that you choose or you're deciding to use, there may be a balance, in terms of iterating faster but still maintaining the transparency and control that you may be looking for. Some implementations of AutoML provide limited visibility into the background experiments, which may produce a really performant model, but that model is often hard to understand, explain, or reproduce manually. Alternatively, there are implementations of AutoML, that not only provide the best model, but they also provide all of the candidates and the full source code that was used to create that model. This is valuable for being able to understand and explain your model, but it also allows you to take that model and potentially further optimize it for extra performance by doing things like applying some of that additional domain knowledge or doing some additional feature engineering on top of the recommended feature engineering code. In this section, I walked through the task, or the steps, in the machine learning workflow that are often requiring a lot of resources, not only in terms of human time to perform these tasks, but also in terms of compute cycles or resource costs. Using solutions that take advantage of automated machine learning helps you avoid those challenges by using machine learning to automate either all or part of your model building activities.
    



### B] Build, train and deploy Ml pipeline using BERT:
In this section, you will build a custom model using BERT algorithm and build an end-to-end ml pipeline for a text review classifier. 
> Part A]: In this part, you will learn how to generate machine learning features from raw data and share those features with other teams using a scalable feature store. 
> Part B]: In this week you will learn how to use these features to train our model at scale in the cloud. Deep dive into Bert model architecture and learn how to build and train a custom BERT model which is usually a two-step pre-training model in an unsupervised learning and fine-tune model for a specific language task. 
> Part C]: 


### C] Optimize Ml models and deploy human-in-the-loop pipelines:
-
__Important links:__
- [Dataset](https://www.kaggle.com/datasets/nicapotato/womens-ecommerce-clothing-reviews)
- [AWS SDK](https://github.com/aws/aws-sdk-pandas)
- [AWS Glue](https://aws.amazon.com/glue/)
- [AWS Athena](https://aws.amazon.com/athena/)
- [Matplotlib](https://matplotlib.org/)
- [Seaborn](https://seaborn.pydata.org/)
- [Pandas](https://pandas.pydata.org/)
- [NumPy](https://numpy.org/)
- [AWS Wrangler](https://aws-sdk-pandas.readthedocs.io/en/stable/)
- [use case](https://aws.amazon.com/sagemaker/canvas/customers/#samsung)
