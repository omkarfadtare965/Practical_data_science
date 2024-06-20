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
__Difference between Data lake and Data warehouse:__
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

![image](https://github.com/omkarfadtare/Practical_data_science/assets/154773580/a4df9583-d801-421b-ac1b-37e82103bbea)

![image](https://github.com/omkarfadtare/Practical_data_science/assets/154773580/39244b29-7303-4ab3-827a-e3c9d973ee7f)

![image](https://github.com/omkarfadtare/Practical_data_science/assets/154773580/7613eb99-7f6b-4f4e-b10d-5e47e3edd61c)

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
- After completing an Amazon SageMaker Autopilot job, several resources and artefacts are automatically generated, including data transformation and configuration code, data exploration, and candidate generation notebooks. These notebooks provide visibility into the data exploration activities, detailing steps for each candidate model, including data preprocessors, algorithms, and hyperparameter settings. The generated artefacts help in refining models and are stored in the specified S3 bucket. Autopilot runs multiple experiments with various combinations of data transformation codes, algorithms, and hyperparameters to identify the best-performing model. Trained model artefacts and a leaderboard with metrics for each candidate are also stored in S3. Notebooks are accessible from both S3 and the SageMaker Studio console.
- Autopilot generates multiple model candidate pipelines which basically refers to an automatically generated sequence of steps that includes data preprocessing, feature engineering, model training, and hyperparameter tuning.

## Model Hosting: Deploying ML model for consumption

![image](https://github.com/omkarfadtare/Practical_data_science/assets/154773580/2acbf119-324e-472d-a2f0-a9859a1a0eaa)

- Autopilot in Amazon SageMaker helps identify the best-performing model for real-time predictions, which is crucial for handling customer feedback in e-commerce. By capturing real-time customer feedback from diverse channels such as emails, chat messages, calls, mobile app posts, social networks, and partner websites an e-commerce company can predict sentiment in real time. This enables swift responses to negative reviews, such as automatically engaging customer support or addressing product issues by removing problematic items from the catalogue. This proactive approach helps the business stay attuned to market trends and customer behaviour, allowing for timely interventions that can enhance customer satisfaction and prevent issues from escalating.
- Real-time prediction requires a model serving stack that includes your trained model and a hosting stack, typically involving a proxy or web server that interacts with your model serving code, allowing applications to make real-time API requests. SageMaker model hosting simplifies this process by managing instance types, and Docker containers for inference, while also supporting automatic scaling to effectively handle varying workloads.
- When using SageMaker model hosting, you choose the instance type and specify the count, along with selecting a Docker container image for inference. SageMaker then handles the creation and deployment of the model to the endpoint.

__Deploying a candidate pipeline generated by Autopilot in Amazon SageMaker Inference pipeline:__
- The inference pipeline allows you to host multiple models behind a single endpoint in SageMaker. This means you can deploy and manage multiple stages of your machine learning process (like data transformation, model inference, and post-processing) under one endpoint.
- Autopilot generates candidate pipelines combining feature engineering, algorithms, and hyperparameter configurations. The best-performing pipeline model can be deployed using SageMaker's inference pipeline feature, which allows hosting multiple stages (like data transformation, model inference, and post-processing) under one endpoint. However, the Inference pipeline has multiple containers. These include:
  > ___`Data transformation container`___ will perform the same transformations on your dataset that were used for training. It ensures that your prediction request data is in the correct format for inference. This container processes and transforms the input data so that it matches the format expected by the model, allowing for accurate and reliable predictions.
  
  > ___`Algorithm container`___ contains the trained model artefacts that were selected as the best-performing model based on your hyperparameter optimization tuning jobs. 
  
  > ___`Inverse label transformer container`___ is used to post-process predictions into a readable format that can be understood by the application consuming the model's output. It specifically converts numerical predictions, which may represent categories or labels in a machine-readable format (such as indices or one-hot encodings), back into their original non-numerical label values that are meaningful to humans.
- When an inference request is sent, data first undergoes preprocessing by the data transformation container to ensure it's in the correct format. The preprocessed data then flows sequentially through the remaining containers in the pipeline. Finally, the output of the last container in the pipeline (in this case, the inverse label transformer) is returned as the result to the client application that made the inference request.

![image](https://github.com/omkarfadtare/Practical_data_science/assets/154773580/2fdf26b4-5fc3-4138-800b-81f2dba6390b)

## Built-in Algorithms:
__Benifits of using Built-in algorithms:__
- Built-in algorithms require minimal coding experience. You can quickly start running experiments by simply providing the input dataset, setting hyperparameters, and specifying the compute resources (e.g., number and type of compute instances).
- They enable rapid prototyping and validation of ideas without the need for extensive development work. This accelerates the initial stages of machine learning projects.
- These algorithms are pre-tuned for common machine learning tasks, offering optimized performance out of the box. They are designed to efficiently handle large datasets and complex computations.
- Built-in algorithms are designed to leverage available hardware effectively, including support for GPUs and multi-instance scaling. This ensures efficient use of computational resources and faster processing times.
- They can scale seamlessly across multiple compute instances, which is particularly useful for handling large datasets and high-throughput requirements without additional configuration.
- Built-in algorithms are well-integrated with the broader ecosystem of machine learning platforms (such as AWS SageMaker). This ensures seamless workflows from data ingestion to model deployment and monitoring.
- These algorithms come pre-tuned for a variety of standard machine learning tasks such as classification, regression, clustering, and recommendation systems. This reduces the need for extensive hyperparameter tuning and custom configurations.
- Using built-in algorithms ensures a consistent and reliable approach to machine learning, as they are thoroughly tested and maintained by the platform providers.
- Built-in algorithms come with comprehensive documentation, examples, and community support, making it easier to troubleshoot issues and implement best practices.
- Machine learning platforms often ensure that built-in algorithms comply with security and regulatory standards, providing an additional layer of trust and safety for sensitive data and applications.
- By managing the underlying infrastructure and providing optimized performance, built-in algorithms can be more cost-effective than developing and maintaining custom algorithms.
- By handling the complexities of algorithm implementation and optimization, built-in algorithms allow data scientists and developers to focus more on business logic and the specific problem they are trying to solve.

__When to Choose Built-in Algorithms:__
- When you need to rapidly test and validate a machine learning concept.
- When you are addressing common problems like classification, regression, clustering, or recommendation systems.
- When your team lacks extensive machine learning expertise or coding skills.
- When you need to handle large datasets or require high-throughput processing.
- When you want to leverage optimized performance on available hardware, especially with large datasets.
- When your project has tight deadlines.

 __When to Choose Custom Code:__
- If your problem requires specific algorithms or techniques not available in the built-in offerings.
- When you need to perform custom preprocessing, feature engineering, or use a novel model architecture.
- When you need fine-grained control over model performance and resource utilization.
- When your solution must deeply integrate with other custom software or hardware components.
- When you are conducting cutting-edge research or developing new machine learning methodologies.
- When your project is expected to evolve significantly over time, requiring ongoing adaptation and enhancement.
 
__SageMaker Built-in Algorithms vs Script Mode vs Bring Your Own Container (BYOC):__
| ___`Feature/Criteria`___ | ___`SageMaker Built-in Algorithms`___                         | ___`	SageMaker Script Mode`___                                                                                           | ___`SageMaker Bring Your Own Container (BYOC)`___                                      |
|--------------------------|---------------------------------------------------------------|--------------------------------------------------------------------------------------------------------------------------|----------------------------------------------------------------------------------------|
| Ease of Use              | Very high - minimal setup required                            | Moderate - requires coding in supported frameworks                                                                       | Low - requires Docker container setup and management                                   |
| Customization            | Low - limited to predefined algorithms                        | Moderate - custom training scripts allowed                                                                               | Very high - full control over the environment                                          |
| Supported Frameworks     | Predefined algorithms in SageMaker                            | Popular ML frameworks (TensorFlow, PyTorch, MXNet)                                                                       | Any framework that can run in a Docker container                                       |
| Control Over Environment | Minimal control                                               | Moderate control (training script)                                                                                       | Full control (entire Docker container)                                                 |
| Setup Time               | Very quick                                                    | Moderate                                                                                                                 | High - need to build and manage Docker images                                          |
| Scalability              | Built-in scalability                                          | Scalable with some configuration                                                                                         | Scalable but requires more setup and management                                        |
| Performance Optimization | Pre-optimized for common tasks                                | Custom optimization possible through scripts                                                                             | Fully customizable optimization                                                        |
| Dependency Management    | Managed by SageMaker                                          | Customizable via training script                                                                                         | Fully customizable within Dockerfile                                                   |
| Use Cases                | Standard ML tasks like classification, regression, clustering | Custom ML tasks needing more control over training logic                                                                 | Specialized tasks needing full environment customization                               |
| Skill Level Required     | Beginner                                                      | Intermediate                                                                                                             | Advanced                                                                               |
| Example Algorithms       | Linear Learner, XGBoost, BlazingText, etc.                    | Custom TensorFlow or PyTorch models                                                                                      | Any model, including those requiring special libraries                                 |
| Deployment               | Simplified via SageMaker interface                            | More involved but handled by SageMaker infrastructure                                                                    | Most involved; user handles deployment using Docker                                    |
| Integration              | Seamless with SageMaker and other AWS services                | Good integration with some manual setup                                                                                  | Full control over integration, requires manual setup                                   |
| Use of GPUs              | Supported where applicable                                    | Supported, user must configure                                                                                           | Supported, user must configure                                                         |
| Automatic Scaling        | Supported                                                     | Supported                                                                                                                | Supported, but requires more configuration                                             |
| Cost                     | Typically lower due to simplified management                  | Variable depending on custom setup                                                                                       | Potentially higher due to custom management and setup                                  |
| Best for                 | Standard ML tasks with minimal customization needs.           | Custom training logic with a need for more control over the training script while leveraging SageMaker’s infrastructure. | Highly specialized tasks requiring full control over the environment and dependencies. |

__Usecases and Algorithms:__
| ___`Task`___             | ___`Use case`___                                                            | ___`Built-in Algorithms`___      |
|--------------------------|-----------------------------------------------------------------------------|----------------------------------|
| Classification           | Predict if an item belongs to a category: an email spam filter              | XGBoost, KNN                     |
| Regression               | Predict numeric or continuous value: estimate the value of the house        | Linear regression, XGBoost       |
| Time series forecasting  | Predict sales on a new product based on previous sales data                 | DeepAR forecasting               |
| Dimensionality reduction | Drop weak features such as the color of the car when predicting its mileage | PCA                              |
| Anomaly detection        | Detect abnormal behaviour                                                   | RCF (Random cut forest)          |
| Clustering               | Group high/medium/low spending customers from transaction details           | Kmeans                           |
| Topic modeling           | Organize a set of documents into topics based on words and phrases          | LDA, NTM (Neural topic model)    |
| Content moderation       | Image classification                                                        | Full training, Transfer learning |
| Object detection         | Detect people and object in an image                                        | Object detection algorithm       |
| Computer vision          | Self driving cars identify  objects in their path                           | Semantic segmentation            |
| Machine translation      | Convert spanish to english                                                  | Seq-to-Seq                       |
| Text summarization       | Summarize a research paper                                                  | Seq-to-Seq                       |
| Speech to text           | Transcribe call center conversations                                        | Seq-to-Seq                       |
| Text classification      | Classify reviews into categories                                            | Blazing text classifier          |


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
