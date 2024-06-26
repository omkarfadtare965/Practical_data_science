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
- [AWS Data Wrangler](https://github.com/aws/aws-sdk-pandas)
- [AWS Glue](https://aws.amazon.com/glue/)
- [AWS Athena](https://aws.amazon.com/athena/)
- [Matplotlib](https://matplotlib.org/)
- [Seaborn](https://seaborn.pydata.org/)
- [Pandas](https://pandas.pydata.org/)
- [NumPy](https://numpy.org/)
- [Measure Pretraining Bias - Amazon SageMaker](https://docs.aws.amazon.com/sagemaker/latest/dg/clarify-measure-data-bias.html)
- [SHAP](https://shap.readthedocs.io/en/latest/)
- [Amazon SageMaker Autopilot](https://aws.amazon.com/sagemaker/canvas/)
- [Word2Vec algorithm](chrome-extension://efaidnbmnnnibpcajpcglclefindmkaj/https://arxiv.org/pdf/1301.3781)
- [AWS SDK for pandas](https://aws-sdk-pandas.readthedocs.io/en/stable/)
- [usecases](https://aws.amazon.com/sagemaker/canvas/customers/#samsung)

## Build Train and Deploy Ml pipeline using BERT:
### Feature engineering:
- Feature engineering is the preocess of converting raw data from one or more resources into meaningful features that can be used for training machine learning models. You will apply your domain knowledge, business accuman and statistics to convert the features in to a features that can be  readily useful by a machine learning algorithm that you chose to solve a specific problem.
- You perform feature engineering with two main goals in mind, first You are preparing your data so that it best fits the machine learning algorithm of your choice. Second by preparing ypur dat a you are trying to imporve the performance of the machine learning model.
- There are typically three steps involved in feature engineering: Feature selection, feature creation and feature transformation. All these steps may or may not be applicable to your specific usecase depending on the raw datset that you start with.

 ### Feature engineering steps:
 - ___`Feature selection:`___ Here you identify the appropriate data attributes or features to include in your training dataset as well as filter out any redundant and irrelevant features
 - You perform feature selection with goal of reducing the feature dimentionality so that the reduced feature set can help train your model much more quickly.
 - How do you select features to include in your dataset? ne of the technique to use is feature importance score that we have already learned about. Which will indicate how imp or relevant each one of the feature is to the final model as indicated by the importance score.
 - Keep in mind that using feature importance score is only one of the technique that you can use to select appropriate features to include in your training dataset
 - ___`Feature creation`___ Here you can combine existing features ti create new features or you can infer new attributes from existing attributes. In our usecase you can infer product review sentiment based on existing feature rating. The idea here is by creating and using these new features you need your machine learning model to produce more accurate predictions.
 - ___`Feature transformation`___ This would involve calculating the missing feature values using a technique like imputation or scaling numerical feature values using techniques like standardization and normalization and finally converting non numerical features into numerical values so that the algorithm canmake sense of these non numerical features.
 - It also involves categorical encoding ad embedding, normalization, standardization imutation and one hot encoding.
### Next step: Pipeline
- Combined with the feature engineering steps along with the additional step to split the dataset into training and testing data you can built a feature engineering pipeline
- a generic feature engineering pipeline would look like
- ![image](https://github.com/omkarfadtare/Practical_data_science/assets/154773580/bc59cb3d-881e-46ba-8fae-c665eca00638)
- Start by selecting appropraiate features along with creating and selecting proper labels, The dataset is then balanced so that there is a correct representation from all classes of the labels this is followed by spliting the dataset into training validation and test dataset and finally you perform the transformation techniques that you have learned so for on these dataset.

### Spliting of the dataset:
- Typically in machine learning you use a data set for both train the model as well as evaluate the model. Start by spolitting up the entire dataset into training data nad test data a major portion of training data is used to train your machine learning model. during the training process a small portion of the training data called validation data is used to frequently evaluate the model and tune the hyperparameters.Now the test data is the data that the model has never seen during the training threfore the test data is used to evaluate the trained model.
- Input of feature engineering pipeline is a raw data and the output is a set of features that are ready to be used for training the machine learning model.

### Question:
- How do you take this generic pipeline and apply it to the specific usecase.

### BERT algorithm and transforming raw product review text data into embeddings. 
- BERT stands for bidirectional encoder representation from transformers which is a neural network-based technique for training NLP-based models

### Difference between balzing text classifier and BERT
- Balzing text is based on word2vec whereas BERT is based on transformer architecture
- Both blazing text and BERT generate word embedings however blazing text operates at word level whereas BERT oprates at senetence level. Additionally using a bidirectional nature of the trnasformer architecture BERT can capture context of the word. 
- Clarifiation: Blazing text learn word level embeddings for all the words that reincluded in training corpus. These vectors or embeddings are then projected into a high dimentional vector space. So similar words will generates a vectors that are close together and these are represented as very close to other in the learned vector space. The embedding that is generated by the blazing text for the word dress is close to regardless of the word dress word apears in the sentence the embedding generated for that particular sentence is alwas going to the same which means the blazing text is not really capturing the contex of the word in the senetnce.
- In contrast the input for BERT is not the word dress but the sentence itself. the output is once again the embedding but this time embdding is based on three indivicual components that is tokens segment and the position.
- For example lets say we have two sentences I love the dress. and I liove the dress but not the price. the context of the word dress is different in these two sentences Bert can take into considaration the words that come prior to the word dress as well as the words that follow the word dress.
- Using this bidirectional nature of the transformer architecture Bert is able to capture the context. SO the embeddings that are generated for the word dress in these two sentences will be completely different . However the length of these embeddings in these two sentences is going to be fixed.  With BERT you encode sequences. Below show the end to end of converting a sequence into a bert embeddings that can be readily used by the BERT algorithm.

![image](https://github.com/omkarfadtare/Practical_data_science/assets/154773580/c194d5dd-f36a-4aca-b74f-7bd807581718)

### BERT EXample:
- The input is and input sequence. The input sequence could also consist of two different senetnces.The step is to apply the word piece tokenization; is a technique used to segment words into subwords and is based on pretrained model with dimensions of 768. Tokens are generated form the input sentence. In addition to the tokens coming from the individual words of the sentence. You will also see special token CLS. CLS specifies that this is a classification problem, If my input sequence consisted of multiple sentences then i would see another special token SEP that seperates the from the individual senetences.
- Once i have the word piece tokens the next step is to apply token embedding, to determine the token embedding from the individual tokens all i have tolook in to token in the 768 dimension vector that i mensioned before here the token CLS gets an embedding of 101. Thatis the position of CLS in that 768 dimension.
- Next tspe is to perform segment embedding.Segement emedding gets much more important when there are multiple sentencs in the input sequence. The segment ID 0 represnts that a sentence is a first sentence in the sequence and similarly the segmanet embedding of 1 respresents that it is a second senetnce in the input sequence.
- Here I have only one sentence in input sequence so i for all the individual tokens i get a segment embedding of 0.
- Next step is to apply position embedding, the position emebedding determines the index position of individual token in the input sequence. Here, my input sequence consists of four tokens. So based on a zero-based index, you can see the position embedding tokens for all the tokens.
- Once I have the three individual embeddings, it's time to pull all of these together. The final step includes determining an element wise sum of the position, segment and token embedding that have been previously determined.
-  So the final embedding is of the dimension 1, 4, 768 in this particular case. And that makes sense because I started with one input sequence that consisted of three different words and I applied the Word Piece Tokenization that has pre-trained models of dimension 768.
-  
![image](https://github.com/omkarfadtare/Practical_data_science/assets/154773580/cd6494da-19c0-498a-bede-5a69ecbbfd83)

### Libraries and API to generate BERT embeddings from raw text programmatically:
- For this use the SCikit learn library, from the scikit learn you will use the RoBERTa tokenizer class. RoBERTa model is built on top of BERT model. It modifies few hyper parameter and the way the model is trained. It also uses a lot more training data than original bert model, this results in significant performnance improvement in variety of NLP task campared to the original BERT model.

> COde to use RoBERTa model:
```python
from transformers import RebertaTokenize
PRE_TRAINED_MODEL_NAME = 'roberta-base'

tokenizer = RobertaTokenizer.from_pretrained(PRE_TRAINED_MODEL_NAME)

def convert_to_bert_input_ids(...):
    encode_plus = tokenizer.encode_plus(
    review, add_Special_tokens=True, max_length=128, return_token_type_ids=False, padding='max_length, return_attension_mask=True, return_tensors='pt', truncation=True'

return encode_plus['input_ids'].flatten().tolist()
```
- A brief note on the maximum sequence length parameter. This is a hyper parameter that is available on both BERT and RoBERTa models. The max of sequence length parameter specifies the maximum number of tokens that can be passed into BERT model with a single sample. To determine the right value for this hyper parameter, you can analyze your data set. Here you can see the word distribution of the product review data set. This analysis indicates that all of our reviews consist of 115 or less words. Now, it's not a 1-1 mapping between the word count and the input token count, but it can be a good indication. the real challenge comes in when you have to generate these embeddings at scale.
- The challenge is performing feature engineering at scale, and to address the challenge, you will use Amazon SageMaker processing. Amazon SageMaker processing allows you to perform data related tasks such as, preprocessing, postprocessing, and model evaluation at scale. SageMaker processing provides this capability by using a distributed cluster.

![image](https://github.com/omkarfadtare/Practical_data_science/assets/154773580/78ad0e88-d8b5-4ebf-b57a-7a2c7676dc9d)

- By specifying some parameters, you can control how many notes and the type of the notes that make up the distributed cluster. Sagemaker processing job executes on the distributed cluster that you configure. Sagemaker processing provides a built-in container for Sklearn.
-  Sagemaker processing expects data to be in S3 bucket. So you specify the S3 location where you're on, input data is stored and Sagemaker processing execute the Sklearn script on the raw data. Finally, the output, which consists of the embeddings, is persisted back into an S3 bucket.
> Code to use SAgeMAker processing with sci-kit learn
```Python
from sagemaker.sklearn.processing import sklearnprocessor
from sagemaker.processing import processinginput, processingoutput

processor = SKLEarnprocessor(
             framework_version = 'SCIKIT_LEARN_VERSION, role = role, instance_type = 'ml.c5.4xlarge',instance_count=2')

processor.run(<parameters>)

code = 'preprocess-scikit-text-tobert.py'
inputs = [ProcessingInput(input_name = 'raw-input-data', source = raw_input_data_s3_uri, ... )]

outputs = [ProcessingOutput(output_name = 'bert-train', s3_upload_mode = 'EndOfJob', sources = s3 path)]
```

### FEature store:
- A lot of effort goes into feature engineering. It would save you a lot of time, if you can store the results of feature engineering efforts, and reuse those results, so that you don't have to run the feature engineering pipeline again and again. It would save time not only for you, but for any other teams in your organization, that may want to use the same data and same features, for their own machine learning projects.
- Feature store, at a very high level, is a repository to store engineered features.
- For such a feature store, there are three high level characteristics that you want to consider. First, you want the feature store to be centralized, so that multiple teams can contribute their features to this centralized repository.
- Second, you want the features from the feature store to be reusable. This is to allow reuse of engineered features, not just across multiple phases of a single machine learning project, but across multiple machine learning projects.
- And finally, you want the feature store to be discoverable, so that any team member can come in and search for the features they want, and use the search results in their own machine learning projects.
- Now, if you extend the feature engineering pipeline that you reviewed before to include a feature store, it would look just like this.

![image](https://github.com/omkarfadtare/Practical_data_science/assets/154773580/396950bc-c1b7-474a-8cf4-5e74cef24a76)

- Besides the high level characteristics of being centralized, being reusable, and discoverable, other functional capabilities for a feature store include the ability to create the feature store itself, as well as the ability to create and ingest individual features, retrieve the features, as well as delete the features once they are obsolete.
- you can architect design and build such a feature store using mechanisms, like a database for persistence and APIs for creating, retrieving, and deleting the features. Or, you can use a AWS tool like Amazon Sagemaker Feature Store, which is a managed service that provides a purpose-built feature store.

### Amazon SageMaker Feature Store
- Sagemaker Feature Store is a fully managed service that provides purpose-built feature store. SageMaker Feature Store provides you with a centralized repository to securely save and serve features from. SageMaker Feature Store provides you with the capabilities to reuse the features, not just across a single machine learning project, but across multiple projects.
-  A typical challenge that data scientist see is training an inference skew that could result from discrepancies in the data used for training and the data used for inferencing. Sagemaker Feature Store helps reduce the skew by reusing the features across training and inference traces and by keeping the features consistent.
- SageMaker Feature Store provides the capabilities to query for the features both in real time and batch. The ability to creating for features in real time suppose use cases such as near real time ML predictions. Similarly, the ability to look up features in batch mode can be used to support use cases, such as model training.
> To start using the Feature Store APIs:
```python
from sagemaker.feature_store.feature_group import  FeatureGroup

reviews_feature_group_name = "reviews+distilbert+max_seq_length_128"
reviews_feature_group = FeatureGroupname=..., feature_definations=..., sagemaker_session=sagemaker_session(

reviews_feature_group.create(s3_uri = "s3 path".format(bucket,prefix), record_identifier_name=record_identifier_feature_name, event_time_feature_name=event_time_feature_name, role_arn=role)

# Ingesting features to feature group
reviwes_feature_group.ingest(data_frame=df_records, max_workers=3, wait=True)

# to retrieve features:
reviews_feature_store_query = reviews_feature_group.athena_query()

reviews_feature_store_table = reviews_store_query.table_name

query_string = 'SQL Query'.format(reviews_feature_store_table)

reviews_feature_store_query.run(query_string=..., ...)
```
- This results the queried features in a DataFrame format. Now, you have an option to convert that DataFrame into a CSV file and save the CSV file wherever you need to. In fact, you can store the CSV file into an S3 location and use that as a direct input to a training job on SageMaker.
- So far, you have seen the APIs to use with SageMaker feature store. If you'd like more of a visual approach, you can view the feature store and the featured groups created in a SageMaker studio environment.
- the feature definitions capture the featured name and the feature type.  The SageMaker Studio environment also provides you with queries that you can use to interactively explore the feature groups. You can take the queries provided in this environment and run them in any query interface of your choice.
- you're storing the embeddings as one of the features and have the ability to creating the feature group to retrieve those input IDs.

### Train and debug a custom machine learning model:
- For the fine tuning of the RoBERTa model is a supervised learning step. You will use the engineered features from the product reviews data, together with the sentiment label, as your training data. You will validate the model performance after each epoch using the validation data set. An epoch is a full pass through the training data set.  In this step, you calculate the validation accuracy and the validation loss.
- Once you finish training your model, you will use the test data set the model hasn't seen before to calculate the final model metrics, such as test accuracy and test loss.

### Difference between Built in algorithms and Pre-trained models
- In pre-trained model You will provide specific text data, the product reviews data, to adapt the model to your text domain and also provide your task and model training code. Telling the pretrained model to perform a text classification task, with the three sentiment classes

![image](https://github.com/omkarfadtare/Practical_data_science/assets/154773580/e4e07baf-8860-4f82-88c8-6071ed577b5f)

### Model pretraining Model fine tunning:
- Pretraining is an unsupervised learning. In which it will only learn the vector representaion of the words.
- pretrained and all key models have been trained on large text corpus, such as large book collections or Wikipedia. In this unsupervised learning step, the model builds vocabulary of tokens, from the training data, and learns the vector representations. You can also pretrain NLP models on specific language data.
- There are many more pretrained models available, that focus on specific text domains and use cases:
  
![image](https://github.com/omkarfadtare/Practical_data_science/assets/154773580/1b2ba168-b1d5-45b1-9725-f3bfb704db7b)

### Model fine tunning:
- Think of fine tunning as a transfer learning in NLP. It's a machine learning technique where a model is trained on one task and then repurposed on a second related task.
- For example,  If you work with English product reviews as your training data, you can use an English language model, pretrained for example, on Wikipedia, and then fine tune it to the English product reviews.
- The assumption here is that the majority of words used in the product reviews have been learned already from the English Wikipedia. As part of the fine tuning step, you would also train the model on your specific NLP task. In the product reviews example, adding a text classifier layer to the pretrained model that classifies the reviews into positive, neutral, and negative sentiment classes.
- Fine tuning is generally faster than pretraining, as the model doesn't have to learn millions or billions of BERT vector representations. Also note that fine tuning is a supervised learning step, as you fit the model using labeled training data.

### Where to find pretrained models:
- Many of the popular machine learning frameworks, such as PyTorch, TensorFlow, and Apache mxnet, have dedicated model huts, or zoos, where you can find pretrained models.
- The open source NLP project, Hugging Face, also provides an extensive model hub with over 8,000 pretrained NLP models. If you want to deploy pretrained models straight into your AWS account, you can use SageMaker JumpStart to get easy access to pretrained text and vision models.
- JumpStart works with PyTorch Hub and TensorFlow Hub and lets you deploy supported models in one click into the SageMaker model hosting environment. JumpStart provides access to over a 100 pretrained vision models, such as Inception V3, ResNet 18, and many more. JumpStart also lists over 30 pretrained text models from PyTorch Hub and TensorFlow Hub, including a variety of BERT models.
- In one click, you can deploy the pretrained model in your AWS account, or you can select the model and fine tune it to your data set. JumpStart also provides a collection of solutions for popular machine learning use cases, such as, for example, fraud detection in financial transactions, predictive maintenance, demand forecasting, churn prediction, and more.
-  When you choose a solution, JumpStart provides a description of the solution and the launch button. There's no extra configuration needed. Solutions launch all of the resources necessary to run the solution, including training and model hosting instances.
-  After launching the solution, JumpStart provides a link to a notebook that you can use to explore the solutions' features. If you don't find a suitable model via JumpStart, you can also pull in other pretrained models via custom code.

### Implementing pretraining and fine tunnning with BERT models
- While you can use BERT as is without training from scratch, BERT uses word masking and next sentence prediction in parallel to learn and understand language. As BERT sees new text, the model masks 15 percent of the words in each sentence. BERT then predicts the masked words and corrects itself, meaning it updates the model weights when it predicts incorrectly. This step is called masked language model or masked LM. 
- Masking forces the model to learn the surrounding words for each sentence. At the same time, BERT is masking and predicting words, or to be more precise, input tokens. It is also performing next sentence prediction, or NSP, on pairs of input sequences. To perform NSP, BERT randomly chooses 50 percent of the sentence pairs and replaces one of the two sentences with a random sentence from another part of the document.
- BERT then predicts if the two sentences are a valid sentence pair or not. BERT again will correct itself when it predicts incorrectly. Both of those training tasks are performed in parallel to create a single accuracy score for the combined training efforts. This results in a more robust model capable of performing word and sentence level predictive tasks.
- this pre-training step is implemented as unsupervised learning. The input data is large collections of unlabeled text.
- BERT has already been pre-trained on millions of public documents from Wikipedia and the Google Books corpus, the vocabulary and learned representations are indeed transferable to a large number of NLP and NLU tasks across a wide variety of domains.
- In the fine-tuning step, you also configure the model for the actual NLP task, such as question and answer, text classification, or a named entity recognition. Fine-tuning is implemented as supervised learning and no masking or next sentence prediction happens. As a result, fine-tuning is very fast and requires a relatively small number of samples or product reviews, in our case.
-  For our usecase we will take the pre-trained RoBERTa model from the Hugging Face model hub and fine tune it to classify the product reviews into the three sentiment classes.

### Train a custom model with Amazon Sagemaker:
- you learned how to train our product reviews text classifier using the building SageMaker blazing text algorithm. This time I show you how to train or fine tune the text classifier with a custom model code for the pre trained bert model you pull from the hugging face model hub. This option is also called bring your on script or a script mode in SageMaker.

![image](https://github.com/omkarfadtare/Practical_data_science/assets/154773580/20dff86a-d9f4-4f8f-a867-6da3b6cc1722)

- To start training a model in SageMaker, you create a training job. The training job includes The URL of the amazon S3 bucket where you have stored the training data. The compute resources that you want SageMaker to use for the model training. Compute resources are ml compute instances that are managed by SageMaker.  The URL of the S3 Bucket where you want to store the output of the training job. The Amazon elastic container registry or Amazon ECR path, where the training code image is stored. SageMaker provides built in docker images that include deep learning framework libraries and other dependencies needed for model training and inference. Using script mode, you can leverage these pre built images for many popular frameworks, including TensorFlow, pyTorch, and Mxnet.
- After you create the training job, SageMaker launches the ml compute instances and uses the training code and the training data set to train the model. It saves the resulting model artifacts and other outputs in the S3 bucket you specify for the purpose.

![image](https://github.com/omkarfadtare/Practical_data_science/assets/154773580/d14bf5c9-eb4d-4b4a-8048-5c07ac763f6b)

> steps you need to perform:
  - 1) first you need to configure the training validation and test data set
  - 2) You also need to specify which evaluation metrics to capture, for example the validation loss and validation accuracy.
  - 3) Next you need to configure the model type of parameters such as number of epochs, learning rate etc.
  - 4) Then you need to write and provide the custom model training script used to fit the model.

1) data set and the evaluation metrics:
- You can use SageMaker training input class to configure a data input flow for the training. Below is the code to configure training input objects to use the training validation and test data splits uploaded to an S3 bucket.
```python
from sagemaker.inputs import TrainingInput

s3_input_train_data = TrainingInput(s3_data=path)
s3_input_validation_data = TrainingInput(s3_data=path)
s3_input_test_data = TrainingInput(s3_data=path)
```
- Define regex expression to capture the values of these metrics from the Amazon cloudwatch logs.
```python
metric_definitions = [
{'Name': 'validation:loss', 'Regex':'val_loss([0-9\\.]+)'}, {'Name': 'validation:accuracy', 'Regex':'val_Acc: ([0-9\\.]+)'}]
```
2) Configure the model Hyperparametrs:
- Model hyper parameters include, for example, number of epochs, the learning rate, batch sizes for training, and validation data, and more. One important type of parameter for bert models is the maximum sequence length. the maximum sequence length refers to the maximum number of input tokens you can pass to the bert model per sample. I choose the value of 128 because the word distribution of the reviews showed that one 100% of the reviews in the training data said have 115 words or less.
```python
hyperparametrs = {'epochs':3,'learning_rate':2e-5, 'training_batch_size': 256, 'train_steps_per_epochs': 50, 'validation_batch_size': 256, 'validation_steps_per_epochs':50, 'max_seq_length': 128}
```
3) Provide your custom training script:
- First you import the hugging phase, transform a library. Hugging face provides pretrained RobertaModel for sequence classification that already pre configured roberta for tax classification tasks, let's download the model conflict for this RobertaNodel. You can do this by calling RobertaConfig from pre-trained and simply provide the model name in this example, roberta-base. You can then customize the configuration by specifying the number of labels for the classifier. You can set non-labels to three representing the three sentiment classes. The ID to label and label to ID parameters. Let you map the zero based index to the actual class, label of -1 for the negative class. The label of 0 for the neutral class and the label of 1 for the positive class. You then download the pretrained RobertaModel from the hugging face library with the command. RobertaForSequenceClassification.
```python
from transformers import RobbertaModel, RobertaConfig
from transformers import RobertaForSequenceClassification

config = RobertaConfig.from_pretrained('roberta-base', num_labels=3, id2label={0: -1, 1:0, 2:1},
label2id={-1:0,0:1,1:2})

model = RobertaFOrSequenceClassification.from_pretrained('roberta-base',config = config)
model = train_model(model, ...)
```
- With a pre-trained model at hand, you need to write the code to fine-tune the model here called train model. Below is the code to fine tune the model using pytorch
```python
def train_model(model, train_data_loader, df_train, val_data_loader, def_val, args):
     loss_function = nn.CrossEntropyLoss()
     Optimizer = optim.Adam(param=model.parameters(), lr = args.learning_rate)
- Then you write the training code:
```python
for epoch in range(args.epochs):
    print('EPOCH -- {}'.format(epoch))
    for i, (sent, label) in enumerate(train_data_loader):
        if i< args.train_steps_per_epoch:
            model.train()
            optimizer.zero_grad()
            sent = sent.squeeze(0)
            output = model(sent)[0]
            _, predicted = torch.max(output, label)
            loss.backword()
            optimizer.step()
            
return model
```
4) FIt the model
```python
from sagemaker.pytorhc import PyTorch as PyTorchEstimator
estimator = PyTorchEstimator(
    entry_point = 'train.py',
    source_dir = 'src',
    role = role,
    instance_count=1,
    instance_type = 'ml.c5.9xlarge',
    framewor_version = <PYTORCH_VERSION>,
    hyperparameters = hyperparameters,
    metric_definitions = metric_definitions)
    
estimator.fit(...) # To start the fine tunning of the model
```
### DEbugging and profiling 
- Training machine learning models is difficult and often a OPEC process and especially training deep learning models usually takes a long time with several training iterations and different combinations of hyper parameters before your model yields the desired accuracy.
- system resources could be inefficiently used, making the model training expensive and compute intensive.
- Debugging and profiling your model training gives you visibility and control to quickly troubleshoot and take corrective measures if needed. For example, capturing metrics in real time during training can help you to detect common training errors such as the gradient values becoming too large or too small. Common training errors include vanishing or explode ingredients.
-  Deep neural networks typically learn through back propagation, in which the models losses trace back through the network. The neurons weights are modified in order to minimize the loss. If the network is too deep, however, the learning algorithm can spend its whole lost touch it on the top layers and waits in the lower layers, never get updated. That's the vanishing gradient problem.
-  In return, the learning algorithm might trace a series of errors to the same neuron resulting in a large modification to that neurons wade that it imbalances the network. That's the exploding gradient problem. Another common error is bad initialization. Initialization assigns random values to the model parameters. If all parameters have the same initial value, they received the same gradient and the model is unable to learn.
-  Initializing parameters with values that are too small or too large may lead to vanishing or exploding gradients again. And then overfitting, the training loop consists of training and validation. If the model's performance improves on a training set but not on a validation data set, it's a clear indication that the model is overfitting. If the model's performance initially improves on the validation set but then begins to fall off, training needs to stop to prevent the overfitting.
-  All these issues impact your model's learning process. Debugging them is usually hard and even harder when you run distributed training. Another area you want to track is the system resource utilization monitoring and profiling. System resources can help you answer how many GPU, CPU, network and memory resources your model training consumes more. Specifically, it helps you to detect and alert you on bottlenecks so you can quickly take corrective actions.
-  These could include I/O bottlenecks when loading your data. CPU or memory bottlenecks when processing the data and GPU bottlenecks or maybe underutilization during model training.

![image](https://github.com/omkarfadtare/Practical_data_science/assets/154773580/b548379a-9c44-47e7-af2a-313be17d2cce)

- If you encounter any model training errors or a system bottlenecks, you want to be informed, so you can take corrective actions. For example, why not stop the model training as soon as the model starts overfitting. This can help save both time and money and it's not only about stopping the model training when an issue is found, maybe you also want to send a notification via email or via text message in that case.

### Implementation of debugging and profiling
- SageMaker Debugger automatically captures real time metrics during the model training such as training and validation loss and accuracy, confusion matrices and learn ingredients to help you improve model accuracy. The metrics from Debugger can also be visualized in SageMaker Studio for easy understanding.
- Debugger can also generate warnings and remediation advice when common training problems are detected. Also Debugger automatically monitors and profiles your system resources such as CPU, GPU, network and memory in real time. And provides recommendations on reallocation of these resources.
- This enables you to use your resources more efficiently during the model training and helps to reduce costs and resources.
- Debugger captures real time debugging data during the model training and stores this data in your security S3 Bucket.  The captured data includes system metrics, framework metrics, and output tensors. System metrics include for example hardware resource utilization data such as CPU, GPU, and memory utilization. Network metrics as well as data input and output or I/O metrics. Framework metrics could include convolutional operations in the forward pass, batch normalization operations in backward pass. And a lot of operations between steps and gradient descent algorithm operations to calculate and update the loss function.
- And finally the output tensors. Output tensors are collections of model parameters that are continuously updated during the back propagation and optimization process, of training machine learning and deep learning models. The captured data for output tensors includes scalar values such as for accuracy and loss, and matrices for example representing weights, gradients, input layers and output layers.
- Now while the model training is still in progress, Debugger also reads the data from the S3 bucket and already runs a real time continuous analysis through rules.  list of Debugger building rules you can choose from.
![image](https://github.com/omkarfadtare/Practical_data_science/assets/154773580/393c031b-00ba-47e6-a079-0eb5b6c6fde6)
![image](https://github.com/omkarfadtare/Practical_data_science/assets/154773580/a0f2ac4f-9d76-41d8-a4f0-e1ebfa5690e6)
![image](https://github.com/omkarfadtare/Practical_data_science/assets/154773580/45653d51-9dcd-4621-b14e-30ea5a132fff)
![image](https://github.com/omkarfadtare/Practical_data_science/assets/154773580/e32ccd6b-ce7e-4d5e-ba9f-f1996b2445df)
![image](https://github.com/omkarfadtare/Practical_data_science/assets/154773580/614dd344-9be8-47a9-b41b-001f296c5548)
- You can also take corrective actions. In case Debugger detects an issue, for example, the model starts to over fit. You can use Amazon CloudWatch events to create a trigger to send you a text message, email you the status or even stop the training job.  You can also analyze the data in your notebook environment using the Debugger SDK. Or you can visualize the training metrics and system resources using the SageMaker Studio IDE.
![image](https://github.com/omkarfadtare/Practical_data_science/assets/154773580/b45f518e-a587-4c69-841c-3a986f09b22b)

> Code to leverage the building rules to watch for common training errors.
```python
from sagemaker.debugger import Rule, rule_configs
rules = [
    Rule.sagemaker(rule_config.loss_not_decreasing(),
    Rule.sagemaker(rule_config.overtraining()]

from sagemaker.pytorch import PyTorch as PyTorchEstimator
estimator = PyTorchEstimator(
    entry_point = 'train.py',
    ...,
    rules = rules
    )
    
# Profile system and framework matrics for your training jobs
from sagemaker.debugger import ProfileRule, rule_configs
rules = [
    ProfileRule.sagemaker(rule_config.LowGPUUtilization()),
    ProfileRule.sagemaker(rule_config.ProfileerReport(),
    ...,
    ]
    
# COnfig profiler
from sagemaker.debugger import ProfileConfig, FrameworkProfile
profiler_config = ProfilerConfig(
    system_monitor_interval_millis=500,
    framework_profile_params = FrameworkProfile(num_steps =10))
    
from sagemaker.pytorch import PyTorch as PyTorchEstimator
estimator = PyTorchEstimator(
    entry_point = 'train.py', ...,
    rules = rules,
    profiler_config = profiler_config)
```

- Select the rules you want to evaluate, such as loss_not_decreasing or model starts to over train.
- Then pass the rules with the rules parameter in your estimator. SageMaker will then start a separate processing job for each rule you specify in parallel to your training job.The processing job will collect the relevant data and observe the metrics.
- To profile the system and framework metrics for your training jobs, you need to perform very similar steps.First you select the rules to observe again. Debugger comes with a list of building rules you can select such as check for low GPU utilization.  If you select the ProfilerReport rule, the rule will invoke all of the building rules for monitoring and profiling. By default, Debugger collects system metrics every 500 milliseconds. And basic output tensors that is scalar outputs such as loss and accuracy every 500 steps. you can modify the configuration if needed.To enable the framework profiling, configure the framework profile params parameter Then pass the rules and profiler_config in the estimator as shown earlier.
- Note that the list of selected rules can contain both the debugging rules together with the profiling rules.

 ### how can you analyze the results? 
 - For any SageMaker training job, the Debugger profiler report rule invokes all of the monitoring and profiling rules and aggregates the rule analysis into a comprehensive report.You can download the Debugger profiling report while you're training job is running or after the job has finished from S3.
 -  At the top of the report Debugger provides a summary of your training job.

 ![image](https://github.com/omkarfadtare/Practical_data_science/assets/154773580/c28c5a64-172f-4d4c-a226-b107583dd4a3)

- In rule summary section, Debugger aggregates all of the real evaluation results, analysis, rule descriptions and the suggestions.

![image](https://github.com/omkarfadtare/Practical_data_science/assets/154773580/34263498-c410-474e-9fd7-1ad1e78fe780)

- The report also shows system resource utilization such as CPU and network utilization over time.

 ![image](https://github.com/omkarfadtare/Practical_data_science/assets/154773580/24d21da8-ca0a-45cb-843a-effcd83831c1)

- Debugger also creates a system utilization heat map.

![image](https://github.com/omkarfadtare/Practical_data_science/assets/154773580/deb9dbc0-af07-4579-ac81-238ad0f0c261)

- In the sample shown here, I have run the training job on a MLC59 X large instance which consists of 36 VCPU's. The heat map here shows how each VCPU was utilized over time. The darker the color, the higher the utilization. If you see resources being underutilized, you could scale down to use a smaller instance type, and save cost and run the training job more efficiently

### MLOps overview:
- Automated pipelines actually span all of the workflow steps, including ingest and analyze, prepare and transform, train and tune, and finally deploy and manage.
- How does the broader concept of MLOps relate to building automated machine learning pipelines? MLOps builds on DevOps practices that encompass people, process, and technology. 
- However, MLOps also includes considerations and practices that are really unique to machine learning workloads. All of these practices aim to be able to deliver machine learning workloads quickly to production while still maintaining high quality consistency and ensuring end-to-end traceability.

### key considerations in ensuring your models have a path to production(difference between SLDC and MLDC)

![image](https://github.com/omkarfadtare/Practical_data_science/assets/154773580/21bcb73a-cd1e-43bd-85c1-74df17fc143c)

- It's important to consider that the machine learning development life cycle is very different than the software development life cycle for a variety of reasons.
- First, the model development life cycle is difficult to plan for from a project management perspective. It typically includes longer experimentation cycles than you would see in a standard agile software development process.
- Also the development of machine learning models includes data tasks like feature engineering and data preparation. ou also have data processing code, as well as new inputs and artifacts to consider for versioning. You also have additional pipeline task as well.
-  When you start to look at automating the machine learning workflow, the inputs and artifacts that are generated across these tasks result in multiple disparate pipelines with dependencies that can be a bit more challenging, stitched together than a typical software development workflow.
- Second, some models exist by themselves where you might be manually reading prediction requests and getting responses through a batch process or even within your notebook on an ad hoc basis.  This is especially true in research environments.
- However, in many cases, a model is typically a small part of an overall solution that incorporates machine-learning. While that model is still a very key component to that solution, most often there is a need for other components that need to be built or integrated.
-  As an example, consider your product review use case and your model that is predicting the classes of sentiment for a product review. That model itself will be able to classify the sentiment related to a product, but you also need to consider how that prediction will actually be used and potentially integrated into other existing applications.
-  For this, there may be additional tasks like creating a rest API as a common interface for other applications to integrate with your model or even building applications that can respond to those reviews. This could mean creating automation to initiate back-end processes that allow for customer support engineers to quickly react and respond to any negative reviews.
-  This brings me to the third consideration where typically multiple personas span the machine learning development lifecycle, and all are really needed to ultimately be able to build, deploy, integrate, and operate a machine learning workload.
- This can create challenges as these personas often have competing priorities and needs. There may also be skill gaps in building an operating machine learning workloads.
- As an example, a data scientist may not have a traditional IT background. While they may be very comfortable in creating a model that meets the performance objectives that have been identified for your particular machine learning use case, they may not know how to host that model in a way that it can be consumed by other applications or other systems.
- In this case, there may be a need to have a deployment engineer that is also engaged to help in building out the infrastructure and the resources that are needed to operate and host that model.
- Also, I mentioned that you might need to integrate that hosted model with another application. In this case, you're likely to depend on a software engineer to perform that integration. If there isn't a cross-functional team with the same project goals in place, competing priorities and skill gaps across these personas make it really difficult to provide that path to production for your model.
- Finally, many teams have processes in place supporting different regulatory or even internal corporate requirements. This means that when you're creating your machine learning pipeline, sometimes you also need to be able to ensure that traditional practices can be included inside the steps of your pipeline.
- Something like change management as an example here. This may mean that within your pipeline, you're going to automatically open a change ticket anytime a new model gets deployed to production. Or maybe it's a manual approval that's required before your model can deploy to production.
- All of these processes may need to be incorporated inside your machine learning pipeline. 
- Considering and understanding all of these different aspects are key in ensuring you're able to provide a path to production of your model. 
-  I just covered the considerations for providing a path to production for your machine learning workloads and some of the common challenges that teams run into.
- MLOps aims to provide that path to production by reducing manual hand-offs between the steps in your workflow, increasing automation within those steps in your workflow, and then going a step further to orchestrate the steps across your workflow.
- But you don't want to just apply automation, you also want to improve the quality of your models. To do that, you need to establish mechanisms and quality gates inside your machine learning pipeline.

### Workflow tasks without MLOps:
- Here's an example of a workflow starting with data ingestion and analysis. Here, a data engineer may create a raw dataset and manually send it to a data scientist.
- Then a data scientist is typically going to iterate through their model-building activities. This includes performing feature engineering, and data transformations, as well as experimenting with multiple hyperparameters and algorithms across their different experiments as they run through their model training and tuning activities as well. They typically iterate through these activities until they have a candidate model that is performing well according to their evaluation metric.
- At that point, a data scientist may hand that model off to a deployment team or an ML engineer who's responsible for deploying that model. If there's been limited communication between these teams up until this point in time, I often see this part result in a lot of delays because that model is essentially a black box to that deployment engineer.
- This means there's very limited visibility into how that model was built, how you would consume that model, and then how you monitor that model.
- To add to that, traditional deployment teams may not have a lot of experience in deploying and operating machine learning workloads. Once the deployment engineer has deployed the model, a software engineer often needs to create or make changes to applications that are going to then use that model.
- Finally, someone ultimately needs to operate that model in production. This typically means ensuring the right level of monitoring is set up, which can be challenging as the team that's operating the model may not be familiar with machine learning workloads or how to monitor a model. This can also include things like identifying and setting up the model retraining strategy as well.
- As you can see here, having a disconnected workflow with multiple manual hand-offs and limited cross team collaboration could really slow down your ability to get a model to production quickly and the ability to continue to iterate on that model as needed.

![image](https://github.com/omkarfadtare/Practical_data_science/assets/154773580/0ba68279-11ff-48ba-8bea-0ca94e2d0979)


### Let's now take a look at a view that incorporates cross team collaboration with automation to reduce those hand-offs and delays in a workflow.
-  You can see in this view that you also incorporate a centralized Model Registry. the Model Registry holds key metadata and information about how your model was built, how the model performed with evaluation metrics. It's no longer a manual hand-off of a black box model. A Model Registry can also be used to trigger downstream deployment workflows as well.
-  Once that model has been registered and approved, it can then trigger that downstream automated pipeline to deploy your model. That deployment pipeline typically includes the steps that are needed to package your model and then deploy it to one or more environments. Depending on the use case, that deployment may require additional code or packaging for consumption.
- Often a software engineer will be needed here to provide the steps and the code that is needed to create or update the API that will then be used to create a RESTful interface or a commonly defined interface for integrating with other applications.
- Finally, for operating your model, the correct monitors need to be identified and put in place early through your cross-functional teams. These monitors include traditional systems or performance monitors, but they also include model specific monitors like checking for things like model quality drift or data drift.
- As you can see here, visibility into those monitors is provided back to the personas that are involved in creating that end-to-end workflow.
- This is because there are some aspects of that monitoring that different personas may need more visibility into. As an example here, some logs may be generated by model monitors that the data scientist or machine learning engineers want to have visibility into
- In this case it's key to make sure that you're providing that back as a feedback mechanism and providing that transparency and visibility to those personas.

![image](https://github.com/omkarfadtare/Practical_data_science/assets/154773580/fe2da7e5-b246-4c10-b9b7-e6439558b717)

- I've talked about automation within each of these workload steps, but how do you coordinate performing the steps across your workflow? This brings us to orchestration.

### Orchestration vs Automation
- With automation, you're typically looking at automating the step and the tasks within that step, that are required to accept an input or inputs and then ultimately produce some type of output or artifact.
- As an example, your data preparation step may include one or more tasks that can help you automate and produce the intended output. In this example, you have your raw dataset input, and that's automatically ingested by the data processing step that is then responsible for taking that raw data, and transforming it into the format that can be consumed by your algorithm.
- This step then produces an output artifact. In this case, it's your transformed data that can then be used and consumed by the next step in your pipeline. This would be your training and your validation datasets. It's important to not only automate these individual tasks, but also the end-to-end machine learning workflow as well.
- To automate the steps across your end-to-end workflow, you also need to add a layer that can provide overall orchestration, in defining when and how these individual steps with automated task are run.
- Automation is great for reducing cycle time and deploying more quickly, but what about improving and ensuring the quality of your model? Your orchestration layer can also provide those quality gates that I talked about a bit before.
- You can use this orchestration layer to implement quality gates between your steps to determine when the pipeline should proceed to the next step in your machine learning pipeline.

![image](https://github.com/omkarfadtare/Practical_data_science/assets/154773580/18ee4d81-e99c-4eb6-b1d7-1e6c111cab5b)


- ___Quality gates___ The term quality gates refers to having an automated, or manual checkpoint within your pipeline that gates whether the next step in your pipeline can run based on some type of criteria or a conditional step that you define.
- ___types of quality gates that can be used and included inside your machine learning pipelines___ As an example for data curation, you may have a quality gate that governs and restricts access to the data that can be used. For model building, you're typically setting up a minimum threshold for your model evaluation metrics. This typically means establishing a metric that you're trying to optimize for, so something like accuracy or F1 score.
- Then establishing the lower boundary for that metric as a gate to proceed to the next step in your deployment pipeline. For model deployment, you can use something like A/B deployment strategy where you have model version 1 that's deployed into production, and you want to slowly deploy model version 2 to serve a portion of the traffic that's reaching your model.
- You can then evaluate how model version 2 is performing relative to model version 1. If model version 2 is performing better than model version 1, you can then start to shift more traffic to that model version 2. For model integration, you want to make sure that the application that's consuming your model is able to get prediction requests bag
- In these tests you're often making sure to check for things like your inference code is synchronized with your model and potentially the API. Then finally for model monitoring, you want to monitor for standard metrics, so things like CPU, GPU, or memory utilization. However, you also want to set up those monitors that are monitoring specifically for your model.
- Things like data drift, which can indicate that the data that you used to train your model now looks much different than the actual ground truth data. In this case, you want to set up the monitor with alerts so that you can get notified of any potential issues with model performance. This allows you to take action such as start of retraining pipeline.
- there is a lot to consider when operationalizing your machine learning workloads and applying MLOps practices like, creating machine learning pipelines.

![image](https://github.com/omkarfadtare/Practical_data_science/assets/154773580/0212ed33-5d4f-4114-a894-4d46c5912ea7)

### Creating a Machine learning pipelines:
- We will cover creating machine learning pipelines as a way to iterate quickly, with reduced handoffs and reduce manual effort. quality gates into your pipeline so that you can not only iterate quickly, but also iterate with improved quality and traceability.
- If you look at the high level tasks in your machine learning workflow, when you create a pipeline, you aim to automate the task and orchestrate the sequence and conditions of running that task. You also need to consider your pipeline triggers.
- In the case of software development life cycles(SDLC), it's usually pretty clear when you have a commit to a source code repository, you're going to automatically start an automated bill. However in the case of a machine learning pipeline you have multiple potential triggers. Such as a change to algorithm, hyper parameters or code, or alternatively having new training data, which could also be another potential trigger for your machine learning pipeline.
- Another key goal in building out an effective pipeline is establishing traceability, so ideally you have a central view of how pipeline artifacts were built. This is important not only for visibility and support, but also in the event of needing to recover any resource or artifact that's part of your pipeline.
- Let's take a closer look at each of these tasks and the considerations for each.
- ___Data task or specifically Data ingestion for model development:___   I work with a lot of teams. Where in this view you reach out to your data engineer, you ask them for some data, sometimes you need to go through a series of security approvals as well. And then ultimately get a data set that you can use for your model building activities.
- However there are challenges with this model as you can imagine. First, it can really slow down model development time. Second, it can also result in limited traceability with these manual handoffs of data. Third, it also makes model development difficult in terms of automating any kind of retraining workflows.While you may be able to do some of those initial model build activities, retraining pipelines in this particular context are very difficult to support.
- It's generally recommended to establish a data lake that provides governed access to data scientists or automated processes. This allows data scientists to quickly start a model development activities, and it also ensures there's traceability for that data because you know which data scientists have checked out specific data sets. This also allows you to create a model retraining workflow as well where the consumer in this particular case is the actual deployment pipeline or the machine learning pipeline, as opposed to the data scientist.

![image](https://github.com/omkarfadtare/Practical_data_science/assets/154773580/7cd78239-baa4-4a24-8519-80eaa1e955a1)

![image](https://github.com/omkarfadtare/Practical_data_science/assets/154773580/49b72a2c-9073-402b-8b6e-1aca23c971c0)


- ___data pre-processing and feature engineering___ After you get the data, a typical machine learning pipeline starts with the task that's needed to then take that raw data. And then transform it into a format that the algorithm can understand and that you can use for your model training or building activities. As we've discussed in the previous session's, your data pre-processing and feature engineering, can include any process or set of tasks. Whether it's a Python script or even another model that's being used to transform your data into the features, that will ultimately be used for training your model.
- When you automate your data processing and feature engineering, you have a key input and that's your raw data. And this typically gets fed in through automation, and then that automation will extract and transform those features, ultimately producing artifacts that will be used in training. The artifacts that are produced in your data preparation step include your training, validation, and test data sets.
  
![image](https://github.com/omkarfadtare/Practical_data_science/assets/154773580/c987a15a-a7c1-4db3-beb5-67c19392bbe2)

- traceability being a benefit of machine learning pipelines. Traceability is partially a result of the versioning of the inputs and the outputs for each workflow step. For your data task, you have code versioning for the code that you used to transform the data, but you also have data versioning.
- So as you can see in this slide, an example here would be that maybe you have that raw data set on input that's version one, and then you perform some feature transformations to ultimately create your output artifacts. So those output artifacts in this case, have aversion associated with them as well. So in this case your training validation and test data sets are your artifacts and maybe be at version 13. And this could be because you've had multiple iterations of feature engineering and data transformations, until you got it into the format that you really want it to be. The key here is that all of your inputs and artifacts that are produced as part of an automated step, should have versions associated with them. 

![image](https://github.com/omkarfadtare/Practical_data_science/assets/154773580/703ce7e9-b14c-47d3-a36a-8a72f5177da7)

- Finally for your data task, there's a number of validation tests that could also be incorporated into your machine learning pipeline and automated to improve quality.
- A few examples here include data quality checks where you can implement automated steps that gather statistics about your input data to look for any signals of potential data quality issues. So things like a specific attribute having an unexpectedly large number of missing values on input.
- Another example here would be checking for indicators of statistical bias. Finally data schema is something that you can include in your automated checks as well. So in this case you can embed a quality check inside your pipeline that ensures that the data structure and schema that's coming in on input, is in the format is expected to be.
- The key here is not only performing these tasks but also automating them as part of your machine learning pipeline.
- ___Training model___  the output of your previous task, then becomes the input into the next task. So in this case the output was your training validation and test data sets, which will be used in training and tuning your model as well as evaluating your model.
- The output includes a number of potential candidates until you find the best performing model according to your evaluation criteria. it will typically include evaluation metrics such as training metrics and model validation metrics at a foundational level.

![image](https://github.com/omkarfadtare/Practical_data_science/assets/154773580/abe5c8fc-52f9-4136-bb48-fbf884aee2b0)

- ___Model deployment task___ For a model deployment task, you're taking that model artifact and you're deploying it for consumption. This can take two different forms, you can either deploy in batch mode where you're basically wanting to send in batch records for prediction, and then receive batch responses back.
- Or you can deploy your model for a real-time or persistent endpont. An endpoint can consistently serve prediction requests and responses through a serving stack.  And the serving stack typically includes a proxy, a web server, that can then accept and respond to your requests coming in on input.

![image](https://github.com/omkarfadtare/Practical_data_science/assets/154773580/17a34040-fb76-4e9b-89ff-1bdad578ba48)

- When you're setting up your model deployment task as part of your machine learning pipeline. It's important to understand how your model will be consumed when you're setting up that model deployment task as part of your pipeline. Because as you can see here, the surrounding resources that need to be built, will differ between these two different forms.
- ___Operating task___ Finally once you've deployed and likely integrated your model into a broader solution, you have your operating task to consider. The output of your previous set of tasks is a deployed model or models that are then available for consumption. 
- Although operating tasks are at the end of our workflow here, when you're setting up your pipeline you really need to be considering your operating tasks early so that they can be incorporated into your workload early.
- As an example if there's runtime data that you need to capture, you need to ensure that the code is in place to capture and log that data. As an example here you could be capturing the request for predictions that come in, as well as the prediction responses that get returned.
-  In this case for your product review use case, you may have a product review coming in that's entered through, say a web page and it ultimately gets classified as either negative, neutral or positive. You may want to capture that prediction request coming in, but you may also want to capture that prediction response going back out. And store it in a secondary data store to be able to perform additional reporting or analysis like determining whether a specific vendor supplying your product has potential quality issues.
-  Also your operating tests include setting up monitors and alerts so that you can ensure that your deployed model has ongoing monitors that look for signs of things like model degradation. And also monitor the health of the systems that are supporting your model. So looking for common system metrics like CPU utilization or GPU utilization, that are supporting your machine learning model.

![image](https://github.com/omkarfadtare/Practical_data_science/assets/154773580/2f09cd76-60b0-45ee-ba94-620746779e9e)


-  One common challenge I run into is being able to have central visibility into your machine learning workflow, across all of the different personas and roles that are typically involved in an end to end machine learning workflow.
- This is a challenge whether it's having visibility into the status of your pipeline, so knowing when a specific version of a model is deployed to production. Or having visibility into system performance for debugging or even having visibility into model performance to see how your model is performing over time.
-  In this case, dashboards can serve as a central feedback mechanism for your machine learning pipelines as well as your deployed models. 

![image](https://github.com/omkarfadtare/Practical_data_science/assets/154773580/71e49fed-2e20-483c-b511-31b09513df4c)

### Model Orchestration
- I've talked about the steps in creating your machine learning pipelines and the tests that can be automated within those steps, but how do you bring it all together into an end to m pipeline? This is where pipeline orchestration comes in because you need a way to not only automate the tasks within your steps, but you also need a way to coordinate those steps that are performed across your end to end workflow.
- There are a lot of great choices for pipeline orchestration but orchestration essentially allows you to manage the end to end traceability, of your pipeline. But focusing on using automation to capture the specific inputs and outputs of a given task, and the artifacts that are then produced by those tasks, and then bringing them all together in a controlled end to end pipeline.

### Model lineage and Artificate tracking
- Bringing all of your automated tasks and steps together in an end-to-end pipeline, allows you to trace the inputs that are used by each step in your pipeline, as well as the artifacts or outputs that are produced by each step in your pipeline.
- Model lineage essentially refers to understanding and tracking all of the inputs that were used to create a specific version of a model.
- There are typically many inputs that go into training a specific version of a model. These inputs include things like the version of the data that was used to train the model in combination with the versions of the code and the hyperparameters that were used to build the model. However, inputs also includes things like the versions of the algorithms or the frameworks that were used. Depending on how you're building your model, this can also include things like the version of your docker images that were used for training, as well as different versions of packages or libraries that were used. it's basically all of the data that tells us exactly how a specific version of a model was built.
  
![image](https://github.com/omkarfadtare/Practical_data_science/assets/154773580/7d2d8ad2-f8d6-4902-8db2-9380aba65969)

- For an example;  you can see all of the model inputs that were used to create version 26 of this model. Each of these inputs has a version or multiple versions associated with it. An input may even have some additional metadata as well.
-  As an example, for your Python code, you probably have a commit hash for the source code commit that was used to commit this particular piece of code. But you may also want to capture additional metadata, like the name of the source code repository, so that all of these inputs together are the main data points that allow you to capture the information and provide a complete picture about how this model was actually built.
-  You also typically want to capture information about the trained model artifact itself as well. Things like the evaluation metrics for that particular version, as well as the location of the model artifacts. As you can see, this is a lot of information to track
- Where does that information about model lineage get stored, and how do you capture all of this information as part of your machine learning workflow? This is where model registry comes in. A model registry is a central store for managing model metadata and model artifacts.
  
![image](https://github.com/omkarfadtare/Practical_data_science/assets/154773580/ee212e04-cb31-427f-89c5-8c4f97689722)

- When you incorporate a model registry into your automated pipeline, it can provide traceability and auditability for your models, allowing you to effectively manage models, especially when you begin to manage at scale and you have tens or hundreds or even thousands of models.
- A model registry also gives you the visibility into how each of the model versions was built. It also typically includes other metadata, such as information about environments where a particular version of a model is deployed into.
- Model registry is a centrally managed model metadata and model artifact tracks which models are deploy across environments
- Keep in mind though, that a model is one artifact that's produced as part of your machine learning pipelines. There's other outputs and artifacts that are produced that you also want to consider for complete end-to-end traceability.


### an example of artifacts that are produced in your machine learning workflow and why artifact tracking is so important.
- Artifact is the output of a step or task that can be consumed by the next step in pipeline or deployed directly for consumption.
- In below you can see your machine learning workflow with corresponding tasks. Let's assume that these tasks have been automated, and you're now orchestrating these tasks into your machine learning pipeline.
- or each task, you have a consumable artifact that becomes the input into the next task. Each of these artifacts has different versions associated with them. For your data task, your process training dataset is an artifact from this task. In your model-building task, your model artifact that is produced becomes input into your model deployment task. A machine learning pipeline really provides a consistent mechanism to capture the metadata and the versions of the inputs that are consumed by each step, as well as the artifacts that are produced by each step.

![image](https://github.com/omkarfadtare/Practical_data_science/assets/154773580/56f5c2b9-ef69-4a24-9a5c-c14204ec5bed)

___But why is all this so important?___
- operational efficiency is one key reason. When you need to go and debug something, it's important to know what version is deployed at any given time, as well as what versions of the inputs were used to create that deployable artifact or the consumable artifact.
-  It's also important for the reliability of your workload. Because what if, for example, a human goes in and inadvertently deletes a live endpoint? Without knowing exactly how that endpoint was built, it's difficult, if not impossible, to recover that endpoint without disruption in your service.

### How to create Machine learning pipeline with amazon sagemaker pipelines:
- Sagemaker Pipelines allows you to create automated workflows using a Python SDK, that's purpose-built for automating model-building tasks. You can also visualize your workflows inside Amazon SageMaker Studio. Pipelines also includes the ability to natively integrate with SageMaker Model Registry.  This allows you to capture some of that Model Metadata that I previously discussed, like the location of your training model artifact in S3, or key information about your trained model, so things like model evaluation metrics. Model Registry also allows you to choose the best performing model that you want to approve for deployment. Finally, SageMaker Projects allows you to extend your pipelines, and incorporate CI/CD practices into your machine learning pipelines. This includes things like source and version control for that true end-to-end traceability.

### Features or components of pipelines:
- ___Pipelines___ First, you have pipelines, which allows you to build automated Model Building workflows using your Python SDK. Again, these workflows can be visualized inside SageMaker Studio.
- ___SAgemaker Model registry___ Second, you have SageMaker Model Registry, which stores Metadata about the model and has built-in capabilities to include Model Deployment approval workflows as well.
- ___Sagemaker projects___ Finally, you have Projects which includes built-in project templates, as well as the ability to bring your own custom templates that establish and pre-configured a pattern for incorporating CI/CD practices into your Model Building Pipelines and your Model Deployment Pipelines.
- SageMaker Pipelines provides the ability to work with all three of these components.

![image](https://github.com/omkarfadtare/Practical_data_science/assets/154773580/1c655b9f-5cc6-45b8-8561-52e82d72f8af)


### Sagamker pipelines:

![image](https://github.com/omkarfadtare/Practical_data_science/assets/154773580/7d5aad76-4a5a-4e2c-a709-da05553f7b97)

- SageMaker Pipelines allows you to take those machine learning workflow tasks, that I've been talking about and automate them together, into a pipeline that's built through code.
-  A Python SDK is provided, so that you can build and configured these workflows. The pipeline visualizations again, which are similar to what you see here, are all provided through SageMaker Studio.
-  Pipelines provides a server-less option for creating and managing automated machine learning pipelines. Meaning, you don't have to worry about any of the Infrastructure or managing any of the servers that are hosting the actual pipeline.

![image](https://github.com/omkarfadtare/Practical_data_science/assets/154773580/1551f2a5-ccb5-4a78-a84d-671ae3b3b79f)





