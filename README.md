# Practical data science Specialization:
- In this course, you will learn how to take your data science project from idea to production. You'll discover how to build, deploy, and scale data science projects, serving thousands of models to millions of end-users. You will learn to analyze and clean datasets, extract relevant features, train models, and construct automated pipelines to orchestrate and scale your data science projects. You will also learn to tackle complex data science challenges with sophisticated tools.
- One of the biggest benefits of running data science projects in the cloud is the agility and elasticity it offers, allowing you to scale and process virtually any amount of data.

![image](https://github.com/omkarfadtare/Practical_data_science/assets/154773580/cc5d2bf9-5e51-4d90-928d-6da501560da6)

__Course [1]:__
- In course1, you will learn how to ingest the data into a central repository and explore the data using various tools, explore and analyze the dataset using interactive queries and learn how to visualize the results.
- You will perform exploratory data analysis and detect statistical data biases.
- Next, you will learn to train machine learning models using automated machine learning techniques and build a multi-class text classification model using state-of-the-art algorithms.

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




