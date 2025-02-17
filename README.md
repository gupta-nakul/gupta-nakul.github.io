# Data Science Student

## Soft Skills
- Problem Solving
- Effective Communication
- Teamwork and Collaboration
- Meticulous
- Curious to learn

## Education
**M.S., Computational Data Science	| Purdue University (_May 2025_)**
- Relevant Coursework includes Statistical Machine Learning, Computer Vision, Data Mining, Deep Learning, Applied Regression Analysis, Time Series and Applications, Applied Decision Theory and Bayesian Statistics

---

**B.Tech., Computer Science and Engineering | Indraprastha Institute of Information Technology Delhi (IIIT-Delhi) (_June 2021_)**
- Relevant Coursework includes Linear Algebra, Probability and Statistics, Calculus, Machine Learning, Information Retrieval, Computer Networks

## Work Experience
**Research Assistant @ Kelley School of Business, Indiana University (_November 2024 - Present_)**
- Conducted in-depth mediation analysis on forklift maintenance data from a major logistics company, identifying key relationships and patterns to optimize maintenance processes, resulting in a 15% decrease in downtime.
- Researched mediation analysis methods and assumptions, transforming raw data to fit the model requirements and uncovering insights that led to a 20% increase in efficiency.

---

**Data Engineer @ Scry AI (_June 2021 - July 2023_)**
- Led the design and development of CoAP Protocol in Java, using Object-Oriented Programming principles and best practices, to reduced data processing time by 30% for the data received from field-deployed sensors, working seamlessly with a team consisting of a Senior Developer and a new Software Engineer.
- Guided an intern by assigning sub-tasks related to backend improvements, enhancing my skills in people management and task prioritization. Supervised a new Software Engineer, assigning small tasks to familiarize them with the project flow and efficiently transferring knowledge.
- Designed and implemented a Data Retention Plugin for H2 database (SQL-based), reducing storage overhead and improving query efficiency.
- Managed the design, development, and testing of the MQTT Protocol flow to process around a million data points received from NEOM's smart meters using using Java, Spring Boot, and Kafka.
- Refactored Spring Boot-based REST APIs to update the MongoDB database, enhancing my debugging and troubleshooting skills. This optimization facilitated future updates to the APIs.

## Artificial Intelligence/Machine Learning Projects
**Hospital RAG Chatbot with Langchain & Neo4j**, **(_December 2024 - Present_)**

[Github](https://github.com/gupta-nakul/langchain-neo4j-rag-chatbot)
1. AI-Powered Hospital Chatbot for Efficient Healthcare Assistance:
Developed a Retrieval-Augmented Generation (RAG) chatbot using LangChain and Neo4j, enabling seamless Q&A across patient reviews (unstructured data) and hospital system records (structured data) in one interface. The chatbot leverages large language model (LLM) to provide accurate and context-aware responses in real time.
2. Notable Capabilities & Results:
  - Text-to-Cypher: Automatically converts user questions into Cypher queries for deeper insights.
  - Dynamic Few-Shot Prompting: Fetches example queries from a vector store, boosting accuracy and reducing effort.
  - Streamlined Workflow: Achieved faster response times and more informative answers for healthcare inquiries.
3. Containerized & Scalable Deployment:
Leveraged Docker Compose for easy setup of the chatbot services, Neo4j, and Streamlit UIs—ensuring consistent environments, straightforward scaling, and simplified maintenance for future expansions.

---

**Text-to-Image Synthesis using DF-GAN**, CSCI 657 **(_October 2024 - December 2024_)**

[Github](https://github.com/gupta-nakul/Text-to-Image-Synthesis-using-DF-GAN)
1. Transforming Text Descriptions into Realistic Images:
Developed a deep learning-based text-to-image synthesis model using DF-GAN, capable of generating high-resolution (256×256) images from textual descriptions. This system enhances AI-generated visual content by ensuring the generated images are semantically accurate and visually realistic.
2. One-Stage GAN Architecture with Enhanced Semantic Fusion:
  - Utilized Deep Fusion Blocks (DFBlock) to integrate text and image features effectively.
  - Implemented Matching-Aware Gradient Penalty (MA-GP) for improved semantic alignment between text and generated images.
  - Outperforms traditional stacked architectures by directly synthesizing high-quality images, reducing feature entanglements.
3. State-of-the-Art Performance in Image Generation:
  - Achieved Frechet Inception Distance (FID) of 14.90, surpassing AttnGAN and StackGAN++ in realism and diversity.
  - Improved Inception Score (IS) from 4.36 to 5.10, demonstrating better image clarity and alignment with input text.
  - DF-GAN’s lightweight architecture (19M parameters) makes it efficient yet highly effective for text-to-image tasks.

---

**NHANES Data Analysis: Association Between Caffeine Intake and Gallstone Formation**, STAT 512 **(_October 2024 - December 2024_)**

[Github](https://github.com/gupta-nakul/NHANES-Caffeine-Gallstone-Analysis)
1. Investigating the Link Between Caffeine Intake and Gallstones:
Analyzed NHANES 2017-2018 data to examine the association between caffeine consumption and gallstone formation, considering demographic, dietary, and lifestyle factors. Applied logistic regression models to explore this relationship across different caffeine intake levels.
2. Key Findings: No Clear Association, but Gender Matters:
The study found no significant direct association between caffeine intake and gallstone risk. However, gender played a key role, with men showing a potential protective trend at higher caffeine intake levels, while no clear trend was observed in women.
3. Statistical Insights: Risk Factors Beyond Caffeine:
  - Women had 2.5× higher odds of gallstones compared to men.
  - Older age & higher BMI significantly increased gallstone risk.
  - Black participants had a lower risk of gallstones than other racial groups.
  - Findings suggest that hormonal differences may influence gallstone development, warranting further research into gender-specific effects.

---

**Polyp Segmentation in Colonoscopy Images using Deep Learning**, INFO 518 **(_March 2024 - May 2024_)**

[Github](https://github.com/gupta-nakul/Polyp-Segmentation-in-Colonoscopy-Images)
1. Polyp Segmentation for Early Cancer Detection:
Developed a deep learning-based polyp segmentation system using DeepLabV3+ (ResNet50) and ResUNet++ to improve the detection of polyps in colonoscopy images. The project enhances early diagnosis of colorectal cancer by providing accurate and automated segmentation of polyps.
2. Advanced Techniques: Custom Augmentation & Stochastic Activations:
Applied custom data augmentation (random crop, cutout, brightness shift, etc.), expanding the dataset 31× to improve model robustness.
Integrated stochastic activation functions (ReLU, Leaky ReLU, ELU, Mish, etc.) to dynamically optimize feature extraction and improve model generalization.
3. Significant Performance Gains in Segmentation Accuracy:
  - ResUNet++ with stochastic activations achieved the highest accuracy, with a Dice Score of 0.8347 and IoU of 0.8153.
  - DeepLabV3+ showed a 7% improvement in Dice Score after custom enhancements.
  - Ensemble learning techniques helped reduce model variance and improved segmentation results.
  - Findings highlight the effectiveness of deep learning in medical image analysis, paving the way for improved clinical decision-making.

---

**Credit Card Fraud Transaction Detection**, CSCI 573 **(_October 2023 - December 2023_)**

[Github](https://github.com/gupta-nakul/fraud-detection)
1. Fraud Detection for Secure Transactions:
Developed a machine learning-based fraud detection system using Logistic Regression, XGBoost, and LightGBM to identify fraudulent credit card transactions. The project leverages data mining, feature selection (PCA), and class balancing techniques (SMOTE, weight balancing) to enhance fraud detection accuracy.
2. Advanced Feature Engineering & Model Optimization:
  - Applied PCA-based dimensionality reduction to retain essential transaction features while improving computational efficiency.
  - Addressed class imbalance using SMOTE and weight balancing to enhance model robustness in detecting fraud.
  - Evaluated multiple models to determine the best-performing fraud detection system, prioritizing recall to minimize false negatives.
3. Optimized Model Performance & Key Insights:
  - LightGBM emerged as the best model, achieving higher recall while balancing precision.
  - Weight balancing outperformed SMOTE, leading to better fraud detection with fewer false positives.
  - Fraudulent transactions were successfully detected with optimized precision-recall tradeoffs, reducing financial risk in digital transactions.

---

**Metaphor Detection**, CSCI 578 **(_October 2023 - December 2023_)**
- Performed data preprocessing and Exploratory Data Analysis (EDA) to clean and understand the data.
- Utilized Word2Vec and BERT embeddings to capture semantic meaning and context of the sentence.
- Developed multiple ML models, including XGBoost and Logistic Regression, to identify metaphors in text.
- Implemented Bidirectional LSTM as the hero model to leverage sequential information and achieve superior performance in identifying metaphors.

## Leadership Experience
**Senior Supervisor @ Indiana University Testing Center(_January 2024 - Present_)**
- Trained 10+ proctors, enabling skill development and decision taking; successfully guided 4 to become senior proctors and 2 to transition into supervisory roles within 4 months.
- Implemented real-time decision-making strategies to efficiently manage up to 1000+ incoming students daily.
- Boosted operational efficiency by 30% through effective student segregation strategies across multiple labs, reducing wait times and streamlining flow of students.

## Technical Skills
**Languages**
- Python
- Java
- R
- Cypher
- SQL

**Technologies**
- Neo4j
- PostgreSQL
- MongoDB
- Apache Spark
- Apache Kafka
- PyTorch
- Tensorflow
- Scikit-learn
- Amazon Sagemaker
- FastAPI
- Docker
- Git
- Numpy
- Pandas
- NLTK
- Jupyter Notebook
- Seaborn
- Matplotlib
- Maven
- RESTful API
- Spring boot Framework
- Retrievel-Augmented Generation (RAG)
- Large Language Models (LLMs)
- Natural Language Processing (NLP)
- Machine Learning (ML)
- Deep Learning (DL)
- Regression analysis
- Clustering techniques
- K Nearest Neighbors (KNN)
