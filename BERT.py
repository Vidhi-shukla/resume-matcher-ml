# Skills Dictionary
skill_lemma_dict = {
    # Programming Languages
    "python": ["python", "py", "pandas", "numpy", "scipy", "flask", "django"],
    "java": ["java", "jvm", "spring", "hibernate", "maven", "gradle"],
    "javascript": ["javascript", "js", "node.js", "react.js", "vue.js", "angular", "typescript"],
    "csharp": ["c#", ".net", "asp.net", "entity framework"],
    "cpp": ["c++", "cpp", "qt", "boost"],
    "php": ["php", "laravel", "symfony"],
    "ruby": ["ruby", "rails"],
    "go": ["go", "golang"],
    "swift": ["swift", "ios"],
    "kotlin": ["kotlin", "android"],
    "scala": ["scala", "akka", "play framework"],
    "rust": ["rust"],
    "perl": ["perl"],

    # Frameworks and Libraries
    "react": ["react", "react.js", "react native"],
    "angular": ["angular"],
    "vue": ["vue", "vue.js"],
    "jquery": ["jquery"],
    "bootstrap": ["bootstrap"],
    "express": ["express"],
    "spring boot": ["spring boot"],
    "tensorflow": ["tensorflow"],
    "keras": ["keras"],
    "pytorch": ["pytorch"],
    "scikit-learn": ["scikit-learn"],

    # Databases
    "sql": ["sql", "mysql", "postgresql", "oracle", "sql server"],
    "nosql": ["nosql", "mongodb", "cassandra", "redis", "neo4j", "couchdb"],
    "mysql": ["mysql"],
    "postgresql": ["postgresql"],
    "mongodb": ["mongodb"],
    "redis": ["redis"],
    "sqlite": ["sqlite"],
    "oracle": ["oracle"],
    "snowflake": ["snowflake"],
    "redshift": ["redshift"],
    "greenplum": ["greenplum"],
    "teradata": ["teradata"],

    # DevOps and Cloud Platforms
    "aws": ["aws", "ec2", "s3", "lambda", "rds"],
    "azure": ["azure", "azure devops"],
    "google cloud": ["google cloud", "gcp", "app engine", "kubernetes engine"],
    "docker": ["docker", "docker-compose"],
    "kubernetes": ["kubernetes", "k8s"],
    "jenkins": ["jenkins"],
    "travis ci": ["travis ci"],
    "gitlab ci": ["gitlab ci"],
    "circleci": ["circleci"],

    # Web Technologies
    "html": ["html"],
    "css": ["css", "sass", "less"],
    "rest api": ["rest", "restful", "json", "xml"],
    "graphql": ["graphql"],

    # Software Development Methodologies
    "agile": ["agile", "scrum", "kanban"],
    "devops": ["devops", "site reliability"],

    # Soft Skills
    "leadership": ["leadership", "management"],
    "communication": ["communication", "teamwork"],
    "problem-solving": ["problem-solving", "analytical skills"],
    "adaptability": ["adaptability", "flexibility"],
    "teamwork": ["teamwork"],
    "leadership": ["leadership"],
    "project management": ["project management", "pm"],
    "creativity": ["creativity"],
    "critical thinking": ["critical thinking"],
    "emotional intelligence": ["emotional intelligence", "eq"],
    "negotiation": ["negotiation"],
    "decision making": ["decision making"],
    
    # Machine Learning
    "machine learning": ["machine learning", "ml"],
    "deep learning": ["deep learning", "dl"],
    "reinforcement learning": ["reinforcement learning", "rl"],
    "supervised learning": ["supervised learning"],
    "unsupervised learning": ["unsupervised learning"],
    "semi-supervised learning": ["semi-supervised learning"],
    "natural language processing": ["natural language processing", "nlp"],
    "computer vision": ["computer vision"],
    "speech recognition": ["speech recognition"],
    "anomaly detection": ["anomaly detection"],
    "generative adversarial networks": ["gan", "generative adversarial networks"],
    "transfer learning": ["transfer learning"],
    "feature engineering": ["feature engineering"],
    "model optimization": ["model optimization"],
    "model deployment": ["model deployment"],
    "edge AI": ["edge ai", "edge computing"],
    "federated learning": ["federated learning"],
    "explainable AI": ["explainable ai", "xai"],

    # ML Frameworks and Libraries
    "tensorflow": ["tensorflow", "tf"],
    "keras": ["keras"],
    "pytorch": ["pytorch"],
    "scikit-learn": ["scikit-learn", "sklearn"],
    "pandas": ["pandas"],
    "numpy": ["numpy"],
    "scipy": ["scipy"],
    "matplotlib": ["matplotlib"],
    "seaborn": ["seaborn"],
    "xgboost": ["xgboost"],
    "lightgbm": ["lightgbm"],
    "opencv": ["opencv"],
    "spacy": ["spacy"],
    "nltk": ["nltk"],
    "gensim": ["gensim"],
    "huggingface": ["huggingface", "transformers"],
    "fastai": ["fastai"],
    "caffe": ["caffe"],
    "theano": ["theano"],
    "dlib": ["dlib"],
    "mlflow": ["mlflow"],
    "pycaret": ["pycaret"],
    "streamlit": ["streamlit"],
    "dash": ["dash"],
    
    # Time Series Analysis & Forecasting
    "time series": ["time series", "time series analysis"],
    "arima": ["arima"],
    "prophet": ["prophet"],
    "lstm": ["lstm"],
    
    # Deep Learning Specific Technologies
    "convolutional neural networks": ["cnn", "convolutional neural networks"],
    "recurrent neural networks": ["rnn", "recurrent neural networks"],
    "long short-term memory": ["lstm"],
    "transformers": ["transformers"],
    "bert": ["bert"],
    "gpt": ["gpt", "gpt-2", "gpt-3"],

    # Visualization and Reporting Tools
    "tableau": ["tableau"],
    "power bi": ["power bi"],
    "qlik": ["qlik", "qlikview", "qliksense"],
    "looker": ["looker"],
    
    # Big Data
    "big data": ["big data"],
    "hadoop": ["hadoop"],
    "spark": ["spark"],
    "kafka": ["kafka"],
    "hive": ["hive"],
    "flink": ["flink"],
    "elastic search": ["elastic search", "elasticsearch"],
    "solr": ["solr"],
    "cassandra": ["cassandra"],
    "hbase": ["hbase"],
    "neo4j": ["neo4j"],

    # Cloud and DevOps
    "aws": ["aws", "amazon web services"],
    "azure": ["azure"],
    "gcp": ["gcp", "google cloud platform"],
    "docker": ["docker"],
    "kubernetes": ["kubernetes", "k8s"],
    "jenkins": ["jenkins"],
    "ci/cd": ["ci/cd", "continuous integration", "continuous deployment"],
    "terraform": ["terraform"],
    "ansible": ["ansible"],
    "cloudformation": ["cloudformation"],
    "openstack": ["openstack"],

    # Data Engineering and ETL Tools
    "airflow": ["airflow"],
    "luigi": ["luigi"],
    "talend": ["talend"],
    "pentaho": ["pentaho"],
    "informatica": ["informatica"],

    # Miscellaneous
    "git": ["git", "version control", "svn"],
    "project management": ["jira", "trello", "asana"],
    "ui/ux": ["ui", "ux", "user interface", "user experience"],
    "cybersecurity": ["cybersecurity", "penetration testing", "encryption"],
    "data analysis": ["data analysis", "excel", "power bi", "tableau", "data analytics"],
    "data visualization": ["data visualization"],
    "data mining": ["data mining"],
    "statistical analysis": ["statistical analysis", "statistics"],
    "ethics in AI": ["ethics in ai", "ai ethics"],
    "quantum computing": ["quantum computing"],
    "blockchain": ["blockchain"],
    "augmented reality": ["augmented reality", "ar"],
    "virtual reality": ["virtual reality", "vr"],
    "internet of things": ["iot", "internet of things"],
    "robotics": ["robotics"],
    "drones": ["drones", "uav"],
    "penetration testing": ["penetration testing"],
    "blockchain": ["blockchain"],
    "cryptocurrency": ["cryptocurrency", "bitcoin", "ethereum"],
    "quantum computing": ["quantum computing"],
    "bioinformatics": ["bioinformatics"],
    "digital twin": ["digital twin"],
    "autonomous vehicles": ["autonomous vehicles", "self-driving cars"]
}

# Regex Skills Dictionary
skill_regex_dict = {
    "python": r"\b(python|py|pandas|numpy|scipy|flask|django)\b",
    "java": r"\b(java|jvm|spring|hibernate|maven|gradle)\b",
    "javascript": r"\b(javascript|js|node\.js|react\.js|vue\.js|angular|typescript)\b",
    "csharp": r"\b(c#|\.net|asp\.net|entity framework)\b",
    "cpp": r"\b(c\+\+|cpp|qt|boost)\b",
    "php": r"\b(php|laravel|symfony)\b",
    "ruby": r"\b(ruby|rails)\b",
    "go": r"\b(go|golang)\b",
    "swift": r"\b(swift|ios)\b",
    "kotlin": r"\b(kotlin|android)\b",
    "scala": r"\b(scala|akka|play framework)\b",
    "rust": r"\b(rust)\b",
    "perl": r"\b(perl)\b",
    "react": r"\b(react|react\.js|react native)\b",
    "angular": r"\b(angular)\b",
    "vue": r"\b(vue|vue\.js)\b",
    "jquery": r"\b(jquery)\b",
    "bootstrap": r"\b(bootstrap)\b",
    "express": r"\b(express)\b",
    "spring boot": r"\b(spring boot)\b",
    "tensorflow": r"\b(tensorflow)\b",
    "keras": r"\b(keras)\b",
    "pytorch": r"\b(pytorch)\b",
    "scikit-learn": r"\b(scikit-learn)\b",
    "sql": r"\b(sql|mysql|postgresql|oracle|sql server)\b",
    "nosql": r"\b(nosql|mongodb|cassandra|redis|neo4j|couchdb)\b",
    "mysql": r"\b(mysql)\b",
    "postgresql": r"\b(postgresql)\b",
    "mongodb": r"\b(mongodb)\b",
    "redis": r"\b(redis)\b",
    "sqlite": r"\b(sqlite)\b",
    "oracle": r"\b(oracle)\b",
    "snowflake": r"\b(snowflake)\b",
    "redshift": r"\b(redshift)\b",
    "greenplum": r"\b(greenplum)\b",
    "teradata": r"\b(teradata)\b",
    "aws": r"\b(aws|ec2|s3|lambda|rds)\b",
    "azure": r"\b(azure|azure devops)\b",
    "google cloud": r"\b(google cloud|gcp|app engine|kubernetes engine)\b",
    "docker": r"\b(docker|docker-compose)\b",
    "kubernetes": r"\b(kubernetes|k8s)\b",
    "jenkins": r"\b(jenkins)\b",
    "travis ci": r"\b(travis ci)\b",
    "gitlab ci": r"\b(gitlab ci)\b",
    "circleci": r"\b(circleci)\b",
    "html": r"\b(html)\b",
    "css": r"\b(css|sass|less)\b",
    "rest api": r"\b(rest|restful|json|xml)\b",
    "graphql": r"\b(graphql)\b",
    "agile": r"\b(agile|scrum|kanban)\b",
    "devops": r"\b(devops|site reliability)\b",
    "leadership": r"\b(leadership|management)\b",
    "communication": r"\b(communication|teamwork)\b",
    "problem-solving": r"\b(problem-solving|analytical skills)\b",
    "adaptability": r"\b(adaptability|flexibility)\b",
    "teamwork": r"\b(teamwork)\b",
    "project management": r"\b(project management|pm)\b",
    "creativity": r"\b(creativity)\b",
    "critical thinking": r"\b(critical thinking)\b",
    "emotional intelligence": r"\b(emotional intelligence|eq)\b",
    "negotiation": r"\b(negotiation)\b",
    "decision making": r"\b(decision making)\b",
    "machine learning": r"\b(machine learning|ml)\b",
    "deep learning": r"\b(deep learning|dl)\b",
    "reinforcement learning": r"\b(reinforcement learning|rl)\b",
    "supervised learning": r"\b(supervised learning)\b",
    "unsupervised learning": r"\b(unsupervised learning)\b",
    "semi-supervised learning": r"\b(semi-supervised learning)\b",
    "natural language processing": r"\b(natural language processing|nlp)\b",
    "computer vision": r"\b(computer vision)\b",
    "speech recognition": r"\b(speech recognition)\b",
    "anomaly detection": r"\b(anomaly detection)\b",
    "generative adversarial networks": r"\b(gan|generative adversarial networks)\b",
    "transfer learning": r"\b(transfer learning)\b",
    "feature engineering": r"\b(feature engineering)\b",
    "model optimization": r"\b(model optimization)\b",
    "model deployment": r"\b(model deployment)\b",
    "edge AI": r"\b(edge ai|edge computing)\b",
    "federated learning": r"\b(federated learning)\b",
    "explainable AI": r"\b(explainable ai|xai)\b",
    "tensorflow": r"\b(tensorflow|tf)\b",
    "keras": r"\b(keras)\b",
    "pytorch": r"\b(pytorch)\b",
    "scikit-learn": r"\b(scikit-learn|sklearn)\b",
    "pandas": r"\b(pandas)\b",
    "numpy": r"\b(numpy)\b",
    "scipy": r"\b(scipy)\b",
    "matplotlib": r"\b(matplotlib)\b",
    "seaborn": r"\b(seaborn)\b",
    "xgboost": r"\b(xgboost)\b",
    "lightgbm": r"\b(lightgbm)\b",
    "opencv": r"\b(opencv)\b",
    "spacy": r"\b(spacy)\b",
    "nltk": r"\b(nltk)\b",
    "gensim": r"\b(gensim)\b",
    "huggingface": r"\b(huggingface|transformers)\b",
    "fastai": r"\b(fastai)\b",
    "caffe": r"\b(caffe)\b",
    "theano": r"\b(theano)\b",
    "dlib": r"\b(dlib)\b",
    "mlflow": r"\b(mlflow)\b",
    "pycaret": r"\b(pycaret)\b",
    "streamlit": r"\b(streamlit)\b",
    "dash": r"\b(dash)\b",
    "time series": r"\b(time series|time series analysis)\b",
    "arima": r"\b(arima)\b",
    "prophet": r"\b(prophet)\b",
    "lstm": r"\b(lstm)\b",
    "convolutional neural networks": r"\b(cnn|convolutional neural networks)\b",
    "recurrent neural networks": r"\b(rnn|recurrent neural networks)\b",
    "long short-term memory": r"\b(lstm)\b",
    "transformers": r"\b(transformers)\b",
    "bert": r"\b(bert)\b",
    "gpt": r"\b(gpt|gpt-2|gpt-3)\b",
    "tableau": r"\b(tableau)\b",
    "power bi": r"\b(power bi)\b",
    "qlik": r"\b(qlik|qlikview|qliksense)\b",
    "looker": r"\b(looker)\b",
    "big data": r"\b(big data)\b",
    "hadoop": r"\b(hadoop)\b",
    "spark": r"\b(spark)\b",
    "kafka": r"\b(kafka)\b",
    "hive": r"\b(hive)\b",
    "flink": r"\b(flink)\b",
    "elastic search": r"\b(elastic search|elasticsearch)\b",
    "solr": r"\b(solr)\b",
    "cassandra": r"\b(cassandra)\b",
    "hbase": r"\b(hbase)\b",
    "neo4j": r"\b(neo4j)\b",
    "aws": r"\b(aws|amazon web services)\b",
    "azure": r"\b(azure)\b",
        "gcp": r"\b(gcp|google cloud platform)\b",
    "docker": r"\b(docker)\b",
    "kubernetes": r"\b(kubernetes|k8s)\b",
    "jenkins": r"\b(jenkins)\b",
    "travis ci": r"\b(travis ci)\b",
    "gitlab ci": r"\b(gitlab ci)\b",
    "circleci": r"\b(circleci)\b",
    "ci/cd": r"\b(ci/cd|continuous integration|continuous deployment)\b",
    "terraform": r"\b(terraform)\b",
    "ansible": r"\b(ansible)\b",
    "cloudformation": r"\b(cloudformation)\b",
    "openstack": r"\b(openstack)\b",
    "airflow": r"\b(airflow)\b",
    "luigi": r"\b(luigi)\b",
    "talend": r"\b(talend)\b",
    "pentaho": r"\b(pentaho)\b",
    "informatica": r"\b(informatica)\b",
    "git": r"\b(git|version control|svn)\b",
    "project management": r"\b(project management|pm)\b",
    "ui/ux": r"\b(ui|ux|user interface|user experience)\b",
    "cybersecurity": r"\b(cybersecurity|penetration testing|encryption)\b",
    "data analysis": r"\b(data analysis|excel|power bi|tableau|data analytics)\b",
    "data visualization": r"\b(data visualization)\b",
    "data mining": r"\b(data mining)\b",
    "statistical analysis": r"\b(statistical analysis|statistics)\b",
    "ethics in AI": r"\b(ethics in ai|ai ethics)\b",
    "quantum computing": r"\b(quantum computing)\b",
    "blockchain": r"\b(blockchain)\b",
    "augmented reality": r"\b(augmented reality|ar)\b",
    "virtual reality": r"\b(virtual reality|vr)\b",
    "internet of things": r"\b(iot|internet of things)\b",
    "robotics": r"\b(robotics)\b",
    "drones": r"\b(drones|uav)\b",
    "penetration testing": r"\b(penetration testing)\b",
    "cryptocurrency": r"\b(cryptocurrency|bitcoin|ethereum)\b",
    "bioinformatics": r"\b(bioinformatics)\b",
    "digital twin": r"\b(digital twin)\b",
    "autonomous vehicles": r"\b(autonomous vehicles|self-driving cars)\b"
}   

# PDF Text Extraction
import os
import PyPDF2

def extract_text_from_pdf(file_path):
    text = ''
    try:
        with open(file_path, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            for page in reader.pages:
                text += page.extract_text() + "\n"
    except Exception as e:
        print(f"Error processing file {file_path}: {e}")
    return text

def extract_texts_from_folder(folder_path):
    pdf_texts = {}
    for filename in os.listdir(folder_path):
        if filename.endswith('.pdf'):
            file_path = os.path.join(folder_path, filename)
            pdf_texts[filename] = extract_text_from_pdf(file_path)
    return pdf_texts

# Usage
folder_path = 'C:/Users/Vidhi/Documents/PROJECTS/RESUME MATCH/RESUME'  
pdf_texts = extract_texts_from_folder(folder_path)

# To print the texts
for filename, text in pdf_texts.items():
    print(f"Text from {filename}:")
    print(text)
    print("--------------------------------------------------\n")


# %% [markdown]
# ## Text Cleaning
# ___

# %%
import re

def clean_function(resumeText):
    # Normalize unicode characters
    resumeText = re.sub(r'[^\x00-\x7f]', r' ', resumeText)

    # Lowercase the text for uniformity
    resumeText = resumeText.lower()

    # Remove URLs
    resumeText = re.sub(r'http\S+', ' ', resumeText)

    # Remove email addresses
    resumeText = re.sub(r'\S*@\S*\s?', '', resumeText)

    # Remove RT, cc, and other Twitter-specific artifacts (if any)
    resumeText = re.sub(r'\brt\b|cc', ' ', resumeText)

    # Remove hashtags and mentions
    resumeText = re.sub(r'#\S+|@\S+', ' ', resumeText)

    # Remove punctuations and special characters
    resumeText = re.sub(r'[^\w\s]', ' ', resumeText)

    # Remove numbers or standardize them based on your need
    # Uncomment the next line to remove
    # resumeText = re.sub(r'\b\d+\b', ' ', resumeText)

    # Replace multiple spaces with a single space
    resumeText = re.sub(r'\s+', ' ', resumeText).strip()

    return resumeText

# %% [markdown]
# ## Skill Extraction
# ___

# %%
import spacy
import re

# Load the spaCy model
nlp = spacy.load("en_core_web_sm")

def skill_extraction(text, skill_lemma_dict):
    doc = nlp(text)
    extracted_skills = set()

    for token in doc:
        # Check each token against the skill variations using regex
        for skill, variations in skill_lemma_dict.items():
            for variation in variations:
                # Prepare regex pattern for the variation
                pattern = r'\b' + re.escape(variation) + r'\b'
                if re.search(pattern, token.text, re.IGNORECASE):
                    extracted_skills.add(skill)
                    break  # Break if a match is found to avoid redundant checks

    return extracted_skills


# Job Description Processing
import spacy
import re

nlp = spacy.load("en_core_web_sm")

def extract_skills_and_keywords(text, skill_regex_dict):
    doc = nlp(text)
    extracted_skills = set()
    general_keywords = set()

    # Regex-based skill extraction
    for token in doc:
        for skill, pattern in skill_regex_dict.items():
            if re.search(pattern, token.text, re.IGNORECASE):
                extracted_skills.add(skill)

    # General keyword extraction based on part-of-speech
    for token in doc:
        if token.pos_ in ['NOUN', 'PROPN', 'ADJ', 'VERB']:  # Adjust POS tags as needed
            # Add to general_keywords only if not already in extracted_skills
            keyword = token.text.lower()
            if keyword not in extracted_skills:
                general_keywords.add(keyword)

    return extracted_skills, general_keywords


job_description_text = input("Enter the job description: ")
cleaned_job_description = clean_function(job_description_text)
extracted_skills, general_keywords = extract_skills_and_keywords(cleaned_job_description, skill_regex_dict)
print("Extracted Skills from Job Description:", extracted_skills)
print("General Keywords from Job Description:", general_keywords)


# BERT Embedding and Similarity Calculation
from transformers import AutoTokenizer, AutoModel
import torch
import numpy as np
import torch
from sklearn.metrics.pairwise import cosine_similarity

tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
model = AutoModel.from_pretrained('bert-base-uncased')

def get_bert_embedding(text):
    inputs = tokenizer(text, return_tensors='pt', padding=True, truncation=True, max_length=512)
    outputs = model(**inputs)
    return outputs.last_hidden_state.mean(dim=1).squeeze()

job_desc_embedding = get_bert_embedding(cleaned_job_description)

# Generate embeddings for each resume
resume_embeddings = {}
for filename, text in pdf_texts.items():
    cleaned_text = clean_function(text)
    resume_embeddings[filename] = get_bert_embedding(cleaned_text)
    
from sklearn.metrics.pairwise import cosine_similarity

def calculate_similarity(embedding1, embedding2):
    return cosine_similarity(embedding1.reshape(1, -1), embedding2.reshape(1, -1))[0][0]

def calculate_similarity(embedding1, embedding2):
    # Detach PyTorch tensors and convert them to NumPy arrays
    embedding1 = embedding1.detach().numpy()
    embedding2 = embedding2.detach().numpy()
    
    return cosine_similarity(embedding1.reshape(1, -1), embedding2.reshape(1, -1))[0][0]

# Calculate similarity scores for each resume
similarity_scores = {}
for name, resume_emb in resume_embeddings.items():
    similarity_scores[name] = calculate_similarity(job_desc_embedding, resume_emb)

# Sort resumes by similarity score
sorted_resumes = sorted(similarity_scores.items(), key=lambda x: x[1], reverse=True)

for resume, score in sorted_resumes:
    print(f"Resume: {resume}, Score: {score}")

from transformers import AutoTokenizer, AutoModel
import torch
import numpy as np
import torch
from sklearn.metrics.pairwise import cosine_similarity

tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
model = AutoModel.from_pretrained('bert-base-uncased')

def get_bert_embedding(text):
    inputs = tokenizer(text, return_tensors='pt', padding=True, truncation=True, max_length=512)
    outputs = model(**inputs)
    return outputs.last_hidden_state.mean(dim=1).squeeze()

job_desc_embedding = get_bert_embedding(cleaned_job_description)

# Generate embeddings for each resume
resume_embeddings = {}
for filename, text in pdf_texts.items():
    cleaned_text = clean_function(text)
    resume_embeddings[filename] = get_bert_embedding(cleaned_text)
    
from sklearn.metrics.pairwise import cosine_similarity

def calculate_similarity(embedding1, embedding2):
    return cosine_similarity(embedding1.reshape(1, -1), embedding2.reshape(1, -1))[0][0]

def calculate_similarity(embedding1, embedding2):
    # Detach PyTorch tensors and convert them to NumPy arrays
    embedding1 = embedding1.detach().numpy()
    embedding2 = embedding2.detach().numpy()
    
    return cosine_similarity(embedding1.reshape(1, -1), embedding2.reshape(1, -1))[0][0]

# Calculate similarity scores for each resume
similarity_scores = {}
for name, resume_emb in resume_embeddings.items():
    similarity_scores[name] = calculate_similarity(job_desc_embedding, resume_emb)

# Sort resumes by similarity score
sorted_resumes = sorted(similarity_scores.items(), key=lambda x: x[1], reverse=True)

for resume, score in sorted_resumes:
    print(f"Resume: {resume}, Score: {score}")

# GUI
import tkinter as tk
from tkinter import filedialog, scrolledtext

# GUI Application
class ResumeMatcherApp:
    def __init__(self, root):
        self.root = root
        root.title("Resume Matcher")

        # Job Description Input
        tk.Label(root, text="Job Description:").pack()
        self.job_desc_text = scrolledtext.ScrolledText(root, height=10)
        self.job_desc_text.pack()

        # Folder Selection
        tk.Label(root, text="Select Folder with Resumes:").pack()
        self.folder_path = tk.Entry(root)
        self.folder_path.pack()
        tk.Button(root, text="Browse", command=self.browse_folder).pack()

        # Process Button
        tk.Button(root, text="Process Resumes", command=self.process_resumes).pack()

        # Output
        tk.Label(root, text="Matched Resumes:").pack()
        self.output_text = scrolledtext.ScrolledText(root, height=10)
        self.output_text.pack()

    def browse_folder(self):
        folder_selected = filedialog.askdirectory()
        self.folder_path.delete(0, tk.END)
        self.folder_path.insert(0, folder_selected)

    def process_resumes(self):
        # Clear output
        self.output_text.delete(1.0, tk.END)

        # Get and clean job description
        job_description = self.job_desc_text.get("1.0", tk.END)
        cleaned_job_description = clean_function(job_description)
        job_desc_embedding = get_bert_embedding(cleaned_job_description)

        # Process resumes in the selected folder
        folder = self.folder_path.get()
        pdf_texts = extract_texts_from_folder(folder)
        resume_embeddings = {filename: get_bert_embedding(clean_function(text))
                             for filename, text in pdf_texts.items()}

        # Calculate similarity scores for each resume
        similarity_scores = {name: calculate_similarity(job_desc_embedding, resume_emb)
                             for name, resume_emb in resume_embeddings.items()}

        # Sort and display resumes
        sorted_resumes = sorted(similarity_scores.items(), key=lambda x: x[1], reverse=True)
        for resume, score in sorted_resumes:
            self.output_text.insert(tk.END, f"Resume: {resume},  Score: {score}\n")

if __name__ == "__main__":
    root = tk.Tk()
    app = ResumeMatcherApp(root)
    root.mainloop()


