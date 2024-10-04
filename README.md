# Resume Matcher Using BERT and SBERT Models

![IMG_202410050218160](https://github.com/user-attachments/assets/702242b3-6700-4b4e-b724-263d583186ee)


## Project Overview

**Resume Matcher** is a project designed to match resumes to job descriptions using state-of-the-art Natural Language Processing (NLP) techniques. The tool utilizes **BERT** (Bidirectional Encoder Representations from Transformers) and **SBERT** (Sentence-BERT) to capture the deeper semantic meaning in the text. By converting resumes and job descriptions into numerical embeddings, the system can efficiently compute similarity scores and rank resumes according to their relevance to a given job description. 

The key objective of this project is to automate and improve the hiring process by providing a data-driven solution that matches job requirements with the most suitable candidates.

## How It Works

1. **Text Extraction**:
   * Resumes are typically in PDF format and the project uses libraries like **PyPDF2** to extract the textual content from these PDFs.
   * The job descriptions are provided via text input or uploaded as a file.
   * This step ensures that both the job description and the resumes are ready for analysis by extracting all necessary textual information.

2. **Text Preprocessing**:
   * The extracted text from both resumes and job descriptions is cleaned using several preprocessing techniques:
     * Removal of special characters, punctuation, URLs, emails, and other irrelevant elements.
     * Conversion of text to lowercase to standardize the format.
     * Tokenization, which splits the text into individual words or subwords, depending on the model being used (BERT or SBERT).
   * This preprocessing step ensures that the text is in a clean, normalized form suitable for input into machine learning models.

3. **Embedding Generation**:
   * **BERT** and **SBERT** models are used to convert the preprocessed text into **embeddings**. These embeddings are high-dimensional vector representations that capture the meaning of the text.
   * **BERT** is a transformer model pre-trained on a large corpus of text. It can generate context-aware embeddings for each word in the resume and job description.
   * **SBERT** is an extension of BERT that generates sentence-level embeddings, making it particularly effective for comparing entire resumes to job descriptions.
   * These embeddings allow the system to understand not just the words used but the context and meaning behind them.

4. **Cosine Similarity**:
   * Once the embeddings are generated, **cosine similarity** is used to calculate the similarity between each resume and the job description.
   * Cosine similarity is a metric that measures how close two vectors (in this case, the resume and job description embeddings) are in high-dimensional space. A score closer to 1 indicates a strong match, while a score closer to 0 indicates a weak match.
   * This allows the system to quantify how well a candidate’s resume matches the job description based on the semantic meaning of their content.

5. **Ranking Resumes**:
   * The resumes are ranked based on their cosine similarity scores, with the highest-scoring resumes being the most relevant to the job description.
   * This ranking helps the recruiter quickly identify the top candidates for the position without manually reviewing each resume.
   * The ranked list can be displayed in the application’s GUI, allowing for easy interaction and further analysis.

## Features

* **Advanced NLP Models**: The project leverages cutting-edge NLP models, **BERT** and **SBERT**, for semantic understanding and matching.
* **PDF Resume Parsing**: Extracts text from resumes in PDF format using **PyPDF2**, making the tool versatile in handling common resume formats.
* **Text Cleaning and Preprocessing**: Cleans, tokenizes, and normalizes text to prepare it for deep learning models.
* **Cosine Similarity**: A powerful method for comparing the semantic similarity between job descriptions and resumes, providing highly accurate matching.
* **Resume Ranking**: Provides a ranked list of resumes based on their relevance to the job description, making the recruitment process faster and more efficient.
* **User Interface**: Includes a simple, user-friendly **Tkinter**-based GUI that allows users to input job descriptions and upload resumes for analysis.

## Dependencies

* **Transformers**: Provides pre-trained BERT and SBERT models for generating embeddings.
* **Sentence-Transformers**: Used for SBERT-based sentence-level embeddings and similarity calculations.
* **scikit-learn**: Used for calculating cosine similarity between resume and job description embeddings.
* **PyPDF2**: Extracts text from PDF resumes for processing.
* **Tkinter**: Used to build the user-friendly graphical interface that allows recruiters to interact with the tool.

## Results

The project demonstrates that **BERT** and **SBERT** models outperform traditional methods like **Bag-of-Words** in accurately matching resumes to job descriptions. The use of these advanced models provides deeper semantic understanding, which leads to more relevant and precise candidate matches.

* **BERT and SBERT models** effectively capture the contextual meaning of both resumes and job descriptions, resulting in higher accuracy when matching candidates to roles.
* **Cosine similarity** between the job description and resumes allows for efficient ranking, ensuring the most relevant candidates are prioritized.
* The **ranking system** enables recruiters to quickly identify top candidates, making the hiring process faster and more reliable.
* **SBERT** provides better results in sentence-level similarity comparisons, offering an advantage in matching resumes to more detailed or nuanced job descriptions.

In summary, **BERT** and **SBERT** models significantly improve the resume matching process compared to traditional methods, providing a more sophisticated and reliable approach to shortlisting candidates.
