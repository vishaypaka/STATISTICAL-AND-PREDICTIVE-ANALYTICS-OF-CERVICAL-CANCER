<h1>Statistical and Predictive Analytics of Cervical Cancer</h1>

<p>This repository contains the complete implementation of a project aimed at predicting cervical cancer. The project incorporates data cleaning, statistical analysis, and predictive modeling using various machine learning techniques. The outcomes of this project include insights on major predictors and an easy-to-use prediction tool.</p>

<h2>Project Overview</h2>
<p>The dataset (<code>cervicalcancer.csv</code>) comprises records related to cervical cancer risk factors. It has been cleaned and analyzed to identify significant predictors and develop a prediction model. The project follows a systematic approach, including data preprocessing, statistical analysis, and model evaluation to create a robust prediction system.</p>

<h2>Steps in the Project</h2>
<ol>
  <li><strong>Data Cleaning:</strong> The dataset is cleaned using Python to handle missing values, transform data types, and filter irrelevant columns. The cleaned dataset is saved for further analysis (<code>STEP 1 - Data Cleaning.py</code>).</li>
  <li><strong>Descriptive Statistics:</strong> Statistical analysis and data visualization are conducted in R to explore the data, perform correlation analysis, and identify key predictors (<code>STEP 2 - Descriptive Statistics.R</code>).</li>
  <li><strong>Predictive Modeling:</strong> Various machine learning models are implemented in Python to predict cervical cancer outcomes. The best-performing model is selected, and a prediction tool is developed (<code>STEP 3 - Predictive Modelling and Tool.py</code>).</li>
</ol>

<h2>Key Features</h2>
<ul>
  <li><strong>Data Cleaning:</strong> Removal of null values and unclean records, data type transformation, and handling missing data using statistical methods.</li>
  <li><strong>Statistical Analysis:</strong> 
    <ul>
      <li>Summary statistics and correlation analysis to identify relationships among variables.</li>
      <li>Visualizations such as histograms, density plots, and heatmaps to understand data distribution.</li>
    </ul>
  </li>
  <li><strong>Machine Learning Models:</strong>
    <ul>
      <li>Logistic Regression</li>
      <li>K-Nearest Neighbors</li>
      <li>Support Vector Machine (Kernel and Linear)</li>
      <li>Decision Trees</li>
      <li>Random Forest</li>
      <li>Gradient Boosting</li>
      <li>Bagging Aggregating Classifier</li>
    </ul>
  </li>
</ul>

<h2>Technologies Used</h2>
<ul>
  <li><strong>Languages:</strong> Python, R</li>
  <li><strong>Libraries:</strong> pandas, numpy, matplotlib, seaborn, scikit-learn (Python); ggplot2, dplyr, caret, corrplot (R)</li>
</ul>

<h2>How to Use</h2>
<ol>
  <li>Clone the repository and ensure all required libraries are installed.</li>
  <li>Run the scripts in the following order:
    <ul>
      <li><code>STEP 1 - Data Cleaning.py</code>: Set up the working directory and clean the data.</li>
      <li><code>STEP 2 - Descriptive Statistics.R</code>: Analyze the cleaned dataset and save the results.</li>
      <li><code>STEP 3 - Predictive Modelling and Tool.py</code>: Train machine learning models and use the prediction tool.</li>
    </ul>
  </li>
</ol>

<h2>Key Insights</h2>
<ul>
  <li>Significant predictors include age, number of pregnancies, and smoking history.</li>
  <li>Linear Support Vector Machine provided the highest accuracy among tested models.</li>
  <li>Statistical analysis and feature selection reduced overfitting issues.</li>
</ul>

<h2>Future Enhancements</h2>
<ul>
  <li>Extend the dataset with more records for better generalization.</li>
  <li>Incorporate deep learning techniques for improved accuracy.</li>
  <li>Develop an interactive web application for the prediction tool.</li>
</ul>

<p>For a detailed explanation, visit the <a href="https://mason.gmu.edu/~vpaka2/index.html">project website</a>.</p>
