# Mental Health in the Tech Industry Project

![image](https://github.com/user-attachments/assets/046d700d-b25f-4629-934d-26f17916502a)


## Introduction

This project contains analysis of Mental Health in the Tech Industry dataset, downloaded from:
https://www.kaggle.com/datasets/anth7310/mental-health-in-the-tech-industry as a mental_health.sqlite file.
These surveys were designed to evaluate people's perceptions of mental health and the prevalence of mental health disorders within the technology industry.

## Dataset Information

This data is from Open Source Mental Illness (OSMI) using survey data from years 2014, 2016, 2017, 2018 and 2019. Each survey measures and attitudes towards mental health and frequency of mental health disorders in the tech workplace.

The SQLite database contains 3 tables with respecitive column names:

* **Survey** (PRIMARY KEY INT SurveyID, TEXT Description)
* **Question** (PRIMARY KEY QuestionID, TEXT QuestionText)
* **Answer** (PRIMARY/FOREIGN KEY SurveyID, PRIMARY KEY UserID, PRIMARY/FOREIGN KEY QuestionID, TEXT AnswerText)

## Data Analysis & Scope

<ins>The analysis is structured into different parts:</ins>

**1. Data Loading**
   * 1.1 Import necessary modules, libraries and packages
   * 1.2 Connecting to the SQL Database
   * 1.3 Explore the Database and Table Structure and get a general idea about the data
   * 1.4 Explore the relevant questions to be answered and work towards a final Pivot Query
   * 1.5 Sociodemographic Feature Analysis
   * 1.6 Mental Health Prevalence Analysis
   * 1.7 Create a Final Query for EDA

**2. Data Cleaning and Preprocessing with the Final DataFrame for EDA and Correlation Analysis**  
   * 2.1 Missing and Invalid Values  
   * 2.2 Verify Data Type for each column  
   * 2.3 Duplicate Analysis  
   * 2.4 Dealing and Treating the Outliers  

**3. EDA: Bivariant and Correlation Analysis**  

**4. Summary of EDA and Insights**  

**5. Improvements**  


<ins>Scope of this analysis:</ins>

It's important to note that our analysis specifically focused on a subset of the overall survey data, including only respondents from the United States who indicated they worked in the tech industry. This filtering decision allowed for more targeted insights into mental health factors within the US tech sector, but it also means our findings cannot be generalized to the global tech industry or to non-tech sectors. The original dataset contained respondents from multiple countries (including United Kingdom, Canada, Germany, and others) and both tech and non-tech companies, but our analytical focus was intentionally narrowed to reduce potential confounding variables related to different national healthcare systems, cultural attitudes toward mental health, and industry-specific workplace environments. This US tech industry focus should be considered when interpreting all demographic distributions and correlations presented in this analysis.

## Project Objectives and Expected Insights
- 📌Get a general overview of the dataset and understand its structure
- 📌Understand the distribution of data including identifying and addressing outliers
- 📌Undertake univariate analysis of the whole dataset including visualizations:
  * Demographic breakdowns (age, gender, company size, employment status)
  * Mental health condition prevalence in the tech industry

- 📌Undertake bivariate analysis including visualizations to explore key relationships:

* Correlation between mental health diagnosis, current conditions, and treatment 
* Relationship between company resources and mental health discussions 
* Impact of company size on mental health benefits and resources 
* Connection between remote work patterns and company characteristics 
* Work interference differences between treated and untreated conditions 


- 📌Identify potential gaps in mental health support systems, particularly:

* How company size mediates both remote work prevalence and mental health resource availability
* Treatment effectiveness in reducing work interference
* Barriers to mental health discussions in different workplace environments

- 📌Provide actionable insights for tech companies to improve mental health support:

* Targeted approaches for different company sizes
* Strategies to support remote workers who may have less access to formal resources
* Ways to bridge the gap between formal benefits and cultural comfort with mental health discussions

  

## Prerequisites

* Python 3.x
* Required Python packages:
  * pandas
  * numpy
  * seaborn
  * matplotlib
  * statsmodel
  * GoogleTranslator
  * pycountry
  * langid
  * plotly
* Jupyter Notebook
* SQLite

## Requirements

### Installation Instructions and Cloning the Repository

Follow these steps to set up the project environment and install the required dependencies:

1. Clone the repository:
    ```bash
    git clone https://github.com/LauraLechermann/SQL_Python_Project__Mental_Health_Tech_Industry.git
    ```
2. Navigate to the project directory:
    ```bash
    cd Mental_health_Tech_Industry
    ```
3. (Optional) Create and activate a virtual environment:
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```
4. Install the required dependencies:
    ```bash
    pip install -r requirements.txt
    ```
5. You are now ready to run the project!
   
7. Follow these steps to open and run the Jupyter Notebook:
   
   Start the Jupyter Notebook by running the following command in your terminal:
   ```bash
     jupyter notebook Mental_health_Tech_Industry.ipynb
   ```
 This will open Jupyter Notebook in your default web browser.


## Importing the original dataset into Jupyter Notebooks for EDA:

* Download the Mental Health dataset from: https://www.kaggle.com/datasets/anth7310/mental-health-in-the-tech-industry as a `mental_health.sqlite` file and save the file in the same directory as the Jupyter Notebook file
* Call `sqlite3.connect()` to create a connection to the database. The returned Connection object `conn` represents the connection to the on-disk database.
In order to  fetch results from SQL queries a database cursor, `con.cursor()` needs to be created:

```bash
# Connect to the database
conn = sqlite3.connect('mental_health.sqlite')
cursor = conn.cursor()

# Check file details
print(f"File exists: {os.path.exists('mental_health.sqlite')}")
print(f"File size: {os.path.getsize('mental_health.sqlite')} bytes")

# Check all tables
cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
tables = cursor.fetchall()
print("Tables in the database:")
for table in tables:
    print(f"- {table[0]}")
```
* Load the final Pivot Query/dataset into a Pandas DataFrame for further data processing
* Proceed with data cleaning using code that inspects duplicates, missing values and outliers before proceeding with the exploratory data analysis (EDA)

## Visualizations/Graphs

The Jupyter Notebook contains visualizations and graphs plotted with funtions that can be found in a separate `viz_utils.py` file. Each visualation function is called separately in the Jupyter Notebook file, e.g. when visualizing the age groups as a bar plot:

```bash
viz.plot_bar_age_groups(age_groups, filename='age_groups.png')
```
If needed, `filename='age_groups.png` can be used to save each plot as a png. file for further use. If this is not needed it can be removed.

When running the Jupyter Notebook file, make sure the `viz_utils.py` function is in the same directory as the Jupyter Notebook file to run the analysis successfully!
