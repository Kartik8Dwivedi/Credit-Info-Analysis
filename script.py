#!/usr/bin/env python
# coding: utf-8

# # Understanding the content of the notebook
# 
# Home Credit is an international consumer finance provider with operations in multiple European and Asian countries. That focus on lending primarily to people with little or no credit history.Established in 1997 as a non-banking lender in Czech Republic
# 
# ## Our Prime Objective 
# Predict how capable each applicant is of repaying a loan
# 
# ## Benifits to the company by this analysis
# They can carefully cater to those in need and create appropiate services for the appropiate people.

# ## Preparing the dataset
# The dataset to be analyzed application_train.csv will be downloaded from the official page of the dataset from kaggle

# In[1]:


#imports
import missingno as msno
import pandas as pd 
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline 
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)

# In[2]:


df=pd.read_csv("/kaggle/input/home-credit-default-risk/application_train.csv")
df.head()

# In[3]:


df.info()

# In[4]:


df.describe()

# In[5]:


#Checking missing values
df.isnull().sum().head(50)

# In[6]:


msno.matrix(df.sample(500))

# In[7]:


total = df.isnull().sum().sort_values(ascending=False) # Total rows that are null in each collumn
percent = (df.isnull().sum()/df.isnull().count()).sort_values(ascending=False) # Percentage of what's missing and total
missing_df = pd.concat([total, percent], axis=1, keys=['Total', 'missing_ratio'])
missing_df.head(50)

# # Setting the problem
# Based on the overview of the data, set our own issues and questions.

# 1. Which metrics increases the probability of a client paying the loan?
# 2. What type of clients have a probability of taking a loan?

# #  Data exploration
# We will explore data to solve the problem or question set in the above section. Creating at least five tables and graphs to use in our exploration.

# ## 1 - client with payment difficulties: he/she had late payment more than X days on at least one of the first Y installments of the loan in our sample, 0 - all other cases

# In[8]:


# How many loans have been payed?
colors = ['#1b9e77', '#a9f971', '#fdaa48','#6890F0','#A890F0']
paid_unpaid = df["TARGET"].value_counts().plot(kind='bar',color = colors)
a1 = df["TARGET"].value_counts()
print("1 - client with payment difficulties: he/she had late payment more than X days on at least one of the first Y installments of the loan in our sample, 0 - all other cases")
print(a1)

# In[9]:


# Gender os dataset
gender_dataset = df["CODE_GENDER"].value_counts().plot(kind='pie',autopct = '%1.0f%%',title='Gender distribution in the dataset')
df["CODE_GENDER"].value_counts()

# In[10]:


# Gender, those who pay
gender_no_pay = df.loc[df['TARGET']==1,'CODE_GENDER']
gender_no_pay.value_counts().plot(kind='pie',autopct = '%1.0f%%',title='Gender distribution for clients with paying difficulties')

# In[11]:


gender_no_pay.value_counts().plot(kind='bar',title='Gender distribution for clients with paying difficulties',color = colors[2:])

# In[12]:


# Gender distribution of clients without payment difficulties
gender_pay = df.loc[df['TARGET'] == 0, 'CODE_GENDER']
gender_pay.value_counts().plot(kind='bar', title='Gender distribution without paying difficulties')
plt.show()


# In[13]:


# pie version
gender_pay.value_counts().plot(kind='pie', autopct='%1.0f%%')
plt.title('Gender distribution without paying difficulties')
plt.ylabel('')
plt.show()

# In[14]:


# Family status of clients with payment difficulties/ Those having a family
family_pay = df.loc[df['TARGET'] == 1, 'NAME_FAMILY_STATUS']
family_pay.value_counts().plot(kind='pie', autopct='%1.0f%%')
plt.title('Family status distribution of clients with payment difficulties')
plt.ylabel('')
plt.show()

# In[15]:


# Those who own a family

family_not_pay = df.loc[df['TARGET'] == 0, 'NAME_FAMILY_STATUS']
family_not_pay.value_counts().plot(kind='pie', autopct='%1.0f%%')
plt.title('Family status distribution of clients without difficulties paying')
plt.ylabel('')  # Optional: hides the default y-label
plt.show()

# In[16]:


# How many children

family_pay = df.loc[df['TARGET'] == 0, 'CNT_CHILDREN']
family_pay.value_counts().plot(kind='bar', color=colors)
plt.title('Number of children distribution for no difficulties paying clients')
plt.xlabel('Number of Children')
plt.ylabel('Count')
plt.show()

# In[17]:


# How many children

family_not_pay = df.loc[df['TARGET'] == 1, 'CNT_CHILDREN']
family_not_pay.value_counts().plot(kind='bar', color=colors)
plt.title('Number of children distribution for clients with payment difficulties')
plt.xlabel('Number of Children')
plt.ylabel('Count')
plt.show()

# In[18]:


# how many Family members client have
family_members = df['CNT_FAM_MEMBERS'].value_counts().plot(kind='bar', color=colors)
plt.title('Number of family members distribution')
plt.xlabel('Family Members')
plt.ylabel('Count')
plt.show()

# In[19]:


# how many Family members client have
family_no_pay = df.loc[df['TARGET']==1,'CNT_FAM_MEMBERS']
family_no_pay.value_counts().plot(kind='bar',color=colors,title='Number of family members distribution for with difficulties paying clients')

# In[20]:


# how many Family members client have

family_pay = df.loc[df['TARGET']==0,'CNT_FAM_MEMBERS']
family_pay.value_counts().plot(kind='bar',color=colors,title='Number of family members distribution for without difficulties paying clients')

# In[21]:


def distribution(column, colors, difficulties,title,graph_type='bar'):
    # how many Family members client have
    distribution = df.loc[df['TARGET']==difficulties,column]
    return distribution.value_counts().plot(kind='bar',color=colors,title=title)

# In[22]:


# Income
distribution('NAME_INCOME_TYPE', colors,1, 'bar', 'INCOME TYPE for members with dificulties')


# In[23]:


# Income
distribution('NAME_INCOME_TYPE', colors,0, 'bar', 'INCOME TYPE for members without dificulties')


# In[24]:


#Occupation type
distribution('OCCUPATION_TYPE', colors,1, 'bar', 'Occupation TYPE for members with dificulties')


# In[25]:


#Occupation type
distribution('OCCUPATION_TYPE', colors,0, 'bar', 'Occupation TYPE for members without dificulties')


# In[26]:



distribution('NAME_EDUCATION_TYPE', colors,1, 'bar', 'Occupation TYPE for members with dificulties')


# In[27]:



distribution('NAME_EDUCATION_TYPE', colors,0, 'bar', 'Occupation TYPE for members without dificulties')


# In[28]:


#NAME_HOUSING_TYPE

distribution('NAME_EDUCATION_TYPE', colors,0, 'bar', 'Name housing TYPE for members without dificulties')


# In[29]:



distribution('NAME_EDUCATION_TYPE', colors,1, 'bar', 'Occupation TYPE for members with dificulties')


# In[30]:



distribution('NAME_EDUCATION_TYPE', colors,1, 'bar', 'Occupation TYPE for members with dificulties')

