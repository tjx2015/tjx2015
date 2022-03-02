# Databricks notebook source
# MAGIC %md
# MAGIC ## San Francisco Crime Data Analysis 

# COMMAND ----------

# DBTITLE 1,Executive Summary
# MAGIC %md 
# MAGIC ### Objective: 
# MAGIC 1. Preliminary analysis using historical record to understand criminal patterns and trends in San Francisco 
# MAGIC 2. Provide actionable recommendations to public health & safety agencies (law enforcement, suicide hotlines, etc.) and helpful guidance to local residents and tourists 
# MAGIC 
# MAGIC ### Data source: Police Department Incident Reports
# MAGIC - Link: https://data.sfgov.org/Public-Safety/Police-Department-Incident-Reports-Historical-2003/tmnf-yvry
# MAGIC - Time Period: January 2003 - May 2018
# MAGIC 
# MAGIC ### Critical Steps in the Analysis
# MAGIC - Section 0: Data collection and preprocessing
# MAGIC - Section 1: Crime category and quantity analysis
# MAGIC - Section 2: Geographical analysis
# MAGIC - Section 3: Weekly cycle analysis (case study: San Francisco downtown)
# MAGIC - Section 4: Monthly cycle analysis (period: 2015 - 2018)
# MAGIC - Section 5: Daily cycle analysis (case study: pre-holiday December 15 of 2015/2016/2017)
# MAGIC - Section 6: Daily cycle analysis (case study: top 3 dangerous district)
# MAGIC - Section 7: Resolution impact analysis 

# COMMAND ----------

# MAGIC %md 
# MAGIC ### Section 0: Data collection and preprocessing

# COMMAND ----------

# DBTITLE 1,Import package 
from csv import reader
from pyspark.sql import Row 
from pyspark.sql import SparkSession
from pyspark.sql.types import *
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import warnings

import os
os.environ["PYSPARK_PYTHON"] = "python3"

# COMMAND ----------

# DBTITLE 1,Import data
# Run with caution: loading may take long due to data size
# import urllib.request
# urllib.request.urlretrieve("https://data.sfgov.org/api/views/tmnf-yvry/rows.csv?accessType=DOWNLOAD", "/tmp/myxxxx.csv")
# dbutils.fs.mv("file:/tmp/myxxxx.csv", "dbfs:/laioffer/spark_hw1/data/sf_03_18.csv")
# display(dbutils.fs.ls("dbfs:/laioffer/spark_hw1/data/"))

data_path = "dbfs:/laioffer/spark_hw1/data/sf_03_18.csv"

# COMMAND ----------

# DBTITLE 1,Get DataFrame and SQL
from pyspark.sql import SparkSession
spark = SparkSession \
    .builder \
    .appName("crime analysis") \
    .config("spark.some.config.option", "some-value") \
    .getOrCreate()

df_opt1 = spark.read.format("csv").option("header", "true").load(data_path)
display(df_opt1)
df_opt1.createOrReplaceTempView("sf_crime")

# COMMAND ----------

# DBTITLE 1,Understanding the Data and Missing Data Check
#Missing value percentage.
df_opt1_pandas = df_opt1.toPandas()
df_opt1_pandas.info()
column_names = df_opt1_pandas.columns.tolist()
column_names_df = pd.DataFrame(column_names, index = column_names, columns = ['Name'])
total = df_opt1_pandas.isnull().sum().rename('Total')
percent = (df_opt1_pandas.isnull().sum()/df_opt1_pandas.isnull().count()).rename('Percent')
missing_data = pd.concat([column_names_df, total,percent], axis = 1)
display(missing_data)

# COMMAND ----------

# DBTITLE 1,Visualization of Missing Data
import matplotlib.pyplot as plt
import seaborn as sns
fig = plt.figure(figsize=(10,4))
sns.heatmap(df_opt1_pandas.isna(), cmap = ['#000080', '#FFC300'])

# COMMAND ----------

# MAGIC %md 
# MAGIC ### Section 1: Crime category and quantity analysis

# COMMAND ----------

# DBTITLE 1,Spark DataFrame version
q1_result = df_opt1.groupBy('category').count().orderBy('count', ascending=False)
display(q1_result)

# COMMAND ----------

# DBTITLE 1,Spark SQL version
q1_sql_result = spark.sql("SELECT category, COUNT(*) AS count FROM sf_crime GROUP BY category ORDER BY Count DESC")
display(q1_sql_result)

# COMMAND ----------

# DBTITLE 1,Visualization
import seaborn as sns
fig_dims = (10,4)
fig = plt.subplots(figsize=fig_dims)
spark_df_q1_plot = q1_result.toPandas()
chart = sns.barplot(x = 'category', y = 'count', palette= 'Blues_r',data = spark_df_q1_plot[0:10] )
chart.set_xticklabels(chart.get_xticklabels(), rotation=45, horizontalalignment='right', fontsize=10)
plt.xlabel('Types of Crimes')
plt.ylabel('# of crime incidents (2003 - May 2018) ')
plt.title('Crime Category Analysis')

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ### Takeaways:
# MAGIC Quantity should not be the only way to measure crimes. 
# MAGIC - Certain severe crimes such as murder, rape and missing person didn't even make it to the top 10 but should be categorized as severe and benchmarked against other cities
# MAGIC - Crimes such as theft and assaults, if with certain scale (in monetary value or lives) or when reaching a certain number, should be addressed systematically with government programs  
# MAGIC - Certain crimes are hard to define or a combination of several categories and thus grouped as a miscelaneous option called "other offenses". More in-depth analysis should be done 
# MAGIC 
# MAGIC In further investigation, propose prioritizing top items with different dimensions of consideration, such as 
# MAGIC - high quantity and consistency (theft)
# MAGIC - high severity (murder)
# MAGIC - public complaints (human feces on sidewalks in early 2018 grouped in vandalism)  
# MAGIC 
# MAGIC Futhermore, geographical analysis should be done on where crimes happen within San Francisco

# COMMAND ----------

# MAGIC %md
# MAGIC ### Section 2: Geographical analysis

# COMMAND ----------

# DBTITLE 1,Spark DataFrame Version
q2_df_result = df_opt1.groupBy('PdDistrict').count().orderBy('Count', ascending=False)
display(q2_df_result)

# COMMAND ----------

q2_df_result = df_opt1.groupBy('PdDistrict').count().orderBy('Count', ascending=False)
display(q2_df_result)

# COMMAND ----------

# DBTITLE 1,Spark SQL Version
q2_sql_result = spark.sql("SELECT PdDistrict, COUNT(*) AS Count FROM sf_crime GROUP BY PdDistrict ORDER BY Count DESC")
display(q2_sql_result)

# COMMAND ----------

# DBTITLE 1,Visualization
fig_dims = (10,4)
fig = plt.subplots(figsize=fig_dims)
q2_df_result_plot = q2_df_result.toPandas()
chart = sns.barplot(x = 'PdDistrict', y = 'count', palette= 'Blues_r',data = q2_df_result_plot )
chart.set_xticklabels(chart.get_xticklabels(), rotation=45, horizontalalignment='right')
plt.title('Geographical Analysis')
plt.xlabel('Regions in San Francisco')
plt.ylabel('# of crime incidents (2003 - May 2018)')

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ### Takeaways:
# MAGIC Quantity should not be the only way to measure whether a region is safe or not
# MAGIC - analysis should be done with combination of section 1 (type of crimes)
# MAGIC - it is also important to understand the size and location of the district
# MAGIC 
# MAGIC With 15 years of data, further analysis should be done on the following dimension:
# MAGIC - trend analysis on how crime rates have changed over the years per region
# MAGIC - case studies should be done for certain region of importance to understand crime cycles

# COMMAND ----------

# MAGIC %md
# MAGIC ### Section 3: Weekly cycle analysis (case study: San Francisco downtown)
# MAGIC 
# MAGIC Assumption: 
# MAGIC - SF downtown is defiend by the range of spatial location with approximation of a rectangle.
# MAGIC - San Francisco Latitude and longitude coordinates are: 37.773972, -122.431297. X and Y represents each. 
# MAGIC - So we assume SF downtown spacial range: X (-122.4213,-122.4313), Y(37.7540,37.7740). 

# COMMAND ----------

# DBTITLE 1,Zoom in on relevant data with date/time information
df_opt2 = df_opt1[['IncidntNum', 'Category', 'Descript', 'DayOfWeek', 'Date', 'Time', 'PdDistrict', 'Resolution', 'Address', 'X', 'Y', 'Location']]
display(df_opt2)
df_opt2.createOrReplaceTempView("sf_crime")

# COMMAND ----------

# DBTITLE 1,Convert Date to date format and extract month and year
from pyspark.sql.functions import hour, date_format, to_date, month, year
# add new columns to convert Date to date format
df_new = df_opt2.withColumn("IncidentDate",to_date(df_opt2.Date, "MM/dd/yyyy")) 
# extract month and year from incident date
df_new = df_new.withColumn('Month',month(df_new['IncidentDate']))
df_new = df_new.withColumn('Year', year(df_new['IncidentDate']))
display(df_new.take(5))

# COMMAND ----------

# DBTITLE 1,Crop out San Francisco Downtown and Observe all Sundays: DataFrame Version
sf_downtown = (df_new.X > -122.4313) & (df_new.X < -122.4213) & (df_new.Y < 37.7740) & (df_new.Y > 37.7540 )
q3_df_result = df_new.filter((df_new.DayOfWeek == "Sunday") & (sf_downtown)).groupby('IncidentDate','DayOfWeek').count().orderBy('IncidentDate')
display(q3_df_result)

# COMMAND ----------

sf_downtown = (df_new.X > -122.4313) & (df_new.X < -122.4213) & (df_new.Y < 37.7740) & (df_new.Y > 37.7540 )
q3_df_result_02 = df_new.filter(sf_downtown).groupby('IncidentDate','DayOfWeek').count().orderBy('IncidentDate')
display(q3_df_result_02)

# COMMAND ----------

# DBTITLE 1,Crop out San Francisco downtown: SQL Version
q3_sql_result = spark.sql("SELECT IncidentDate, DayOfWeek, COUNT(*) AS Count FROM sf_crime WHERE DayOfWeek = 'Sunday' \
                          AND X > -122.4313 AND X < -122.4213 AND Y > 37.7540 AND Y < 37.7740 \
                          GROUP BY IncidentDate, DayOfWeek ORDER BY IncidentDate")
display(q3_sql_result)



# COMMAND ----------

allcrime = spark.sql("SELECT DayOfWeek,COUNT(*) AS crime_cnt FROM sf_crime GROUP BY 1")
display(allcrime) #294592 sunday

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ### Takeaways:
# MAGIC In this preliminary analysis, at least from the total volumn perspective, day of the week doesn't have obvious impact on crime rate
# MAGIC Further analysis can be done for certain holidays of the year or significant events (e.g., sport) worth celebration.
# MAGIC e.g., the peak value on June 30th, 2013 happens to coincide with two significant events of celebration:
# MAGIC 1. San Francisco Giants beat Colorado Rockies
# MAGIC 2. San Francisco Pride Parade
# MAGIC 
# MAGIC Another suggestion is to zoom out from weekly analysis to monthly analysis to seek patterns in data

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ###Section 4: Monthly cycle analysis (period: 2015 - 2018)

# COMMAND ----------

# DBTITLE 1,Select range of data (>2014)
years = [2015, 2016, 2017, 2018]
df_years = df_new[df_new.Year.isin(years)]
display(df_years.take(5))

# COMMAND ----------

# DBTITLE 1,Plot monthly trend
spark_df_q4 = df_years.groupby(['Year', 'Month']).count().orderBy('Year','Month')
display(spark_df_q4)

# COMMAND ----------

# DBTITLE 1,Show Monthly Count
df_opt2.createOrReplaceTempView('sf_crime')

q4_sql_result = spark.sql("""
                       SELECT SUBSTRING(Date,1,2) AS Month, SUBSTRING(Date,7,4) AS Year, COUNT(*) AS Count
                       FROM sf_crime
                       GROUP BY Year, Month
                       HAVING Year in (2015, 2016, 2017, 2018) 
                       ORDER BY Year, Month
                       """)
display(q4_sql_result)

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ### Takeaways:
# MAGIC Overall there has been a gradual decrease in crime volume from 2015 to 2018, specifically,
# MAGIC - crime volumn from 2015 to 2017 was very high, especially the theft crime, and there has been a downward trend in 2018, especially in May. 
# MAGIC - crime rate was high in 2015, which may partially due to the 47th Act signed by the governor in the California referendum in 2014, which led to a large number of theft and robbery crimes. 
# MAGIC - decrease in crime volumn in 2018 may be due to the San Francisco Police Department increasing uniformed police patrols, hence violence and theft activities have been greatly reduced. In addition, the San Francisco Police Department stepped up its crackdown on the drug trade, which is also one of the reasons for the decline in crime rate.

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ### Section 5: Daily cycle analysis (case study: pre-holiday December 15 of 2015/2016/2017)

# COMMAND ----------

# DBTITLE 1,Extract hours from incident time
from pyspark.sql.functions import to_timestamp
# add new columns to convert Time to hour format
df_new1 = df_new.withColumn('IncidentTime', to_timestamp(df_new['Time'],'HH:mm')) 
# extract hour from incident time
df_new1 = df_new1.withColumn('Hour',hour(df_new1['IncidentTime']))
display(df_new1.take(5))


# COMMAND ----------

# DBTITLE 1,See crime volume by the hour for December 15th in 2015 - 2017
dates = ['12/15/2015','12/15/2016','12/15/2017']
df_days = df_new1[df_new1.Date.isin(dates)]
spark_df_q5_1 = df_days.groupby('Hour','Date').count().orderBy('Date','Hour')
display(spark_df_q5_1)

# COMMAND ----------

# MAGIC %md
# MAGIC #### Takeaways:
# MAGIC - the two peak periods of crime are 12:00 and 18:00, corresponding to peak hours for ourdoor dining 
# MAGIC - the nonpeak period of crime is around 2:00 and 6:00, which makes sense as most people are asleep
# MAGIC - it is worth it to recommend tourists and locals be alert during those hours and increase patrol during those hours

# COMMAND ----------

# MAGIC %md
# MAGIC #### Section 6: Daily cycle analysis (case study: top 3 dangerous district)
# MAGIC - step 1: find out the regions with highest crime volume
# MAGIC - step 2: drill down to the crime category of each region

# COMMAND ----------

# DBTITLE 1,Spark DataFrame Version By Region
spark_df_q6_s1 = df_new.groupby('PdDistrict').count().orderBy('count',ascending = False)
display(spark_df_q6_s1)


# COMMAND ----------

# DBTITLE 1,Identify the top 3 regions with highest crime volume
top3_danger = df_new.groupby('PdDistrict').count().orderBy('count',ascending = False).head(3)
top3_danger_district = [top3_danger[i][0] for i in range(3)]
top3_danger_district

# COMMAND ----------

# DBTITLE 1,Drill down to Crime Category for the top 3 regions by the hours
spark_df_q6_s2 = df_new1.filter(df_new1.PdDistrict.isin('SOUTHERN', 'MISSION', 'NORTHERN')).groupby('Category','Hour').count().orderBy('Category','Hour')
display(spark_df_q6_s2)

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ### Takeaways:
# MAGIC - Theft, assult and drug/narcotic remain to be the top categories for the three regions, compared to overall region
# MAGIC - Peak and non-peak hours for crime is consistent with previous analysis

# COMMAND ----------

# MAGIC %md 
# MAGIC #### Section 7: Resolution impact analysis

# COMMAND ----------

# DBTITLE 1,Get Resolution categories
# MAGIC %sql select distinct(resolution) as resolve from sf_crime

# COMMAND ----------

# DBTITLE 1,Analyze resolution for each crime category
import pyspark.sql.functions as f
from pyspark.sql.window import Window
resolution_func = udf (lambda x: x != 'NONE')
spark_df_q7 = df_new.withColumn('IsResolution', resolution_func(f.col('Resolution')))
spark_df_q7 = spark_df_q7.groupBy('category', 'Resolution', 'IsResolution').count().withColumnRenamed('count', 'resolved').orderBy('category')
spark_df_q7 = spark_df_q7.withColumn('total', f.sum('resolved').over(Window.partitionBy('category')))\
             .withColumn('percentage%', f.col('resolved')*100/f.col('total'))\
             .filter(spark_df_q7.IsResolution == True).orderBy('percentage%', ascending=False)
display(spark_df_q7)

# COMMAND ----------

# DBTITLE 1,Compared Y-o-Y Crime by Category
df_opt2.createOrReplaceTempView('sf_crime')

q7_sql_result_02 = spark.sql("""
                       SELECT DISTINCT(category) as type, COUNT(*) as Count, SUBSTRING(Date,7,4) AS Year 
                       FROM sf_crime
                       GROUP BY type, year
                       HAVING Year in (2015, 2016, 2017, 2018) 
                       ORDER BY Year, Count DESC
                       """)
display(q7_sql_result_02)

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ### Takeaways:
# MAGIC - The top four categories of crimes resolved are STOLEN PROPERTIES, WARRANTS, DRIVING UNDER THE INFLUENCE, DRUG/NARCOTIC.
# MAGIC - The categories which less than 10% of crimes resolved are RECOVERED VEHICLE, VEHICLE THEFT, and LARCENY/THEFT.
# MAGIC - Can increase the police force against theft crimes

# COMMAND ----------

# MAGIC %md
# MAGIC ### Conclusion. 
# MAGIC 
# MAGIC ### Key Takeaways:
# MAGIC - Quantity should not be the only dimension to measure crimes, or whether a region is safe or not.         
# MAGIC - Potential Improvement: Severity, impact and combinations of crimes should be taken into consideration for more insights, with consideration of trend analysis 
# MAGIC - Day of week or month is not sensitive to crime rate, but significant events or holiday seasons could increase crime volume of certain categories significantly.
# MAGIC - Recommendation: increase policy patrol during significant events and pre-holiday seasons
# MAGIC - 12:00pm and 18:00pm are peak crime hours and 2:00AM - 6:00AM is non-peak crime hours. 
# MAGIC - Recommendation: increase policy patrol during peak hours 
# MAGIC - Theft, assault and drug/narcotic are the top categories of crimes with lowest resolution.
# MAGIC - Recommendation: increase policy allocation and crack down of these crime categories with better collaboration of local community 
