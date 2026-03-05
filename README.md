# CoffeeCoders
The most caffeinated coders of Harris '26.

The Coffee Coders are made up of two groups from the University of Chicago Harris School of Public Policy's Data Visualization Course, groups 24 and 49.
Coffee Coders Group 24:Dominick Sanakiewicz (dominicksanakiewicz) & Abraham Sadat (abrahamsadat96) 
Coffee Coders Group 49: Amanda Gu (AmandaAtHarris) & Zhen Zang (UChiZhen)

This project uses data to explore education outcomes across Cook County using data from 2018-2024.

With the exception of food deserts, which are fixed to 2019, the scope of our data is from 2018 to 2024. Our data was sourced from sources across all levels of government. We used the Census’s 5 Year American Community Survey data to capture household income and demographics per census tract. Finally, we used the USDA’s Economic Research Survey (ERS) to capture food desert data per census tract. We used a couple of different metrics constructed to model segregation that we encountered in the literature. These include the segregation quotient from Aguirre-Nuñez, Carlos, et al (2024) and the dissimilarity index Green (2022). We used this data as our independent variables. 

Our dependent variables came from the Illinois School Board of Education (ISBE) School Report Card. They were: chronic absenteeism, four-year graduation rate, English Language Arts (ELA) proficiency, and math proficiency. The ISBE defines chronic absenteeism as a student being absent for 10 or more percent of the school year, or 18 days. Both ELA and math proficiency are derived from standardized tests.

All of our data is publicly available.

The data processing flow is as follows: our pre processing file is called preprocessing.py , this cleaned all of our data which is in the FINAL_data folder. Our machine learning pipeline is called ml_pipeline.py, this runs our elastic net. Finally our write up is called final_project.qmd

This is the link to our dashboard*: https://coffinatededu.streamlit.app/

*please note that the dashboard must be "woken up" before running. This is a Streamlit feature, not a bug.

# Y Variables Data Update
y_variables_long.csv includes main y variables (ELA Proficiency & Math Proficiency), and sub y variables (chronic absenteeism, and graduation rate only for high schools).

cook_county_schools_master.csv includes the school id and other identifiers.

# X Variables Data Update and X & Y merge for school layer
x_data_dictionary.csv includes the x variables used for school layer and the report shows the quality of the data. And I merged the x and y variables and standardized the school id
