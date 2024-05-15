
# Project - Transport
#
# 1. Statement of the problem
# 1.1 Analysis of Israel Travel Research Data
# 1.2 Create a machine learning model to predict the number of passengers in Israel.
#
# 2. Data to analyze
# 2.1 The dataset is based on 2019 weekday travel data to Israel.
# 2.2 There are 1270 zones in Israel. One municipality may have several zones.
# 2.3 There is hourly travel data from each zone to each zone.
# 2.4 Added statistics for each zone to the main trip data set, such as geodata for zones, number of residents, average age,
#     religious residents, Arab residents, number of workers, etc.
# 2.5 In total, the complete data set contains 1,521,699 rows × 30 columns.
# 2.6 All data is published on the government website: https://info.data.gov.il/.
#
# 3. The main goal of the work
# The main goal of the work is to create a model for predicting the number of trips in the morning 
# from a given specific zone in Israel to each other zone in accordance with the data of the corresponding zones.


# Import the  models
import pickle
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

# To run the models, we selected data from a large dataset:
# zones located from the center of Tel Aviv at a distance - 20 km.
# zones from which passengers depart are defined as - residential only
# pairs of zones from which passengers departed and arrived – at least one person
path = './data/'
GushDan_df = pd.read_pickle(path + "GushDan.pkl")

#  Descriptions of Data fields
# Some features were not taken into the final version of the model due to their low impact on the final result (===>).:
feature_columns = [0] * 41

# feature_columns[0] = 1  # fromZone  - unique identifier of the zone from which the trip began ( Zone - A )    ||===> 0%
# feature_columns[1] = 1  # ToZone - unique identifier of the zone in which the trip ended ( Zone - B )     ||===> 0%
feature_columns[2] = 1  # Distance_A_B   - geographical distance between zones A and B in meters.   ||===> 62.2%
# feature_columns[3] = 1  # Distance_CenterTA_A      ||===> 0%
# feature_columns[4] = 1  # Distance_CenterTA_B      ||===> 0%
feature_columns[5] = 1  # Morning_rush_6_9 - number of passengers traveling during the morning rush hour (from 6:00 to 9:00) from zone A to zone B     ||===> 0%
feature_columns[6] = 1  # LRT_Distance_A  - geographical distance between zone A and the nearest train LRT station in meters.    ||===> 1.2%
# feature_columns[7] = 1  # TMA_Distance_A - geographical distance between zone A and the nearest heavy passenger transport lines in meters.     ||===> 0.4%
feature_columns[8] = 1  # MotoRate_A - level of vehicles data - MotoRate (number of vehicles per 1000 people) for zone A     ||===> 0.8%
feature_columns[9] = 1  # Density_jobs3_A - Density of jobs in zone A    ||===> 0.9%
feature_columns[10] = 1  # Industrial_Distance_A - Distance from zone A to the nearest industrial zone in meters. If zone A borders an industrial zone, then the distance between the zones is 0.     ||===> 1.2%
feature_columns[11] = 1  # Population_A - Number of inhabitants in zone A.     ||===> 3.7%
# feature_columns[12] = 1  # Average_age_A - The mean age of residents of zone A     ||===> 0.5%
# feature_columns[13] = 1  # Level_ultra-Orthodox_A  -  Level of ultra-Orthodox Jewish population in area A. Where level equal to 1 is the largest level of ultra-Orthodox Jewish population, level equal to 5 there is a relatively small number of ultra-Orthodox residents.    ||===> 0%
feature_columns[14] = 1  # Workers_A - Number of workers living in the zone.     ||===> 5.8%
# feature_columns[15] = 1  # Area_m2_A  - Area of zone A in square meters    ||===> 0.6%
# # feature_columns[16] = 1  # Pop_Density_Sqkm_A  - Population density per square meter.    ||===> 0.6%
# feature_columns[17] = 1  # Socio_Economic_Index_A  - Socio-economic index of zone A, where 1 is the lowest level and 10 is the highest socio-economic level.    ||===> 1.4%
# feature_columns[18] = 1  # X_A  - longitude of the central point of zone A in which the trip began.     ||===> 0%
# feature_columns[19] = 1  # Y_A  - latitude of the central point of zone A in which the trip began.    ||===> 0%
feature_columns[20] = 1  # LRT_Distance_B  - geographical distance between zone B and the nearest train LRT station in meters.   ||===> 1.3%
feature_columns[21] = 1  # TMA_Distance_B  - geographical distance between zone B and the nearest heavy passenger transport lines in meters.    ||===> 0.2%
feature_columns[22] = 1  # MotoRate_B - level of vehicles data - MotoRate (number of vehicles per 1000 people) for zone B     ||===> 1%
feature_columns[23] = 1  # Density_jobs3_B  - Density of jobs in zone B    ||===> 2.6%
feature_columns[24] = 1  # Industrial_Distance_B  - Distance from zone B to the nearest industrial zone in meters. If zone B borders an industrial zone, then the distance between the zones is 0.    ||===> 1.4%
feature_columns[25] = 1  # Population_B - Number of inhabitants in zone B     ||===> 3.6%
feature_columns[26] = 1  # Average_age_B  - The mean age of residents of zone A    ||===> 0.5%
feature_columns[27] = 1  # Level_ultra-Orthodox_B  -  Level of ultra-Orthodox Jewish population in area B. Where level equal to 1 is the largest level of ultra-Orthodox Jewish population, level equal to 5 there is a relatively small number of ultra-Orthodox residents.    ||===> 0%    ||===> 0%
feature_columns[28] = 1  # Workers_B  - Number of workers living in the zone.    ||===> 4.7%
feature_columns[29] = 1  # Area_m2_B  - Area of zone B in square meters    ||===> 1.6%
feature_columns[30] = 1  # Pop_Density_Sqkm_B - Population density per square meter.     ||===> 1%
feature_columns[31] = 1  # Socio_Economic_Index_B  - Socio-economic index of zone B, where 1 is the lowest level and 10 is the highest socio-economic level.    ||===> 1.1%
# feature_columns[32] = 1  # X_B - longitude of the central point of zone B     ||===> 0%
# feature_columns[33] = 1  # Y_B -  latitude of the central point of zone B   ||===> 0%
# feature_columns[34] = 1  # Municipality_category_A - Municipal number.     ||===> 0.6%
feature_columns[35] = 1  # Municipality_category_B - Municipal number.     ||===> 0.6%
# feature_columns[36] = 1  # Land_use_category_A - Land use type    ||===> 0%
feature_columns[37] = 1  # Land_use_category_B -  Land use type.    ||===> 0.7%
# feature_columns[38] = 1  # Municipality_B - Name of municipality.    ||===> 0%
# feature_columns[39] = 1  # Area_detail_B - Name of the area.     ||===> 0%
# feature_columns[40] = 1  # geometry -  Coordinates of the boundaries of the Area.   ||===> 0%

mask = np.array(feature_columns).astype(bool)
model_df = GushDan_df.loc[:, mask]



# Split the data into training and test sets
X = model_df.drop(['Morning_rush_6_9'], axis=1)
y = model_df['Morning_rush_6_9']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

# Run the Random Forest model with optimal hyperparameters
n_estimators = 300
max_depth = 20
min_samples_split = 10
min_samples_leaf = 5

random_forest_model = RandomForestRegressor(n_estimators=n_estimators,
                                  max_depth=max_depth,
                                  min_samples_split=min_samples_split,
                                  min_samples_leaf=min_samples_leaf,
                                  random_state=42)

random_forest_model.fit(X_train, y_train)
random_forest_predictions = random_forest_model.predict(X_test)
random_forest_train = random_forest_model.predict(X_train)
random_forest_rmse = mean_squared_error(y_test, random_forest_predictions, squared=False)
random_forest_r2 = r2_score(y_test, random_forest_predictions)
random_forest_r2_train = r2_score(y_train, random_forest_train)
print(f'Results of Random Forest model :')
print(f"Random Forest RMSE: {random_forest_rmse:.2f}")
print(f"Random Forest R-squared test: {random_forest_r2:.2f}")
print(f"Random Forest R-squared train: {random_forest_r2_train:.2f}")

# Save the model to a pickle file
with open('./shared_volume/random_forest_model.pkl', 'wb') as file:
    pickle.dump(random_forest_model, file)

# Preparation of data for all districts in the Gush Dan region, including statistical and geographical data
feature_columns = [0] * 41

feature_columns[1] = 1  # ToZone      
feature_columns[20] = 1  # LRT_Distance_B      
feature_columns[21] = 1  # TMA_Distance_B      
feature_columns[22] = 1  # MotoRate_B      
feature_columns[23] = 1  # Density_jobs3_B      
feature_columns[24] = 1  # Industrial_Distance_B      
feature_columns[25] = 1  # Population_B      
feature_columns[26] = 1  # Average_age_B      
feature_columns[27] = 1  # Level_ultra-Orthodox_B      
feature_columns[28] = 1  # Workers_B      
feature_columns[29] = 1  # Area_m2_B      
feature_columns[30] = 1  # Pop_Density_Sqkm_B      
feature_columns[31] = 1  # Socio_Economic_Index_B      
feature_columns[32] = 1  # X_B      
feature_columns[33] = 1  # Y_B      
feature_columns[35] = 1  # Municipality_category_B      
feature_columns[37] = 1  # Land_use_category_B      
feature_columns[38] = 1  # Municipality_B      
feature_columns[39] = 1  # Area_detail_B      
feature_columns[40] = 1  # geometry      

mask = np.array(feature_columns).astype(bool)
model_mask_df = GushDan_df.loc[:, mask]

unique_df = model_mask_df.drop_duplicates()
GushDan_B = unique_df.reset_index(drop=True)

with open('./shared_volume/GushDan_B.pkl', 'wb') as file:
  pickle.dump(GushDan_B, file)
