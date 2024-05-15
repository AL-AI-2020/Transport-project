import pickle
from pyproj import Transformer
import geopandas as gpd
from shapely.geometry import Point
import pandas as pd
import matplotlib.pyplot as plt


# Loading a saved model random_forest_model
with open('random_forest_model.pkl', 'rb') as file:
     random_forest_model = pickle.load(file)

# Loading the saved dataset GushDan_B
with open('GushDan_B.pkl', 'rb') as file:
     GushDan = pickle.load(file)

print(f'GushDan: {GushDan.info()}')

# Loading the saved dataset Municipality_df
with open('Municipality_df.pkl', 'rb') as file:
     Municipality_df = pickle.load(file)


# Loading the saved dataset Industrial_Zones_df
with open('Industrial_Zones_df.pkl', 'rb') as file:
     Industrial_Zones_df = pickle.load(file)        
        
# Loading the saved dataset LRT_stations_df
with open('LRT_stations_df.pkl', 'rb') as file:
     LRT_stations_df = pickle.load(file)               

# Loading the saved dataset TMA_stations_df
with open('TMA_lines_df.pkl', 'rb') as file:
     TMA_lines_df = pickle.load(file)             

# Loading the saved dataset MotoRate_df
with open('MotoRate_df.pkl', 'rb') as file:
     MotoRate_df = pickle.load(file)      

# Loading the saved dataset Density_jobs_df.pkl
with open('Density_jobs_df.pkl', 'rb') as file:
     Density_jobs_df = pickle.load(file)      
        
new_column_order = [
                'Distance_A_B',
                'LRT_Distance_A',
                'MotoRate_A',
                'Density_jobs3_A',
                'Industrial_Distance_A',
                'Population_A',
                'Workers_A',
                'LRT_Distance_B',
                'TMA_Distance_B',
                'MotoRate_B',
                'Density_jobs3_B',
                'Industrial_Distance_B',
                'Population_B',
                'Average_age_B',
                'Level_ultra-Orthodox_B',
                'Workers_B',
                'Area_m2_B',
                'Pop_Density_Sqkm_B',
                'Socio_Economic_Index_B',
                'Municipality_category_B',
                'Land_use_category_B'
                ]


def get_forecast(X_coordinates = 32.11492,
                 Y_coordinates = 34.82364,
                 Population_A = 13424, 
                 Level_ultra_Orthodox_A = 5,
                 Workers_A = 6007, 
                 Average_age_A = 36,
                 Area_m2_A = 1434523,
                 max_zones = 3 ,
                 zone_name = "New zone"):
    


    GushDan['Population_A'] = Population_A
    GushDan['Level_ultra-Orthodox_A'] = Level_ultra_Orthodox_A
    GushDan['Workers_A'] = Workers_A
    GushDan['Average_age_A'] = Average_age_A
    GushDan['Area_m2_A'] = Area_m2_A

    transform_coordinats(X_coordinates, Y_coordinates)

    distance_A_B(GushDan)

    industrial_Distance(GushDan)

    LRT_Distance(GushDan)

    TMA_Distance(GushDan)

    MotoRate(GushDan)

    Density_jobs(GushDan)

    Pop_Density_Sqkm(GushDan, Population_A, Area_m2_A)
    
    X_new = GushDan[new_column_order]
    
    y_pred_new = predict(X_new)
    
    top_df = plot_predict(y_pred_new, max_zones, zone_name )

    return top_df

# Convert from GPS coordinates to Israel Coordinates
def transform_coordinats(X_coordinates, Y_coordinates ):
    global x0 , y0, point_0 # Declare x0 , y0 - a variable as global
    TRAN_4326_TO_2039 = Transformer.from_crs("EPSG:4326", "EPSG:2039")

    def transform_EPSG(lat, lon):
        return TRAN_4326_TO_2039.transform(lat, lon)

    x0 , y0 = transform_EPSG(X_coordinates, Y_coordinates)
    # print('Point : GPS coordinates (EPSG:4326) => Israel coordinates (EPSG:2039)  ')
    # print(f" A: {X_coordinates, Y_coordinates} ==> ({x0} , {y0})")

    # Create a Point_0 object from the coordinates
    point_0 = Point(x0, y0)

# Calculate distance for each points of GushDan to Point_0
def distance_A_B(GushDan):
    def calculate_distance(x1, y1, x2, y2):
        return ((x1 - x2) ** 2 + (y1 - y2) ** 2) ** 0.5

    # Calculate distance for each points of GushDan to Point_0

    GushDan['Distance_A_B'] = GushDan.apply(lambda row: calculate_distance(row['X_B'], row['Y_B'], x0, y0), axis=1)

    return GushDan

# Calculation of the distance to the nearest Industrial Zone
def industrial_Distance(GushDan):
    Industrial_Distance_A = Industrial_Zones_df.geometry.apply(lambda x: x.distance(point_0)).min()
    GushDan['Industrial_Distance_A'] = Industrial_Distance_A

    return GushDan

# Calculation of minimum distance to the nearest LRT station
def LRT_Distance(GushDan):
    LRT_Distance_A = LRT_stations_df.geometry.apply(lambda x: x.distance(point_0)).min()
    GushDan['LRT_Distance_A'] = LRT_Distance_A
  
    return GushDan

# Calculation minimum distance to the nearest TMA line
def TMA_Distance(GushDan):
    TMA_Distance_A = TMA_lines_df.geometry.apply(lambda x: x.distance(point_0)).min()
    GushDan['TMA_Distance_A'] = TMA_Distance_A

    return GushDan

# Calculation MotoRate for Zone 0
def MotoRate(GushDan):
    # Load MotoRate_df
    # MotoRate_df = gpd.GeoDataFrame(MotoRate_df)
    # Check containment and retrieve MotoRate
    # point_series = gpd.GeoSeries(point_0, name='point_new')
    containing_polygon_df = MotoRate_df[MotoRate_df.geometry.contains(point_0)]

    if not containing_polygon_df.empty:
        containing_polygon_index = containing_polygon_df.index[0]
        containing_MotoRate = MotoRate_df['MOTORATE40'][containing_polygon_index]
    #     print(f"The point ({point_new}) is located in polygon with MotoRate value: {containing_MotoRate}")
    else:
    #     print(f"The point ({point_new}) does not lie within any polygon in MotoRate_df")
        containing_MotoRate = 0


    GushDan['MotoRate_A'] = containing_MotoRate

    return GushDan  

# Calculation Density_jobs for Zone 0
def Density_jobs(GushDan):
    # Density_jobs_df = gpd.GeoDataFrame(Density_jobs_df)
    # Check containment and retrieve 
    # point_series = gpd.GeoSeries(point_0, name='point_new')
    containing_polygon_jobs_df = Density_jobs_df[Density_jobs_df.geometry.contains(point_0)]

    if not containing_polygon_jobs_df.empty:
        containing_polygon_jobs_index = containing_polygon_jobs_df.index[0]
        Density_jobs_new = Density_jobs_df['JOB30_DENS'][containing_polygon_jobs_index]
    else:
        Density_jobs_new = 0

    GushDan['Density_jobs3_A'] = Density_jobs_new

    return GushDan  

# Calculation Pop_Density_Sqkm for Zone 0
def Pop_Density_Sqkm(GushDan, Population_A, Area_m2_A):
    GushDan['Pop_Density_Sqkm_A'] = Population_A / Area_m2_A
   
    return GushDan  


def predict(X_new):
    y_pred_new = random_forest_model.predict(X_new)
    # print(f"Prediction: {y_pred_new}")
    return(y_pred_new)

def plot_predict(y_pred_new, max_zones, zone_name):
    
    data = {'y_predict': y_pred_new}
    y_pred_df = pd.DataFrame(data)

    merged_df = pd.concat([y_pred_df, GushDan], axis=1)

    sorted_df = merged_df.sort_values(by='y_predict', ascending=False)
    top_df = sorted_df.head(max_zones)

    top_df['y_predict'] = top_df['y_predict'].round().astype(int)



    top_df = gpd.GeoDataFrame(top_df, geometry=top_df["geometry"])

    # Normalize values
    norm = plt.Normalize(min(top_df['y_predict']), max(top_df['y_predict']))
    top_df['color'] = norm(top_df['y_predict'])

    font_size = 15  # 

    # Creating a figure and axes
    fig, ax = plt.subplots(figsize=(25, 20))

    # Display the geometry of each Municipality in its own color
    color = Municipality_df['color']
    Municipality_df.plot(ax=ax, color=color, edgecolor='black', linewidth=0.1, alpha=0.3, zorder=5) 

    # Display geometry with color differentiation
    top_df.plot(ax=ax, column='color', cmap='Reds', edgecolor='black', linewidth=1)


    # Plot the point with a red marker (or customize marker style and color)
    point = gpd.GeoSeries(Point(x0, y0))
    point.plot(ax=ax, marker='o', markersize=100, color='red', label='Point (x0, y0)')

    # Adding y_predict values for each region
    for idx, row in top_df.iterrows():
        x, y = row.geometry.centroid.x, row.geometry.centroid.y
        # Check if an area is within given boundaries
        if 170000 <= x <= 205000 and 640000 <= y <= 680000:
            text = f"{int(row['y_predict'])}"
            ax.text(x, y, text, ha='center', va='center', fontsize=font_size, color='black')


    # Adding a title and color scale
    ax.set_title(f'Forecast of passengers leaving {zone_name}  in the morning for the Gush-Dan areas', fontsize=font_size*2 )

    sm = plt.cm.ScalarMappable(cmap='Reds', norm=norm)
    sm.set_array([])
    fig.colorbar(sm, label='y_predict', ax=ax)

    ax.legend([f'{zone_name}'], fontsize=font_size*2)



    ax.axis('off')
    ax.set_xlim([170000, 205000])
    ax.set_ylim([640000, 680000])

    # plt.show()

    # Save chart as JPEG image
    fig.savefig('./static/y_predict.jpg', dpi=300, bbox_inches='tight') 

    ##########
    top_df = top_df[['Municipality_B', 'Area_detail_B','y_predict']]
    # Rename columns
    top_df.rename(columns={'y_predict': 'Predict of number passengers', 'Municipality_B': 'Municipality' , 'Area_detail_B': 'Area name'}, inplace=True)

    print(f'top_df')
    print(f"{top_df}")

    return top_df



###################
if __name__ == '__main__':
    print('eof forecast')