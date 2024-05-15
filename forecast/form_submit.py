
from forecast import get_forecast
from flask import Flask, render_template, request

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('form.html')  


@app.route('/getForecast')
def process_forecast():

    zone_name = request.args.get('zone_name')
    longitude_str = request.args.get('longitude')
    latitude_str = request.args.get('latitude')
    area_m2_str = request.args.get('area_m2')
    population_str = request.args.get('population')
    workers_str = request.args.get('workers')
    average_age_str = request.args.get('average_age')
    level_orthodox_str = request.args.get('level_orthodox')
    max_zones_str = request.args.get('max_zones')


    zone_name = zone_name if zone_name else 'Tsrifin. Rishon LeZion'
    longitude = float(longitude_str) if longitude_str else 31.967768  
    latitude = float(latitude_str) if latitude_str else 34.840902   
    area_m2 = int(area_m2_str.replace(',', '')) if area_m2_str else 900000  
    population = int(population_str.replace(',', '')) if population_str else 27000 
    workers = int(workers_str.replace(',', '')) if workers_str else 8250
    average_age = int(average_age_str.replace(',', '')) if average_age_str else 32
    level_orthodox = int(level_orthodox_str.replace(',', '')) if level_orthodox_str else 5
    max_zones = int(max_zones_str.replace(',', '')) if max_zones_str else 5

    top_areas_df = get_forecast(longitude, latitude, population, level_orthodox, workers, average_age, area_m2, max_zones, zone_name )

    # Convert the DataFrame to a list of dictionaries
    top_df = top_areas_df.to_dict(orient='records')

    print(20*'++++')
    print(f'top_df_list: {top_df}')



    return render_template('answer_forecast.html', answer=zone_name , top_df=top_df)



if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8081, debug=False)