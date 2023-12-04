import folium
from folium.plugins import HeatMap
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from geopy.distance import geodesic
import requests
import shap
import matplotlib.pyplot as plt
import seaborn as sns
import folium
from flask import Flask, request, jsonify

from folium.plugins import HeatMap
app = Flask(__name__)

df = pd.read_csv('Book1.csv')
print(f"Shape of the DataFrame after loading: {df.shape}")
df = df[df['date'] != 'date']
print(f"Shape of the DataFrame after removing 'date' rows: {df.shape}")
df['date'] = pd.to_datetime(df['date'], dayfirst=True)
df = df.sort_values('date')
df.columns = df.columns.str.strip()
columns_to_convert = ['pm25', 'pm10', 'o3', 'no2', 'so2', 'co']
for col in columns_to_convert:
    df[col] = pd.to_numeric(df[col], errors='coerce')
df = df.dropna()
print(f"Shape of the DataFrame after dropping NaN rows: {df.shape}")
X = df.drop(columns=['date', 'pm10', 'location'])
y = df['pm10']
model = RandomForestRegressor(n_estimators=100)
model.fit(X, y)
area_coords = {
    'btm': (12.9154, 77.6101),
    'silk board': (12.9172, 77.6227),
    'hebbal': (13.0382, 77.5919),
    'peenya': (13.0329, 77.5274),
    'hombegowda': (12.9615, 77.6017),
    'jayanagar': (12.9299, 77.5824),
    'bapuji nagar': (12.9636, 77.5375),
    'saneguravahalli': (12.9775, 77.5492),
    'city railway': (12.9779, 77.5667),
    'bwssb': (12.9613, 77.5871),
}
feature_importances = model.feature_importances_
importance_df = pd.DataFrame({
    'feature': X.columns,
    'importance': feature_importances
}).sort_values(by='importance', ascending=False)
print(importance_df)
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X)
def find_closest_area(user_lat, user_lon):
    min_distance = float('inf')
    closest_area = None
    for area, (area_lat, area_lon) in area_coords.items():
        distance = geodesic((user_lat, user_lon), (area_lat, area_lon)).km
        if distance < min_distance:
            min_distance = distance
            closest_area = area
    return closest_area

def display_pollutant_bar_graph_for_area(closest_area):
    user_data = df[df['location'] == closest_area]
    if user_data.empty:
        return {
            'error': f"No data available for the specified location: {closest_area}"
        }

    pollutants = ['co', 'no2', 'so2', 'pm25', 'pm10']
    graph_data = {
        'date': user_data['date'].tolist(),
    }

    for pollutant in pollutants:
        graph_data[pollutant] = user_data[pollutant].tolist()

    return graph_data

def get_aqi_prediction_for_user(user_lat, user_lon, closest_area, age, fitness_level, activity_type, health_profile, has_allergies):
    recommendations = []
    recommendations.append("Your Current Location is: " + closest_area)
    user_data = df[df['location'] == closest_area]
    if user_data.empty:
        return {
            'error': f"No data available for the specified location: {closest_area}"
        }
    latest_data = user_data.iloc[-1]
    features = latest_data.drop(['date', 'pm25', 'location'])
    predicted_pm25 = model.predict([features])[0]
    predicted_aqi = "Your Personalised AQI is " + str(predicted_pm25)
    recommendations.append(predicted_aqi)
    dominant_pollutant = "PM2.5"
    recommendations.append("General precautions advised.")
    explanatory_insight = f"High levels of {dominant_pollutant} due to industrial emissions in the area."
    recommendations.append(explanatory_insight)
    if age == "child":
        if predicted_pm25 <= 50:
            recommendations.append("For children, current air quality is good and within threshold limit.")
            if fitness_level == "high":
                recommendations.append( "Perfect air quality, no restrictions or recommendations.")
                if activity_type == "indoor":
                    recommendations.append("Indoor activities are safe. You can also consider going outdoors as the air quality is good.")
                elif activity_type == "outdoor":
                    recommendations.append("Good time for outdoor activities. Air quality is in the safe range.")
            elif fitness_level == "low":
                recommendations.append("The air quality is good. However, always monitor your health when performing activities.")
                if activity_type == "indoor":
                    recommendations.append("Indoor activities are safe.")
                elif activity_type == "outdoor":
                    recommendations.append( "You can enjoy outdoor activities. However, take breaks if needed.")
        elif predicted_pm25 <= 100:
            recommendations.append("Children should be cautious as air quality is moderate.")
            if fitness_level == "high":
                recommendations.append("Children with good fitness can handle moderate air quality, but it's best to limit prolonged exposure.")
                if activity_type == "indoor":
                    recommendations.append("Indoor activities are preferred.")
                elif activity_type == "outdoor":
                    recommendations.append("Limit time spent outdoors. Short activities or games are okay.")
            elif fitness_level == "low":
                recommendations.append("Children with lower fitness should avoid strenuous activities.")
                if activity_type == "indoor":
                    recommendations.append("Stick to indoor activities.")
                elif activity_type == "outdoor":
                    recommendations.append("Consider moving outdoor activities indoors.")
        else:
            recommendations.append("For children, the current air quality is not ideal.")
            if fitness_level == "high":
                recommendations.append("Even though they are active, outdoor physical activities should be minimized.")
                if activity_type == "indoor":
                    recommendations.append("Indoor play is advised.")
                elif activity_type == "outdoor":
                    recommendations.append("Limit outdoor playtime and avoid strenuous activities.")
            elif fitness_level == "low":
                recommendations.append("Avoid any strenuous activities.")
                if activity_type == "indoor":
                    recommendations.append("Indoor activities are safe.")
                elif activity_type == "outdoor":
                    recommendations.append("Consider staying indoors.")
    elif age == "adult":
        if predicted_pm25 <= 50:
            recommendations.append("For adults, the current air quality is excellent.")
            if fitness_level == "high":
                recommendations.append("Ideal conditions for outdoor exercise or sports.")
                if activity_type == "indoor":
                    recommendations.append("Indoor activities are safe, but consider going outdoors to make the most of the clean air.")
                elif activity_type == "outdoor":
                    recommendations.append("A great time for outdoor workouts or activities.")
            elif fitness_level == "low":
                recommendations.append("The air quality is good. Do activities at your own pace.")
                if activity_type == "indoor":
                    recommendations.append("Indoor activities are safe.")
                elif activity_type == "outdoor":
                    recommendations.append("You can enjoy light outdoor activities.")
        elif predicted_pm25 <= 100:
            recommendations.append("Adults should note that the air quality is moderate.")
            if fitness_level == "high":
                recommendations.append( "While you can handle moderate air quality, be cautious during prolonged exposure.")
                if activity_type == "indoor":
                    recommendations.append("Indoor activities remain unaffected.")
                elif activity_type == "outdoor":
                    recommendations.append("Outdoor activities are okay, but consider taking breaks.")
            elif fitness_level == "low":
                recommendations.append("Be cautious and monitor how you feel.")
                if activity_type == "indoor":
                    recommendations.append("Indoor activities are safe.")
                elif activity_type == "outdoor":
                    recommendations.append("Consider shorter or lighter outdoor activities.")
        else:
            recommendations.append("For adults, be cautious as the air quality deteriorates.")
            if fitness_level == "high":
                recommendations.append("Limit time spent outdoors, especially during workouts.")
                if activity_type == "indoor":
                      recommendations.append("Engage in indoor workouts.")
                elif activity_type == "outdoor":
                      recommendations.append("Consider modifying or postponing outdoor activities.")
                elif fitness_level == "low":
                      recommendations.append("Avoid long exposure outdoors.")
                      if activity_type == "indoor":
                          recommendations.append("Stay indoors.")
                      elif activity_type == "outdoor":
                          recommendations.append("Limit time outside.")
    elif age == "elderly":
        if predicted_pm25 <= 50:
            recommendations.append("For the elderly, the current air quality is excellent.")
            if fitness_level == "high":
                recommendations.append("Ideal conditions for outdoor walks or light exercises.")
                if activity_type == "indoor":
                    recommendations.append("Indoor activities are safe, but consider taking a short walk outside to enjoy the clean air.")
                elif activity_type == "outdoor":
                    recommendations.append("A great time for a gentle stroll or outdoor relaxation.")
            elif fitness_level == "low":
                recommendations.append( "The air quality is good. Engage in activities that you're comfortable with.")
                if activity_type == "indoor":
                    recommendations.append("Indoor activities are safe.")
                elif activity_type == "outdoor":
                    recommendations.append("You can enjoy short outdoor activities.")
        elif predicted_pm25 <= 100:
            recommendations.append("Elderly individuals should note that the air quality is moderate.")
            if fitness_level == "high":
                recommendations.append("Be cautious during prolonged outdoor activities.")
                if activity_type == "indoor":
                    recommendations.append("Indoor activities are safe.")
                elif activity_type == "outdoor":
                    recommendations.append("Outdoor activities are okay, but avoid strenuous exercises.")
            elif fitness_level == "low":
                recommendations.append("Limit outdoor exposure.")
                if activity_type == "indoor":
                    recommendations.append("It's best to stay indoors.")
                elif activity_type == "outdoor":
                    recommendations.append("If necessary, limit your time outside.")
        else:
            recommendations.append("Elderly individuals should be very cautious as air quality is poor.")
            if fitness_level == "high":
                recommendations.append("Avoid strenuous activities, even indoors.")
                if activity_type == "indoor":
                    recommendations.append("Relaxing indoor activities are best.")
                elif activity_type == "outdoor":
                    recommendations.append("Avoid going outside.")
            elif fitness_level == "low":
                recommendations.append("Limit any form of strenuous activities.")
                if activity_type == "indoor":
                    recommendations.append("Stay indoors and relax.")
                elif activity_type == "outdoor":
                    recommendations.append("It's best to stay indoors.")
    if health_profile == "asthma":
        recommendations.append("If asthmatic, carry your inhaler and monitor respiratory symptoms closely.")
    if has_allergies == 1:
        recommendations.append("If you have allergies, consider wearing a mask during outdoor activities.")
    features_df = pd.DataFrame([features], columns=X.columns)
    shap_values_for_prediction = explainer.shap_values(features_df)[0]
    shap_values_df = pd.DataFrame({
        'feature': X.columns,
        'shap_value': shap_values_for_prediction
    })
    features = ['pm25', 'co', 'o3', 'no2', 'so2']
    importances = [0.629064, 0.150933, 0.085591, 0.083306, 0.051106]
    plot_feature_importance(importance_df)
    return (recommendations)
    plt.show()
def plot_feature_importance(importance_df):
    plt.figure(figsize=(10, 6))
    sns.barplot(x='importance', y='feature', data=importance_df, palette='viridis')
    plt.xlabel('Importance')
    plt.ylabel('Feature')
    plt.title('Feature Importance')
    plt.show()
def create_aqi_heatmap(locations, aqi_values):
    m = folium.Map(location=[12.971598, 77.594562], zoom_start=12)  # Centered around Bangalore for example
    heat_data = [[location[0], location[1], aqi] for location, aqi in zip(locations, aqi_values)]
    HeatMap(heat_data).add_to(m)
    return m
# Prediction endpoint
# Prediction endpoint
@app.route('/')
def hello_world():
    return 'Hello World!'
@app.route('/predict_aqi', methods=['POST'])
def predict_aqi():
    if request.method == 'POST':
        content_type = request.headers.get('Content-Type')
        if content_type is None or content_type != 'application/json':
            return jsonify({'error': 'Unsupported Media Type: Content-Type must be application/json'}), 415

        data = request.get_json()
        user_lat = data.get('user_lat')
        user_lon = data.get('user_lon')
        age = data.get('age')
        fitness_level = data.get('fitness_level')
        activity_type = data.get('activity_type')
        health_profile = data.get('health_profile')
        has_allergies = data.get('has_allergies')

        if any(param is None for param in [user_lat, user_lon, age, fitness_level, activity_type, health_profile, has_allergies]):
            return jsonify({'error': 'Missing required parameters'}), 400

        closest_area = find_closest_area(user_lat, user_lon)
        prediction = get_aqi_prediction_for_user(user_lat, user_lon, closest_area, age, fitness_level, activity_type, health_profile, has_allergies)

        return jsonify({'prediction': prediction})
    else:
        return jsonify({'error': 'Method Not Allowed'}), 405


@app.route('/heatmap', methods=['GET'])
def get_heatmap():
    locations = [area_coords[location] for location in df['location']]
    aqi_values = df['pm25'].astype(float).values
    heatmap = create_aqi_heatmap(locations, aqi_values)

    return heatmap._repr_html_()

if __name__ == '__main__':
    app.run(debug=True)
