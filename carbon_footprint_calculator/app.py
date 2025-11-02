from flask import Flask, render_template, request, jsonify

app = Flask(__name__)

# Define the calculation logic
def calculate_carbon_footprint(household_size, gas_usage, electricity_usage, airline_miles, car_count):
    # Constants (values in kg CO2e)
    CO2_per_gas_unit = 2.2  # Example value, adjust as needed
    CO2_per_electricity_unit = 0.5  # Example value, adjust as needed
    CO2_per_airline_mile = 0.15  # Example value, adjust as needed
    CO2_per_car_km = 0.24  # Example value, adjust as needed

    # Calculate total carbon footprint
    total_gas_emissions = household_size * gas_usage * CO2_per_gas_unit
    total_electricity_emissions = household_size * electricity_usage * CO2_per_electricity_unit
    total_airline_emissions = airline_miles * CO2_per_airline_mile
    total_car_emissions = car_count * household_size * CO2_per_car_km

    total_carbon_footprint = (total_gas_emissions + total_electricity_emissions +
                              total_airline_emissions + total_car_emissions)

    return total_carbon_footprint

# Create a route to display the home page
@app.route('/')
def index():
    return render_template('index.html')

# Create a route to calculate the carbon footprint
@app.route('/calculate', methods=['POST'])
def calculate():
    data = request.get_json()
    household_size = data['household_size']
    gas_usage = data['gas_usage']
    electricity_usage = data['electricity_usage']
    airline_miles = data['airline_miles']
    car_count = data['car_count']

    carbon_footprint = calculate_carbon_footprint(household_size, gas_usage, electricity_usage, airline_miles, car_count)

    return jsonify({'carbon_footprint': carbon_footprint})

if __name__ == '__main__':
    app.run(debug=True)
