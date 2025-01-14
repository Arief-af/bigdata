from flask import Flask, request, jsonify, render_template
from flask_cors import CORS  # Import CORS
import pandas as pd
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.arima.model import ARIMA
import matplotlib.pyplot as plt
import os

app = Flask(__name__)

# Enable CORS for the entire app
CORS(app)

@app.route('/forecast_plot', methods=['GET'])
def get_forecast_plot():
    image_path = "static/forecast_plot.png"
    if os.path.exists(image_path):
        return send_from_directory(directory="static", path="forecast_plot.png", as_attachment=False)
    else:
        return jsonify({"error": "Forecast plot image not found."}), 404

# Fungsi untuk memproses data dan melakukan peramalan
def forecast_production(file_path, steps):
    # Load the data
    df = pd.read_csv(file_path)
    df = df[['Tahun', 'Produksi']]
    
    # Check for stationarity using ADF Test
    adf_result = adfuller(df['Produksi'])
    is_stationary = adf_result[1] <= 0.05

    # Perform differencing if data is not stationary
    if not is_stationary:
        df['Differenced'] = df['Produksi'].diff()

        # Plot Differenced Data
        plt.figure(figsize=(10, 6))
        plt.plot(df["Tahun"][1:], df["Differenced"][1:], marker="o", color="orange")
        plt.title("Data Setelah Differencing")
        plt.xlabel("Tahun")
        plt.ylabel("Perubahan Produksi (Ton)")
        plt.grid()
        plt.savefig("static/differenced_data.png")
        plt.close()

        # ACF and PACF plots
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        plot_acf(df["Differenced"].dropna(), ax=axes[0])
        plot_pacf(df["Differenced"].dropna(), ax=axes[1])
        axes[0].set_title("ACF")
        axes[1].set_title("PACF")
        plt.savefig("static/acf_pacf_plots.png")
        plt.close()

        data_to_fit = df['Differenced'].dropna()
    else:
        data_to_fit = df['Produksi']

    # Fit ARIMA model
    model = ARIMA(data_to_fit, order=(1, 3, 1))
    results = model.fit()

    # Forecast for the specified steps
    forecast = results.get_forecast(steps=steps)
    forecast_mean = forecast.predicted_mean
    forecast_mean = forecast_mean.astype(int)

    # Generate dynamic forecast years
    forecast_years = list(range(df['Tahun'].iloc[-1] + 1, df['Tahun'].iloc[-1] + 1 + steps))
    dfd = pd.read_csv(file_path)
    df = dfd[['Tahun', 'Produksi']]
    
    # Plot the forecast
    plt.figure(figsize=(10, 6))
    df['Produksi'] = df['Produksi'].astype(int)
    plt.plot(dfd['Tahun'], dfd['Produksi'], marker='o', label='Data Historis')
    print(dfd['Produksi'])
    print(forecast_mean)
    plt.plot(forecast_years, forecast_mean, marker='o', color='red', label=f'Forecast ({steps} tahun)')
    plt.title("Peramalan Produksi")
    plt.xlabel("Tahun")
    plt.ylabel("Produksi (Ton)")
    plt.legend()
    plt.grid()
    plt.savefig("static/forecast_plot.png")
    plt.close()

    # Return the forecast as a dictionary
    forecast_results = [{"year": year, "forecast": value} for year, value in zip(forecast_years, forecast_mean)]
    return forecast_results

@app.route('/')
def index():
    # Cek apakah file gambar exist untuk plot differencing dan forecast
    differenced_image = None
    forecast_image = None

    if os.path.exists("static/differenced_data.png"):
        differenced_image = "/static/differenced_data.png"

    if os.path.exists("static/forecast_plot.png"):
        forecast_image = "/static/forecast_plot.png"

    return render_template('index.html', differenced_image=differenced_image, forecast_image=forecast_image)

# Route untuk upload file dan melakuk.an peramalan
@app.route('/forecast', methods=['POST'])
def forecast_endpoint():
    # Ambil 'steps' dari body JSON request
    data = request.get_json()
    steps = int(data.get('steps', 10))  # Default steps = 10

    # Simpan file sementara
    file_path = 'data1.csv'

    try:
        # Lakukan peramalan
        results = forecast_production(file_path, steps)
        return jsonify({
            "message": "Peramalan berhasil",
            "forecast": results
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    

if __name__ == '__main__':
    app.run(debug=True)
