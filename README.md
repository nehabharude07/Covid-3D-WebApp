# COVID-19 3D Atlas (upgraded)

    pip install -r requirements.txt
    python app.py                 # http://127.0.0.1:5000
    python build_static.py        # optional: standalone covid_3d_atlas.html

- 3D globe: bars per country (height + colour = cumulative cases/deaths), weekly time slider with Play.
- 3D forecast: last 150 days + 30-day forecast with 80% range, orbit/zoom/hover.
- Forecast is a numpy/pandas baseline (damped trend + weekly pattern); no Prophet needed.
