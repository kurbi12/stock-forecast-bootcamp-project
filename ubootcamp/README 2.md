# Stock Forecast – Bootcamp Mini Proje (Tamamlanmış Sürüm)

Bu proje, **bir sonraki gün (t+1) kapanış fiyatını** tahmin eden *üretime yakın* bir mini uygulamadır.
Hem **Jupyter Notebook** hem de **komut satırı** (CLI) ile çalışır.

## Özellikler
- Veri: Yahoo Finance (`yfinance`) ile otomatik indirme
- Özellik mühendisliği: `lag1/5/10`, `sma5/10`, `vol5`, `rsi14`
- Modeller: `LinearRegression` ve `RandomForestRegressor`
- Kayıtlar: `outputs/` klasörüne **metrikler**, **grafik** ve **tahmin** dosyası yazılır
- Model kayıt: `joblib` ile model kaydetme/yükleme

## Hızlı Başlangıç (CLI)
```bash
pip install -r requirements.txt

# Eğitim + değerlendirme
python -m src.train --ticker AAPL --start 2018-01-01 --horizon 1 --outdir outputs

# Grafik + rapor için grafiği oluşturur (outputs/plot.png)
# Metrikler outputs/metrics.json dosyasına yazılır

# Tahmin (kayıtlı modeli kullanır)
python -m src.infer --model outputs/best_model.joblib --ticker AAPL --start 2018-01-01 --horizon 1 --outdir outputs
# Tahmin dosyası: outputs/next_day_prediction.json
```

## Jupyter Notebook
Notebook sürümü için `stock_forecast.ipynb` dosyasını kullanın (ayrı olarak eklenecek).

## Kullanım Notları
- BIST için: sembolleri `THYAO.IS`, `GARAN.IS` gibi `.IS` uzantısıyla yazın.
- Varsayılan olarak en iyi model **RandomForest** seçilir ve `outputs/best_model.joblib` dosyasına kaydedilir.
- Metrikler: `MAE` ve `R²` karşılaştırması yapılır, `outputs/metrics.json` içine yazılır.
- Grafik: Son 200 gün için **Gerçek vs Tahmin** grafiği `outputs/plot.png` olarak üretilir.

## Proje Yapısı
```
stock-forecast-bootcamp/
├── README.md
├── requirements.txt
├── .gitignore
├── LICENSE
├── stock_forecast.ipynb            # (ayrı olarak eklendi)
├── outputs/                         # (çıktılar buraya yazılır)
├── docs/
│   └── example_plot.png            # (opsiyonel örnek)
├── scripts/
│   └── run_all.sh                  # Hızlı çalıştırma betiği
└── src/
    ├── __init__.py
    ├── data.py
    ├── model.py
    ├── train.py
    └── infer.py
```

## Hızlı Çalıştırma (tek komutla)
```bash
bash scripts/run_all.sh AAPL
```
> İlk argüman semboldür (varsayılan `AAPL`). Betik, eğitimi çalıştırır ve ardından tahmini üretir.

## Uyarı
Bu proje **eğitim amaçlıdır**, yatırım tavsiyesi değildir.
