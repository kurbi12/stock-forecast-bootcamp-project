from dataclasses import dataclass
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor

@dataclass
class Models:
    lr: LinearRegression
    rf: RandomForestRegressor

def build_models():
    lr = LinearRegression()
    rf = RandomForestRegressor(
        n_estimators=300,
        max_depth=8,
        random_state=42,
        n_jobs=-1
    )
    return Models(lr=lr, rf=rf)
