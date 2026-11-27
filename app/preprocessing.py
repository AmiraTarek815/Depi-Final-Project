import numpy as np

bool_cols = [
    "has_two_wheeler",
    "any_overturned",
    "any_left_road",
    "has_pedestrian",
    "is_weekend",
]


def transform_features(df):
  df = df.copy()
  df["number_of_vehicles"] = df["number_of_vehicles"].clip(upper=8)
  df["max_engine_cc"] = np.log1p(df["max_engine_cc"].clip(upper=20000))
  df["mean_casualty_age"] = df["mean_casualty_age"].clip(lower=0, upper=100)

  for col in bool_cols:
    df[col] = df[col].astype(int)
  return df