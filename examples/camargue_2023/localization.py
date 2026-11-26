"""Site coordinates and time zone."""

import pandas as pd

localisation = {
    "latitude": 43.734286, "longitude": 4.570565, "altitude": 0, "timezone": "Europe/Paris",
}


def date(month, day, hour):
    return pd.Timestamp(year=2023, month=month, day=day, hour=hour, tz=localisation["timezone"])
