import pandas as pd
import pytest
from retailiq.transform import to_daily_sales


def test_wide_to_long():
    wide = pd.DataFrame({"item_id": ["A"], "store_id": ["S1"], "d_1": [2], "d_2": [0]})
    calendar = pd.DataFrame({"d": ["d_1", "d_2"], "date": ["2011-01-29", "2011-01-30"]})
    out = to_daily_sales(wide, calendar)
    assert len(out) == 2
    assert out["UNITS_SOLD"].tolist() == [2, 0]
    assert set(out.columns) == {"ITEM_ID", "STORE_ID", "DAY_KEY", "SALE_DATE", "UNITS_SOLD"}


def test_missing_calendar_day_fails():
    wide = pd.DataFrame({"item_id": ["A"], "store_id": ["S1"], "d_1": [2]})
    with pytest.raises(ValueError, match="no matching calendar"):
        to_daily_sales(wide, pd.DataFrame({"d": ["d_2"], "date": ["2011-01-30"]}))


def test_negative_sales_fails():
    wide = pd.DataFrame({"item_id": ["A"], "store_id": ["S1"], "d_1": [-1]})
    with pytest.raises(ValueError, match="negative sales"):
        to_daily_sales(wide, pd.DataFrame({"d": ["d_1"], "date": ["2011-01-29"]}))
