import pytest
from retailiq.inventory import recommend_order


def test_order_recommendation():
    # 70/week => 10/day; 7-day lead + 3-day safety => target 100
    r = recommend_order(70, 20, 7, 3)
    assert r['recommended_order'] == 80
    assert r['inventory_position'] == 20
    assert r['stock_cover_days'] == 2


def test_incoming_stock_reduces_order():
    assert recommend_order(70, 20, 7, 3, on_order=30)['recommended_order'] == 50


def test_backorders_raise_order():
    assert recommend_order(70, 20, 7, 3, backorders=5)['recommended_order'] == 85


def test_zero_demand():
    r = recommend_order(0, 0, 7)
    assert r['recommended_order'] == 0
    assert r['stock_cover_days'] is None


@pytest.mark.parametrize('forecast', [-1, float('nan'), float('inf')])
def test_invalid_forecast(forecast):
    with pytest.raises(ValueError):
        recommend_order(forecast, 1, 3)


def test_invalid_stock():
    with pytest.raises(ValueError):
        recommend_order(10, -1, 7)
