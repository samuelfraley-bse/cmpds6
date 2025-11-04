import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import datetime as dt
import pandas as pd
import pytest
from geopy.distance import distance

from hw4 import count_simba, get_day_month_year, compute_distance, sum_general_int_list


# TESTS FOR count_simba
def test_count_simba_basic():
    sentences = [
        "Simba and Nala are lions.",
        "I laugh in the face of danger.",
        "Hakuna matata",
        "Timon, Pumba and Simba are friends, but Simba could eat the other two.",
    ]
    assert count_simba(sentences) == 3


def test_count_simba_zero_occurrences():
    sentences = ["Hello world", "No Simba here"]
    assert count_simba(sentences) == 1 


def test_count_simba_empty_list():
    assert count_simba([]) == 0


def test_count_simba_case_sensitive():
    sentences = ["simba is small", "SIMBA is big", "Simba is correct"]
    assert count_simba(sentences) == 1  


# TESTS FOR get_day_month_year
def test_get_day_month_year_basic():
    dates = [dt.date(2020, 5, 17), dt.date(2021, 6, 18), dt.date(2022, 7, 19)]
    df = get_day_month_year(dates)

    assert list(df.columns) == ["day", "month", "year"]
    assert df.shape == (3, 3)
    assert df.iloc[0]["day"] == 17
    assert df.iloc[1]["month"] == 6
    assert df.iloc[2]["year"] == 2022


def test_get_day_month_year_empty():
    df = get_day_month_year([])
    assert isinstance(df, pd.DataFrame)
    assert df.empty


def test_get_day_month_year_single_date():
    date = [dt.date(2000, 1, 1)]
    df = get_day_month_year(date)
    assert df.iloc[0].to_dict() == {"day": 1, "month": 1, "year": 2000}


# TESTS FOR compute_distance
def test_compute_distance_basic():
    coords = [((41.23, 23.5), (41.5, 23.4))]
    result = compute_distance(coords)

    assert isinstance(result, list)
    assert len(result) == 1
    assert pytest.approx(result[0], rel=1e-2) == distance(coords[0][0], coords[0][1]).km


def test_compute_distance_multiple_pairs():
    coords = [
        ((41.23, 23.5), (41.5, 23.4)),
        ((52.38, 20.1), (52.3, 17.8)),
    ]
    result = compute_distance(coords)
    assert len(result) == 2
    assert all(isinstance(d, float) for d in result)


def test_compute_distance_empty():
    assert compute_distance([]) == []


# TESTS FOR sum_general_int_list
def test_sum_general_int_list_simple():
    lst = [1, 2, 3, 4]
    assert sum_general_int_list(lst) == 10


def test_sum_general_int_list_nested():
    lst = [[2], 4, 5, [1, [2], [3, 5, [7, 8]]], 1]
    assert sum_general_int_list(lst) == 37


def test_sum_general_int_list_deeply_nested():
    lst = [[[1], [[2]], [[[3]]]]]
    assert sum_general_int_list(lst) == 6


def test_sum_general_int_list_empty():
    assert sum_general_int_list([]) == 0


def test_sum_general_int_list_mixed_types():
    lst = [1, [2, "hello", [3.5, [4]]]]
    assert sum_general_int_list(lst) == 7


