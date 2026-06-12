import numpy as np
import pandas as pd
import scipy.stats


def test_column_names(data: pd.DataFrame) -> None:
    """Test if the DataFrame has the expected column names."""
    expected_columns = [
        "id",
        "name",
        "host_id",
        "host_name",
        "neighbourhood_group",
        "neighbourhood",
        "latitude",
        "longitude",
        "room_type",
        "price",
        "minimum_nights",
        "number_of_reviews",
        "last_review",
        "reviews_per_month",
        "calculated_host_listings_count",
        "availability_365",
    ]

    these_columns = data.columns.values
    assert list(expected_columns) == list(these_columns)


def test_neighborhood_names(data: pd.DataFrame) -> None:
    """Test if neighborhood group names are within expected values."""
    known_names = ["Bronx", "Brooklyn", "Manhattan", "Queens", "Staten Island"]
    neigh = set(data["neighbourhood_group"].unique())

    assert set(known_names) == set(neigh)


def test_proper_boundaries(data: pd.DataFrame) -> None:
    """Test proper longitude and latitude boundaries for NYC properties."""
    idx = data["longitude"].between(-74.25, -73.50) & data["latitude"].between(
        40.5, 41.2
    )

    assert np.sum(~idx) == 0


def test_similar_neigh_distrib(
    data: pd.DataFrame, ref_data: pd.DataFrame, kl_threshold: float
) -> None:
    """Apply a threshold on KL divergence for neighborhood distribution."""
    dist1 = data["neighbourhood_group"].value_counts(normalize=True).sort_index()
    dist2 = ref_data["neighbourhood_group"].value_counts(normalize=True).sort_index()

    assert np.isclose(dist1.sum(), 1.0)
    assert np.isclose(dist2.sum(), 1.0)
    assert dist1.index.equals(dist2.index)

    kl_div = scipy.stats.entropy(dist1, dist2, base=2)

    assert np.isfinite(kl_div)
    assert kl_div < kl_threshold


def test_row_count(data: pd.DataFrame) -> None:
    """Test that the dataset has a reasonable number of rows."""
    assert 15000 < data.shape[0] < 1000000


def test_price_range(data: pd.DataFrame, min_price: float, max_price: float) -> None:
    """Test that all listing prices are within the accepted range."""
    assert data["price"].between(min_price, max_price).all()
