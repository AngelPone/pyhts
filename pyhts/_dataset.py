import pandas as pd
import pkg_resources

__all__ = ["load_tourism"]


def load_tourism() -> pd.DataFrame:
    """Load the tourism dataset.

    :return:
    """
    with pkg_resources.resource_stream(__name__, "data/Tourism.csv") as f:
        return pd.read_csv(f)