import pandas as pd
import pkg_resources


def load_tourism() -> pd.DataFrame:
    """Load tourism Dataset

    :return:
    """
    stream = pkg_resources.resource_stream(__name__, "data/Tourism.csv")

    return pd.read_csv(stream)