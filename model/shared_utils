
class TreatmentCoder:
    """
    class to encode and decode the multiple treatments
    encoder: encode list of treatment to one string
    decoder: decode the string of treatment to list of treatment
    """
    @staticmethod
    def name_encoder(treatment=[]):
        return '-'.join(treatment)

    @staticmethod
    def name_decoder(name=''):
        return name.split('-')


def breakdown_ts(ts, date1):
    """
    function to breakdown the time series by date
    :param ts: dataframe, timeXregion
    :param date1: date1 to breakdown
    :return: three dataframes
    """
    if date1 < ts.index.min() or date1 > ts.index.max():
        raise ValueError('The date is not in the range')
    ts.sort_index(inplace=True)
    pre = ts.loc[ts.index < date1].copy()
    post = ts.loc[ts.index >= date1].copy()
    return pre, post
