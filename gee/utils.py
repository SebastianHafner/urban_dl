import ee


def compute_time_series_metrics(time_series: ee.ImageCollection, bands: list, metrics: list) -> ee.Image:

    time_series_metrics = []
    if 'mean' in metrics:
        time_series_metrics.append(time_series.reduce(ee.Reducer.mean()))
    if 'median' in metrics:
        time_series_metrics.append(time_series.reduce(ee.Reducer.median()))
    if 'max' in metrics:
        time_series_metrics.append(time_series.reduce(ee.Reducer.max()))
    if 'min' in metrics:
        time_series_metrics.append(time_series.reduce(ee.Reducer.min()))
    if 'stdDev' in metrics:
        time_series_metrics.append(time_series.reduce(ee.Reducer.stdDev()))

    time_series_metrics = ee.Image.cat(time_series_metrics)
    # debug0_names = time_series_metrics.bandNames().getInfo()
    new_order = [f'{band}_{metric}' for band in bands for metric in metrics]
    time_series_metrics = time_series_metrics.select(new_order)
    # debug1_names = time_series_metrics.bandNames().getInfo()

    return time_series_metrics

