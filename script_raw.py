#!/usr/bin/env python

"""Script to convert raw trades into ohlc, renko and volume resampled
"""

from datetime import date, timedelta
import pickle
import timeit

from raw_trades_processor import RawTradesProcessor


tic=timeit.default_timer()

directory = '/mnt/d/data'
pair = 'BTCBUSD'
period_ohlc = '15T'
renko_size = 50
renko_starting_price = 20500  # be very careful with this
log_size = 0.0024  # 0.24%
vol_size = 1200
dollar_size = 30000000

d1 = date(2023,1,30)
d2 = date(2023,1,31)

# this will give you a list containing all of the dates
dd = [d1 + timedelta(days=x) for x in range((d2-d1).days + 1)]

d = dd[0]
processor = RawTradesProcessor(f'{directory}/{pair}-trades-{str(d)}.zip')

# ohlc
processor.raw_to_ohlc(period_ohlc).to_csv(
    f'{directory}/{str(d.year)}/{str(d.month)}/{pair}-ohlc-{period_ohlc}-{str(d)}.csv', index=False)

# renko
df_renko, df_rem_renko, prev_close_renko, prev_trend_renko, prev_cumdiff_renko = processor.raw_to_renko(
    renko_size, custom_starting_price=renko_starting_price)  # just to be prettier
df_renko.to_csv(
    f'{directory}/{str(d.year)}/{str(d.month)}/{pair}-renko-{renko_size}-{str(d)}.csv', index=False)

# log renko
df_log, df_rem_log, prev_close_log, prev_trend_log, prev_cumdiff_log = processor.raw_to_renko_log(log_size)
df_log.to_csv(
    f'{directory}/{str(d.year)}/{str(d.month)}/{pair}-log-{log_size}-{str(d)}.csv', index=False)

# volume
df_vol, df_rem_vol, prev_cumvol = processor.raw_to_volume_bars(vol_size)
df_vol.to_csv(
    f'{directory}/{str(d.year)}/{str(d.month)}/{pair}-vol-{vol_size}-{str(d)}.csv', index=False)

# dollar / volume quoted
df_dollar, df_rem_dollar, prev_cumdollar = processor.raw_to_dollar_bars(dollar_size)
df_dollar.to_csv(
    f'{directory}/{str(d.year)}/{str(d.month)}/{pair}-dollar-{dollar_size}-{str(d)}.csv', index=False)

print(f'Success: {directory}/{pair}-trades-{str(d)}.zip')

for d in dd[1:]:
    processor = RawTradesProcessor(f'{directory}/{pair}-trades-{str(d)}.zip')

    # ohlc
    processor.raw_to_ohlc(period_ohlc).to_csv(
        f'{directory}/{str(d.year)}/{str(d.month)}/{pair}-ohlc-{period_ohlc}-{str(d)}.csv', index=False)

    # renko
    df_renko, df_rem_renko, prev_close_renko, prev_trend_renko, prev_cumdiff_renko = processor.raw_to_renko(
        renko_size, df_rem_renko, prev_close_renko, prev_trend_renko, prev_cumdiff_renko)
    df_renko.to_csv(
        f'{directory}/{str(d.year)}/{str(d.month)}/{pair}-renko-{renko_size}-{str(d)}.csv', index=False)

    # log renko
    df_log, df_rem_log, prev_close_log, prev_trend_log, prev_cumdiff_log = processor.raw_to_renko_log(log_size)
    df_log.to_csv(
        f'{directory}/{str(d.year)}/{str(d.month)}/{pair}-log-{log_size}-{str(d)}.csv', index=False)

    # volume
    df_vol, df_rem_vol, prev_cumvol = processor.raw_to_volume_bars(vol_size)
    df_vol.to_csv(
        f'{directory}/{str(d.year)}/{str(d.month)}/{pair}-vol-{vol_size}-{str(d)}.csv', index=False)

    # dollar / volume quoted
    df_dollar, df_rem_dollar, prev_cumdollar = processor.raw_to_dollar_bars(dollar_size)
    df_dollar.to_csv(
        f'{directory}/{str(d.year)}/{str(d.month)}/{pair}-dollar-{dollar_size}-{str(d)}.csv', index=False)

    print(f'Success: {directory}/{pair}-trades-{str(d)}.zip')

with open(f'{directory}/{pair}-remains-{str(d)}.pkl', 'wb') as f:
    pickle.dump(
        [
        df_rem_renko, prev_close_renko, prev_trend_renko, prev_cumdiff_renko,
        df_rem_log, prev_close_log, prev_trend_log, prev_cumdiff_log,
        df_rem_vol, prev_cumvol,
        df_rem_dollar, prev_cumdollar
        ],
        f)
print(f'Success pkl: {directory}/{pair}-remains-{str(d)}.pkl')

toc=timeit.default_timer()

print("Total Time: " + str(int(toc - tic)) + 's')
