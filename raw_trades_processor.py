#!/usr/bin/env python

from numba import njit
import numpy as np
import pandas as pd
from typing import Tuple

class RawTradesProcessor:
    """
    Helper class to process "raw"/tick trades data retrieved from 
    binance (https://data.binance.vision/?prefix=data/spot/daily/trades/)
    into:
    - OHLC Candles
    - Renko Bars (all bars have the same absolute price difference)
    - Log Renko Bars (all bars have the same percentage price difference)
    - Volume Bars (all bars have the same volume)
    - Dollar Bars (all bars have the same quoted volume)
    
    Usage:
        processor = RawTradesProcessor(csv)

        ohlc = processor.raw_to_ohlc(args) 
        renko = processor.raw_to_renko(args) 
        log = processor.raw_to_renko_log(args)
        volume = processor.raw_to_volume_bars(args) 
        dollar = processor.raw_to_dollar_bars(args) 

    Args:
        csv (str): CSV's path to be read, can be .zip or .csv
    """
    def __init__(self, csv: str) -> None:
        self.csv = csv
        self._read_csv()
        self._aggregate_raw_trades()

    def _read_csv(self) -> None:
        """reads csv from disk, and store it as instance variable
        """
        header_raw_trades = [
            'id_trade', 'price', 'volume', 'volume_quoted', 'timestamp', 'is_buyer_maker','is_best_match']
        self.df = pd.read_csv(self.csv, names=header_raw_trades)

    def _aggregate_raw_trades(self) -> None:
        """Aggregates raw trades where the price, timestamp and is_buyer_maker
        is the same. 
        Computes the maker, taker volumes and number of trades.
        Stores the final aggregated dataframe as instance variable
        """
        self.df_agg = self.df.groupby(
            by = ['price', 'timestamp', 'is_buyer_maker'], 
            as_index=False, 
            sort=False
            ).agg(
                timestamp = ('timestamp', 'first'),  
                id_1st_trade = ('id_trade', 'first'),
                id_last_trade = ('id_trade', 'last'),
                price = ('price', 'first'),  
                volume = ('volume', 'sum'),
                volume_quoted = ('volume_quoted', 'sum'),
                is_buyer_maker = ('is_buyer_maker', 'first'),
        )   

        self.df_agg['timestamp'] = pd.to_datetime(self.df_agg['timestamp'], unit='ms')
        self.df_agg['volume_maker'] = np.where(self.df_agg['is_buyer_maker'], self.df_agg['volume'], 0)
        self.df_agg['volume_taker'] = np.where(self.df_agg['is_buyer_maker'], 0, self.df_agg['volume'])
        self.df_agg['volume_qt_maker'] = np.where(self.df_agg['is_buyer_maker'], self.df_agg['volume_quoted'], 0) 
        self.df_agg['volume_qt_taker'] = np.where(self.df_agg['is_buyer_maker'], 0, self.df_agg['volume_quoted'])
        self.df_agg['num_trades'] = self.df_agg['id_last_trade'] - self.df_agg['id_1st_trade'] + 1  # + itself
    
    def raw_to_ohlc(self, period: str) -> pd.DataFrame:
        """Converts raw trades to OHLC candles

        Args:
            period (str): Period of candles to resample to. Ex: 5min, 15min, 1h etc
                Note that the period format is "Offset aliases", ex 'T' is minute, 'H' hour, 'S' seconds etc
                see: https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html#offset-aliases

        Returns:
            pd.DataFrame: converted ohlc dataframe
        """
        df = self.df_agg.copy()
        df = df.set_index('timestamp', drop=False)
        
        df_ohlc = df.resample(period
            ).agg(
                open_time = ('timestamp', 'first'),
                close_time = ('timestamp', 'last'), 
                open =('price', 'first'),
                high = ('price', 'max'),
                low = ('price', 'min'),
                close =('price', 'last'),
                volume = ('volume', 'sum'),
                volume_quoted = ('volume_quoted', 'sum'),
                volume_maker = ('volume_maker', 'sum'),
                volume_taker = ('volume_taker', 'sum'),
                volume_qt_maker = ('volume_qt_maker', 'sum'),
                volume_qt_taker = ('volume_qt_taker', 'sum'),
                num_trades = ('num_trades', 'sum'),
        )

        return df_ohlc

    def _prepend_custom_price(self, df: pd.DataFrame, custom_starting_price: float) -> pd.DataFrame:
        """When the custom_starting_price is used (renko and log 
        transformations) this method is called to add a false "new trade" 
        to top of Dataframe with all values set to 0, except the custom price

        Args:
            df (pd.Dataframe): local df
            custom_starting_price (float): desired starting price

        Returns:
            pd.Dataframe: dataframe with new row prepended
        """
        lst = [
            df['timestamp'][0],
            0,  # id 1st trade
            0,  # id last trade
            custom_starting_price,  # price
            0,  # volume
            0,  # volume quoted
            False,  # is buyer maker
            0,  # volume maker
            0,  # volume taker
            0,  # volume quoted maker
            0,  # volume quoted taker
            0,  # number of trades
        ]
        df_new = pd.DataFrame([lst], columns=df.columns)
        df = pd.concat([df_new , df], ignore_index=True)

        return df 

    def _get_df(self, df_prev_remains: pd.DataFrame, custom_starting_price: float = None) -> pd.DataFrame:
        """Method used to get the local temporary dataframe, ready to work.
        This is, if there are no previous remains passed, it returns only a copy
        of instance's aggregated dataframe, otherwise it returns both dataframe
        merged.

        In case a custom_starting_price has been passed it also concatenates it 
        to the top of dataframe as well.

        Args:
            df_prev_remains (pd.DataFrame): dataframe with previous remains
            custom_starting_price (float, optional): desired starting price

        Returns:
            pd.DataFrame: dataframe ready to work, process, transform
        """
        if df_prev_remains.empty:
            df = self.df_agg.copy()
        else:
            df = pd.concat([df_prev_remains, self.df_agg], ignore_index=True, copy=True)
        
        if custom_starting_price:
            df = self._prepend_custom_price(df, custom_starting_price)

        return df

    @staticmethod 
    @njit
    def _replicate_groups(groups: np.ndarray) -> np.ndarray:
        """Receives an 1D array of groups where 0 means no group and replaces 
        the 0 by the corresponding group. Replaces the last rows by nans, as 
        they dont belong to any group. 
        Note: 0 means that no group was assigned.
        
        Example:
        In:  0,0,1,0,0,2,0
        Out: 1,1,1,2,2,2,Nan

        Args:
            groups (np.ndarray): floats array with groups in form 0,0,1,0,0,2,0

        Returns:
            np.ndarray: groups array in form 1,1,1,2,2,2,Nan
        """
        uniques = np.unique(groups)
        uniques = np.append(uniques, uniques[-1] + 1)
        uniques = np.delete(uniques, 0)  # group 0 means that no group was defined

        j = 0
        for i in range(groups.shape[0]):
            if groups[i] == uniques[j]:
                j += 1
            else:
                groups[i] = uniques[j]
        
        # group of trades that didn't fill a complete bar/candle/group
        # replaced by nan, to be ignored by groupby
        groups[ groups == uniques[-1] ] = np.nan
        
        return groups

    def _get_remaining_rows(
            self,
            df: pd.DataFrame,
            groups: np.ndarray,
            cumsums_arr: np.ndarray,
            values_arr: np.ndarray
    ) -> Tuple[pd.DataFrame, float]:
        """Gets the last rows of a DataFrame that didn't form any group, which
        means that no brick, candle, etc is possible to form from this rows.
        So, this rows will be passed as a separated variable to process later.
        It will also be passed the last cumsum of the last row with a group.

        Args:
            df (pd.DataFrame): local dataframe
            groups (np.ndarray): 1D array with groups
            cumsums_arr (np.ndarray): 1D array of cumsums or cumdiffs or etc
            values_arr (np.ndarray): 1D array of values or diffs or etc

        Returns:
            Tuple[pd.DataFrame, float]: Tuple containing, respectively, a 
            Dataframe with the remaining rows and a float with last value/diff 
        """
        inds_nans = np.isnan(groups)
        first_nan = inds_nans.argmax()

        # gives the remainder of the cumsum when the previous group was formed
        # ex: if the group closed at 50, and the cumsum was 50.14 this will get 
        # the remainder of 0.14
        # obtained at the first nan, values with nans means that these lines did 
        # not form a group and will be added to the next file
        cumsum_remainder = cumsums_arr[ first_nan ] - values_arr[ first_nan ]
        df_remaining = df[inds_nans].copy()

        return df_remaining, cumsum_remainder

    def _append_new_rows(self, df: pd.DataFrame, new_rows:list) -> pd.DataFrame:
        """Appends the list new_rows to df and returns a DataFrame with both,
        new_rows in the end of dataframe

        Args:
            df (pd.DataFrame): local dataframe
            new_rows (list): list of new_rows with the same dimensions as df

        Returns:
            pd.DataFrame: merged dataframe, with new_rows at the end
        """
        df_new = pd.DataFrame(new_rows, columns=df.columns)
        df = pd.concat([df, df_new], ignore_index=True, copy=True)
        
        return df
    
    @staticmethod 
    @njit
    def _dynamic_cumdiff(arr: np.ndarray, maximum: float, cumsum: float = 0.0) -> Tuple[np.ndarray, np.ndarray]:
        """Dynamic cumulative sum of differences. Computes differences and then 
        sums them until the maximum value is reached. When the max value is 
        reached, the cumsum is truncated at the maximum and the remaining value
        is forwarded to the next iteration. 
        Optimized with numba njit.

        Args:
            arr (np.ndarray): 1D array to be difirenced and cumsumed
            maximum (float): maximum value of the cumsum
            cumsum (float, optional): Initial value of the cumsum. 
            Defaults to 0.0.

        Returns:
            Tuple[np.ndarray, np.ndarray]: Tuple with the following order:
            1D array with cumsums of differences and 1D array with differences.
        """        
        diffs = np.diff(arr)
        diffs = np.append(0, diffs)  # to keep shape

        multiple = 0.0
        remainder = 0.0
        cumsums_arr = np.empty(diffs.shape[0], dtype=np.float64)
        
        for i in range(diffs.shape[0]):
            cumsum += diffs[i]

            # abs must be used because floor division of a negative number floors it down. 
            # Ex: -1.14 // 1 = -2 with a remainder of 0.86 instead of -1 with a remainder of -0.14
            absolute = abs(cumsum)

            if absolute >= maximum: 
                remainder = absolute % maximum
                multiple = absolute - remainder
                
                if cumsum < 0:
                    multiple = -multiple
                    remainder = -remainder
            
                cumsum = remainder        
                cumsums_arr[i] = multiple
        
            else:
                cumsums_arr[i] = cumsum  
            
        return cumsums_arr, diffs

    def _append_row_renko(self, lst: list, df: pd.DataFrame, group: float, open: float, close: float) -> None:
        """Appends new row to lst with the shape of local renko dataframe.
        Auxiliary method used when a single trade makes several blocks.
        Doesn't return anything but appends row to lst.

        Args:
            lst (list): list to be appended
            df (pd.DataFrame): local dataframe
            group (float): group value of this row
            open (float): open price of this row
            close (float): close price of this row
        """
        lst.append([ 
            df['timestamp'],
            df['id_1st_trade'],
            df['id_last_trade'],
            df['price'],
            0,  # volume
            0,  # volume_quoted
            df['is_buyer_maker'],
            0,  # volume maker
            0,  # volume taker
            0,  # volume qt maker
            0,  # volume qt taker
            1,  # num trades
            open,  # renko open
            close,  # renko close
            group,  # group      
        ])

    def _compute_renko_groups(
        self, 
        df: pd.DataFrame, 
        cumdiffs_arr: np.ndarray, 
        brick_size: float, 
        open: float = None,
        uptrend: bool = None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, list, float, bool]:
        """Computes groups according to renko rules to later be used in groupby.
        General renko rules: price has to rise at least a brick size (ex: 50) 
        to form a new block if it is an uptrend, has to fall at least a brick
        size in a downtrend. To invert a trend the price has to change by two
        brick sizes. 

        Args:
            df (pd.DataFrame): local dataframe
            cumdiffs_arr (np.ndarray): array with dynamic cumsums of differences
            brick_size (float): desired brick size
            open (float, optional): last brick's close
            uptrend (bool, optional): last trend

        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray, list, float, bool]: Tupple
            with the following order: 1D array of computed groups, 1D array of 
            opens, 1D array of closes, list with new rows, last close and last 
            trend.
            Last line of code:
            "groups_arr, opens_arr, closes_arr, new_rows, close, uptrend"
        """
        if open is None:
            open = df['price'][0]

        cumdiff = 0.0
        absolute = 0.0
        group = 1.0
        close = open
        new_rows = []  # new rows that are formed when a single trade makes several bricks
        groups_arr = np.zeros(cumdiffs_arr.shape[0], dtype=np.float64) 
        closes_arr = np.zeros(cumdiffs_arr.shape[0], dtype=np.float64)  
        opens_arr = np.zeros(cumdiffs_arr.shape[0], dtype=np.float64)   

        # indices of cumdiffs where a new block may be formed, there is no need to iterate over every row
        inds = np.where(np.remainder(cumdiffs_arr, brick_size) == 0)[0]  

        for i in inds:
            cumdiff += cumdiffs_arr[i]

            brick_step = np.sign(cumdiff) * brick_size
            quotient = cumdiff // brick_size
            
            # it is always int as this for is only iterating over inds where a block may be formed
            absolute = int(abs(quotient))  

            if  (absolute >= 1 and uptrend is None):  # trend not defined yet

                if brick_step > 0:
                    uptrend = True
                else:
                    uptrend = False 

                groups_arr[i] = group
                group += 1

                for _ in range(absolute - 1):
                    close = open + brick_step
                    self._append_row_renko(new_rows, df.iloc[i], group, open, close)
                    open = close
                    group += 1
                
                close = open + brick_step
                opens_arr[i] = open
                closes_arr[i] = close        
                open = close
                cumdiff = 0.0

            elif (  # trend stays the same
                (quotient >= 1.0 and uptrend) or  
                (quotient <= -1.0 and not uptrend)
                ):  

                groups_arr[i] = group
                group += 1

                for _ in range(absolute - 1):
                    close = open + brick_step
                    self._append_row_renko(new_rows, df.iloc[i], group, open, close)
                    open = close
                    group += 1       
        
                close = open + brick_step
                opens_arr[i] = open
                closes_arr[i] = close        
                open = close
                cumdiff = 0.0

            elif (absolute >= 2.0): # trend inverts

                uptrend = not uptrend

                groups_arr[i] = group
                group += 1
                open += brick_step

                for _ in range(absolute - 2):
                    close = open + brick_step
                    self._append_row_renko(new_rows, df.iloc[i], group, open, close)
                    open = close
                    group += 1      

                close = open + brick_step
                opens_arr[i] = open
                closes_arr[i] = close        
                open = close
                cumdiff = 0.0

        return groups_arr, opens_arr, closes_arr, new_rows, close, uptrend

    def _update_df_renko(self, df:pd.DataFrame, opens: np.ndarray, closes: np.ndarray, groups: np.ndarray) -> None:
        """Adds new columns to dataframe with corresponding labels.

        Args:
            df (pd.DataFrame): local dataframe
            opens (np.ndarray): 1D array of open prices
            closes (np.ndarray): 1D arrays of close prices
            groups (np.ndarray): 1D array of groups
        """
        df['r_open'] = opens
        df['r_close'] = closes
        df['groups'] = groups

    def _groupby_renko(self, df: pd.DataFrame) -> pd.DataFrame:
        """Groupby local dataframe by groups to renko dataframe

        Args:
            df (pd.DataFrame): local dataframe

        Returns:
            pd.DataFrame: renko dataframe
        """
        df_renko = df.reset_index() \
        .groupby(
            by='groups',
            as_index=False,
            sort=False
        ).agg(
            open_time = ('timestamp', 'first'),
            close_time = ('timestamp', 'last'),
            open =('r_open', 'last'),
            close =('r_close', 'last'),
            high = ('price', 'max'),
            low = ('price', 'min'),
            volume = ('volume', 'sum'),
            volume_quoted = ('volume_quoted', 'sum'),
            volume_maker = ('volume_maker', 'sum'),
            volume_taker = ('volume_taker', 'sum'),
            volume_qt_maker = ('volume_qt_maker', 'sum'),
            volume_qt_taker = ('volume_qt_taker', 'sum'),
            num_trades = ('num_trades', 'sum'),
            id_last_trade = ('id_last_trade', 'last'),
        ).sort_values(
            by = ['id_last_trade', 'volume', 'groups'], 
            ignore_index = True
        ).drop(
            ['groups', 'id_last_trade'],
            axis = 1
        )

        return df_renko

    def raw_to_renko(
            self, 
            brick_size: float, 
            df_prev_remains: pd.DataFrame = pd.DataFrame(),
            open: float = None,
            last_trend: bool = None,
            last_cumdiff: float = 0.0,
            custom_starting_price: float = None,
    ) -> Tuple[pd.DataFrame, pd.DataFrame, float, bool, float]:
        """Converts instance dataframe to renko style dataframe.
        Accepts and returns remaining variables with remaining values to be
        compatible with next renko, to be possible to compute renko on the next
        day keeping coherence. 

        Args:
            brick_size (float): desired brick size. Ex: 10, 25, 50
            df_prev_remains (pd.DataFrame, optional): Dataframe with previous 
            remaining rows. Defaults to pd.DataFrame().
            open (float, optional): Previous last closing price. Defaults
            to None.
            last_trend (bool, optional): Previous last trend. Defaults to None.
            last_cumdiff (float, optional): Previous last cumsum of differences.
            Defaults to 0.0.
            custom_starting_price (float, optional): Desired starting price, to
            make graphs prettier. BE VERY CAREFUL USING THIS, it introduces 
            error in system. Defaults to None.

        Returns:
            Tuple[pd.DataFrame, pd.DataFrame, float, bool, float]: Tuple with
            following order: Renko DataFrame, Remaining Rows Dataframe, Last
            closing price, last trend, last cumsum of differences.
            Last line of code:
            "df_renko, df_remaining, last_close, last_trend, last_cumdiff"
        """
        df = self._get_df(df_prev_remains, custom_starting_price)

        prices = df['price'].to_numpy()
        cumdiffs_arr, diffs_arr = self._dynamic_cumdiff(prices, brick_size, last_cumdiff)

        groups_arr, opens_arr, closes_arr, new_rows, last_close, last_trend = self._compute_renko_groups(
            df, cumdiffs_arr, brick_size, open, last_trend)
        
        groups_arr = self._replicate_groups(groups_arr)

        df_remaining, last_cumdiff = self._get_remaining_rows(df, groups_arr, cumdiffs_arr, diffs_arr)

        self._update_df_renko(df, opens_arr, closes_arr, groups_arr)

        if new_rows:
            df = self._append_new_rows(df, new_rows)

        df_renko = self._groupby_renko(df)

        return df_renko, df_remaining, last_close, last_trend, last_cumdiff

    def raw_to_renko_log(
            self, 
            percentage: float, 
            df_prev_remains: pd.DataFrame = pd.DataFrame(),
            open: float = None,
            last_trend: bool = None,
            last_cumdiff: float = 0.0,
            custom_starting_price: float = None,
    ) -> Tuple[pd.DataFrame, pd.DataFrame, float, bool, float]:
        """Converts instance dataframe to renko style dataframe with logarithmic
        differences. Every brick's open price differs from close by a fixed 
        percentage.
        Accepts and returns remaining variables with remaining values to be
        compatible with next renko, to be possible to compute renko on the next
        day keeping coherence. 

        Args:
            percentage (float): desired percentage between 0-1 to form bricks. 
            Example: 0.15 -> 15%
            df_prev_remains (pd.DataFrame, optional): Dataframe with previous 
            remaining rows. Defaults to pd.DataFrame().
            open (float, optional): Previous last closing price. Defaults
            to None.
            last_trend (bool, optional): Previous last trend. Defaults to None.
            last_cumdiff (float, optional): Previous last cumsum of differences.
            Defaults to 0.0.
            custom_starting_price (float, optional): Desired starting price, to
            make graphs prettier. BE VERY CAREFUL USING THIS, it introduces 
            error in system. Defaults to None.

        Returns:
            Tuple[pd.DataFrame, pd.DataFrame, float, bool, float]: Tuple with
            following order: Renko Log DataFrame, Remaining Rows Dataframe, Last
            closing price, last trend, last cumsum of differences.
            Last line of code:
            "df_log, df_remaining, last_close, last_trend, last_cumdiff"
        """
        df = self._get_df(df_prev_remains, custom_starting_price)
        
        if open is None:
            open = df['price'][0]

        # convert prices to log
        brick_size = np.log(1 + percentage)
        open = np.log(open)
        prices = np.log(df['price'].to_numpy())

        cumdiffs_arr, diffs = self._dynamic_cumdiff(prices, brick_size, last_cumdiff)

        groups_arr, opens_arr, closes_arr, new_rows, last_close, last_trend = self._compute_renko_groups(
            df, cumdiffs_arr, brick_size, open, last_trend)
        
        groups_arr = self._replicate_groups(groups_arr)

        df_remaining, last_cumdiff = self._get_remaining_rows(df, groups_arr, cumdiffs_arr, diffs)

        self._update_df_renko(df, opens_arr, closes_arr, groups_arr)

        if new_rows:
            df = self._append_new_rows(df, new_rows)

        df_log = self._groupby_renko(df)

        # convert prices back to normal, only open and close prices are affected
        df_log['open'] = np.exp(df_log['open'])
        df_log['close'] = np.exp(df_log['close'])

        return df_log, df_remaining, last_close, last_trend, last_cumdiff

    @staticmethod
    @njit
    def _dynamic_cumsum_volume(
        main_vol: np.ndarray,
        maximum: float,
        all_volumes: np.ndarray,
        cumsum: float = 0.0
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Dynamic cumulative sum of volumes. It sums main volume until maximum 
        value is reached. When the max value is reached, the cumsum is truncated
        at the maximum and the remaining value is forwarded to the next 
        iteration. 
        The other volumes are summed and truncated according to main volume,
        they have theirs cumsums in proportion to main cumsum.

        Args:
            main_vol (np.ndarray): 1D array with volume where the cumsum will 
            be referring
            maximum (float): maximum value of sum, where sum will be truncated 
            all_volumes (np.ndarray): all volumes, to be sumed as main volume
            cumsum (float, optional): Initial value of cumsum Defaults to 0.0.

        Returns:
            Tuple[np.ndarray, np.ndarray]: Tupple in the following order, 1D 
            array with cumsum of the main volume, ND array with the cumsum of 
            all other volumes.
        """
        multiple = 0.0
        remainder = 0.0
        cumsum_others = np.zeros(all_volumes.shape[1], dtype=np.float64)
        remainder_others = np.empty(all_volumes.shape[1], dtype=np.float64)
        cumsums = np.empty(main_vol.shape, dtype=np.float64)
        cumsums_others = np.empty(all_volumes.shape, dtype=np.float64)

        for i in range(main_vol.shape[0]):
            cumsum += main_vol[i]
            cumsum_others += all_volumes[i]

            if cumsum >= maximum:
                remainder = cumsum % maximum
                multiple = cumsum - remainder
                
                # computes the remaining cumsum (in proportion to the main volume) on other volumes
                remainder_others = cumsum_others * remainder / cumsum 

                cumsum = remainder        
                cumsums[i] = multiple

                cumsums_others[i] = cumsum_others - remainder_others
                cumsum_others = remainder_others

            else:
                cumsums[i] = cumsum
                cumsums_others[i] = cumsum_others
                
        return cumsums, cumsums_others

    def _compute_volume_groups(
            self,
            df: pd.DataFrame,
            main_cumvol: np.ndarray,
            brick_size: float,
            all_cumvols: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, list]:
        """Computes groups according to volume cumulative sum. Every group has 
        the same volume.

        Args:
            df (pd.DataFrame): local dataframe
            main_cumvol (np.ndarray): main volume
            brick_size (float): desired brick size
            all_cumvols (np.ndarray): all other volumes

        Returns:
            Tuple[np.ndarray, np.ndarray, list]: Tuple with the following order:
            1D array with groups, ND array with all_volumes and a list with new
            rows.
        """
        quotient = 0
        g = 1.0
        new_rows = []  # new rows that are formed when a single trade makes several bricks
        row = np.empty(all_cumvols.shape[1], dtype=np.float64)
        groups_arr = np.zeros(main_cumvol.shape, dtype=np.float64)

        # indices of cumdiffs multiple of brick_size, this is, inds where a new block may be formed
        inds = np.where(np.remainder(main_cumvol, brick_size) == 0)[0]
        
        for i in inds:
            quotient = int(main_cumvol[i] // brick_size)
            row = all_cumvols[i]

            if  (quotient >= 1): 
                row = row / quotient  # volume is the same on every rows
        
                groups_arr[i] = g
                g += 1

                for _ in range(quotient-1):
                    new_rows.append([ 
                        df['timestamp'][i],
                        df['id_1st_trade'][i],
                        df['id_last_trade'][i],
                        df['price'][i],
                        row[0],  # volume
                        row[1],  # volume_quoted
                        df['is_buyer_maker'][i],
                        row[2],  # volume maker
                        row[3],  # volume taker
                        row[4],  # volume qt maker
                        row[5],  # volume qt taker
                        1,  # num trades
                        g,  # group      
                    ])

                    g += 1
                
                all_cumvols[i] = row
        
        # it is only needed to return the all_cumvols as the main vol should have been incorporated in this
        # one before this method
        return groups_arr, all_cumvols, new_rows

    def _update_df_volume(
            self,
            df: pd.DataFrame,
            all_volumes: np.ndarray,
            groups: np.ndarray
    ) -> None:
        """Adds new columns to dataframe with corresponding labels.

        Args:
            df (pd.DataFrame): local dataframe
            all_volumes (np.ndarray): matrix (ND array) with all volumes
            groups (np.ndarray): array of groups
        """
        df['volume'] = all_volumes[:,0]
        df['volume_quoted'] = all_volumes[:,1]
        df['volume_maker'] = all_volumes[:,2]
        df['volume_taker'] = all_volumes[:,3]
        df['volume_qt_maker'] = all_volumes[:,4]
        df['volume_qt_taker'] = all_volumes[:,5]
        df['groups'] = groups

    def _groupby_volume(self, df: pd.DataFrame) -> pd.DataFrame:
        """Groupby local dataframe by groups to volume resampled dataframe

        Args:
            df (pd.DataFrame): local dataframe

        Returns:
            pd.DataFrame: volume resampled dataframe
        """
        df_volume = df.reset_index(
        ).groupby(
            by='groups',
            as_index=False,
            sort=True,
        ).agg(
            open_time = ('timestamp', 'first'),
            close_time = ('timestamp', 'last'),
            open =('price', 'last'),
            close =('price', 'last'),
            high = ('price', 'max'),
            low = ('price', 'min'),
            volume = ('volume', 'last'),
            volume_quoted = ('volume_quoted', 'last'),
            volume_maker = ('volume_maker', 'last'),
            volume_taker = ('volume_taker', 'last'),
            volume_qt_maker = ('volume_qt_maker', 'last'),
            volume_qt_taker = ('volume_qt_taker', 'last'),
            num_trades = ('num_trades', 'sum'),
        ).drop(
            'groups',
            axis=1
        )
    
        return df_volume
    
    def raw_to_volume_bars(
            self, 
            brick_size: float, 
            df_prev_remains: pd.DataFrame = pd.DataFrame(),
            prev_cumvol: float = 0.0,
    ) -> Tuple[pd.DataFrame, pd.DataFrame, float]:
        """Converts instance dataframe to volume resampled dataframe (every bar
        has the same volume).
        Accepts and returns remaining variables with remaining values to be
        compatible with next dataframe, to be possible to compute next
        resampling on the next day keeping coherence.

        Args:
            brick_size (float): desired volume brick size
            df_prev_remains (pd.DataFrame, optional): Dataframe with previous 
            remaining rows. Defaults to pd.DataFrame().
            prev_cumvol (float, optional): Previous last cumsum of volume. 
            Defaults to 0.0.

        Returns:
            Tuple[pd.DataFrame, pd.DataFrame, float]: Tuple with the folloing
            order: Volume Resampled Dataframe, Remaining Rows DataFrame and last
            cumsum of volume.
        """
        df = self._get_df(df_prev_remains)

        all_volumes = df[
            ['volume', 'volume_quoted', 'volume_maker', 'volume_taker', 'volume_qt_maker', 'volume_qt_taker']
        ].to_numpy()

        volume = df['volume'].to_numpy()
        cumvol, cumvol_all = self._dynamic_cumsum_volume(volume, brick_size, all_volumes, prev_cumvol)
        cumvol_all[:,0] = cumvol

        groups_arr, cumvol_all, new_rows = self._compute_volume_groups(df, cumvol, brick_size, cumvol_all)
        groups_arr = self._replicate_groups(groups_arr)

        df_remaining, prev_cumvol = self._get_remaining_rows(df, groups_arr, cumvol, volume)

        self._update_df_volume(df, cumvol_all, groups_arr)

        if new_rows:
            df = self._append_new_rows(df, new_rows)

        df_volume = self._groupby_volume(df)

        return df_volume, df_remaining, prev_cumvol

    def raw_to_dollar_bars(
            self, 
            brick_size: float, 
            df_prev_remains: pd.DataFrame = pd.DataFrame(),
            prev_cumvol: float = 0.0,
    ) -> Tuple[pd.DataFrame, pd.DataFrame, float]:
        """Converts instance dataframe to dollar resampled dataframe (every bar
        has the same quoted volume).
        Accepts and returns remaining variables with remaining values to be
        compatible with next dataframe, to be possible to compute next
        resampling on the next day keeping coherence.

        Args:
            brick_size (float): desired dollar/volume quoted brick size
            df_prev_remains (pd.DataFrame, optional): Dataframe with previous 
            remaining rows. Defaults to pd.DataFrame().
            prev_cumvol (float, optional): Previous last cumsum of quoted 
            volume. Defaults to 0.0.

        Returns:
            Tuple[pd.DataFrame, pd.DataFrame, float]: Tuple with the folloing
            order: Dollar Resampled Dataframe, Remaining Rows DataFrame and last
            cumsum of volume.
        """

        df = self._get_df(df_prev_remains)

        all_volumes = df[
            ['volume', 'volume_quoted', 'volume_maker', 'volume_taker', 'volume_qt_maker', 'volume_qt_taker']
        ].to_numpy()

        volume = df['volume_quoted'].to_numpy()
        cumvol, cumvol_all = self._dynamic_cumsum_volume(volume, brick_size, all_volumes, prev_cumvol)
        cumvol_all[:,1] = cumvol

        groups_arr, cumvol_all, new_rows = self._compute_volume_groups(df, cumvol, brick_size, cumvol_all)
        groups_arr = self._replicate_groups(groups_arr)

        df_remaining, prev_cumvol = self._get_remaining_rows(df, groups_arr, cumvol, volume)

        self._update_df_volume(df, cumvol_all, groups_arr)

        if new_rows:
            df = self._append_new_rows(df, new_rows)

        df_volume = self._groupby_volume(df)

        return df_volume, df_remaining, prev_cumvol