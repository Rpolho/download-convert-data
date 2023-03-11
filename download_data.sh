#!/bin/bash

#
# Script to download data from Binance platform (https://data.binance.vision/?prefix=data/spot/daily/)
# Useful to automatically download multiple zips
#
# Usage:
# - chose your preferences below
# - run this script by: ./download_data.sh
# - you can time it by: time ./download_data.sh
#

# chose your preferences
first_date=2022-11-1  # first date to download
second_date=2023-3-1 # last date to download
directory="/mnt/d/data/" # basis directory 
pair="ETHUSDT" # coin pair
periods=("1s" "5m" "15m") # klines periods  

baseurl="https://data.binance.vision/data/spot/daily"

# get current directory and date to write logs
current_date=$(date +%Y%m%d%H)
current_directory="$PWD"
mkdir -p "${current_directory}/logs"

# change directory to basis directory
cd ${directory} 

# check dates
start_date=$(date -I -d "$first_date") || exit -1
end_date=$(date -I -d "$second_date")  || exit -1

if [ "$start_date" \> "$end_date" ]; then
    echo "First Date Larger Than Second Date!"
    exit -1
fi

#
# Download candles / klines for every period
# https://data.binance.vision/data/spot/daily/klines/BTCUSDT/1s/BTCUSDT-1s-2023-03-08.zip
#
for period in ${periods[@]}; do
    d="$start_date"
    until [ "$d" \> "$end_date" ]; do # until d larger than end date

        file_name="${pair}-${period}-$d.zip"
        url="${baseurl}/klines/${pair}/${period}/${file_name}"
        
        # download zip
        response=$(wget --server-response --quiet ${url} 2>&1 | awk 'NR==1{print $2}')
        if [ ${response} == '404' ]; then
            echo "! File does not exist: ${url}" | tee -a "${current_directory}/logs/error-${current_date}.txt"
        else
            echo "${file_name} Downloaded .zip" | tee -a "${current_directory}/logs/log-${current_date}.txt"
        fi

        # download checksum
        response=$(wget --server-response --quiet ${url}.CHECKSUM 2>&1 | awk 'NR==1{print $2}')
        if [ ${response} == '404' ]; then
            echo "! File does not exist: ${url}.CHECKSUM" | tee -a "${current_directory}/logs/error-${current_date}.txt"
        else
            echo "${file_name} Downloaded .CHECKSUM" | tee -a "${current_directory}/logs/log-${current_date}.txt"
        fi

        # check if checksum matches
        checksum=$(sha256sum -c ${directory}/${file_name}.CHECKSUM | awk 'NR==1{print $2}') 
        if [ ${checksum} = 'OK' ]; then
            echo "${file_name} CHECKSUM OK" | tee -a "${current_directory}/logs/log-${current_date}.txt"
        else
            echo "! CHECKSUM ERROR! ${file_name}" | tee -a "${current_directory}/logs/error-${current_date}.txt"
        fi

        # update date
        d=$(date -I -d "$d + 1 day")

    done
done

# uncomment to download raw trades 
#
# Download raw trades for every period
# https://data.binance.vision/data/spot/daily/trades/BTCUSDT/BTCUSDT-trades-2023-03-09.zip
#
# d="$start_date"
# until [ "$d" \> "$end_date" ]; do # until d larger than end date

#     file_name="${pair}-trades-$d.zip"
#     url="${baseurl}/trades/${pair}/${file_name}"
        
#     # download zip
#     response=$(wget --server-response --quiet ${url} 2>&1 | awk 'NR==1{print $2}')
#     if [ ${response} == '404' ]; then
#         echo "! File does not exist: ${url}" | tee -a "${current_directory}/logs/error-${current_date}.txt"
#     else
#         echo "${file_name} Downloaded .zip" | tee -a "${current_directory}/logs/log-${current_date}.txt"
#     fi

#     # download checksum
#     response=$(wget --server-response --quiet ${url}.CHECKSUM 2>&1 | awk 'NR==1{print $2}')
#     if [ ${response} == '404' ]; then
#         echo "! File does not exist: ${url}.CHECKSUM" | tee -a "${current_directory}/logs/error-${current_date}.txt"
#     else
#         echo "${file_name} Downloaded .CHECKSUM" | tee -a "${current_directory}/logs/log-${current_date}.txt"
#     fi

#     # check if checksum matches
#     checksum=$(sha256sum -c ${directory}/${file_name}.CHECKSUM | awk 'NR==1{print $2}') 
#     if [ ${checksum} = 'OK' ]; then
#         echo "${file_name} CHECKSUM OK" | tee -a "${current_directory}/logs/log-${current_date}.txt"
#     else
#         echo "! CHECKSUM ERROR! ${file_name}" | tee -a "${current_directory}/logs/error-${current_date}.txt"
#     fi

#     # update date
#     d=$(date -I -d "$d + 1 day")

# done