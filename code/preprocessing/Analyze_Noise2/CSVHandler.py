"""CSVIngestor class: create storms from csv file."""

from StormModel import StormModel
from RawNoiseModel import RawNoiseModel
from NoiseModel import NoiseModel
from typing import List
import pandas


class CSVHandler():
    """CSVIngestor class definition."""

    @classmethod
    def parseStorm(cls, path: str) -> List[StormModel]:
        """Parse a csv file contains storms.

        :param str: path to the csv file.
        """

        storms = []
        df = pandas.read_csv(path, header=0, low_memory=False, skiprows=[1])
        # print(df.columns.tolist()) 

        for index, row in df.iterrows():
            new_storm = StormModel(row['SID'], row['ISO_TIME'], row['LAT'], row['LON'])
            #if new_storm.season.isdigit() and new_storm.season.isdigit() and new_storm.season.isdigit() and new_storm.season.isdigit() and new_storm.season.isdigit():
            #    if int(new_storm.season) > 2007 and new_storm.isInEastSea(): #get storm in East sea from 2008
            storms.append(new_storm)

        return storms
    
    @classmethod
    def exportStorm(cls, list: List[StormModel]):
        """Export storm list to lighter csv file"""
        df = {'SID': [], 'ISO_TIME': [], 'LAT': [], 'LON': []}

        # Update DataFrame with data from storm list
        for storm in list:
            df['SID'].append(storm.sid)
            df['ISO_TIME'].append(storm.time)
            df['LAT'].append(storm.lat)
            df['LON'].append(storm.lon)

        # Write DataFrame to csv file
        pandas.DataFrame(df).to_csv("../data/storm.csv")

    @classmethod
    def parseNoise(cls, path: str) -> List[RawNoiseModel]:
        """Parse a csv file contains storms.

        :param str: path to the csv file.
        """

        noises = []
        df = pandas.read_csv(path, header=0, low_memory=False)
        # print(df.columns.tolist()) 

        for index, row in df.iterrows():
            new_storm = RawNoiseModel(row.iloc[0], row.iloc[1], row.iloc[2])
            noises.append(new_storm)

        return noises
    
    @classmethod
    def exportNoise(cls, rawList: List[RawNoiseModel], list: List[NoiseModel], path: str):
        """Export storm list to lighter csv file"""
        # df = {'STORM_FILE_PATH': [], 'POSITION': [], 'IMPACT_STORM': [], 'IMPACT_STORM_INFO': [],'PASSED': [], 'EXPECTED_VALUE': [], 'TRACKING_AREA': []}

        # df = {'STORM_FILE_PATH': [], 'POSITION': [], 'IMPACT_STORM_INFO': [],'PASSED': [], 'EXPECTED_VALUE': [], 'TRACKING_AREA': [], 'DISTANCE (degree)': [], 'DISTANCE (km)': []}

        df = {'DISTANCE(degree)': [], 'DISTANCE(km)': []}
        # Update DataFrame with data from raw list
        for raw in rawList:
            df['STORM_FILE_PATH'].append(raw.storm_filepath)
            # df['POSITION'].append(raw.position)
            # df['IMPACT_STORM'].append(raw.impactStorm)

        # Update DataFrame with data from check result list
        for result in list:
            df['IMPACT_STORM_INFO'].append(result.impactStormsStr)
            # df['PASSED'].append(result.passed)
            df['EXPECTED_VALUE'].append(result.expectNoiseFlag)
            df['TRACKING_AREA'].append(f'min_lat: {result.trackingArea["min_lat"]} - max_lat: {result.trackingArea["max_lat"]} - min_lon: {result.trackingArea["min_lon"]} - max_lat: {result.trackingArea["max_lon"]}')
            df['DISTANCE(degree)'].append(result.distanceInDegree)
            df['DISTANCE(km)'].append(result.distanceInKm)

        # Write DataFrame to csv file
        pandas.DataFrame(df).to_csv(path)