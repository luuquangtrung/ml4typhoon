from StormModel import StormModel
from CSVHandler import CSVHandler
from typing import List

class StormManager:
    """Initialization function."""

    def __init__(self):
        """Intialization function for StormManager.

        :param sid: .
        :param season: 
        ...
        """

        # self._storms = CSVHandler.parseStorm("./Resources/ibtracs_ALL_list_v04r00.csv")
        # CSVHandler.exportStorm(self._storms)
        self._storms = CSVHandler.parseStorm("/N/u/tqluu/BigRed200/workspace/libtcg_Dataset/data/raw/ibtracs/ibtracs.ALL.list.v04r00.csv")

    def getStormList(self) -> List[StormModel]:
        return self._storms
    
    def getStormByTime(self, time) -> StormModel:
        """Find and return a storm by its time"""
        
        for storm in self._storms:
            if storm.time == time: return storm
        return None
    
    def getStormBySid(self, sid) -> StormModel:
        """Find and return a storm by its time"""
        
        for storm in self._storms:
            if storm.sid == sid: return storm
        return None
    
    def getStormBySidAndTime(self, sid, time) -> StormModel:
        """Find and return a storm by its sid and time"""
        for storm in self._storms:
            if storm.sid == sid and storm.time == time: return storm
        return None