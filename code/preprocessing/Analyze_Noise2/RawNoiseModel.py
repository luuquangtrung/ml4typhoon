"""This class is used to encapsulate the attributes for storms."""

class RawNoiseModel:
    """Initialization function."""

    def __init__(self, storm_filepath, position, impactStorm):
        """Intialization function for NoiseModel.

        :param sid: .
        :param season: 
        ...
        """
        self.storm_filepath = storm_filepath.strip()
        self.position = position.strip()
        self.impactStorm = impactStorm.strip()

    def __repr__(self) -> str:
        """Represent the content of NoiseModel."""
        return f'"{self.storm_filepath}" - {self.isNoise} - {self.impactStorm} '  