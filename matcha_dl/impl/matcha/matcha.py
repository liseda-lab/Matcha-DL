from pathlib import Path

from matcha_dl.core.contracts.matcha import IMatcha


class Matcha(IMatcha):

    @property
    def matcha_path(self) -> Path:
        """
        Get the path to the matcha directory.

        Returns:
            Path: The path to the matcha directory.
        """
        return (Path(__file__).parent / "matcha").resolve()

    @property
    def jar_path(self) -> Path:
        """
        Get the path to the matcha.jar file.

        Returns:
            Path: The path to the matcha.jar file.
        """
        return (self.matcha_path / "matcha.jar").resolve()
