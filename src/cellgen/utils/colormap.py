from matplotlib.colors import LinearSegmentedColormap

class CellColorMaps:
    """Color maps for different cell types."""

    NEUTROPHIL = LinearSegmentedColormap.from_list(
        "Neutrophil",
        [(0.62, 0.54, 0.51), (0.2, 0.07, 0.52)],
        N=20
    )

    CELL_BODY = LinearSegmentedColormap.from_list(
        "CellBody",
        [(0.94, 0.83, 0.73), (0.84, 0.68, 0.66)],
        N=20
    )

    MONOCYTE = LinearSegmentedColormap.from_list(
        "Monocyte",
        [(0.64, 0.55, 0.61), (0.16, 0.1, 0.53)],
        N=20
    )

    RED_BLOOD_CELL_BODY = LinearSegmentedColormap.from_list(
        "RedBloodCell",
        [(124/255, 108/255, 124/255), (195/255, 180/255, 183/255)],
        N=20
    )

class CellSolidColors:
    """Solid colors for different cells."""

    MONOCYTE_CYTOPLASM = [0.451, 0.576, 0.702]

    NEUTROPHIL_CYTOPLASM = [0.859, 0.745, 0.596]
