from typing import List, Optional, Tuple, Union

import numpy as np
from sklearn.cluster import KMeans

from drawing import Drawing

INIT = ["random", "uniform", "ctc"]


class KMeansClustering:
    """
    KMeans clustering with k clusters being the characters in the sample (k = num chars)
    """

    y_multiplier: float
    x_multiplier: float
    use_stroke_info: bool
    stroke_multiplier: float
    normalise: bool
    # Horizontal offset for centroids when CTC logits are used, because they tend to be
    # towards the beginning of a character, hence it is helpful to move them slightly
    # for the clustering to be more effective. Value is normalised based on the width.
    # on drawing width / token count
    ctc_offset: float
    init: str

    def __init__(
        self,
        x_multiplier: float = 1.0,
        y_multiplier: float = 0.04,
        use_stroke_info: bool = True,
        stroke_multiplier: float = 224,
        normalise: bool = False,
        ctc_offset: float = 0.1,
        init: str = "ctc",
    ):
        assert (
            init in INIT
        ), f"init={repr(init)} is not supported, must be one of {' | '.join(INIT)}"
        self.x_multiplier = x_multiplier
        self.y_multiplier = y_multiplier
        self.use_stroke_info = use_stroke_info
        self.stroke_multiplier = stroke_multiplier
        self.normalise = normalise
        self.ctc_offset = ctc_offset
        self.init = init

    def segment(
        self, drawing: Drawing, text: str, ctc_logits: Optional[List[int]] = None
    ) -> List[int]:
        points = drawing.all_points()

        num_chars = len(text.replace(" ", ""))

        if self.normalise:
            width = drawing.get_bounding_box().width()
            height = drawing.get_bounding_box().height()
            total_strokes = len(drawing.strokes)
        else:
            width = 1
            height = 1
            total_strokes = 1

        def create_features(
            x: float,
            y: float,
            stroke: float,
            # TODO: Figure out whether index is used at all.
            index: float,
        ) -> Union[Tuple[float, float, float, float], Tuple[float, float]]:
            x_mul = self.x_multiplier * (len(text) if self.normalise else 1) / width
            y_mul = self.y_multiplier / height
            stroke_mul = self.stroke_multiplier / total_strokes
            if self.use_stroke_info:
                return (x * x_mul, y * y_mul, stroke * stroke_mul, index)
            return (x * x_mul, y * y_mul)

        init: Union[str, np.ndarray] = "k-means++"
        centroids: List = []
        if self.init == "uniform":
            # Distribute the centroids across image, assuming all characters have same
            # width
            char_width = drawing.get_bounding_box().width() / len(text)
            index_width = len(points) / num_chars
            centroid_y = drawing.get_bounding_box().height() / 2
            stroke_width = len(drawing.strokes) / num_chars

            for index, c in enumerate(text):
                if c != " ":
                    centroids.append(
                        create_features(
                            x=index * char_width + char_width / 2,
                            y=centroid_y,
                            stroke=len(centroids) * stroke_width + stroke_width / 2,
                            index=len(centroids) * index_width + index_width / 2,
                        )
                    )
            init = np.array(centroids)
        elif self.init == "ctc":
            assert ctc_logits is not None, 'ctc_logits are required for init="ctc"'
            assert len(text) == len(
                ctc_logits
            ), "ctc_logits must have the same length as the text"

            for i in range(len(ctc_logits)):
                if text[i] != " ":
                    centroid = points[ctc_logits[i]]
                    centroids.append(
                        create_features(
                            x=centroid.x
                            + self.ctc_offset
                            * drawing.get_bounding_box().width()
                            / len(text),
                            y=centroid.y,
                            stroke=centroid.stroke,
                            index=centroid.index,
                        )
                    )
            init = np.array(centroids)
        elif self.init != "random":
            raise ValueError(f"init={repr(self.init)} is not supported")

        features = np.array(
            [
                create_features(
                    x=point.x, y=point.y, stroke=point.stroke, index=point.index
                )
                for point in points
            ]
        )
        kmeans = KMeans(
            n_clusters=num_chars,
            init=init,
            # Annoying, but needs to be set manually to 1 if the init is
            # a list of centroids, otherwise there are tons of warnings.
            n_init=10 if isinstance(init, str) else 1,
        )
        kmeans.fit(features)

        return kmeans.labels_
