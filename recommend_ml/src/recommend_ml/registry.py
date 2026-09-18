"""モデル名 -> インスタンス生成の対応表。

CLI から `--models random,most_popular` のように指定するための入口。
手法を足すときはここに 1 行足す。
"""

from __future__ import annotations

from collections.abc import Callable

from .models.base import Recommender
from .models.baseline import MostPopularRecommender, RandomRecommender
from .models.knn import ItemKNNRecommender, UserKNNRecommender
from .models.linear import EASERecommender
from .models.mf import ALSRecommender, BPRRecommender

MODEL_REGISTRY: dict[str, Callable[..., Recommender]] = {
    "random": RandomRecommender,
    "most_popular": MostPopularRecommender,
    "item_knn": ItemKNNRecommender,
    "user_knn": UserKNNRecommender,
    "als": ALSRecommender,
    "bpr": BPRRecommender,
    "ease": EASERecommender,
}


def build_model(name: str, **kwargs: object) -> Recommender:
    if name not in MODEL_REGISTRY:
        known = ", ".join(sorted(MODEL_REGISTRY))
        raise KeyError(f"unknown model: {name} (known: {known})")
    return MODEL_REGISTRY[name](**kwargs)
