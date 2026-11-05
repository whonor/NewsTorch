"""Runtime composition of NewsTorch news and user encoders.

This model is deliberately configuration-driven: adding a new pairing of
already registered encoders does not require a dedicated model class.
"""

from __future__ import annotations

from typing import Dict, Tuple, Type

import torch.nn as nn

from config import Config
from models.modules import newsEncoders, userEncoders
from models.modules.click_predictor import DotProduct
from models.modules.nrms_variants import (
    CNNNewsEncoder,
    RNNNewsEncoder,
    RNNUserEncoder,
)


EncoderType = Type[nn.Module]

_NEWS_ENCODERS: Dict[str, EncoderType] = {
    "attention": newsEncoders.MHSA,
    "rnn": RNNNewsEncoder,
    "cnn": CNNNewsEncoder,
}

_USER_ENCODERS: Dict[str, EncoderType] = {
    "attention": userEncoders.MHSA,
    "rnn": RNNUserEncoder,
}


def _normalise_name(name: str) -> str:
    if not isinstance(name, str) or not name.strip():
        raise ValueError("Encoder names must be non-empty strings")
    return name.strip().lower().replace("_", "-")


def _register(
    registry: Dict[str, EncoderType],
    name: str,
    encoder_class: EncoderType,
    *,
    replace: bool,
) -> None:
    key = _normalise_name(name)
    if not isinstance(encoder_class, type) or not issubclass(encoder_class, nn.Module):
        raise TypeError("encoder_class must be a torch.nn.Module subclass")
    if key in registry and not replace:
        raise ValueError("Encoder %r is already registered" % key)
    registry[key] = encoder_class


def register_news_encoder(
    name: str, encoder_class: EncoderType, *, replace: bool = False
) -> None:
    """Register a news encoder for subsequent runtime compositions."""

    _register(_NEWS_ENCODERS, name, encoder_class, replace=replace)


def register_user_encoder(
    name: str, encoder_class: EncoderType, *, replace: bool = False
) -> None:
    """Register a user encoder for subsequent runtime compositions."""

    _register(_USER_ENCODERS, name, encoder_class, replace=replace)


def available_encoders() -> Dict[str, Tuple[str, ...]]:
    """Return stable, sorted names accepted by :class:`Composable`."""

    return {
        "news": tuple(sorted(_NEWS_ENCODERS)),
        "user": tuple(sorted(_USER_ENCODERS)),
    }


def _resolve_encoder(
    registry: Dict[str, EncoderType], name: str, role: str
) -> Tuple[str, EncoderType]:
    key = _normalise_name(name)
    try:
        return key, registry[key]
    except KeyError as error:
        choices = ", ".join(sorted(registry))
        raise ValueError(
            "Unknown %s encoder %r; available encoders: %s" % (role, name, choices)
        ) from error


class Composable(nn.Module):
    """Recommendation model assembled from registered encoder components."""

    def __init__(self, config: Config):
        super().__init__()
        news_name, news_encoder_class = _resolve_encoder(
            _NEWS_ENCODERS, config.news_encoder_type, "news"
        )
        user_name, user_encoder_class = _resolve_encoder(
            _USER_ENCODERS, config.user_encoder_type, "user"
        )

        self.news_encoder = news_encoder_class(config)
        if not hasattr(self.news_encoder, "news_embedding_dim"):
            raise TypeError(
                "%s must expose news_embedding_dim" % news_encoder_class.__name__
            )
        self.user_encoder = user_encoder_class(self.news_encoder, config)
        self.click_predictor = DotProduct()
        self.news_encoder_type = news_name
        self.user_encoder_type = user_name
        self.composition = "%s-%s" % (news_name, user_name)
        self.use_user_embedding = False
        self.model_name = config.model
        self.batch_size = config.batch_size
        self.config = config

    def initialize(self) -> None:
        self.news_encoder.initialize()
        self.user_encoder.initialize()

    def forward(
        self,
        user_ID,
        user_category,
        user_subCategory,
        user_title_text,
        user_title_mask,
        user_title_entity,
        user_content_text,
        user_content_mask,
        user_content_entity,
        user_history_mask,
        user_history_graph,
        user_history_category_mask,
        user_history_category_indices,
        news_category,
        news_subCategory,
        news_title_text,
        news_title_mask,
        news_title_entity,
        news_content_text,
        news_content_mask,
        news_content_entity,
    ):
        del user_ID
        user_embedding = None
        news_representation = self.news_encoder(
            news_title_text,
            news_title_mask,
            news_title_entity,
            news_content_text,
            news_content_mask,
            news_content_entity,
            news_category,
            news_subCategory,
            user_embedding,
        )
        if news_representation.dim() == 2:
            news_representation = news_representation.unsqueeze(1)

        user_representation = self.user_encoder(
            user_title_text,
            user_title_mask,
            user_title_entity,
            user_content_text,
            user_content_mask,
            user_content_entity,
            user_category,
            user_subCategory,
            user_history_mask,
            user_history_graph,
            user_history_category_mask,
            user_history_category_indices,
            user_embedding,
            news_representation,
        )
        if user_representation.dim() != 2:
            raise ValueError(
                "User encoder must return [batch, news_embedding_dim], got %r"
                % (tuple(user_representation.shape),)
            )
        if user_representation.size(1) != news_representation.size(2):
            raise ValueError(
                "Encoder dimensions do not match: user=%d, news=%d"
                % (user_representation.size(1), news_representation.size(2))
            )
        logits = self.click_predictor(
            user_representation.unsqueeze(1),
            news_representation.permute(0, 2, 1),
        )
        # NewsTorch training supplies several candidates, while dev/test
        # scores one candidate per row and expects a rank-1 score vector.
        return logits.squeeze(1) if logits.size(1) == 1 else logits
