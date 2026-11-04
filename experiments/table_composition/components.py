"""Study-local adapters; importing this module does not edit the core registry."""

from torch import nn

from models import SOTAComposition as composition_module
from models.modules import newsEncoders, userEncoders


class EncodedHistory(nn.Module):
    """Bridge the native user-encoder interface to precomputed history vectors."""

    def __init__(self, width):
        super().__init__()
        self.news_embedding_dim = width
        self.value = None

    def forward(self, *unused):
        return self.value


class NativeUserModule(nn.Module):
    """Reuse native NAML/LSTUR user encoders without copying their algorithms."""

    encoder_class = None

    def __init__(self, config, news_dim):
        super().__init__()
        self.history = EncodedHistory(news_dim)
        self.encoder = self.encoder_class(self.history, config)
        self.user_embedding = (
            nn.Embedding(config.user_num, news_dim)
            if self.encoder_class is userEncoders.LSTUR else None
        )
        self.dropout = nn.Dropout(config.dropout_rate)
        self.auxiliary_loss = None

    def initialize(self):
        self.encoder.initialize()
        if self.user_embedding is not None:
            nn.init.uniform_(self.user_embedding.weight, -0.1, 0.1)
            nn.init.zeros_(self.user_embedding.weight[0])

    def forward(self, history_embedding, candidate_embedding, history_mask,
                user_id, **unused):
        self.history.value = history_embedding
        user = None
        if self.user_embedding is not None:
            user = self.dropout(self.user_embedding(user_id))
        # Native user encoders use title_text only for its batch dimension;
        # EncodedHistory supplies the actual news representations.
        try:
            result = self.encoder(
                history_embedding, None, None, None, None, None, None, None,
                history_mask, None, None, None, user, candidate_embedding,
            )
        finally:
            self.history.value = None
        return result.unsqueeze(1)


class NAMLUserModule(NativeUserModule):
    encoder_class = userEncoders.ATT


class LSTURUserModule(NativeUserModule):
    encoder_class = userEncoders.LSTUR


def install():
    """Register reusable components in this process only."""
    composition_module._TEXT_NEWS_ENCODERS['lstur'] = newsEncoders.CNN
    composition_module._USER_MODULE_CLASSES.update(
        naml=NAMLUserModule, lstur=LSTURUserModule,
    )
