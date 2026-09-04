class GlyphoError(Exception):
    pass


class GlyphoNotFoundError(GlyphoError):
    pass


class RecognitionError(GlyphoError):
    pass

