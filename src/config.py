class Config(object):

    def __init__(self):
        super().__init__()

    def __repr__(self):
        return '\n'.join(['%s: %s' % (key, str(value)) for key, value in self.__dict__.items()])
