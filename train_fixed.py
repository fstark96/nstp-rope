import sys
class FakeProfile:
    def run(*args, **kwargs): pass
    def runctx(*args, **kwargs): pass
sys.modules['profile'] = FakeProfile()

# Now import the training script
exec(open('train_wikitext.py').read())
