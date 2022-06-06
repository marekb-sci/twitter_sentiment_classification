import sys, os
sys.path.insert(0, os.path.abspath('.'))

from pytest import mark


from train import create_dataset

def test_dataset():

    filename = 'sample_data/sample_for_training.csv'

    ds = create_dataset(filename, labels=[0,1])
    assert ds[0]['text'] == 'heading home from the beach to chill with my dad '
    assert ds[0]['label'] == 1
    assert len(ds) == 11


@mark.skip("Not written")
def test_data_preprocessing():
	pass

@mark.skip("Not written")
def test_inference():
	pass