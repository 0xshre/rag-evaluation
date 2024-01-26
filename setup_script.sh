mkdir data
mkdir data/raw
mkdir data/processed

wget https://huggingface.co/datasets/wikitext/resolve/main/wikitext-2-raw-v1/test-00000-of-00001.parquet
wget https://huggingface.co/datasets/wikitext/resolve/main/wikitext-2-raw-v1/train-00000-of-00001.parquet
wget https://huggingface.co/datasets/wikitext/resolve/main/wikitext-2-raw-v1/validation-00000-of-00001.parquet

mv test-00000-of-00001.parquet data/raw/test.parquet
mv train-00000-of-00001.parquet data/raw/train.parquet
mv validation-00000-of-00001.parquet data/raw/valid.parquet