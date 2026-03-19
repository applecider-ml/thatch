# Batch Processing

THATCH includes a pipeline tracker for processing many transients on a long-running machine.

## Initialize

```bash
python -m thatch.tracker init thatch_hst_transients.csv -o tracker.parquet
```

## Run

```bash
python -m thatch.tracker run tracker.parquet --datadir data/ --max-objects 10
```

## Check Status

```bash
python -m thatch.tracker status tracker.parquet
```

The pipeline is fully resumable — kill and restart at any time.
