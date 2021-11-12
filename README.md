# TreePointerNet
This is the code for TreePonterNet as described in our TMLR paper "Identifying Axiomatic Mathematical Transformation Steps using Tree-Structured Pointer Networks" (https://openreview.net/forum?id=gLQ801ewwp).
The model is based on the TreeTransformer from https://github.com/nxphi47/tree_transformer

# Installation

Install fairseq
```bash
# install the latest pytorch first
pip install --upgrade fairseq==0.6.2
pip install nltk

git clone https://github.com/sj-w/tree_pointer_net.git
```

# Convert Data into Binary Fairseq Format
The data must be available in a format such that it can be read by NLTKs tree parser.
To convert the data into the binary format run the following script:

```python /src/preprocess_nstack2seq_merge.py --source-lang src --target-lang tgt --user-dir . --trainpref /data/$path/train --testpref /data/$path/test --validpref /data/$path/valid --destdir /data/$path/bin --joined-dictionary --no_remove_root --workers 10 --output-format binary --no_collapse #--srcdict /data/$path/dict.src.txt```

Uncomment the last argument in case you are converting multiple files separately, i.e. reuse the dictionary created during the first run to make sure that the tokens are mapped to the same indices again.

# Training the Model
The model can be trained like any other fairseq model, e.g.
```
fairseq-train \    
 /data/$path/bin \
 --user-dir src \
 --task nstack_merge2seq \
 --arch pointer_transformer \
 --optimizer adam \
 --max-tokens 1024 \
 --criterion cross_entropy \
 --source-lang src \
 --log-interval 100 \
 --share-all-embeddings \
 --append-eos-to-target \
 --max-epoch 100 \
 --source-lang src \
 --target-lang tgt \
 --save-dir /data/checkpoints
```

# Inference
To generate predictions simply use fairseq-generate, for example:
`fairseq-generate /data/$path/bin --path /data/checkpoints/checkpoint_best.pt --gen-subset test --task nstack_merge2seq --user-dir src --append-eos-to-target --batch-size 1 --beam 1 --nbest 1`
