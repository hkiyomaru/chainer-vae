# Variational AutoEncoder

## Paper

[Semi-supervised Learning with Deep Generative Models](http://papers.nips.cc/paper/5352-semi-supervised-learning-with-deep-generative-models.pdf)

## Environment

* OS: Ubuntu14.04
* Python: 2.7.9
* Chainer: 1.17.0

## Getting Started

Install all dependent package.

```
$ virtualenv venv
$ . venv/bin/activate
$ pip install -r requirements.txt
```

And, create dataset to train.

```
$ cd source/
$ ./data.sh
```

## Train

```
$ python train.py
```

## Test
