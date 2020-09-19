# Life CNN

This project is inspired by [It's Hard for Neural Networks To Learn the Game of Life](https://arxiv.org/abs/2009.01398).
Authors want to train CNN models predicting results after `n` update steps without intermidate information, but models will fail to converge on large `n` values.

Given that the rule is simple and can be described by a CNN model, this conclusion is somewhat counterintuitive.
This project is meant to figure out the reason.

## Innovations

- Add residual links
